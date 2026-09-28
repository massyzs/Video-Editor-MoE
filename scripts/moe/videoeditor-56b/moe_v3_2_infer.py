#!/usr/bin/env python
"""videoeditor-56b inference (benchmark + single video): the 56B DiT is split by block across two GPUs. Self-contained.

  1. Multi-GPU split: 56B in bf16 = 104 GiB does not fit on one 80 GB GPU. With --gpus 2 the DiT is split by block index into
     two shards on cuda:0 / cuda:1 (every block and the head of the second shard get a device-moving pre-hook, because the
     top-level forward rebuilds kwargs inside the loop; only x is passed on from the previous block). MLLM on cuda:1, VAE on cuda:0.
     Split point computed from the checkpoint header: 7.1 GiB per MoE block, 0.3 GiB per dense block, split=15 -> ~58 / 63 GiB.
  2. Load once, many variants x many benches: --variants 'label:cfg:frac:shift,...' and --benches a,b,c; the model is instantiated
     only once (no repeated loading). Per case, the MLLM encoding / VAE encoding / initial noise are computed once and shared by all variants.
  3. Resume: with --skip_existing 1 a job is skipped when its mp4 size/frame count/fps match the arguments and all 49 PNGs exist (bench_output_check.output_exists).
  4. Fast construction: torch.nn.init random initialisation is a no-op during construction (single-threaded xavier over 55.9B parameters
     takes ~10 min); then assert checkpoint keys ⊇ all parameters and overwrite everything. expert_copy_noise is fixed to 0 (no noise at
     inference; with 0, build_dit skips randn and copies the source weights directly).
     --experts_merged defaults to "": build_dit reads no merged-expert file and skips the source-weight self-check; all experts come
     from --ckpt (only the released files are needed).
  5. Single video: --src_video <mp4> --prompt "<instruction>" [--name <out name>] --out_dir <dir> (same frame sampling / naming / CFG handling as the videoeditor-14b inference script).

Usage (2 GPUs with >= 60 GiB free each; <BASE_DIR> contains ckpt/kiwi-edit-instruct-only-videoxfun/, <CKPT_DIR> is the released
videoeditor-56b directory with moe_v3_2_trainable.safetensors + meta.json):
  Single video:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/moe/videoeditor-56b/moe_v3_2_infer.py --base_dir <BASE_DIR> --ckpt <CKPT_DIR> \
        --src_video <input.mp4> --prompt "Remove the dog" --name dog_removed --out_dir <OUT_DIR>
    (multi-step instruction: --prompts "Remove the dog ||| Convert to oil painting style", joined with " Then ";
     CFG / sigma shift: --cfg_text_scale 3 --cfg_text_steps_frac 0.6 --sigma_shift 5 (defaults), or --variants as below)
  Benchmark (each <BENCH_ROOT>/<bench>/cases.json = [{name, src, prompts[]}], src relative to <BENCH_ROOT>/<bench>/):
    CUDA_VISIBLE_DEVICES=0,1 python scripts/moe/videoeditor-56b/moe_v3_2_infer.py --base_dir <BASE_DIR> --ckpt <CKPT_DIR> \
        --bench_root <BENCH_ROOT> --benches openve_bench,coinve_bench_sub,covebench_sub \
        --out_root <OUT_DIR> --variants 'seg_shift5:3:0.6:5,full_shift5:3:1.0:5,seg_shift3:3:0.6:3,full_shift3:3:1.0:3' \
        --gpus 2 --shard_idx 0 --num_shards 4
Output: bench = <out_root>/<bench>/<label>/<case>.mp4; single = <out_dir>/[<label>/]<name>.mp4;
        each with frames/<case>/f000..f048.png + <case>_experts_{attn,ffn}.png + manifest_shard{k}.jsonl
Exit codes: 17 non-finite weights / 18 MLLM self-test failed / 19 degenerate output -- meant for an outer launcher to retry (a retry restarts the whole process).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from moe_v3_2_model import TextEncoder, build_dit, build_mllm, build_vae, load_trainable_weights   # noqa: E402
from videox_fun.moe_common.bench_output_check import output_exists                                                       # noqa: E402

TAG = "[infer-v3.2]"
BENCH_DEFAULT_VARIANTS = "seg_shift5:3:0.6:5,full_shift5:3:1.0:5,seg_shift3:3:0.6:3,full_shift3:3:1.0:3"
COMPOUND_JOINER = " Then "
SOURCE_HUES = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212]], dtype=np.float32)
EXPERT_COLORS = None


def expert_palette(num_experts, copies):
    cols = []
    for j in range(num_experts):
        src, c = j // copies, j % copies
        base = SOURCE_HUES[src % len(SOURCE_HUES)]
        t = 0.0 if copies <= 1 else 0.55 * c / (copies - 1)
        cols.append(base * (1 - t) + 255 * t)
    return np.clip(np.array(cols), 0, 255).astype(np.uint8)


def load_video_frames_first_n(path, num_frames, width, height):
    """Same frame sampling as training: first n frames (trimmed to 4k+1), padded with the last frame; geometry = cover resize (short side fits) + center crop."""
    import decord
    from PIL import Image
    vr = decord.VideoReader(path)
    n = min(num_frames, len(vr))
    while n > 1 and n % 4 != 1:
        n -= 1
    arrs = vr.get_batch(list(range(n))).asnumpy()
    h0, w0 = arrs.shape[1:3]
    scale = max(height / h0, width / w0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    left, top = (nw - width) // 2, (nh - height) // 2
    frames = [Image.fromarray(x).convert("RGB").resize((nw, nh), Image.BILINEAR).crop((left, top, left + width, top + height)) for x in arrs]
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="directory containing moe_v3_2_trainable.safetensors + meta.json")
    p.add_argument("--bench_root", default="", help="bench mode: each <bench_root>/<bench>/cases.json lists the cases (src relative to <bench_root>/<bench>/)")
    p.add_argument("--benches", default="openve_bench,coinve_bench_sub,covebench_sub")
    p.add_argument("--out_root", default="", help="bench-mode output root, organised as <bench>/<variant>/ (defaults to --out_dir)")
    p.add_argument("--variants", default=None,
                   help="comma-separated 'label:cfg_text_scale:cfg_text_steps_frac:sigma_shift'; if omitted, bench mode = " + BENCH_DEFAULT_VARIANTS
                        + ", single mode = one run with --cfg_text_scale/--cfg_text_steps_frac/--sigma_shift, written directly to --out_dir")
    # Single-video mode: with --src_video only that video is run and no bench cases.json is read
    p.add_argument("--src_video", default="", help="source video path; if given, runs in single-video mode")
    p.add_argument("--prompt", default="", help="full editing instruction (write a compound instruction as one sentence, or split it with --prompts)")
    p.add_argument("--prompts", default="", help="sub-instructions separated by |||, joined into one sentence with \" Then \" (same joining as the CoinVE training data)")
    p.add_argument("--name", default="single", help="output file name in single mode (without extension)")
    p.add_argument("--out_dir", default="", help="output directory in single mode (required); can replace --out_root in bench mode")
    p.add_argument("--sigma_shift", type=float, default=5.0, help="used in single mode when --variants is not given")
    p.add_argument("--cfg_text_scale", type=float, default=3.0, help="text CFG scale s: pred = u + s*(c-u); 1 = off (used in single mode when --variants is not given)")
    p.add_argument("--cfg_text_steps_frac", type=float, default=0.6, help="segmented CFG: text CFG only for the first frac of the denoising steps; 1.0 = all steps (used in single mode when --variants is not given)")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--gpus", type=int, default=2, help="1 = whole DiT on one GPU (rejected for 56B); 2 = split into two shards by block")
    p.add_argument("--split_block", type=int, default=15, help="with --gpus 2: index of the first block of the second shard")
    p.add_argument("--base_dir", required=True, help="<BASE_DIR>: contains ckpt/kiwi-edit-instruct-only-videoxfun/ (base DiT, VAE, qwen/, moe_expert_init.safetensors)")
    p.add_argument("--config_path", default="config/wan2.2/wan_civitai_5b.yaml", help="relative to the repository root")
    p.add_argument("--experts_merged", default="", help="merged-expert init file (training only); \"\" = experts come entirely from --ckpt")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--dump_expert_maps", type=int, default=1)
    p.add_argument("--frames_png", type=int, default=1)
    p.add_argument("--skip_existing", type=int, default=1)
    p.add_argument("--fast_init", type=int, default=1, help="1 = skip random init during construction (the checkpoint overwrites everything; asserted)")
    p.add_argument("--min_free_gib", type=float, default=60.0)
    return p.parse_args()


def parse_variants(spec):
    out = []
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        assert len(parts) == 4, f"each --variants item must be label:cfg_scale:cfg_frac:sigma_shift, got {item!r}"
        label = parts[0].strip()
        assert label and "/" not in label and label not in (".", ".."), f"invalid variant label: {label!r}"
        out.append(dict(label=label, cfg_text_scale=float(parts[1]), cfg_text_steps_frac=float(parts[2]), sigma_shift=float(parts[3])))
    assert out and len({v["label"] for v in out}) == len(out), "--variants is empty or has duplicate labels"
    return out


def variant_out_dir(base, label):
    """Single mode: the anonymous variant (label '') writes to base, others to base/<label> (same as the videoeditor-14b inference script)."""
    return os.path.join(base, label) if label else base


class ExpertMapRecorder:
    """Hooks the self-attention and FFN of the given MoE layers; records each token's top-1 expert for one forward pass. Keys ("attn"|"ffn", layer)."""

    def __init__(self, dit, layer_indices):
        self.maps, self.handles = {}, []
        for i in layer_indices:
            for kind, module in (("attn", dit.blocks[i].self_attn), ("ffn", dit.blocks[i].ffn)):
                def hook(mod, inputs, output, _key=(kind, i)):
                    tokens = inputs[0].reshape(-1, inputs[0].shape[-1])
                    scores = torch.softmax(torch.nn.functional.linear(tokens.float(), mod.gate.weight.float()), dim=-1)
                    self.maps[_key] = scores.argmax(dim=-1).cpu()
                self.handles.append(module.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


def save_frames_png(video, out_dir):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    x = (video[0].clamp(0, 1).permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8)
    for i, fr in enumerate(x):
        Image.fromarray(fr).save(os.path.join(out_dir, f"f{i:03d}.png"), compress_level=6)
    return out_dir


def save_expert_map_png(maps, grid, frame_indices, path, patch=16):
    from PIL import Image
    f_p, h_p, w_p = grid
    rows = []
    for layer, m in sorted(maps.items()):
        m = m.view(f_p, h_p, w_p).numpy()
        cols = [np.kron(EXPERT_COLORS[m[fi]], np.ones((patch, patch, 1), dtype=np.uint8)) for fi in frame_indices]
        rows.append(np.concatenate(cols, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


# ------------------------------------------------------------------ model construction + split
def build_dit_fast(ns, dtype, ckpt_file, fast_init):
    """No-op random init during construction, assert the checkpoint covers every parameter, then load it all."""
    from safetensors import safe_open
    t0 = time.time()
    saved = {}
    if fast_init:
        import torch.nn.init as _init
        for fn in ("xavier_uniform_", "xavier_normal_", "normal_", "kaiming_uniform_", "kaiming_normal_", "uniform_", "trunc_normal_"):
            if hasattr(_init, fn):
                saved[fn] = getattr(_init, fn)
                setattr(_init, fn, lambda t, *aa, **kk: t)
    try:
        dit = build_dit(ns, dtype=dtype, verbose=True)
    finally:
        import torch.nn.init as _init
        for fn, f in saved.items():
            setattr(_init, fn, f)
    print(f"{TAG} build_dit {time.time() - t0:.0f}s (fast_init={fast_init})", flush=True)
    with safe_open(ckpt_file, "pt") as f:
        ck_keys = set(f.keys())
    p_names = set(n for n, _ in dit.named_parameters())
    uncovered = sorted(p_names - ck_keys)
    assert not uncovered, f"checkpoint does not cover {len(uncovered)} parameters (they would keep their construction-time values): {uncovered[:5]}"
    print(f"{TAG} coverage check: checkpoint {len(ck_keys)} keys ⊇ model {len(p_names)} parameters", flush=True)
    t1 = time.time()
    load_trainable_weights(dit, ckpt_file)
    print(f"{TAG} weights loaded in {time.time() - t1:.0f}s", flush=True)
    return dit


def _to_dev(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_dev(o, dev) for o in obj)
    if isinstance(obj, dict):
        return {k: _to_dev(v, dev) for k, v in obj.items()}
    return obj


def _mk_move(dev):
    def hook(mod, args, kwargs):
        return _to_dev(args, dev), _to_dev(kwargs, dev)
    return hook


def place_dit(dit, gpus, split):
    """gpus=1: everything on cuda:0 (rejected for 56B; only for small models / self-tests). gpus=2: blocks[:split] + other submodules
    -> cuda:0, blocks[split:] + head -> cuda:1. Every second-shard block and the head get a device-moving hook (the top-level forward
    rebuilds kwargs for each block; only x is chained)."""
    d0 = "cuda:0"
    if gpus == 1:
        n_par = sum(p.numel() for p in dit.parameters())
        total = torch.cuda.get_device_properties(0).total_memory
        if n_par * 2 > 0.92 * total:
            raise RuntimeError(f"{n_par / 1e9:.1f}B × 2 B = {n_par * 2 / 2**30:.0f} GiB does not fit in {total / 2**30:.0f} GiB, use --gpus 2")
        return dit.to(d0), d0, d0
    d1 = "cuda:1"
    for name, mod in dit.named_children():
        if name != "blocks":
            mod.to(d0)
    for i, b in enumerate(dit.blocks):
        b.to(d1 if i >= split else d0)
    dit.head.to(d1)
    for bn, buf in list(dit.named_buffers(recurse=False)):
        setattr(dit, bn, buf.to(d0))
    for i in range(split, len(dit.blocks)):
        dit.blocks[i].register_forward_pre_hook(_mk_move(d1), with_kwargs=True)
    dit.head.register_forward_pre_hook(_mk_move(d1), with_kwargs=True)
    for i in (0, 1):
        free, total = torch.cuda.mem_get_info(i)
        print(f"{TAG} cuda:{i} allocated {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB, free {free / 2**30:.1f} / {total / 2**30:.0f} GiB", flush=True)
    return dit, d0, d1


# ------------------------------------------------------------------ per case: prepare once, reuse for every variant
@torch.no_grad()
def prepare_case(dit, text_encoder, vae, bench_dir, case, a, device, dtype, need_null):
    from videox_fun.data.kiwi_video_utils import frames_to_video_tensor
    src_frames = load_video_frames_first_n(os.path.join(bench_dir, case["src"]), a.num_frames, a.width, a.height)
    prompt = COMPOUND_JOINER.join(case["prompts"])
    ctx_cond = [text_encoder.encode(prompt, src_frames).to(device)]           # MLLM may be on cuda:1; move back to the input device
    ctx_null = [text_encoder.encode_null(src_frames).to(device)] if need_null else None
    src_latents = vae.encode(frames_to_video_tensor(src_frames, dtype=torch.float32).to(device, dtype))[0].sample()
    latent_channels = getattr(vae.config, "z_dim", None) or vae.config.latent_channels
    f_lat, h_lat, w_lat = (a.num_frames - 1) // 4 + 1, a.height // 16, a.width // 16
    generator = torch.Generator("cpu").manual_seed(a.seed)
    latents0 = torch.randn(1, latent_channels, f_lat, h_lat, w_lat, generator=generator, dtype=torch.float32).to(device, dtype)
    patch = dit.patch_size
    grid = (f_lat // patch[0], h_lat // patch[1], w_lat // patch[2])
    return dict(ctx_cond=ctx_cond, ctx_null=ctx_null, src_latents=src_latents, latents0=latents0, grid=grid,
                seq_len=grid[0] * grid[1] * grid[2])


@torch.no_grad()
def sample_case(dit, vae, prep, case, v, a, device, dtype, expert_layers):
    from videox_fun.data.kiwi_flow_match_scheduler import FlowMatchScheduler
    from tqdm import tqdm
    ctx_cond, src_latents, grid, seq_len = prep["ctx_cond"], prep["src_latents"], prep["grid"], prep["seq_len"]
    ctx_null = prep["ctx_null"] if v["cfg_text_scale"] != 1.0 else None
    assert not (v["cfg_text_scale"] != 1.0 and prep["ctx_null"] is None), "variant needs CFG but prepare_case did not compute the unconditional branch"
    latents = prep["latents0"].clone()
    scheduler = FlowMatchScheduler(shift=v["sigma_shift"], sigma_min=0.003 / 1.002, extra_one_step=True)
    scheduler.set_timesteps(a.steps)
    sigmas, timesteps = scheduler.sigmas.to(device), scheduler.timesteps.to(device)
    expert_maps = None
    n_cfg = int(round(v["cfg_text_steps_frac"] * len(timesteps)))
    for i in tqdm(range(len(timesteps)), desc=f"{case['name'][:24]}:{v['label']}", dynamic_ncols=True, leave=False):
        t = timesteps[i].reshape(1).to(device=device, dtype=dtype)
        sigma = sigmas[i].to(device=device, dtype=dtype).reshape(1)
        recorder = ExpertMapRecorder(dit, expert_layers) if (a.dump_expert_maps and i == len(timesteps) // 2) else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_cond = dit(x=[latents[0]], t=t, mllm_context=ctx_cond, seq_len=seq_len, vae_source_input=src_latents, sigma=sigma).to(device)
            if recorder is not None:
                expert_maps = dict(recorder.maps)
                recorder.remove()
            if ctx_null is not None and i < n_cfg:
                pred_uncond = dit(x=[latents[0]], t=t, mllm_context=ctx_null, seq_len=seq_len, vae_source_input=src_latents, sigma=sigma).to(device)
                pred = pred_uncond + v["cfg_text_scale"] * (pred_cond - pred_uncond)
            else:
                pred = pred_cond
        pred = pred.to(dtype)
        sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else torch.zeros_like(sigmas[i])
        latents = latents + pred * (sigma_next.to(dtype) - sigma)
    out = vae.decode(latents.to(vae.dtype))
    video = out.sample if hasattr(out, "sample") else out[0]
    return (video / 2 + 0.5).clamp(0, 1).float().cpu(), expert_maps, grid


# ------------------------------------------------------------------ main
def main():
    a = parse_args()
    dtype = torch.bfloat16
    for _k in ("ckpt", "base_dir", "bench_root", "out_root", "out_dir", "src_video"):   # resolve user paths against the caller's cwd (we chdir to the repo root below)
        if getattr(a, _k):
            setattr(a, _k, os.path.abspath(getattr(a, _k)))
    os.chdir(_repo)
    assert a.gpus in (1, 2)
    single = bool(a.src_video)
    if single:                                                        # single mode (same as the videoeditor-14b inference script)
        prompts = [x.strip() for x in a.prompts.split("|||") if x.strip()] if a.prompts else [a.prompt.strip()]
        assert prompts and prompts[0], "single mode needs --prompt or --prompts"
        assert os.path.isfile(a.src_video), f"--src_video not found: {a.src_video}"
        assert a.out_dir, "single mode needs --out_dir"
        assert a.name and "/" not in a.name, f"invalid --name: {a.name!r}"
        variants = parse_variants(a.variants) if a.variants else [dict(
            label="", cfg_text_scale=float(a.cfg_text_scale), cfg_text_steps_frac=float(a.cfg_text_steps_frac), sigma_shift=float(a.sigma_shift))]
        groups = [None]                                               # single mode has no bench groups
        out_root = ""
        all_cases = [(None, {"name": a.name, "src": a.src_video, "prompts": prompts})]
    else:
        out_root = a.out_root or a.out_dir
        assert a.bench_root and out_root, "bench mode needs --bench_root and --out_root (or --out_dir); for a single video use --src_video/--prompt/--out_dir"
        variants = parse_variants(a.variants if a.variants is not None else BENCH_DEFAULT_VARIANTS)
        groups = [b.strip() for b in a.benches.split(",") if b.strip()]
        all_cases = []
        for b in groups:
            for c in json.load(open(os.path.join(a.bench_root, b, "cases.json"))):
                all_cases.append((b, c))

    def vdir_of(b, v):
        return variant_out_dir(a.out_dir, v["label"]) if b is None else os.path.join(out_root, b, v["label"])

    def src_root_of(b):
        return "" if b is None else os.path.join(a.bench_root, b)    # in single mode src is an absolute path

    assert torch.cuda.device_count() >= a.gpus, f"visible GPUs {torch.cuda.device_count()} < --gpus {a.gpus}"
    # create a CUDA context on every GPU first (the GPU then shows as in use), and check free memory
    holds = []
    for i in range(a.gpus):
        holds.append(torch.zeros(1, device=f"cuda:{i}"))
        free, total = torch.cuda.mem_get_info(i)
        print(f"{TAG} cuda:{i} context created, free {free / 2**30:.1f} / {total / 2**30:.1f} GiB", flush=True)
        if free < a.min_free_gib * 2**30:
            raise SystemExit(f"{TAG} cuda:{i} free memory {free / 2**30:.1f} GiB < {a.min_free_gib}; another process is holding GPU memory")
    from videox_fun.utils.utils import save_videos_grid

    meta = json.load(open(os.path.join(a.ckpt, "meta.json")))
    global EXPERT_COLORS
    EXPERT_COLORS = expert_palette(int(meta["moe_num_experts"]), int(meta.get("expert_copies", 5)))
    ns = SimpleNamespace(
        config_path=a.config_path,
        pretrained_wan_path=os.path.join(a.base_dir, "ckpt/kiwi-edit-instruct-only-videoxfun"),
        mllm_model=os.path.join(a.base_dir, "ckpt/kiwi-edit-instruct-only-videoxfun/qwen"),
        init_weights=os.path.join(a.base_dir, "ckpt/kiwi-edit-instruct-only-videoxfun/moe_expert_init.safetensors"),
        experts_merged=a.experts_merged, use_gradient_checkpointing=0,
        moe_layers=meta["moe_layers"], moe_num_experts=meta["moe_num_experts"], moe_top_k=meta["moe_top_k"],
        moe_aux_loss_alpha=0.0,
        expert_sources=",".join(meta.get("expert_sources", [])), expert_copies=int(meta.get("expert_copies", 5)),
        expert_copy_noise=0.0,      # no noise at inference (the ckpt overwrites every expert; with 0, build_dit copies source weights without randn)
        mllm_attn_impl="sdpa")
    print(f"{TAG} shard {a.shard_idx}/{a.num_shards} ckpt step {meta.get('step')} gpus={a.gpus} split={a.split_block} "
          f"variants {[(v['label'] or '<single>', v['cfg_text_scale'], v['cfg_text_steps_frac'], v['sigma_shift']) for v in variants]} "
          f"{'single ' + a.src_video if single else 'benches ' + str(groups)}", flush=True)

    # ---- enumerate this shard's jobs first; if every output exists, exit without building the model ----
    my_cases = all_cases[a.shard_idx::a.num_shards]
    for b in groups:
        for v in variants:
            os.makedirs(vdir_of(b, v), exist_ok=True)

    def pending_of(b, c):
        return [v for v in variants if not (a.skip_existing and output_exists(
            vdir_of(b, v), c["name"], a.frames_png, a.num_frames, a.width, a.height, a.fps))]

    todo = [(b, c, pending_of(b, c)) for b, c in my_cases]
    n_render = sum(len(p) for _, _, p in todo)
    print(f"{TAG} {len(all_cases)} cases total, {len(my_cases)} in this shard, {n_render} to render (skipping {len(my_cases) * len(variants) - n_render} already done)", flush=True)
    if n_render == 0:
        print(f"{TAG} shard {a.shard_idx} nothing to do, ALL DONE", flush=True)
        return

    # ---- build the model (once) ----
    t_build = time.time()
    dit = build_dit_fast(ns, dtype, os.path.join(a.ckpt, "moe_v3_2_trainable.safetensors"), a.fast_init)
    n_par = sum(p.numel() for p in dit.parameters())
    bad = [n for n, p in dit.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        print(f"{TAG} ⚠ {len(bad)} non-finite weights, e.g. {bad[:3]}, exit 17", flush=True)
        sys.exit(17)
    dit = dit.eval()
    dit.requires_grad_(False)
    dit, device, mllm_dev = place_dit(dit, a.gpus, a.split_block)
    print(f"{TAG} DiT {n_par / 1e9:.2f}B on GPU, build+load+split took {time.time() - t_build:.0f}s", flush=True)

    from PIL import Image as _Image
    _probe = [_Image.new("RGB", (a.width, a.height))] * a.num_frames
    for _attempt in range(1, 4):
        text_encoder = TextEncoder(build_mllm(ns).to(mllm_dev))
        _where = text_encoder.self_test("Add a red ball.", _probe)
        if _where is None:
            print(f"{TAG} MLLM self-test passed (attempt {_attempt}, on {mllm_dev})", flush=True); break
        print(f"{TAG} MLLM self-test failed [{_where}], attempt {_attempt}, rebuilding", flush=True)
        del text_encoder; import gc; gc.collect(); torch.cuda.empty_cache()
    else:
        print(f"{TAG} MLLM self-test failed 3 times in a row, exit 18", flush=True); sys.exit(18)
    vae = build_vae(ns, dtype).to(device).eval()
    for i in range(a.gpus):
        free, total = torch.cuda.mem_get_info(i)
        print(f"{TAG} all models ready: cuda:{i} free {free / 2**30:.1f} / {total / 2**30:.0f} GiB", flush=True)
    moe_layers = list(dit.moe_layers)
    expert_layers = [moe_layers[0], moe_layers[len(moe_layers) // 2], moe_layers[-1]] if moe_layers else []
    print(f"{TAG} moe_layers={moe_layers} top_k={dit.blocks[moe_layers[0]].ffn.top_k if moe_layers else '-'} expert-map layers {expert_layers}", flush=True)

    # ---- render ----
    n_done = 0
    t_start = time.time()
    for ci, (b, case, pend) in enumerate(todo):
        where = case["name"] if b is None else f"{b}/{case['name']}"
        if not pend:
            print(f"{TAG} skip {ci + 1}/{len(todo)} {where}: all variants already have outputs", flush=True)
            continue
        t_prep = time.time()
        prep = prepare_case(dit, text_encoder, vae, src_root_of(b), case, a, device, dtype,
                            need_null=any(v["cfg_text_scale"] != 1.0 for v in pend))
        prep_s = round(time.time() - t_prep, 1)
        for v in pend:
            t0 = time.time()
            vdir = vdir_of(b, v)
            out_mp4 = os.path.join(vdir, case["name"] + ".mp4")
            video, expert_maps, grid = sample_case(dit, vae, prep, case, v, a, device, dtype, expert_layers)
            if not torch.isfinite(video).all() or float(video.std()) < 1e-3:
                print(f"{TAG} ⚠ {where}[{v['label']}] abnormal output (finite={bool(torch.isfinite(video).all())}, std={float(video.std()):.4g}), exit 19", flush=True)
                sys.exit(19)
            save_videos_grid(video, out_mp4, fps=a.fps)
            frames_dir = save_frames_png(video, os.path.join(vdir, "frames", case["name"])) if a.frames_png else None
            rec = {"name": case["name"], "bench": b, "src": case["src"], "prompts": case["prompts"], "out": out_mp4, "ckpt": a.ckpt,
                   "ckpt_step": meta.get("step"), "steps": a.steps, "cfg": f"text_x{v['cfg_text_scale']}_first{v['cfg_text_steps_frac']}",
                   "sigma_shift": v["sigma_shift"], "variant_label": v["label"], "prep_s": prep_s, "seed": a.seed,
                   "elapsed_s": round(time.time() - t0, 1), "frames_dir": frames_dir, "gpus": a.gpus, "split_block": a.split_block}
            if expert_maps:
                for kind in ("attn", "ffn"):
                    maps_k = {layer: m for (k_, layer), m in expert_maps.items() if k_ == kind}
                    png = os.path.join(vdir, case["name"] + f"_experts_{kind}.png")
                    save_expert_map_png(maps_k, grid, [0, grid[0] // 2, grid[0] - 1], png)
                    rec[f"expert_map_{kind}"] = png
                    rec[f"expert_usage_{kind}"] = {str(layer): np.bincount(m.numpy(), minlength=meta["moe_num_experts"]).tolist() for layer, m in maps_k.items()}
            with open(os.path.join(vdir, f"manifest_shard{a.shard_idx}.jsonl"), "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            el = time.time() - t_start
            print(f"{TAG} shard {a.shard_idx} [{n_done}/{n_render}] {where}[{v['label'] or '-'}] {rec['elapsed_s']}s -> {out_mp4} "
                  f"(avg {el / n_done:.0f}s/item, ETA {(n_render - n_done) * el / n_done / 60:.0f} min)", flush=True)
    print(f"{TAG} shard {a.shard_idx} ALL DONE: {n_done} items, {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
