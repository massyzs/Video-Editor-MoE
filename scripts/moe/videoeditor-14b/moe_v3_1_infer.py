#!/usr/bin/env python
"""videoeditor-14b inference: runs bench cases (or a single video) with text CFG, and can plot which expert each token picked (one map for the self-attention gate, one for the FFN gate).

  - Model: build_dit_fast = build_dit (structure; random init is skipped during construction; with --experts_merged empty no merged file is read)
          → assert the checkpoint covers every parameter → load_trainable_weights(<ckpt>/moe_v3_1_trainable.safetensors, the whole DiT); MLLM / VAE frozen
  - Instruction = the full instruction (bench sub-instructions joined with " Then "); unconditional branch = empty instruction
  - Sampling = flow-match Euler, shift 5, 50 steps, pred = u + s·(c − u)
  - Source video: first n frames (trimmed to 4k+1), padded with the last frame, same as training
  - --dump_expert_maps: per case also save two PNGs (_experts_attn.png / _experts_ffn.png): a few MoE layers × a few frames, each patch colored by its top-1 expert (at the middle sampling step)
    Color order = expert order: 0 main red / 1 style green / 2 add yellow / 3 remove blue / 4 replace orange / 5 background purple / 6 direct_finetune cyan
Usage (<BASE_DIR> must contain ckpt/kiwi-edit-instruct-only-videoxfun/; <CKPT_DIR> = the released weight directory with moe_v3_1_trainable.safetensors + meta.json):
  Single video:           CUDA_VISIBLE_DEVICES=0 python moe_v3_1_infer.py --base_dir <BASE_DIR> --ckpt <CKPT_DIR> --src_video <video.mp4> --prompt "Remove the dog" --name dog --out_dir <OUT_DIR>
  Compound instruction:   ... --src_video a.mp4 --prompts "Remove the dog ||| Convert to oil painting style"   (joined into one sentence with " Then ")
  Bench (sharded):        CUDA_VISIBLE_DEVICES=0 python moe_v3_1_infer.py --base_dir <BASE_DIR> --ckpt <CKPT_DIR> --bench_dir <BENCH_DIR with cases.json> --shard_idx 0 --num_shards 15
  Custom case list:       ... --cases_json my_cases.json [--bench_dir <video root dir; omitted = the directory of cases_json>]
  Output dir:             if --out_dir is omitted: <run>/render/<ckpt dir name>/ (convention: each run dir keeps weights in ckpt/, inference results in render/)
"""
import argparse
import json
import math
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

from moe_v3_1_model import TextEncoder, build_dit, build_mllm, build_vae, load_trainable_weights   # noqa: E402

COMPOUND_JOINER = " Then "
EXPERT_COLORS = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                          [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212]], dtype=np.uint8)


def load_video_frames_first_n(path, num_frames, width, height):
    """Frame sampling as in training: first n frames (trimmed to 4k+1), padded with the last frame; geometry = cover resize (short side fits) + center crop."""
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
    p.add_argument("--ckpt", required=True, help="Training ckpt directory (contains moe_v3_1_trainable.safetensors and meta.json)")
    p.add_argument("--bench_dir", default="", help="Bench mode: directory with cases.json and the source videos (relative case src paths are rooted at it); not needed in single-video mode (--src_video)")
    p.add_argument("--out_dir", default="", help="Output directory; omitted = <run>/render/<ckpt dir name>/ (convention: weights in ckpt/, results in render/)")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--base_dir", required=True, help="<BASE_DIR>: its ckpt/kiwi-edit-instruct-only-videoxfun/ holds the base model (config.json, diffusion_pytorch_model.safetensors, moe_expert_init.safetensors, Wan2.2_VAE.pth, qwen/)")
    p.add_argument("--config_path", default="config/wan2.2/wan_civitai_5b.yaml", help="Relative to the repository root")
    p.add_argument("--experts_merged", default="", help="Empty = use only the released checkpoint (all expert weights come from --ckpt); a path = first load the merged initial weights from merge_experts_v3_1.py (same as the training start), then overwrite with --ckpt")
    p.add_argument("--fast_init", type=int, default=1, help="1 = skip random init while constructing the DiT (the checkpoint overwrites every parameter; asserted)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--cfg_text_scale", type=float, default=3.0, help="Text CFG scale s: pred = u + s*(c-u); 1 = off")
    p.add_argument("--cfg_text_steps_frac", type=float, default=0.6, help="Segmented CFG: apply text CFG only in the first frac of denoising steps (sets structure), then turn it off (restores detail, less drift from the source video); 1.0 = all steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_free_gib", type=float, default=60.0, help="refuse to start if the GPU has less free memory than this (GiB)")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--dump_expert_maps", type=int, default=1)
    p.add_argument("--base_only", type=int, default=0, help="1 = the un-finetuned kiwi-edit base (no MoE layers, no trained weights, rest of the pipeline unchanged); --ckpt only needs a placeholder meta.json")
    # Single-video mode: with --src_video only that video is run and the bench cases.json is not read
    p.add_argument("--src_video", default="", help="Source video path; giving it enables single-video mode")
    p.add_argument("--prompt", default="", help="Full edit instruction (write a compound instruction as one sentence, or split it with --prompts)")
    p.add_argument("--prompts", default="", help="Sub-instructions separated by |||, joined into one sentence with \" Then \" (same joining as CoinVE in training)")
    p.add_argument("--name", default="single", help="Output file name in single-video mode (without extension)")
    p.add_argument("--cases_json", default="", help="Custom case list json (same format as the bench cases.json: [{name, src, prompts[]}]); relative paths are rooted at --bench_dir")
    p.add_argument("--frames_png", type=int, default=1, help="1 = also save every output frame as a lossless PNG under <out_dir>/frames/<case>/ (for figures), 0 = mp4 only")
    # Load the model once per process and run several CFG/shift variants (no repeated model loading)
    p.add_argument("--variants", default="", help="'label:cfg_text_scale:cfg_text_steps_frac:sigma_shift', comma-separated, e.g. "
                   "'seg_shift5:3:0.6:5,full_shift5:3:1.0:5'; each variant writes to <out_dir>/<label>/; empty = a single run with --cfg_text_scale/--cfg_text_steps_frac/--sigma_shift, written to <out_dir>")
    p.add_argument("--skip_existing", type=int, default=1, help="1 = skip when video size, frame count and fps match the args and all PNG frames are present")
    return p.parse_args()


def parse_variants(spec, a):
    """Parse --variants; empty string → one anonymous variant (label ''), taken from --cfg_text_scale/--cfg_text_steps_frac/--sigma_shift. No torch dependency, unit-testable."""
    spec = (spec or "").strip()
    if not spec:
        return [dict(label="", cfg_text_scale=float(a.cfg_text_scale), cfg_text_steps_frac=float(a.cfg_text_steps_frac), sigma_shift=float(a.sigma_shift))]
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        assert len(parts) == 4, f"each --variants item must be label:cfg_scale:cfg_frac:sigma_shift, got {item!r}"
        label = parts[0].strip()
        assert label and "/" not in label and label not in (".", ".."), f"invalid variant label: {label!r}"
        out.append(dict(label=label, cfg_text_scale=float(parts[1]), cfg_text_steps_frac=float(parts[2]), sigma_shift=float(parts[3])))
    assert out, "--variants is non-empty but no variant was parsed"
    assert len({v["label"] for v in out}) == len(out), "duplicate variant labels"
    return out


def variant_out_dir(base, label):
    return os.path.join(base, label) if label else base


from videox_fun.moe_common.bench_output_check import output_exists


def model_namespace(a, meta):
    """Arguments needed by build_dit / build_mllm / build_vae (all base weights live under <base_dir>/ckpt/kiwi-edit-instruct-only-videoxfun/)."""
    base_ckpt = os.path.join(a.base_dir, "ckpt/kiwi-edit-instruct-only-videoxfun")
    return SimpleNamespace(
        config_path=a.config_path,
        pretrained_wan_path=base_ckpt,
        mllm_model=os.path.join(base_ckpt, "qwen"),
        init_weights=os.path.join(base_ckpt, "moe_expert_init.safetensors"),
        experts_merged=a.experts_merged, use_gradient_checkpointing=0,
        moe_layers=meta["moe_layers"], moe_num_experts=meta["moe_num_experts"], moe_top_k=meta["moe_top_k"],
        moe_aux_loss_alpha=0.0)


def build_dit_fast(ns, dtype, ckpt_file, fast_init):
    """Disable random init during construction, assert the checkpoint covers every parameter, then load it all (same as videoeditor-56b/moe_v3_2_infer.py)."""
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
    print(f"[infer-v3.1] build_dit {time.time() - t0:.0f}s (fast_init={fast_init})", flush=True)
    with safe_open(ckpt_file, "pt") as f:
        ck_keys = set(f.keys())
    p_names = set(n for n, _ in dit.named_parameters())
    uncovered = sorted(p_names - ck_keys)
    assert not uncovered, f"checkpoint does not cover {len(uncovered)} parameters (they would keep their construction-time values): {uncovered[:5]}"
    print(f"[infer-v3.1] coverage check: checkpoint {len(ck_keys)} keys ⊇ model {len(p_names)} parameters", flush=True)
    t1 = time.time()
    load_trainable_weights(dit, ckpt_file)
    print(f"[infer-v3.1] weights loaded in {time.time() - t1:.0f}s", flush=True)
    return dit


def default_out_dir(ckpt):
    """Default output location when --out_dir is omitted: <run dir>/render/<ckpt dir name>/.
    Convention: each run dir keeps weights in ckpt/ and inference results in render/. ckpt is usually <run>/ckpt/step-N or <run>/ckpt/final_components."""
    ckpt = os.path.abspath(ckpt.rstrip('/'))
    parent = os.path.dirname(ckpt)
    run_dir = os.path.dirname(parent) if os.path.basename(parent) == 'ckpt' else parent
    return os.path.join(run_dir, 'render', os.path.basename(ckpt))


class ExpertMapRecorder:
    """Hooks the self-attention and FFN of the given MoE layers and records each token's top-1 expert (argmax of the gate softmax) in one forward pass.
    maps keys: ("attn", layer) / ("ffn", layer)."""

    def __init__(self, dit, layer_indices):
        self.maps = {}
        self.handles = []
        for i in layer_indices:
            for kind, module in (("attn", dit.blocks[i].self_attn), ("ffn", dit.blocks[i].ffn)):
                def hook(mod, inputs, output, _key=(kind, i)):
                    tokens = inputs[0].reshape(-1, inputs[0].shape[-1])
                    scores = torch.softmax(torch.nn.functional.linear(tokens.float(), mod.gate.weight.float()), dim=-1)
                    self.maps[_key] = scores.argmax(dim=-1).cpu()          # [T]
                self.handles.append(module.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()


def save_frames_png(video, out_dir):
    """Save the output [1, C, T, H, W] ∈ [0,1] frame by frame as lossless PNGs (for figures; straight from the tensor, no mp4 compression): out_dir/f000.png ..., same frame order as the mp4."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    x = (video[0].clamp(0, 1).permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8)   # [T, H, W, C]
    for i, fr in enumerate(x):
        Image.fromarray(fr).save(os.path.join(out_dir, f"f{i:03d}.png"), compress_level=6)
    return out_dir


def save_expert_map_png(maps, grid, frame_indices, path, patch=16):
    """maps: {layer: [T]} top-1 expert; grid = (F_p, H_p, W_p). One row per layer, one column per selected frame, each patch colored by its expert."""
    from PIL import Image
    f_p, h_p, w_p = grid
    rows = []
    for layer, m in sorted(maps.items()):
        m = m.view(f_p, h_p, w_p).numpy()
        cols = [np.kron(EXPERT_COLORS[m[fi]], np.ones((patch, patch, 1), dtype=np.uint8)) for fi in frame_indices]
        rows.append(np.concatenate(cols, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def build_base_dit(ns, dtype):
    """Un-finetuned kiwi-edit base = original weights via from_pretrained + vae_condition (source-video conditioning conv) from moe_expert_init.safetensors; empty moe_layers → all original blocks."""
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from wan_transformer3d_kiwi_moe_v3_1 import KiwiMoeTransformer3DModel
    config = OmegaConf.load(ns.config_path)
    dit_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    dit_dir = os.path.join(ns.pretrained_wan_path, dit_kwargs.get("transformer_low_noise_model_subpath", "./"))
    if not os.path.isfile(os.path.join(dit_dir, "config.json")):
        dit_dir = ns.pretrained_wan_path
    dit_kwargs.update(moe_layers=[], moe_num_experts=7, moe_top_k=3, moe_aux_loss_alpha=0.0)
    dit = KiwiMoeTransformer3DModel.from_pretrained(dit_dir, transformer_additional_kwargs=dit_kwargs)
    init_state = load_file(ns.init_weights)
    cond_state = {k[len("dit."):]: v for k, v in init_state.items() if k.startswith("dit.")}
    _, unexpected = dit.load_state_dict(cond_state, strict=False)
    assert not unexpected, f"init key mismatch: {unexpected[:3]}"
    assert not list(dit.moe_layers), "base_only must not have MoE layers"
    print(f"[infer-v3.1] base model: no MoE layers, {sum(p.numel() for p in dit.parameters()) / 1e9:.2f}B parameters", flush=True)
    return dit.to(dtype=dtype)


@torch.no_grad()
def prepare_case(dit, text_encoder, vae, case, a, device, dtype, need_null):
    """Do the variant-independent part of a case once: read the source video (first N frames + short-side fit + center crop), MLLM encoding
    (conditional + optional unconditional branch), VAE-encode the source, fix the initial noise by --seed (all variants reuse it, so they differ only in CFG/shift)."""
    from videox_fun.data.kiwi_video_utils import frames_to_video_tensor
    src_frames = load_video_frames_first_n(os.path.join(a.bench_dir, case["src"]), a.num_frames, a.width, a.height)
    prompt = COMPOUND_JOINER.join(case["prompts"])
    ctx_cond = [text_encoder.encode(prompt, src_frames)]
    ctx_null = [text_encoder.encode_null(src_frames)] if need_null else None

    src_latents = vae.encode(frames_to_video_tensor(src_frames, dtype=torch.float32).to(device, dtype))[0].sample()
    latent_channels = getattr(vae.config, "z_dim", None) or vae.config.latent_channels
    f_lat, h_lat, w_lat = (a.num_frames - 1) // 4 + 1, a.height // 16, a.width // 16
    generator = torch.Generator("cpu").manual_seed(a.seed)
    latents0 = torch.randn(1, latent_channels, f_lat, h_lat, w_lat, generator=generator, dtype=torch.float32).to(device, dtype)
    patch = dit.patch_size
    grid = (f_lat // patch[0], h_lat // patch[1], w_lat // patch[2])
    seq_len = grid[0] * grid[1] * grid[2]
    return dict(ctx_cond=ctx_cond, ctx_null=ctx_null, src_latents=src_latents, latents0=latents0, grid=grid, seq_len=seq_len)


@torch.no_grad()
def sample_case(dit, vae, prep, case, v, a, device, dtype, expert_layers):
    """Sample a video for one variant v (cfg_text_scale / cfg_text_steps_frac / sigma_shift) from the prepare_case result. Bit-identical to the old run_case."""
    from videox_fun.data.kiwi_flow_match_scheduler import FlowMatchScheduler
    ctx_cond, src_latents, grid, seq_len = prep["ctx_cond"], prep["src_latents"], prep["grid"], prep["seq_len"]
    ctx_null = prep["ctx_null"] if v["cfg_text_scale"] != 1.0 else None
    assert not (v["cfg_text_scale"] != 1.0 and prep["ctx_null"] is None), "variant needs CFG but prepare_case did not compute the unconditional branch"
    latents = prep["latents0"].clone()
    scheduler = FlowMatchScheduler(shift=v["sigma_shift"], sigma_min=0.003 / 1.002, extra_one_step=True)
    scheduler.set_timesteps(a.steps)
    sigmas, timesteps = scheduler.sigmas.to(device), scheduler.timesteps.to(device)

    expert_maps = None
    from tqdm import tqdm
    for i in tqdm(range(len(timesteps)), desc=(case["name"][:24] + (":" + v["label"] if v["label"] else "")), dynamic_ncols=True, leave=False):
        t = timesteps[i].reshape(1).to(device=device, dtype=dtype)
        sigma = sigmas[i].to(device=device, dtype=dtype).reshape(1)
        recorder = ExpertMapRecorder(dit, expert_layers) if (a.dump_expert_maps and i == len(timesteps) // 2) else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_cond = dit(x=[latents[0]], t=t, mllm_context=ctx_cond, seq_len=seq_len, vae_source_input=src_latents, sigma=sigma)
            if recorder is not None:
                expert_maps = dict(recorder.maps)
                recorder.remove()
            if ctx_null is not None and i < int(round(v["cfg_text_steps_frac"] * len(timesteps))):    # segmented CFG: only in the first frac of steps
                pred_uncond = dit(x=[latents[0]], t=t, mllm_context=ctx_null, seq_len=seq_len, vae_source_input=src_latents, sigma=sigma)
                pred = pred_uncond + v["cfg_text_scale"] * (pred_cond - pred_uncond)
            else:
                pred = pred_cond
        pred = pred.to(dtype)
        sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else torch.zeros_like(sigmas[i])
        latents = latents + pred * (sigma_next.to(dtype) - sigma)

    out = vae.decode(latents.to(vae.dtype))
    video = out.sample if hasattr(out, "sample") else out[0]
    video = (video / 2 + 0.5).clamp(0, 1).float().cpu()
    return video, expert_maps, grid


@torch.no_grad()
def run_case(dit, text_encoder, vae, case, a, device, dtype, expert_layers):
    """Legacy interface (single setting): prepare_case + sample_case."""
    v = dict(label="", cfg_text_scale=float(a.cfg_text_scale), cfg_text_steps_frac=float(a.cfg_text_steps_frac), sigma_shift=float(a.sigma_shift))
    prep = prepare_case(dit, text_encoder, vae, case, a, device, dtype, need_null=v["cfg_text_scale"] != 1.0)
    return sample_case(dit, vae, prep, case, v, a, device, dtype, expert_layers)


def main():
    a = parse_args()
    # Make user-given paths absolute against the current directory (we chdir to the repo root below; --config_path is relative to the repo root and left as is)
    for _k in ("ckpt", "out_dir", "src_video", "cases_json", "bench_dir", "base_dir", "experts_merged"):
        if getattr(a, _k):
            setattr(a, _k, os.path.abspath(getattr(a, _k)))
    if not a.src_video:
        if not a.cases_json and not a.bench_dir:
            raise SystemExit("[infer-v3.1] need --src_video + --prompt (single-video mode), or --bench_dir <dir with cases.json> / --cases_json <file> (bench mode)")
        if not a.bench_dir:
            a.bench_dir = os.path.dirname(a.cases_json)
    device, dtype = "cuda", torch.bfloat16
    os.chdir(_repo)
    # Create the GPU context and check free memory first, so we exit before the slow model build instead of OOM-ing at .to(cuda).
    torch.cuda.set_device(0)
    _hold = torch.zeros(1, device=device)
    _free, _total = torch.cuda.mem_get_info()
    print(f"[infer] GPU context created, free memory {_free / 2**30:.1f} / {_total / 2**30:.1f} GiB", flush=True)
    if _free < a.min_free_gib * 2**30:
        raise SystemExit(f"[infer] only {_free / 2**30:.1f} GiB of GPU memory free (<{a.min_free_gib}); another process may be holding GPU memory")
    if not a.out_dir:
        a.out_dir = default_out_dir(a.ckpt)
    os.makedirs(a.out_dir, exist_ok=True)
    from videox_fun.utils.utils import save_videos_grid

    meta = json.load(open(os.path.join(a.ckpt, "meta.json")))
    ns = model_namespace(a, meta)
    print(f"[infer-v3.1] shard {a.shard_idx}/{a.num_shards} building model (ckpt step {meta.get('step')}, base_only={a.base_only}) ...", flush=True)
    if a.base_only:
        dit = build_base_dit(ns, dtype)                              # original kiwi-edit (from_pretrained + vae_condition), no MoE
        a.dump_expert_maps = 0
    else:
        dit = build_dit_fast(ns, dtype, os.path.join(a.ckpt, "moe_v3_1_trainable.safetensors"), a.fast_init)
    dit = dit.to(device).eval()
    # MLLM self-test + rebuild (as in training): run a fixed instruction + blank frames and check the output is finite; a one-off bad state like this once made 4 of 75 bench clips all black (NaN → 0)
    from PIL import Image as _Image
    _probe = [_Image.new("RGB", (a.width, a.height))] * a.num_frames
    for _attempt in range(1, 4):
        text_encoder = TextEncoder(build_mllm(ns).to(device))
        _where = text_encoder.self_test("Add a red ball.", _probe)
        if _where is None:
            print(f"[infer] MLLM self-test passed (attempt {_attempt})", flush=True); break
        print(f"[infer] MLLM self-test failed, first non-finite at [{_where}], attempt {_attempt}; rebuilding it entirely", flush=True)
        del text_encoder; import gc; gc.collect(); torch.cuda.empty_cache()
    else:
        print("[infer] MLLM self-test failed 3 times in a row, exit 18 so the caller can retry", flush=True); sys.exit(18)
    vae = build_vae(ns, dtype).to(device).eval()
    bad = [n for n, p in dit.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        print(f"[infer-v3.1] ⚠ {len(bad)} non-finite weights, e.g. {bad[:3]}, exit 17 so the caller can retry", flush=True)
        sys.exit(17)
    moe_layers = list(dit.moe_layers)
    expert_layers = [moe_layers[0], moe_layers[len(moe_layers) // 2], moe_layers[-1]] if moe_layers else []
    print(f"[infer-v3.1] moe_layers={moe_layers} top_k={dit.blocks[moe_layers[0]].ffn.top_k if moe_layers else '-'} text-CFG x{a.cfg_text_scale} expert-map layers {expert_layers}", flush=True)

    if a.src_video:                                                   # single-video mode
        prompts = [x.strip() for x in a.prompts.split("|||") if x.strip()] if a.prompts else [a.prompt.strip()]
        assert prompts and prompts[0], "single-video mode needs --prompt or --prompts"
        cases = [{"name": a.name, "src": os.path.abspath(a.src_video), "prompts": prompts}]
    else:
        cases = json.load(open(a.cases_json if a.cases_json else os.path.join(a.bench_dir, "cases.json")))
    my_cases = cases[a.shard_idx::a.num_shards]
    variants = parse_variants(a.variants, a)                          # load once, run several variants
    for v in variants:
        os.makedirs(variant_out_dir(a.out_dir, v["label"]), exist_ok=True)
    print(f"[infer-v3.1] {len(cases)} cases in total, {len(my_cases)} in this shard; output {a.out_dir}; variants "
          f"{[(v['label'] or '<single>', v['cfg_text_scale'], v['cfg_text_steps_frac'], v['sigma_shift']) for v in variants]}; skip_existing={a.skip_existing}", flush=True)
    n_done = 0
    for ci, case in enumerate(my_cases):
        pending = [v for v in variants if not (a.skip_existing and output_exists(variant_out_dir(a.out_dir, v["label"]), case["name"], a.frames_png, a.num_frames, a.width, a.height, a.fps))]
        if not pending:
            print(f"[infer-v3.1] shard {a.shard_idx} skip {ci + 1}/{len(my_cases)}: {case['name']} all variants already have outputs", flush=True)
            continue
        t_prep = time.time()
        prep = prepare_case(dit, text_encoder, vae, case, a, device, dtype, need_null=any(v["cfg_text_scale"] != 1.0 for v in pending))
        prep_s = round(time.time() - t_prep, 1)
        for v in pending:
            t0 = time.time()
            vdir = variant_out_dir(a.out_dir, v["label"])
            out_mp4 = os.path.join(vdir, case["name"] + ".mp4")
            video, expert_maps, grid = sample_case(dit, vae, prep, case, v, a, device, dtype, expert_layers)
            # Output self-check: if non-finite or near-constant over the whole clip (all black/gray), write no mp4 and no manifest; exit 19 so the caller retries (retry = rebuild all models)
            if not torch.isfinite(video).all() or float(video.std()) < 1e-3:
                print(f"[infer] ⚠ {case['name']}[{v['label']}] bad output (finite={bool(torch.isfinite(video).all())}, std={float(video.std()):.4g}), not saved, exit 19 so the caller can retry", flush=True)
                sys.exit(19)
            save_videos_grid(video, out_mp4, fps=a.fps)
            frames_dir = save_frames_png(video, os.path.join(vdir, "frames", case["name"])) if a.frames_png else None
            rec = {"name": case["name"], "src": case["src"], "prompts": case["prompts"], "out": out_mp4, "ckpt": a.ckpt,
                   "steps": a.steps, "cfg": f"text_x{v['cfg_text_scale']}_first{v['cfg_text_steps_frac']}", "sigma_shift": v["sigma_shift"], "variant_label": v["label"], "prep_s": prep_s, "seed": a.seed, "variant": ("kiwi_base" if a.base_only else "moe_v3_1"), "elapsed_s": round(time.time() - t0, 1), "frames_dir": frames_dir}
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
            print(f"[infer-v3.1] shard {a.shard_idx} done {ci + 1}/{len(my_cases)}: {case['name']}[{v['label'] or '-'}] {rec['elapsed_s']}s (prep {prep_s}s)", flush=True)
        del prep
    print(f"[infer-v3.1] shard {a.shard_idx} ALL DONE ({n_done} outputs)", flush=True)


if __name__ == "__main__":
    main()
