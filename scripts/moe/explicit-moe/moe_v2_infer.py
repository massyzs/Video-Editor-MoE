#!/usr/bin/env python
"""explicit-moe inference: a single video, or a complex_bench-style case list, with text CFG.
Model: see moe_v2_transformer.py.
  - routing = router-head argmax (deployment mode); sub-instructions sorted by expert name (as in training, so RoPE positions match)
  - base ctx = full instruction (sub-instructions joined with " Then "); unconditional branch = empty instruction (experts unchanged)
  - sampling = flow-match Euler, shift=5, 50 steps; pred = u + s*(c-u)
  - source video: first n frames (trimmed to 4k+1), padded with the last frame, as in training (not linspace)
  - sharding via --shard_idx/--num_shards, one process per GPU; mp4 + manifest are written to --out_dir
Usage:
  single video:  CUDA_VISIBLE_DEVICES=0 python scripts/moe/explicit-moe/moe_v2_infer.py --base_dir <BASE_DIR> --ckpt <CKPT_DIR> \\
                     --src_video <video.mp4> --prompt "Remove the dog" --name dog --out_dir <OUT_DIR>
  compound:      ... --src_video <video.mp4> --prompts "Remove the dog ||| Convert to oil painting style"
                 (each sub-instruction is routed to its own expert; the base stream gets them joined with " Then ")
  benchmark:     ... --bench_dir <BENCH_DIR with cases.json> --out_dir <OUT_DIR> --shard_idx 0 --num_shards 1
  custom cases:  ... --cases_json <cases.json: [{name, src, prompts[]}]> [--bench_dir <root for relative src, default = dir of cases.json>]
  <BASE_DIR>/ckpt/kiwi-edit-instruct-only-videoxfun/ holds the frozen base (config.json, diffusion_pytorch_model.safetensors,
  moe_expert_init.safetensors, Wan2.2_VAE.pth, qwen/). <CKPT_DIR> is the extracted explicit-moe release (moe_v2_state.pt,
  router_head_online.pt, lora_experts/<name>/dit_query_router.safetensors); for a training ckpt dir (step-N / final_components)
  pass --lora_experts_dir and --router_head explicitly.
  Outputs: <OUT_DIR>/<name>.mp4, <OUT_DIR>/frames/<name>/f000.png ... (--frames_png 1), <OUT_DIR>/manifest_shard<i>.jsonl
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

from moe_v2_transformer import DEFAULT_LORA_EXPERT_TEMPLATE, build_model, load_checkpoint     # noqa: E402
from videox_fun.moe_common.bench_output_check import output_exists                           # noqa: E402

COMPOUND_JOINER = " Then "        # matches the training default of --compound_joiner


def load_video_frames_first_n(path, num_frames, width, height):
    """Frame sampling as in training: first n frames (trimmed to 4k+1), padded with the last frame; geometry = short-side-fit resize + center crop."""
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
    frames = [Image.fromarray(x).convert("RGB").resize((nw, nh), Image.BILINEAR)
              .crop((left, top, left + width, top + height)) for x in arrs]
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoint dir containing moe_v2_state.pt: the extracted explicit-moe release, or a training ckpt dir (final_components / step-N)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_dir", required=True, help="<BASE_DIR>: its ckpt/kiwi-edit-instruct-only-videoxfun/ holds the frozen base (config.json, diffusion_pytorch_model.safetensors, moe_expert_init.safetensors, Wan2.2_VAE.pth, qwen/)")
    p.add_argument("--lora_experts_dir", default="", help="LoRA expert dir (file = <dir>/<--lora_expert_template>); default = <ckpt>/lora_experts")
    p.add_argument("--lora_expert_template", default=DEFAULT_LORA_EXPERT_TEMPLATE, help="expert file path relative to --lora_experts_dir, {name} = expert name")
    p.add_argument("--router_head", default="", help="router head .pt (only the structural fields dmodel/heads/classes are used; its weights are overridden by the jointly fine-tuned ones in moe_v2_state.pt); default = <ckpt>/router_head_online.pt")
    # single-video mode: if --src_video is given, run only that video and do not read cases.json
    p.add_argument("--src_video", default="", help="source video path; if given, enables single-video mode")
    p.add_argument("--prompt", default="", help="edit instruction (the whole sentence is one sub-instruction, routed to one expert)")
    p.add_argument("--prompts", default="", help="sub-instructions separated by |||: each is routed to its own expert; the base stream gets the full instruction joined with \" Then \"")
    p.add_argument("--name", default="single", help="output file name in single-video mode (without extension)")
    # bench mode
    p.add_argument("--bench_dir", default="", help="bench mode: dir with cases.json and the source videos (root for relative case src); unused in single-video mode")
    p.add_argument("--cases_json", default="", help="custom case list json (same format as complex_bench/cases.json: [{name, src, prompts[]}]); relative src is resolved against --bench_dir, which defaults to the directory of cases_json")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--config_path", default="config/wan2.2/wan_civitai_5b.yaml")
    p.add_argument("--fusion_heads", type=int, default=24)
    p.add_argument("--experts", default="style,add,remove,replace,background")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--cfg_text_scale", type=float, default=3.0, help="text CFG scale s: pred = u + s*(c-u); 1 = off")
    p.add_argument("--cfg_text_steps_frac", type=float, default=1.0, help="segmented CFG: apply text CFG only in the first frac of denoising steps (same convention as mlp-moe / videoeditor inference, e.g. 0.6); 1.0 = all steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--frames_png", type=int, default=1, help="1 = also save every output frame as lossless PNG to <out_dir>/frames/<case>/, 0 = mp4 only")
    p.add_argument("--skip_existing", type=int, default=1, help="1 = skip a case when video size, frame count and fps match the arguments and all PNG frames exist")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def build_cases(a):
    """Single-video mode -> one case; bench mode -> cases.json (or --cases_json). Called before building the model so bad arguments fail fast."""
    if a.src_video:
        prompts = [x.strip() for x in a.prompts.split("|||") if x.strip()] if a.prompts else [a.prompt.strip()]
        if not (prompts and prompts[0]):
            raise SystemExit("[infer-v2] single-video mode needs --prompt or --prompts")
        if not os.path.isfile(a.src_video):
            raise SystemExit(f"[infer-v2] --src_video not found: {a.src_video}")
        return [{"name": a.name, "src": os.path.abspath(a.src_video), "prompts": prompts}]
    return json.load(open(a.cases_json if a.cases_json else os.path.join(a.bench_dir, "cases.json")))


def save_frames_png(video, out_dir):
    """Save the output [1, C, T, H, W] in [0,1] frame by frame as lossless PNG (straight from the tensor, no mp4 compression): out_dir/f000.png ..., same frame order as the mp4."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    x = (video[0].clamp(0, 1).permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8)   # [T, H, W, C]
    for i, fr in enumerate(x):
        Image.fromarray(fr).save(os.path.join(out_dir, f"f{i:03d}.png"), compress_level=6)
    return out_dir


def build_ns(a, ck_meta):
    """Arguments for build_model: the base is in <base_dir>/ckpt/kiwi-edit-instruct-only-videoxfun/; experts and router head default to --ckpt (release layout)."""
    base_ckpt = os.path.join(a.base_dir, "ckpt/kiwi-edit-instruct-only-videoxfun")
    return SimpleNamespace(
        config_path=a.config_path,
        pretrained_wan_path=base_ckpt,
        mllm_model=os.path.join(base_ckpt, "qwen"),
        init_weights=os.path.join(base_ckpt, "moe_expert_init.safetensors"),
        router_head=a.router_head or os.path.join(a.ckpt, "router_head_online.pt"),
        lora_experts_dir=a.lora_experts_dir or os.path.join(a.ckpt, "lora_experts"),
        lora_expert_template=a.lora_expert_template, experts=a.experts,
        dit_lora_rank=128, dit_lora_target_modules="q,k,v,o,ffn.0,ffn.2", use_gradient_checkpointing=0,
        fusion_start=int(ck_meta["fusion_start"]), fusion_end=int(ck_meta["fusion_end"]),
        fusion_heads=a.fusion_heads, fusion_init="normal", fusion_init_gain=0.05,
        rope_expert_offset0=int(ck_meta.get("rope_expert_offset0", 1)))


def load_model(a, dtype):
    """build_model (frozen DiT + VAE + MLLM + LoRA experts + router-head structure) -> load_checkpoint (fusion layers + jointly fine-tuned router head). All on CPU; the caller does .to(device)."""
    state_path = os.path.join(a.ckpt, "moe_v2_state.pt")
    ck_meta = torch.load(state_path, map_location="cpu")
    ns = build_ns(a, ck_meta)
    del ck_meta
    for what, path in (("--router_head", ns.router_head), ("--lora_experts_dir", ns.lora_experts_dir),
                       ("--base_dir", ns.pretrained_wan_path)):
        if not os.path.exists(path):
            raise SystemExit(f"[infer-v2] {what}: not found: {path}")
    model, vae = build_model(ns, dtype)
    ck = load_checkpoint(model, state_path)
    n_fu = sum(1 for k in ck["trainable"] if k.startswith("fusions."))
    n_rh = sum(1 for k in ck["trainable"] if k.startswith("router_head."))
    assert n_fu > 0, "ckpt has no fusion-layer weights"
    print(f"[infer-v2] ckpt step={ck.get('step')} fusion_layers={len(model.fusions)} absorb_tensors={n_fu} "
          f"router_head={n_rh} shift={a.sigma_shift} text-CFG x{a.cfg_text_scale} (first {a.cfg_text_steps_frac} of steps) "
          f"blocks[{model.fusion_start},{model.fusion_end}) first-n frame sampling", flush=True)
    del ck
    return model, vae


@torch.no_grad()
def run_case(model, vae, case, a, device, dtype):
    from videox_fun.data.kiwi_video_utils import frames_to_video_tensor
    from videox_fun.data.kiwi_flow_match_scheduler import FlowMatchScheduler
    names = model.adapter_names
    src_frames = load_video_frames_first_n(os.path.join(a.bench_dir, case["src"]), a.num_frames, a.width, a.height)
    prompts = list(case["prompts"])

    # routing + ordering consistent with training
    ctxs, eids, logits_list, raw_cls = model.route(prompts, src_frames)
    confs = [float(torch.softmax(l.detach().float(), 1).max()) for l in logits_list]
    if any(c != c for c in confs):     # nan = corrupted weights/features in this process -> exit 17 so the launcher retries
        print(f"[infer-v2] ⚠ router confidence is nan ({case['name']}), exit 17 for launcher retry", flush=True)
        sys.exit(17)
    order = sorted(range(len(eids)), key=lambda j: (names[eids[j]], j))
    ctxs, eids, confs = [ctxs[j] for j in order], [eids[j] for j in order], [confs[j] for j in order]
    adapter_seq = [names[e] for e in eids]
    ctx_experts = [[c] for c in ctxs]                      # K experts, 1 sample each

    ctx_base = [model.text_ctx(COMPOUND_JOINER.join(prompts), src_frames)]
    ctx_null = [model.null_ctx(src_frames)] if a.cfg_text_scale != 1.0 else None

    src_latents = vae.encode(frames_to_video_tensor(src_frames, dtype=torch.float32).to(device, dtype))[0].sample()
    z = getattr(vae.config, "z_dim", None) or vae.config.latent_channels
    f_lat, h_lat, w_lat = (a.num_frames - 1) // 4 + 1, a.height // 16, a.width // 16
    g = torch.Generator("cpu").manual_seed(a.seed)
    latents = torch.randn(1, z, f_lat, h_lat, w_lat, generator=g, dtype=torch.float32).to(device, dtype)
    pch = model.dit.patch_size
    seq_len = (f_lat // pch[0]) * (h_lat // pch[1]) * (w_lat // pch[2])

    scheduler = FlowMatchScheduler(shift=a.sigma_shift, sigma_min=0.003 / 1.002, extra_one_step=True)
    scheduler.set_timesteps(a.steps)
    sigmas, timesteps = scheduler.sigmas.to(device), scheduler.timesteps.to(device)
    n_cfg = int(round(a.cfg_text_steps_frac * len(timesteps)))   # segmented CFG: the unconditional branch runs only in the first n_cfg steps

    from tqdm import tqdm
    for i in tqdm(range(len(timesteps)), desc=case["name"][:32], dynamic_ncols=True, leave=False):
        t = timesteps[i].reshape(1).to(device=device, dtype=dtype)
        sigma = sigmas[i].to(device=device, dtype=dtype).reshape(1)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            pred_c = model.denoise([latents[0]], t, ctx_base, ctx_experts, adapter_seq, seq_len, src_latents, sigma)
            if ctx_null is not None and i < n_cfg:
                pred_u = model.denoise([latents[0]], t, ctx_null, ctx_experts, adapter_seq, seq_len, src_latents, sigma)
                pred = pred_u + a.cfg_text_scale * (pred_c - pred_u)
            else:
                pred = pred_c
        pred = pred.to(dtype)
        sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else torch.zeros_like(sigmas[i])
        latents = latents + pred * (sigma_next.to(dtype) - sigma)

    out = vae.decode(latents.to(vae.dtype))
    video = out.sample if hasattr(out, "sample") else out[0]
    video = (video / 2 + 0.5).clamp(0, 1).float().cpu()
    return video, {"experts_used": adapter_seq, "router_raw": raw_cls,
                   "router_conf": [round(c, 4) for c in confs], "k": len(prompts)}


def main(argv=None):
    a = parse_args(argv)
    # make user-given paths absolute w.r.t. the cwd (we chdir to the repo root below; --config_path is repo-relative, left as is)
    for _k in ("ckpt", "out_dir", "src_video", "cases_json", "bench_dir", "base_dir", "lora_experts_dir", "router_head"):
        if getattr(a, _k):
            setattr(a, _k, os.path.abspath(getattr(a, _k)))
    if not a.src_video:
        if not a.cases_json and not a.bench_dir:
            raise SystemExit("[infer-v2] need --src_video + --prompt (single-video mode), or --bench_dir <dir with cases.json> / --cases_json <file> (bench mode)")
        if not a.bench_dir:
            a.bench_dir = os.path.dirname(a.cases_json)
    cases = build_cases(a)
    device, dtype = a.device, torch.bfloat16
    os.chdir(_repo)
    os.makedirs(a.out_dir, exist_ok=True)
    from videox_fun.utils.utils import save_videos_grid

    print(f"[infer-v2] shard {a.shard_idx}/{a.num_shards} building model ...", flush=True)
    model, vae = load_model(a, dtype)
    model = model.to(device).eval()
    vae = vae.to(device).eval()
    # concurrent multi-process loading from a network filesystem occasionally corrupts weights (nan routing -> all-black video) -> scan all parameters for finiteness; on failure exit 17 so the launcher retries
    bad = [n for n, p_ in model.named_parameters() if not torch.isfinite(p_).all()]
    bad += [f"vae.{n}" for n, p_ in vae.named_parameters() if not torch.isfinite(p_).all()]
    bad += [f"buf.{n}" for n, b_ in model.named_buffers() if b_.is_floating_point() and not torch.isfinite(b_).all()]
    if bad:
        print(f"[infer-v2] ⚠ {len(bad)} non-finite weights (corrupted concurrent load), e.g. {bad[:3]}, exit 17 for launcher retry", flush=True)
        sys.exit(17)
    print("[infer-v2] weight finiteness scan passed", flush=True)

    my_cases = cases[a.shard_idx::a.num_shards]
    print(f"[infer-v2] shard {a.shard_idx}: {len(my_cases)}/{len(cases)} cases -> {a.out_dir}", flush=True)
    mani_path = os.path.join(a.out_dir, f"manifest_shard{a.shard_idx}.jsonl")
    for ci, case in enumerate(my_cases):
        t0 = time.time()
        out_mp4 = os.path.join(a.out_dir, case["name"] + ".mp4")
        if a.skip_existing and output_exists(a.out_dir, case["name"], a.frames_png, a.num_frames, a.width, a.height, a.fps):
            print(f"[infer-v2] shard {a.shard_idx} skip {ci + 1}/{len(my_cases)}: {case['name']} output exists", flush=True)
            continue
        video, info = run_case(model, vae, case, a, device, dtype)
        save_videos_grid(video, out_mp4, fps=a.fps)
        frames_dir = save_frames_png(video, os.path.join(a.out_dir, "frames", case["name"])) if a.frames_png else None
        rec = {"name": case["name"], "src": case["src"], "prompts": case["prompts"], "out": out_mp4,
               "ckpt": a.ckpt, "steps": a.steps, "sigma_shift": a.sigma_shift,
               "cfg": f"text_x{a.cfg_text_scale}_first{a.cfg_text_steps_frac}", "seed": a.seed,
               "num_frames": a.num_frames, "elapsed_s": round(time.time() - t0, 1), "frames_dir": frames_dir, **info}
        with open(mani_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[infer-v2] shard {a.shard_idx} done {ci + 1}/{len(my_cases)}: {case['name']} "
              f"experts={info['experts_used']} conf={info['router_conf']} {rec['elapsed_s']}s", flush=True)
    print(f"[infer-v2] shard {a.shard_idx} ALL DONE", flush=True)


if __name__ == "__main__":
    main()
