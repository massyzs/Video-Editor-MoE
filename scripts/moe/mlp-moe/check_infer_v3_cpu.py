"""CPU self-check of the mlp-moe inference path (no GPU needed): load a real trained ckpt via build_dit + load_trainable_weights,
then run one DiT forward on a tiny latent to verify that the 12-expert, no-shared-expert weight keys match exactly, the output
is finite and the expert routing statistics look sane.
Usage: CUDA_VISIBLE_DEVICES= python check_infer_v3_cpu.py --base_dir <BASE_DIR> --ckpt <CKPT_DIR> [--experts_merged <experts_merged_6.safetensors>]"""
import argparse, os, sys, time, types, torch
_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import json
from moe_v3_model import EXPERT_NAMES, build_dit, load_trainable_weights
a = argparse.ArgumentParser(); a.add_argument("--ckpt", required=True)
a.add_argument("--base_dir", required=True, help="directory containing ckpt/kiwi-edit-instruct-only-videoxfun/")
a.add_argument("--experts_merged", default="", help="empty = use --ckpt only (same as inference); given a merge_experts.py output, load the merged experts first")
a = a.parse_args()
meta = json.load(open(os.path.join(a.ckpt, "meta.json")))
B = a.base_dir
ns = types.SimpleNamespace(config_path=os.path.join(_repo, "config/wan2.2/wan_civitai_5b.yaml"), pretrained_wan_path=f"{B}/ckpt/kiwi-edit-instruct-only-videoxfun",
                           init_weights=f"{B}/ckpt/kiwi-edit-instruct-only-videoxfun/moe_expert_init.safetensors",
                           experts_merged=a.experts_merged, use_gradient_checkpointing=0,
                           moe_layers=meta["moe_layers"], moe_num_experts=meta["moe_num_experts"], moe_top_k=meta["moe_top_k"],
                           moe_aux_loss_alpha=0.0, expert_copy_noise=meta.get("expert_copy_noise", 0.01))
t0 = time.time(); dit = build_dit(ns, torch.bfloat16, verbose=True); print(f"build {time.time() - t0:.0f}s", flush=True)
t0 = time.time()
from safetensors import safe_open
path = os.path.join(a.ckpt, "moe_v3_trainable.safetensors")
with safe_open(path, "pt") as sf: n_keys = len(list(sf.keys()))
state = load_trainable_weights(dit, path)
model_keys = {k for k, _ in dit.named_parameters()}
print(f"ckpt {n_keys} keys, loaded {len(state)} keys, model params {len(model_keys)}; fraction of model params covered by ckpt {len(set(state) & model_keys) / len(model_keys):.4f}; shared keys in ckpt: {sum('ffn.shared' in k for k in state)} ; load {time.time() - t0:.0f}s", flush=True)
dit.eval()
with torch.no_grad():
    f, h, w = 1, 4, 4                                                   # tiny latent: 1 frame 64x64 px -> latent 4x4 -> grid 1x2x2 = 4 tokens
    x = [torch.randn(48, f, h, w, dtype=torch.bfloat16)]
    src = torch.randn(1, 48, f, h, w, dtype=torch.bfloat16)
    ctx = [torch.randn(7, 3072, dtype=torch.bfloat16)]
    t = torch.tensor([500.0], dtype=torch.bfloat16); sigma = torch.tensor([0.5], dtype=torch.bfloat16)
    t0 = time.time()
    y = dit(x=x, t=t, mllm_context=ctx, seq_len=f * (h // 2) * (w // 2), vae_source_input=src, sigma=sigma)
print(f"forward {time.time() - t0:.0f}s: output {tuple(y.shape)} finite {bool(torch.isfinite(y).all())} |y|max {y.float().abs().max():.3f}", flush=True)
sel = [m.last_selected for m in dit.moe_ffns()]
print("per-MoE-layer sum of selection fractions (should be 3):", [round(float(s.sum()), 2) for s in sel][:5], "...; layer 1 per expert:", [round(float(v), 2) for v in sel[0]])
print("CHECK_INFER_DONE")
