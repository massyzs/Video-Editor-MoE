"""videoeditor-14b self-attention GPU parity test (single GPU, real sizes: dim 3072 / 24 heads / head_dim 128 / sequence = 13x15x26 = 5,070 tokens after patchify, random weights, no model loaded):
(Uses the post-patchify token count; the latent voxel count 20,280 would overestimate attention cost 16x.)
 1) flash path (8 padded dims + softmax_scale=1/sqrt(128)) vs fp32 SDPA reference with explicit attn_bias (computed on GPU), elementwise comparison
 2) consistency with the original WanSelfAttention when experts == base (bf16)
 3) forward + backward time and peak memory (batch B), compared with the original WanSelfAttention
Usage: CUDA_VISIBLE_DEVICES=0 python probe_moe_v3_1_gpu.py
"""
import math, os, sys, time, torch, torch.nn.functional as F
_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION")
torch.manual_seed(0)
from wan_transformer3d_kiwi_moe_v3_1 import MoEKVSelfAttention, WanSelfAttention, rope_apply, rope_params, route_tokens
dev = "cuda"; dim, heads, d = 3072, 24, 128
f, h, w = 13, 15, 26; L = f * h * w; B = 4                # 49 frames 480x832 -> latent 13x30x52 -> patch (1,2,2) -> 13x15x26; batch 4 = launcher per-GPU batch
freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1).to(dev)
grid = torch.tensor([[f, h, w]] * B, device=dev); lens = torch.tensor([L] * B, device=dev)
x = torch.randn(B, L, dim, device=dev, dtype=torch.bfloat16)
base = WanSelfAttention(dim, heads, (-1, -1), True, 1e-6).to(dev, torch.bfloat16).eval()      # original; serves as base and reference
attn = MoEKVSelfAttention(dim, heads, (-1, -1), True, 1e-6, num_experts=7, top_k=3, aux_loss_alpha=0.001).to(dev, torch.bfloat16).eval()
with torch.no_grad():
    for name in ("q", "o", "norm_q", "norm_k"):
        getattr(attn, name).load_state_dict(getattr(base, name).state_dict())
    attn.gate.weight.normal_(0, 0.5)
    for i in range(7):
        for mod, src in ((attn.k_experts[i], base.k), (attn.v_experts[i], base.v)):
            mod.load_state_dict(src.state_dict())
            for p in mod.parameters(): p.add_(torch.randn_like(p) * p.float().std().to(p.dtype) * 0.05)
ok = True
def dmax(a, b): return (a.float() - b.float()).abs().max().item()

def reference_fp32(attn, x):
    """fp32 reference with an explicit attn_bias (same concatenation of top_k copies)."""
    a = attn.float(); xf = x.float()
    b, s, n = xf.shape[0], xf.shape[1], a.num_heads
    sh = 1 if a.shared_kv else 0; copies = sh + a.top_k; tokens = xf.reshape(-1, dim)
    scores, w_, idx = route_tokens(a.gate, tokens, a.top_k)
    Kc = torch.zeros(tokens.shape[0], copies, dim, device=dev); Vc = torch.zeros_like(Kc); prior = torch.zeros(tokens.shape[0], copies, device=dev)
    if sh:
        Kc[:, 0] = a.k(tokens); Vc[:, 0] = a.v(tokens)
    for i in range(a.num_experts):
        ti, sl = (idx == i).nonzero(as_tuple=True)
        Kc[ti, sh + sl] = a.k_experts[i](tokens[ti]); Vc[ti, sh + sl] = a.v_experts[i](tokens[ti]); prior[ti, sh + sl] = torch.log(w_[ti, sl].clamp_min(1e-6))
    q = rope_apply(a.norm_q(a.q(xf)).view(b, s, n, d), grid, freqs)
    Kc = rope_apply(a.norm_k(Kc).view(b, s, copies * n, d), grid, freqs).view(b, s * copies, n, d); Vc = Vc.view(b, s * copies, n, d)
    out = torch.zeros(b, s, n, d, device=dev)
    for hh in range(n):                                   # one head at a time to save memory
        logit = torch.einsum("bsd,btd->bst", q[:, :, hh], Kc[:, :, hh]) / math.sqrt(d) + prior.view(b, 1, s * copies)
        out[:, :, hh] = torch.einsum("bst,btd->bsd", torch.softmax(logit, dim=-1), Vc[:, :, hh])
    y = a.o(out.reshape(b, s, -1)); attn.to(torch.bfloat16); return y, idx

with torch.no_grad():
    y_flash = attn(x, lens, grid, freqs, dtype=torch.bfloat16)
    y_ref, idx_ref = reference_fp32(attn, x)
    y_flash2 = attn(x, lens, grid, freqs, dtype=torch.bfloat16)
rel = dmax(y_flash, y_ref) / y_ref.abs().max().item()
print(f"1) flash with extra-dim prior injection vs fp32 explicit-bias reference: max|Δ|={dmax(y_flash, y_ref):.3e} rel {rel:.2e} (|y|max={y_ref.abs().max():.3e}); diff between two flash forwards {dmax(y_flash, y_flash2):.1e}; routing matches {bool((attn.last_topk_idx == idx_ref).all())}")
ok &= rel < 2e-2
# 2) experts == base -> must equal the original WanSelfAttention
with torch.no_grad():
    for i in range(7):
        attn.k_experts[i].load_state_dict(base.k.state_dict()); attn.v_experts[i].load_state_dict(base.v.state_dict())
    y_base = base(x, lens, grid, freqs, dtype=torch.bfloat16); y_eq = attn(x, lens, grid, freqs, dtype=torch.bfloat16)
rel2 = dmax(y_base, y_eq) / y_base.abs().max().item()
print(f"2) experts == base: MoE attention vs original WanSelfAttention: max|Δ|={dmax(y_base, y_eq):.3e} rel {rel2:.2e}"); ok &= rel2 < 2e-2
# 3) time / memory
def bench(mod, name):
    mod.train(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    xg = x.clone().requires_grad_(True)
    for it in range(4):
        if it == 1: torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = mod(xg, lens, grid, freqs, dtype=torch.bfloat16)
        y.float().pow(2).mean().backward()
    torch.cuda.synchronize(); dt = (time.time() - t0) / 3
    print(f"3) {name}: fwd+bwd {dt * 1000:.0f} ms/iter (batch {B}), peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    return dt
t_base = bench(base, "original WanSelfAttention"); t_moe = bench(attn, "MoEKVSelfAttention (top-3, no shared pair)")
print(f"   ratio {t_moe / t_base:.2f}x")
print("RESULT:", "PASS" if ok else "FAIL")
