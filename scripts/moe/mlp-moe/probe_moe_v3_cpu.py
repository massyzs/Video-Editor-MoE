"""CPU equivalence test of the mlp-moe architecture (small config: 8 layers, dim 64, MoE at layers 1,3,5):
 A) With every expert a copy of the original FFN, MoE output == original dense output (weighted mean with sum(g) = 1, unchanged at step 0)
 B) A 1% expert perturbation changes the output only slightly; gate and experts get gradients, and the load-balancing loss gradient reaches the gate
 C) GC on/off agree; every expert is called every step (even with an empty tensor); total usage count = tokens x top_k; per-layer sum of last_selected (selection fraction) = top_k
 D) top_k weight normalization: with all experts identical, output == that expert's output (sum(g) = 1, independent of gate scores)
Usage: VIDEOX_ATTENTION_TYPE=SDPA CUDA_VISIBLE_DEVICES= python probe_moe_v3_cpu.py
"""
import copy, os, sys, torch, torch.nn as nn
_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)
torch.manual_seed(0)
from videox_fun.models.wan_transformer3d_kiwi import KiwiWanTransformer3DModel
from wan_transformer3d_kiwi_moe import KiwiMoeTransformer3DModel, MoEFFN

cfg = dict(model_type="t2v", patch_size=(1, 2, 2), text_len=8, in_dim=4, dim=64, ffn_dim=128, freq_dim=32,
           text_dim=16, out_dim=4, num_heads=2, num_layers=8, eps=1e-6, in_channels=4, hidden_size=64)
MOE_LAYERS, N, K = (1, 3, 5), 4, 3
dense = KiwiWanTransformer3DModel(**cfg)
with torch.no_grad():                                        # Wan init_weights zeros the head; use random weights so the output is non-zero
    dense.head.head.weight.normal_(0, 0.05); dense.head.head.bias.normal_(0, 0.05)
moe = KiwiMoeTransformer3DModel(**cfg, moe_layers=MOE_LAYERS, moe_num_experts=N, moe_top_k=K, moe_aux_loss_alpha=0.01)
missing, unexpected = moe.load_state_dict(dense.state_dict(), strict=False)
# MoE-layer ffn keys differ (shared/experts/gate): dense blocks.i.ffn.* show up as unexpected, MoE ones as missing; both only in MoE layers
assert all(".ffn." in k and any(k.startswith(f"blocks.{i}.") for i in MOE_LAYERS) for k in list(missing) + list(unexpected)), (missing[:3], unexpected[:3])
with torch.no_grad():                                        # MoE layers: every expert copies the dense ffn
    for i in MOE_LAYERS:
        ffn_dense, ffn_moe = dense.blocks[i].ffn, moe.blocks[i].ffn
        for expert in ffn_moe.experts:
            expert.load_state_dict(ffn_dense.state_dict())
dense.requires_grad_(False); moe.requires_grad_(False)
for i in MOE_LAYERS:
    moe.blocks[i].ffn.requires_grad_(True)

B, f, h, w = 2, 2, 4, 6
x = [torch.randn(4, f, 2 * h, 2 * w) for _ in range(B)]
src = torch.randn(B, 4, f, 2 * h, 2 * w); sigma = torch.rand(B); t = torch.rand(B) * 1000
ctx = [torch.randn(7, 64) for _ in range(B)]; seq_len = f * h * w
ok = True
def d(a, b): return (a.float() - b.float()).abs().max().item()

with torch.no_grad():
    y_dense = dense(x=x, t=t, mllm_context=ctx, seq_len=seq_len, vae_source_input=src, sigma=sigma)
    moe.eval(); y_moe = moe(x=x, t=t, mllm_context=ctx, seq_len=seq_len, vae_source_input=src, sigma=sigma)
print(f"A) experts = original copies: MoE output vs dense max|diff|={d(y_dense, y_moe):.2e} (|y|max={y_dense.abs().max():.2e})"); ok &= d(y_dense, y_moe) < 1e-5

# B) perturbation + train mode: small output change, gradient flow
with torch.no_grad():
    for i in MOE_LAYERS:
        for expert in moe.blocks[i].ffn.experts:
            for p in expert.parameters(): p.add_(torch.randn_like(p) * p.std() * 0.01)
moe.train()
for gc in (False, True):
    moe.gradient_checkpointing = gc; moe.zero_grad()
    y = moe(x=x, t=t, mllm_context=ctx, seq_len=seq_len, vae_source_input=src, sigma=sigma)
    y.float().pow(2).mean().backward()
    ffn = moe.blocks[MOE_LAYERS[0]].ffn
    g_gate = ffn.gate.weight.grad.abs().max().item(); g_exp = max(p.grad.abs().max().item() for e in ffn.experts for p in e.parameters())
    g_sh = max(p.grad.abs().max().item() for p in moe.blocks[MOE_LAYERS[-1]].ffn.experts.parameters())   # experts of the last MoE layer must also get gradients
    usage_total = sum(int(f_.last_usage.sum()) for f_ in moe.moe_ffns()); expect = len(MOE_LAYERS) * B * seq_len * K
    print(f"B/C) GC={gc}: relative output change after perturbation {(d(y.detach(), y_dense) / y_dense.abs().max()).item():.3f}; grad gate={g_gate:.2e} experts={g_exp:.2e} last_layer_experts={g_sh:.2e}; "
          f"aux={float(sum(f_.last_aux_loss for f_ in moe.moe_ffns())):.4f}; usage total {usage_total} (expected {expect})")
    ok &= g_gate > 0 and g_exp > 0 and g_sh > 0 and usage_total == expect
    selected_sums = [float(f_.last_selected.sum()) for f_ in moe.moe_ffns()]
    print(f"   per-layer sum of selection fractions {['%.3f' % s for s in selected_sums]} (all should = {K})"); ok &= all(abs(s - K) < 1e-5 for s in selected_sums)
    if gc: y_gc = y.detach()
    else: y_nogc = y.detach()
print(f"   GC on/off output max|diff|={d(y_gc, y_nogc):.2e}"); ok &= d(y_gc, y_nogc) < 1e-5
# Does the aux loss gradient really reach the gate? Recompute with alpha = 0; the gate gradient should differ
for f_ in moe.moe_ffns(): f_.aux_loss_alpha = 0.0
moe.zero_grad(); y = moe(x=x, t=t, mllm_context=ctx, seq_len=seq_len, vae_source_input=src, sigma=sigma); y.float().pow(2).mean().backward()
g_gate0 = ffn.gate.weight.grad.abs().max().item()
print(f"   gate grad with alpha=0 {g_gate0:.2e} vs alpha=0.01 {g_gate:.2e} (should differ: the aux loss gradient reaches the gate)"); ok &= abs(g_gate0 - g_gate) > 1e-9
# D) normalization: set all 4 experts to the same FFN, then y = sum(g)*e = e (holds only if sum(g) = 1)
ffn_d = MoEFFN(dim=64, ffn_dim=128, num_experts=N, top_k=K, aux_loss_alpha=0.0).eval()
with torch.no_grad():
    for e in ffn_d.experts: e.load_state_dict(ffn_d.experts[0].state_dict())
    u = torch.randn(2, 11, 64); y_d = ffn_d(u); y_e = ffn_d.experts[0](u)
print(f"D) all experts identical: MoE output vs that expert's output max|diff|={d(y_d, y_e):.2e} (should be ~0: top_k weights sum to 1)"); ok &= d(y_d, y_e) < 1e-5
print("RESULT:", "PASS" if ok else "FAIL")
