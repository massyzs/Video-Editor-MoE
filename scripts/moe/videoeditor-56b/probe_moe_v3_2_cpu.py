"""videoeditor-56b structure CPU parity test (small config: 8 layers, dim 64, 2 heads, MoE at layers 1,3,5, 35 experts top-3):
 A) with all 35 experts (self-attention k/v and FFN) = base weights, MoE output == original dense model output (no change at step 0)
 B) log ĝ injection via an extra head dim == reference with an explicit attn_bias (same 4 key/value copies concatenated, attn_mask passed to SDPA), elementwise match
 C) after perturbing experts: at alpha=0 both the self-attention gate and the FFN gate get main-loss gradients (log ĝ path is live); experts and shared k/v/q/o all get gradients
 D) per layer, the attention / FFN selected fractions each sum to 3; total usage = number of tokens x 3
 E) outputs match with GC on and off
Usage: VIDEOX_ATTENTION_TYPE=SDPA CUDA_VISIBLE_DEVICES= python scripts/moe/videoeditor-56b/probe_moe_v3_2_cpu.py
"""
import math, os, sys, torch, torch.nn.functional as F
_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)
torch.manual_seed(0)
from videox_fun.models.wan_transformer3d_kiwi import KiwiWanTransformer3DModel
from wan_transformer3d_kiwi_moe_v3_2 import KiwiMoeTransformer3DModel, MoEKVSelfAttention, rope_apply, route_tokens

cfg = dict(model_type="t2v", patch_size=(1, 2, 2), text_len=8, in_dim=4, dim=64, ffn_dim=128, freq_dim=32,
           text_dim=16, out_dim=4, num_heads=2, num_layers=8, eps=1e-6, in_channels=4, hidden_size=64)
MOE_LAYERS, N, K = (1, 3, 5), 35, 3     # videoeditor-56b: 35 experts top-3 (structure test, weight source irrelevant)
dense = KiwiWanTransformer3DModel(**cfg)
with torch.no_grad():                                        # Wan init_weights zeroes the head; use random weights so the output is non-zero
    dense.head.head.weight.normal_(0, 0.05); dense.head.head.bias.normal_(0, 0.05)
moe = KiwiMoeTransformer3DModel(**cfg, moe_layers=MOE_LAYERS, moe_num_experts=N, moe_top_k=K, moe_aux_loss_alpha=0.001)
missing, unexpected = moe.load_state_dict(dense.state_dict(), strict=False)
allowed = lambda k: any(k.startswith(f"blocks.{i}.") for i in MOE_LAYERS) and (".ffn." in k or ".self_attn." in k)
assert all(allowed(k) for k in list(missing) + list(unexpected)), (missing[:3], unexpected[:3])
with torch.no_grad():                                        # MoE layers: every expert copies the dense weights
    for i in MOE_LAYERS:
        bd, bm = dense.blocks[i], moe.blocks[i]
        for e in bm.ffn.experts: e.load_state_dict(bd.ffn.state_dict())
        for ke in bm.self_attn.k_experts: ke.load_state_dict(bd.self_attn.k.state_dict())
        for ve in bm.self_attn.v_experts: ve.load_state_dict(bd.self_attn.v.state_dict())
dense.requires_grad_(False); moe.requires_grad_(True)

B, f, h, w = 2, 2, 4, 6
x = [torch.randn(4, f, 2 * h, 2 * w) for _ in range(B)]
src = torch.randn(B, 4, f, 2 * h, 2 * w); sigma = torch.rand(B); t = torch.rand(B) * 1000
ctx = [torch.randn(7, 64) for _ in range(B)]; seq_len = f * h * w
ok = True
def d(a, b): return (a.float() - b.float()).abs().max().item()
run = lambda m: m(x=x, t=t, mllm_context=ctx, seq_len=seq_len, vae_source_input=src, sigma=sigma)

with torch.no_grad():
    y_dense = run(dense); moe.eval(); y_moe = run(moe)
print(f"A) experts = base copies: MoE output vs dense max|Δ|={d(y_dense, y_moe):.2e} (|y|max={y_dense.abs().max():.2e})"); ok &= d(y_dense, y_moe) < 1e-4

# B) log ĝ injection vs explicit-bias reference (perturb experts first so the copies differ)
with torch.no_grad():
    for i in MOE_LAYERS:
        for mod in list(moe.blocks[i].ffn.experts) + list(moe.blocks[i].self_attn.k_experts) + list(moe.blocks[i].self_attn.v_experts):
            for p in mod.parameters(): p.add_(torch.randn_like(p) * p.std() * 0.05)
        moe.blocks[i].self_attn.gate.weight.normal_(0, 0.5)   # make the gate non-uniform so ĝ values differ clearly
attn = moe.blocks[MOE_LAYERS[0]].self_attn
xs = torch.randn(B, seq_len, 64)
grid = torch.tensor([[f, h, w]] * B); lens = torch.tensor([seq_len] * B)
def reference(attn, x, grid_sizes, freqs):
    """Reference: same 4 k/v copies concatenated, but the prior is an explicit attn_bias (added to the logits); fp32 SDPA."""
    b, s, n, dd = *x.shape[:2], attn.num_heads, attn.head_dim
    sh = 1 if attn.shared_kv else 0
    copies = sh + attn.top_k; tokens = x.reshape(-1, attn.dim)
    scores, w_, idx = route_tokens(attn.gate, tokens, attn.top_k)
    Kc = torch.stack(([attn.k(tokens)] if sh else []) + [torch.zeros_like(tokens)] * attn.top_k, 1)
    Vc = torch.stack(([attn.v(tokens)] if sh else []) + [torch.zeros_like(tokens)] * attn.top_k, 1)
    prior = torch.zeros(tokens.shape[0], copies)
    for i in range(attn.num_experts):
        ti, sl = (idx == i).nonzero(as_tuple=True)
        Kc[ti, sh + sl] = attn.k_experts[i](tokens[ti]); Vc[ti, sh + sl] = attn.v_experts[i](tokens[ti])
        prior[ti, sh + sl] = torch.log(w_[ti, sl].clamp_min(1e-6))
    q = rope_apply(attn.norm_q(attn.q(x)).view(b, s, n, dd), grid_sizes, freqs)
    Kc = rope_apply(attn.norm_k(Kc).view(b, s, copies * n, dd), grid_sizes, freqs).view(b, s * copies, n, dd)
    Vc = Vc.view(b, s * copies, n, dd)
    bias = prior.view(b, 1, 1, s * copies).expand(b, n, s, s * copies)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), Kc.transpose(1, 2), Vc.transpose(1, 2), attn_mask=bias, scale=1 / math.sqrt(dd))
    return attn.o(out.transpose(1, 2).reshape(b, s, -1))
attn.eval()
with torch.no_grad():
    y_impl = attn(xs, lens, grid, moe.freqs, dtype=torch.float32)
    y_ref = reference(attn, xs, grid, moe.freqs)
print(f"B) log ĝ extra-dim injection vs explicit-bias reference: max|Δ|={d(y_impl, y_ref):.2e} (|y|max={y_ref.abs().max():.2e}); layer min ĝ {attn.last_topk_idx.shape} "
      f"sel={[round(v, 2) for v in attn.last_selected.tolist()]}"); ok &= d(y_impl, y_ref) < 1e-4

# C/D/E) training mode: gradient flow, selected fractions, GC
moe.train()
for gc in (False, True):
    moe.gradient_checkpointing = gc; moe.zero_grad()
    y = run(moe); y.float().pow(2).mean().backward()
    blk = moe.blocks[MOE_LAYERS[0]]
    g = lambda mod: max(p.grad.abs().max().item() for p in mod.parameters() if p.grad is not None)
    g_gate_a, g_gate_f = blk.self_attn.gate.weight.grad.abs().max().item(), blk.ffn.gate.weight.grad.abs().max().item()
    g_ke = max(g(e) for e in blk.self_attn.k_experts); g_ve = max(g(e) for e in blk.self_attn.v_experts); g_fe = max(g(e) for e in blk.ffn.experts)
    g_shared = min(g(blk.self_attn.q), g(blk.self_attn.o))      # shared q / o (without a shared k/v pair, self.k/self.v are unused in forward and get no gradient)
    sel_a = [float(m.last_selected.sum()) for m in moe.moe_attns()]; sel_f = [float(m.last_selected.sum()) for m in moe.moe_ffns()]
    aux = float(sum(m.last_aux_loss for m in moe.moe_attns() + moe.moe_ffns()))
    print(f"C/D) GC={gc}: grad gate_attn={g_gate_a:.2e} gate_ffn={g_gate_f:.2e} k_experts={g_ke:.2e} v_experts={g_ve:.2e} ffn_experts={g_fe:.2e} shared(q/o min)={g_shared:.2e}; "
          f"aux total {aux:.5f} (expect ≈{6 * 0.001:.3f}); per-layer selected-fraction sums attn={['%.2f' % v for v in sel_a]} ffn={['%.2f' % v for v in sel_f]}")
    ok &= g_gate_a > 0 and g_gate_f > 0 and g_ke > 0 and g_ve > 0 and g_fe > 0 and g_shared > 0
    ok &= all(abs(v - K) < 1e-5 for v in sel_a + sel_f)
    if gc: y_gc = y.detach()
    else: y_nogc = y.detach()
print(f"E) GC on/off output max|Δ|={d(y_gc, y_nogc):.2e}"); ok &= d(y_gc, y_nogc) < 1e-5
# at alpha=0 the gates still get gradients (from the main loss via log ĝ / ĝ weighting)
for m in moe.moe_attns() + moe.moe_ffns(): m.aux_loss_alpha = 0.0
moe.zero_grad(); y = run(moe); y.float().pow(2).mean().backward()
g0a, g0f = blk.self_attn.gate.weight.grad.abs().max().item(), blk.ffn.gate.weight.grad.abs().max().item()
print(f"   alpha=0 gate grads attn={g0a:.2e} ffn={g0f:.2e} (both should be > 0: main-loss gradient reaches the gates via log ĝ / ĝ)"); ok &= g0a > 0 and g0f > 0
print("RESULT:", "PASS" if ok else "FAIL")
