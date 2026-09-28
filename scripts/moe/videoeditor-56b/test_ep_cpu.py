#!/usr/bin/env python
"""CPU parity test for expert parallelism (EP) (gloo, 4 processes = 2 "nodes" x 2 GPUs). Same expert weights, different inputs per
rank; the EP MoEFFN / MoEKVSelfAttention are compared with the original modules on:
  (1) outputs element-wise (fp32, tolerance 1e-5); (2) gradient w.r.t. input x; (3) expert weight grads: the EP shard after _reduce_grads
     (paired all_reduce + DP sharding, / world) == the matching shard of the global mean of the original per-rank grads; (4) one optimizer
     step: weights after ExpertStore.step == one AdamW step on the same mean grads (torch.optim.AdamW as reference);
  (5) replica self-check verify_replicas passes; (6) aux loss / last_selected / last_topk_idx bit-identical.
Usage: CUDA_VISIBLE_DEVICES= python scripts/moe/videoeditor-56b/test_ep_cpu.py   (CPU only, about 1 minute)"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DIM, FFN, HEADS, N, K, ALPHA = 64, 128, 4, int(os.environ.get("EP_TEST_N", 5)), 3, 0.01
LOCAL = int(os.environ.get("EP_TEST_LOCAL", 2))    # EP_TEST_N/LOCAL/WORLD select the layout; unpaired mode = N divisible by LOCAL (e.g. N=6 LOCAL=3 WORLD=6)
WORLD = int(os.environ.get("EP_TEST_WORLD", 4))
B, GRID = 2, (2, 2, 3)
S = GRID[0] * GRID[1] * GRID[2]


def make_ref(seed):
    import wan_transformer3d_kiwi_moe_v3_2 as T
    torch.manual_seed(seed)
    ffn = T.MoEFFN(DIM, FFN, N, K, ALPHA)
    attn = T.MoEKVSelfAttention(DIM, HEADS, (-1, -1), True, 1e-6, N, K, ALPHA)
    return ffn, attn


def run(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = os.environ.get("EP_TEST_PORT", "29911")
    os.environ["VIDEOX_ATTENTION_TYPE"] = "SDPA"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(4)
    import wan_transformer3d_kiwi_moe_v3_2 as T
    import moe_v3_2_ep as EP
    device = torch.device("cpu")
    layout = EP.EPLayout(N, world, rank, local_world=LOCAL)
    layout.build_groups()
    # ---- reference modules (same weight seed, identical on all ranks)
    ref_ffn, ref_attn = make_ref(0)
    # ---- EP modules + expert store: copy weights from the reference modules
    store = EP.ExpertStore(layout, [0], DIM, FFN, device, dtype=torch.float32)
    with torch.no_grad():
        for e in layout.my_experts:
            w1, b1, w2, b2 = store.ffn_weights(0, e)
            w1.copy_(ref_ffn.experts[e][0].weight); b1.copy_(ref_ffn.experts[e][0].bias); w2.copy_(ref_ffn.experts[e][2].weight); b2.copy_(ref_ffn.experts[e][2].bias)
            wk, bk, wv, bv = store.kv_weights(0, e)
            wk.copy_(ref_attn.k_experts[e].weight); bk.copy_(ref_attn.k_experts[e].bias); wv.copy_(ref_attn.v_experts[e].weight); bv.copy_(ref_attn.v_experts[e].bias)
    EP.install(store, layout)
    ep_ffn = EP.EPMoEFFN(DIM, FFN, N, K, ALPHA); ep_ffn.block_index = 0
    ep_attn = EP.EPMoEKVSelfAttention(DIM, HEADS, (-1, -1), True, 1e-6, N, K, ALPHA); ep_attn.block_index = 0
    with torch.no_grad():
        ep_ffn.gate.weight.copy_(ref_ffn.gate.weight); ep_attn.gate.weight.copy_(ref_attn.gate.weight)
        ep_attn.q.weight.copy_(ref_attn.q.weight); ep_attn.q.bias.copy_(ref_attn.q.bias); ep_attn.o.weight.copy_(ref_attn.o.weight); ep_attn.o.bias.copy_(ref_attn.o.bias)
        ep_attn.norm_q.weight.copy_(ref_attn.norm_q.weight); ep_attn.norm_k.weight.copy_(ref_attn.norm_k.weight)
    assert store.verify_replicas(), "replica self-check failed"
    # ---- inputs (different on each rank)
    torch.manual_seed(100 + rank)
    x_ref = torch.randn(B, S, DIM, requires_grad=True)
    x_ep = x_ref.detach().clone().requires_grad_(True)
    wsum = torch.randn(B, S, DIM)
    d = DIM // HEADS
    freqs = torch.cat([T.rope_params(1024, d - 4 * (d // 6)), T.rope_params(1024, 2 * (d // 6)), T.rope_params(1024, 2 * (d // 6))], dim=1)
    grid_sizes = torch.tensor([GRID] * B, dtype=torch.long)
    seq_lens = torch.tensor([S] * B, dtype=torch.long)
    report = {}
    for name, ref_m, ep_m, call in (("ffn", ref_ffn, ep_ffn, lambda m, x: m(x)),
                                    ("attn", ref_attn, ep_attn, lambda m, x: m(x, seq_lens, grid_sizes, freqs, dtype=torch.float32))):
        ref_m.train(); ep_m.train()
        for p in ref_m.parameters():
            p.grad = None
        x_ref.grad = None; x_ep.grad = None; store.zero_grad()
        y_ref = call(ref_m, x_ref); (y_ref * wsum).sum().backward()
        y_ep = call(ep_m, x_ep); (y_ep * wsum).sum().backward()
        out_diff = (y_ref - y_ep).abs().max().item()
        xg_diff = (x_ref.grad - x_ep.grad).abs().max().item()
        aux_same = torch.equal(ref_m.last_aux_loss, ep_m.last_aux_loss) and torch.equal(ref_m.last_selected, ep_m.last_selected) and torch.equal(ref_m.last_topk_idx, ep_m.last_topk_idx)
        # expert grads: reference = global mean of per-rank grads; EP = shard from _reduce_grads
        ref_grads = {}
        for e in range(N):
            mods = [(f"blocks.0.ffn.experts.{e}.0.weight", ref_ffn.experts[e][0].weight), (f"blocks.0.ffn.experts.{e}.0.bias", ref_ffn.experts[e][0].bias),
                    (f"blocks.0.ffn.experts.{e}.2.weight", ref_ffn.experts[e][2].weight), (f"blocks.0.ffn.experts.{e}.2.bias", ref_ffn.experts[e][2].bias)] if name == "ffn" else \
                   [(f"blocks.0.self_attn.k_experts.{e}.weight", ref_attn.k_experts[e].weight), (f"blocks.0.self_attn.k_experts.{e}.bias", ref_attn.k_experts[e].bias),
                    (f"blocks.0.self_attn.v_experts.{e}.weight", ref_attn.v_experts[e].weight), (f"blocks.0.self_attn.v_experts.{e}.bias", ref_attn.v_experts[e].bias)]
            for key, p in mods:
                g = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                dist.all_reduce(g); ref_grads[key] = g / world
        if store.opt is None:
            store.build_optimizer(lr=1e-3, weight_decay=0.01)
            store.install_grad_hooks()                                       # as in training: hooks reduce during backward (this backward already ran, so _reduce_grads handles the first round)
        sq = store._reduce_grads()
        grads = [m.grad.clone() for m in store.masters]                     # shard grads are already in the pinned m.grad
        g_diff = 0.0
        for key, g in zip(store.shard_keys, grads):
            if (name == "ffn") != (".ffn." in key):
                continue
            L = store.shard_len[key]; full = ref_grads[key].reshape(-1)
            ref_chunk = torch.zeros(L); s0, e0 = layout.node * L, min((layout.node + 1) * L, full.numel())
            if e0 > s0: ref_chunk[:e0 - s0] = full[s0:e0]
            g_diff = max(g_diff, (g - ref_chunk).abs().max().item())
        report[name] = dict(out=out_diff, xgrad=xg_diff, stats_same=aux_same, shard_grad=g_diff)
    # ---- one optimizer step (needs fresh grads: recomputing the ffn grads once is enough)
    store.zero_grad(); x_ep.grad = None
    y = ep_ffn(x_ep); (y * wsum).sum().backward()
    sq = store._reduce_grads()
    grads = [m.grad.clone() for m in store.masters]
    # reference: copy the full weights, one AdamW step on the same mean grads (same hyperparameters as DeepSpeedCPUAdam)
    ref_params = {k: store.tensors[k].detach().clone() for k in store.shard_keys}
    ref_g = {}
    for key, g in zip(store.shard_keys, grads):
        L = store.shard_len[key]; full = torch.zeros(L * layout.n_nodes)
        parts = [torch.zeros(L) for _ in range(layout.n_nodes)]
        dist.all_gather(parts, g.contiguous(), group=layout.dp_group) if layout.n_nodes > 1 else parts.__setitem__(0, g)
        ref_g[key] = torch.cat(parts)[:ref_params[key].numel()].reshape(ref_params[key].shape)
    plist = [torch.nn.Parameter(ref_params[k].clone()) for k in store.shard_keys]
    for p, k in zip(plist, store.shard_keys): p.grad = ref_g[k].clone()
    torch.optim.AdamW(plist, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01).step()
    nsq = store.expert_norm_sq(sq)
    store.step(clip_coef=1.0, lr=1e-3)
    upd_diff = max((store.tensors[k].detach() - p.detach()).abs().max().item() for p, k in zip(plist, store.shard_keys))
    rep_ok = store.verify_replicas()
    report["opt_step_vs_torch_adamw"] = upd_diff; report["replicas_ok_after_step"] = rep_ok
    # ---- full hook-path coverage: joint backward of ffn + attn in one block; here 3 experts x 8 = 24 tensors all arrive -> _reduce_block fires inside the hook (no fallback)
    store.zero_grad(); x_ep.grad = None; store.begin_step()
    with torch.no_grad():                                                    # the optimizer step above only updated the EP weights, not the reference -> first reset EP weights to the reference values
        for e in layout.my_experts:
            w1, b1, w2, b2 = store.ffn_weights(0, e)
            w1.copy_(ref_ffn.experts[e][0].weight); b1.copy_(ref_ffn.experts[e][0].bias); w2.copy_(ref_ffn.experts[e][2].weight); b2.copy_(ref_ffn.experts[e][2].bias)
            wk, bk, wv, bv = store.kv_weights(0, e)
            wk.copy_(ref_attn.k_experts[e].weight); bk.copy_(ref_attn.k_experts[e].bias); wv.copy_(ref_attn.v_experts[e].weight); bv.copy_(ref_attn.v_experts[e].bias)
    for p in list(ref_ffn.parameters()) + list(ref_attn.parameters()):
        p.grad = None
    x_ref.grad = None
    y = ep_ffn(x_ep) + ep_attn(x_ep, seq_lens, grid_sizes, freqs, dtype=torch.float32); (y * wsum).sum().backward()
    hooked = len(store._block_reduced) == 1 and not [b for b in store.blocks if b not in store._block_reduced]
    sq = store._reduce_grads()
    yr = ref_ffn(x_ref) + ref_attn(x_ref, seq_lens, grid_sizes, freqs, dtype=torch.float32); (yr * wsum).sum().backward()
    ref_all = {}
    for e in range(N):
        for key, pp in ([(f"blocks.0.ffn.experts.{e}.0.weight", ref_ffn.experts[e][0].weight), (f"blocks.0.ffn.experts.{e}.0.bias", ref_ffn.experts[e][0].bias),
                         (f"blocks.0.ffn.experts.{e}.2.weight", ref_ffn.experts[e][2].weight), (f"blocks.0.ffn.experts.{e}.2.bias", ref_ffn.experts[e][2].bias),
                         (f"blocks.0.self_attn.k_experts.{e}.weight", ref_attn.k_experts[e].weight), (f"blocks.0.self_attn.k_experts.{e}.bias", ref_attn.k_experts[e].bias),
                         (f"blocks.0.self_attn.v_experts.{e}.weight", ref_attn.v_experts[e].weight), (f"blocks.0.self_attn.v_experts.{e}.bias", ref_attn.v_experts[e].bias)]):
            g = pp.grad.detach().clone() if pp.grad is not None else torch.zeros_like(pp); dist.all_reduce(g); ref_all[key] = g / world
    hook_diff = 0.0
    for key, m in zip(store.shard_keys, store.masters):
        L = store.shard_len[key]; full = ref_all[key].reshape(-1); ref_chunk = torch.zeros(L); s0, e0 = layout.node * L, min((layout.node + 1) * L, full.numel())
        if e0 > s0: ref_chunk[:e0 - s0] = full[s0:e0]
        hook_diff = max(hook_diff, (m.grad - ref_chunk).abs().max().item())
    report["hook_path_block_reduced_in_backward"] = hooked; report["hook_path_shard_grad"] = hook_diff
    ok_hook = hooked and hook_diff < 1e-5
    print(f"rank{rank} {layout.describe()}\n  {report}", flush=True)
    ok = all(report[n]["out"] < 1e-5 and report[n]["xgrad"] < 1e-5 and report[n]["stats_same"] and report[n]["shard_grad"] < 1e-5 for n in ("ffn", "attn")) and upd_diff < 1e-5 and rep_ok and ok_hook
    flag = torch.tensor([0 if ok else 1]); dist.all_reduce(flag)
    if rank == 0:
        print("EP_CPU_PARITY", "PASS" if int(flag.item()) == 0 else "FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, args=(WORLD,), nprocs=WORLD, join=True)
