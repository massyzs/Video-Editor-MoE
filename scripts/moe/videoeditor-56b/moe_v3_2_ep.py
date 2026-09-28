"""videoeditor-56b expert parallelism (EP): replaces ZeRO-3, which shards all 35 experts and re-gathers them in full every step.

Layout (pairwise sharing; with ep_size=24 the cross-node all_to_all bytes of the 3 top-3 copies are barely lower than ZeRO-3's, hence intra-node EP):
  · EP group = the 8 GPUs of one node (all_to_all stays on NVLink); each of the 3 nodes holds a full replica of all 35 experts (expert data parallelism,
    DP group = the 3 GPUs with the same local_rank across nodes).
  · The 8 GPUs of a node form 4 pairs; the 35 experts are split 9/9/9/8 over the pairs. In a pair, k experts = k//2 on the even GPU only + k//2 on the odd GPU
    only + (k%2) with a copy on each ("shared"; the sender splits its tokens between the two GPUs by position parity). Each GPU does 4.5 or 4 experts of work per
    layer (ideal 4.375, at most 3% over) instead of 5 vs 4 (14%).
  · Expert weights are plain bf16 GPU tensors (requires_grad) outside the engine module; DeepSpeed/ZeRO-3 only manages the dense 3.5B + gates; autograd
    produces expert gradients as usual.
  · Expert optimizer: sharded within the DP group (ZeRO-1 style): grads are reduce_scattered over the DP group (shared experts are first all_reduced across the
    pair), ÷world gives the same "gradient averaged over 24 GPUs" as DeepSpeed; each GPU updates only its 1/3 shard (fp32 master weights + Adam m/v in CPU
    pinned memory, DeepSpeedCPUAdam), then the bf16 shards are all_gathered back to full. Shared experts are updated by the even GPU; the odd GPU gets the
    updated weights via broadcast. CPU Adam params per GPU ≈ 4.5 × 14 × 107M / 3 ≈ 2.25B, comparable to 2.33B under ZeRO-3.
  · Clipping: the expert grad sum of squares (each shard counted once) is all_reduced globally and merged with the dense norm by a patch of DeepSpeed's
    unscale_and_clip_grads; the same coefficient is applied to both.
Per-layer data flow (EPMoEFFN / EPMoEKVSelfAttention.forward):
  local gate top-3 → dispatch_by_expert orders (token, slot) → dest GPU = expert holder (shared experts alternate by in-group position parity) → pack sorted
  by dest GPU → exchange counts (all_to_all of a small int tensor) → all_to_all_single (differentiable, torch.distributed.nn.functional) sends tokens and
  expert ids → sort locally by expert id, compute per expert → unsort → all_to_all back → one index_add_ per slot (fp32, no duplicate indices,
  deterministic) / index_put_ to fill K, V.
Gradient-checkpoint recomputation runs the all_to_all again: every GPU does it for the same block in the same order, so this is safe.
Saving: weights = dense safetensors (gathered by DeepSpeed, keys unchanged) + each of node 0's 8 GPUs writes its held experts to
experts_ep_lr{local_rank}.safetensors (same keys as the full file; shared experts written by the even GPU); merge_ep_ckpt.py merges them back into
moe_v3_2_trainable.safetensors for inference. Optimizer = DeepSpeed state (dense) + per-GPU ep_opt_rank{rank}.pt (fp32 master shard + m + v).
"""
import contextlib
import json
import math
import os
import re
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_to_all_single as all_to_all_single_autograd
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from wan_transformer3d_kiwi_moe_v3_2 import (AddAuxiliaryLoss, Fp32Gate, MoEFFN, MoEKVSelfAttention, WanSelfAttention, attention_with_scale,
                                             expert_balance_loss, rope_apply, route_tokens)

TAG = "[moe-v3.2-ep]"
FFN_PROJS = (("ffn.experts.{e}.0.weight", ("ffn", "dim")), ("ffn.experts.{e}.0.bias", ("ffn",)), ("ffn.experts.{e}.2.weight", ("dim", "ffn")), ("ffn.experts.{e}.2.bias", ("dim",)))
KV_PROJS = (("self_attn.k_experts.{e}.weight", ("dim", "dim")), ("self_attn.k_experts.{e}.bias", ("dim",)),
            ("self_attn.v_experts.{e}.weight", ("dim", "dim")), ("self_attn.v_experts.{e}.bias", ("dim",)))


# =========================================================================== #
#  Layout: expert -> GPU, three process groups                                #
# =========================================================================== #
class EPLayout:
    def __init__(self, num_experts, world, rank, local_world=8):
        """EP group = local_world consecutive ranks (with 8 GPUs per node all_to_all stays on NVLink; with 28 GPUs = 4×7 groups of 7 may span nodes);
        n_nodes keeps its old name = number of EP groups (expert DP replicas), node = this rank's group index."""
        assert world % local_world == 0, (world, local_world)
        self.num_experts, self.world, self.rank, self.local_world = num_experts, world, rank, local_world
        self.n_nodes, self.node, self.local_rank = world // local_world, rank // local_world, rank % local_world
        # No pairing when num_experts is divisible by the group size (35 / 7 = 5): each GPU holds the 5 experts with e % local_world == local_rank
        # (round-robin: the 5 copies of one source are numbered consecutively, so each GPU gets 5 different sources, no hot pair); no shared experts, no pair groups.
        self.no_pair = (num_experts % local_world == 0)
        self.owner = {}                     # expert -> (kind, even_lr, odd_lr); kind ∈ {"unique", "shared"}; odd_lr = None for unique
        if self.no_pair:
            for ex in range(num_experts):
                self.owner[ex] = ("unique", ex % local_world, None)
        else:
            assert local_world % 2 == 0, "pairing is required when num_experts is not divisible by the group size, so the group size must be even"
            n_pairs = local_world // 2
            # Experts go round-robin to the 4 pairs by e % n_pairs (the 5 copies of one source are numbered consecutively; a contiguous split would make the
            # direct_finetune pair a 1.3× hot spot); in a pair the first k//2 go to the even GPU, the next k//2 to the odd GPU, 1 left over is shared. 35 / 4 pairs → 9, 9, 9, 8.
            n_assigned = 0
            for p in range(n_pairs):
                mine = [ex for ex in range(num_experts) if ex % n_pairs == p]
                k = len(mine); even, odd = 2 * p, 2 * p + 1
                for ex in mine[:k // 2]:
                    self.owner[ex] = ("unique", even, None)
                for ex in mine[k // 2:2 * (k // 2)]:
                    self.owner[ex] = ("unique", odd, None)
                if k % 2:
                    self.owner[mine[-1]] = ("shared", even, odd)
                n_assigned += k
            assert n_assigned == num_experts, (n_assigned, num_experts)
        assert len(self.owner) == num_experts, (len(self.owner), num_experts)
        self.held = {lr: [] for lr in range(local_world)}       # local_rank -> held experts (incl. shared)
        for ex, (kind, a, b) in self.owner.items():
            self.held[a].append(ex)
            if kind == "shared":
                self.held[b].append(ex)
        self.my_experts = sorted(self.held[self.local_rank])
        self.my_shared = [ex for ex in self.my_experts if self.owner[ex][0] == "shared"]
        # optimizer ownership: unique → holder; shared → even GPU (the odd GPU only receives the broadcast)
        self.opt_owned = [ex for ex in self.my_experts if self.owner[ex][0] == "unique" or self.owner[ex][1] == self.local_rank]
        # routing tables (GPU tensors built per device on demand): base_lr[e] = first holder, is_shared[e]
        self.base_lr = [self.owner[ex][1] for ex in range(num_experts)]
        self.is_shared = [1 if self.owner[ex][0] == "shared" else 0 for ex in range(num_experts)]
        self.pair_even = None if self.no_pair else self.local_rank - (self.local_rank % 2)
        self.ep_group = self.dp_group = self.pair_group = None
        # fewer routing syncs: slot of expert e on holder lr (= its index in that GPU's sorted held list); the sender packs by (dest GPU, slot), the receiver slices by counts alone
        self.n_slots = max(len(v) for v in self.held.values())
        self.slot_of = {}                       # (e, lr) -> slot
        for lr, exs in self.held.items():
            for si, ex in enumerate(sorted(exs)):
                self.slot_of[(ex, lr)] = si

    def build_groups(self, timeout_minutes=60):
        """All ranks create groups in the same order (new_group is a collective); timeout is 60 min like the main process group (with the 10-min default a slow
        checkpoint save would time out in a subgroup first)."""
        if not dist.is_initialized() or self.world == 1:
            return
        from datetime import timedelta
        to = timedelta(minutes=timeout_minutes)
        for n in range(self.n_nodes):
            g = dist.new_group(list(range(n * self.local_world, (n + 1) * self.local_world)), timeout=to)
            if n == self.node:
                self.ep_group = g
        for lr in range(self.local_world):
            g = dist.new_group([n * self.local_world + lr for n in range(self.n_nodes)], timeout=to)
            if lr == self.local_rank:
                self.dp_group = g
        if self.no_pair:
            return                                                          # no pairing: no shared experts, no pair groups (pair_group stays None and every caller takes the None branch)
        for n in range(self.n_nodes):
            for p in range(self.local_world // 2):
                g = dist.new_group([n * self.local_world + 2 * p, n * self.local_world + 2 * p + 1], timeout=to)
                if n == self.node and p == self.local_rank // 2:
                    self.pair_group = g

    def route_tables(self, device):
        """Small routing tensors, built once per device: base_lr[e], is_shared[e], slot[e, lr] (slot of expert e on GPU lr, -1 if not held)."""
        key = str(device)
        if not hasattr(self, "_tables") or key not in self._tables:
            self._tables = getattr(self, "_tables", {})
            slot = torch.full((self.num_experts, self.local_world), -1, dtype=torch.long)
            for (ex, lr), si in self.slot_of.items():
                slot[ex, lr] = si
            self._tables[key] = (torch.tensor(self.base_lr, device=device, dtype=torch.long), torch.tensor(self.is_shared, device=device, dtype=torch.long),
                                 slot.to(device))
        return self._tables[key]

    def describe(self):
        return (f"EP: world {self.world} = {self.n_nodes} groups × {self.local_world}{' (no pairing)' if self.no_pair else ''}, rank {self.rank} (group {self.node}, "
                f"local {self.local_rank}) holds experts {self.my_experts} (shared {self.my_shared}), optimizer owns {self.opt_owned}, 1/{self.n_nodes} shard each")


# =========================================================================== #
#  Expert weights + optimizer                                                 #
# =========================================================================== #
class ExpertStore:
    """Expert weights held by this GPU (bf16 on GPU, requires_grad, not nn.Parameter, invisible to the engine) and their sharded optimizer."""

    def __init__(self, layout, moe_layers, dim, ffn_dim, device, dtype=torch.bfloat16):
        self.layout, self.moe_layers, self.dim, self.ffn_dim, self.device, self.dtype = layout, list(moe_layers), dim, ffn_dim, device, dtype
        self.shapes = {"dim": dim, "ffn": ffn_dim}
        self.tensors = {}                   # key (same as in the full safetensors) -> bf16 GPU tensor
        for block in self.moe_layers:
            for e in layout.my_experts:
                for tmpl, shp in FFN_PROJS + KV_PROJS:
                    key = f"blocks.{block}.{tmpl.format(e=e)}"
                    t = torch.zeros(tuple(self.shapes[s] for s in shp), dtype=dtype, device=device)   # zeros, not empty: a missed load shows up at once in the self-check / loss
                    t.requires_grad_(True)
                    self.tensors[key] = t
        self.numel = sum(t.numel() for t in self.tensors.values())
        self.opt = None
        self.clip_coef = 1.0
        self.last_expert_norm_sq = 0.0
        self.timing = {}

    # ---- weights of one expert in one layer (used by the EP modules' forward)
    def ffn_weights(self, block, e):
        p = f"blocks.{block}.ffn.experts.{e}."
        return self.tensors[p + "0.weight"], self.tensors[p + "0.bias"], self.tensors[p + "2.weight"], self.tensors[p + "2.bias"]

    def kv_weights(self, block, e):
        p = f"blocks.{block}.self_attn."
        return (self.tensors[p + f"k_experts.{e}.weight"], self.tensors[p + f"k_experts.{e}.bias"],
                self.tensors[p + f"v_experts.{e}.weight"], self.tensors[p + f"v_experts.{e}.bias"])

    # ---- load / save weights
    def load_from_safetensors(self, path, verbose=False):
        """Read only the keys this GPU needs from the full moe_v3_2_trainable.safetensors (safe_open reads per key; 24 concurrent processes read ~15 GB each)."""
        from safetensors import safe_open
        t0 = time.time()
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            missing = [k for k in self.tensors if k not in keys]
            assert not missing, f"full checkpoint is missing expert keys {missing[:3]}"
            for k, t in self.tensors.items():
                with torch.no_grad():
                    t.copy_(f.get_tensor(k).to(t.dtype))
        self.check_finite("after load")
        if verbose:
            print(f"{TAG} loaded expert weights: {len(self.tensors)} tensors {self.numel / 1e9:.2f}B in {time.time() - t0:.0f}s", flush=True)

    def load_from_ep_shards(self, step_dir):
        """Load this GPU's experts from an EP checkpoint dir (experts_ep_lr{lr}.safetensors, written by node 0): held keys are in the lr file (shared experts in the
        even GPU's file)."""
        from safetensors import safe_open
        import glob as _glob
        files = sorted(_glob.glob(os.path.join(step_dir, "experts_ep_lr*.safetensors")))     # scan all shards and pick by key, independent of layout (old checkpoints still resume after a layout change)
        assert files, f"no EP shard files in: {step_dir}"
        loaded = set()
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k in self.tensors and k not in loaded:
                        with torch.no_grad():
                            self.tensors[k].copy_(f.get_tensor(k).to(self.dtype))
                        loaded.add(k)
        missing = [k for k in self.tensors if k not in loaded]
        assert not missing, f"EP checkpoint is missing keys {missing[:3]}"
        self.check_finite("after EP checkpoint load")

    def save_shards(self, step_dir):
        """Each of node 0's 8 GPUs writes the experts it holds (shared experts only by the even GPU); other nodes are replicas and do not write. All ranks must call
        this (barrier)."""
        from safetensors.torch import save_file
        if self.layout.node == 0:
            picked = {}
            for e in self.layout.my_experts:
                if self.layout.owner[e][0] == "shared" and self.layout.local_rank % 2 == 1:
                    continue
                for block in self.moe_layers:
                    for tmpl, _ in FFN_PROJS + KV_PROJS:
                        k = f"blocks.{block}.{tmpl.format(e=e)}"
                        picked[k] = self.tensors[k].detach().to(torch.bfloat16).cpu().contiguous()
            os.makedirs(step_dir, exist_ok=True)
            final = os.path.join(step_dir, f"experts_ep_lr{self.layout.local_rank}.safetensors")
            save_file(picked, final + ".tmp")
            os.replace(final + ".tmp", final)                                                       # atomic write: a partial file is never taken for a checkpoint
            with open(os.path.join(step_dir, f"experts_ep_lr{self.layout.local_rank}.json"), "w") as f:
                json.dump({"experts": sorted({expert_id_of(k) for k in picked if ".ffn.experts." in k}), "num_keys": len(picked),
                           "layout": {"local_world": self.layout.local_world, "num_experts": self.layout.num_experts}}, f)
        if dist.is_initialized():
            dist.barrier()

    def check_finite(self, where):
        bad = [k for k, t in self.tensors.items() if not torch.isfinite(t).all()]
        assert not bad, f"{where}: non-finite expert weights {bad[:3]}"

    def checksum(self):
        """fp64 sum over all held experts (self-check: shared experts match across the pair, replicas match across nodes)."""
        return sum(float(t.detach().double().sum()) for t in self.tensors.values())

    def verify_replicas(self):
        """Self-check: shared experts are bitwise identical on both GPUs of a pair; node replicas are bitwise identical (compares each tensor's fp64 sum)."""
        if not dist.is_initialized() or self.layout.world == 1:
            return True
        sums = torch.tensor([float(self.tensors[k].detach().double().sum()) for k in sorted(self.tensors)], dtype=torch.float64, device=self.device)
        ok = True
        if self.layout.dp_group is not None:
            gathered = [torch.zeros_like(sums) for _ in range(self.layout.n_nodes)]
            dist.all_gather(gathered, sums, group=self.layout.dp_group)
            ok &= all(torch.equal(g, sums) for g in gathered)
        if self.layout.my_shared and self.layout.pair_group is not None:
            keys = sorted(k for k in self.tensors if expert_id_of(k) in self.layout.my_shared)      # includes k_experts/v_experts (a substring match would miss k/v)
            s2 = torch.tensor([float(self.tensors[k].detach().double().sum()) for k in keys], dtype=torch.float64, device=self.device)
            g2 = [torch.zeros_like(s2) for _ in range(2)]
            dist.all_gather(g2, s2, group=self.layout.pair_group)
            ok &= torch.equal(g2[0], g2[1])
        flag = torch.tensor([0 if ok else 1], device=self.device)
        dist.all_reduce(flag)
        return int(flag.item()) == 0

    # ---- gradients
    def zero_grad(self):
        for t in self.tensors.values():
            t.grad = None

    # ---- optimizer (sharded within the DP group)
    def build_optimizer(self, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        """For each tensor of the experts this GPU owns (opt_owned) take shard number `node` (flattened, split evenly into n_nodes, zero-padded at the tail);
        fp32 master weights + DeepSpeedCPUAdam state live in CPU pinned memory."""
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        n = self.layout.n_nodes
        self.shard_keys = [k for k in sorted(self.tensors) if expert_id_of(k) in self.layout.opt_owned]
        self.shard_len = {}
        self.masters = []
        pin = self.device.type == "cuda"      # pinned host memory needs a CUDA device; the CPU parity test (test_ep_cpu.py) runs without one
        for k in self.shard_keys:
            t = self.tensors[k]
            L = (t.numel() + n - 1) // n
            self.shard_len[k] = L
            m = torch.zeros(L, dtype=torch.float32, device="cpu", pin_memory=pin)
            flat = t.detach().reshape(-1)
            s, e = self.layout.node * L, min((self.layout.node + 1) * L, t.numel())
            if e > s:
                m[:e - s].copy_(flat[s:e].float())
            m.grad = torch.zeros(L, dtype=torch.float32, device="cpu", pin_memory=pin)
            self.masters.append(nn.Parameter(m, requires_grad=True))
            self.masters[-1].grad = m.grad
        self.opt = DeepSpeedCPUAdam(self.masters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adamw_mode=True)
        self.shard_numel = sum(self.shard_len[k] for k in self.shard_keys)
        self._shard_index = {k: i for i, k in enumerate(self.shard_keys)}
        self._use_rs = dist.is_initialized() and self.layout.n_nodes > 1 and dist.get_backend(self.layout.dp_group) == "nccl"
        # Communication is bucketed per LAYER (560 small per-tensor collectives per step made the first EP run take 28 s/step): per layer one pair all_reduce
        # (shared experts), one DP reduce_scatter, one all_gather, one broadcast, 56 per step in total. Bucket layout: big[n_nodes, Σ_k Ls_k], row j = the j-th
        # shard of each key concatenated; after reduce_scatter this node gets its own row (= its shard of each key, concatenated); all_gather uses the same layout.
        block_of = lambda k: int(k.split(".")[1])
        self.blocks = sorted({block_of(k) for k in self.tensors})
        self.held_keys = {b: sorted(k for k in self.tensors if block_of(k) == b) for b in self.blocks}
        self.owned_keys = {b: [k for k in self.shard_keys if block_of(k) == b] for b in self.blocks}
        self.shared_keys = {b: [k for k in self.held_keys[b] if expert_id_of(k) in self.layout.my_shared] for b in self.blocks}
        self.owned_off, self.owned_total, self.layer_pin = {}, {}, {}
        for b in self.blocks:
            off, offs = 0, {}
            for k in self.owned_keys[b]:
                offs[k] = off
                off += self.shard_len[k]
            self.owned_off[b], self.owned_total[b] = offs, off
            self.layer_pin[b] = torch.empty(off, dtype=self.dtype, device="cpu", pin_memory=pin)      # per-layer pinned staging for copying updated shards back to the GPU
        # collectives issued from hooks run on a side stream so the compute stream does not wait for NCCL (otherwise every layer's backward kernels queue behind
        # all_reduce/reduce_scatter and the 24 GPUs run in per-layer lockstep)
        self.comm_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.begin_step()

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def _grad_of(self, k):
        t = self.tensors[k]
        return t.grad if t.grad is not None else torch.zeros_like(t)

    def _reduce_block(self, b):
        """Gradient handling for all held experts of one layer (block b); called from the hook when the layer's last tensor arrives, or as a fallback after backward:
        (1) shared experts: the 8 tensors are flattened and all_reduced once across the pair; (2) optimizer-owned keys: packed into big[n_nodes, Σ Ls] (row j =
        the j-th shard of each key), one DP reduce_scatter → this node's row = its shard of each key; ÷world (same convention as DeepSpeed), cast to fp32 →
        accumulate the sum of squares → per-key D2H into the pinned m.grad; (3) free all GPU grads of the layer."""
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        L = self.layout
        n = L.n_nodes
        cs = getattr(self, "comm_stream", None)
        if cs is not None:
            cs.wait_stream(torch.cuda.current_stream(self.device))                # the layer's grads are ready on the compute stream
        with (torch.cuda.stream(cs) if cs is not None else contextlib.nullcontext()):   # everything below runs on the side stream: temporaries are allocated there, D2H too
            if self.shared_keys[b] and L.pair_group is not None:
                grads = [self._grad_of(k) for k in self.shared_keys[b]]
                flat = _flatten_dense_tensors(grads)
                dist.all_reduce(flat, group=L.pair_group)
                for k, g, u in zip(self.shared_keys[b], grads, _unflatten_dense_tensors(flat, grads)):
                    if self.tensors[k].grad is None:
                        self.tensors[k].grad = u.clone()
                    else:
                        self.tensors[k].grad.copy_(u)
                del flat, grads
            keys = self.owned_keys[b]
            if keys:
                total = self.owned_total[b]
                big = torch.empty(n, total, dtype=self.dtype, device=self.device)
                for k in keys:
                    Ls, off = self.shard_len[k], self.owned_off[b][k]
                    flat = self._grad_of(k).reshape(-1)
                    if flat.numel() != Ls * n:
                        flat = torch.cat([flat, flat.new_zeros(Ls * n - flat.numel())])
                    big[:, off:off + Ls].copy_(flat.view(n, Ls))
                out = torch.empty(total, dtype=self.dtype, device=self.device)
                if n == 1:
                    out.copy_(big[0])
                elif self._use_rs:
                    dist.reduce_scatter_tensor(out, big.reshape(-1), group=L.dp_group)
                else:
                    flatb = big.reshape(-1).contiguous()
                    dist.all_reduce(flatb, group=L.dp_group)
                    out.copy_(flatb.view(n, total)[L.node])
                del big
                for k in keys:
                    Ls, off = self.shard_len[k], self.owned_off[b][k]
                    g32 = out[off:off + Ls].float().div_(L.world)
                    self._sq += g32.square().sum(dtype=torch.float64)          # no fp64 copy (saves ~350 MB of temporaries per key)
                    self.masters[self._shard_index[k]].grad.copy_(g32, non_blocking=True)   # D2H into pinned memory (same stream, safe by stream order)
                del out
        for k in self.held_keys[b]:
            g = self.tensors[k].grad
            if g is not None and cs is not None:
                g.record_stream(cs)                                            # tell the allocator before freeing: the side stream is still reading it
            self.tensors[k].grad = None                                        # free this layer's bf16 grads right after use
        self._block_reduced.add(b)

    def _on_grad(self, k):
        b = int(k.split(".")[1])
        assert k not in self._done_keys, f"{k} accumulated its grad twice in one backward (it would be dropped or carried into the next step; this design only supports gradient_accumulation_steps=1 and one forward per module per step)"
        self._done_keys.add(k)
        self._block_done[b] = self._block_done.get(b, 0) + 1
        if self._block_done[b] == len(self.held_keys[b]) and b not in self._block_reduced:
            self._reduce_block(b)

    def install_grad_hooks(self):
        """Fixes an OOM at batch 5: register_post_accumulate_grad_hook reduces and frees each layer's expert grads as soon as the layer is complete, without waiting
        for the end of backward, so not all expert grads are resident at the peak (−12~15 GB per GPU), and the DP reduce_scatter overlaps with backward compute.
        Backward runs blocks from last to first; one communication is issued once all 40 tensors of a layer have arrived. All ranks have the same backward graph
        and hook order, so collectives stay in the same order (every held expert is called every step (an empty slice still gets zero grads), so every tensor
        fires exactly once per step)."""
        self._hooks = []
        for k, t in self.tensors.items():
            self._hooks.append(t.register_post_accumulate_grad_hook(lambda p, k=k: self._on_grad(k)))
        self._hooks_installed = True

    def begin_step(self):
        self._sq = torch.zeros((), dtype=torch.float64, device=self.device)
        self._done_keys = set()
        self._block_done = {}
        self._block_reduced = set()

    def _reduce_grads(self):
        """Wrap-up after backward: layers the hooks did not finish (should not happen) are processed in block order; sync D2H; return this GPU's shard sum of
        squares (GPU fp64, not all_reduced)."""
        if not hasattr(self, "_sq"):
            self.begin_step()
        missing = [b for b in self.blocks if b not in self._block_reduced]
        if missing:
            if getattr(self, "_hooks_installed", False):
                print(f"{TAG} rank{self.layout.rank}: warning: expert grad hooks incomplete for {len(missing)} layers (e.g. block {missing[:3]}), processing after backward", flush=True)
            for b in missing:
                self._reduce_block(b)
        left = [k for k, t in self.tensors.items() if t.grad is not None]
        assert not left, f"after reduce, {len(left)} expert grads are still not freed, e.g. {left[:2]}"
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        sq = self._sq
        self.begin_step()
        return sq

    def expert_norm_sq(self, sq):
        """This GPU's shard sum of squares → global all_reduce (each shard lives on exactly one GPU, so every expert parameter is counted once)."""
        if dist.is_initialized():
            dist.all_reduce(sq)
        self.last_expert_norm_sq = float(sq.item())
        return self.last_expert_norm_sq

    def step(self, clip_coef, lr):
        """(Grads are already in the pinned m.grad from _reduce_grads.) Multiply by the clip coefficient on CPU → CPU Adam → bf16 shards H2D via the pinned buffer →
        DP all_gather back to full → shared experts broadcast from the even to the odd GPU."""
        L = self.layout
        n = L.n_nodes
        t0 = time.time()
        self.set_lr(lr)
        if clip_coef != 1.0:
            for m in self.masters:
                m.grad.mul_(clip_coef)
        t1 = time.time()
        self.opt.step()
        t2 = time.time()
        # updated shards back to the GPU + one all_gather per layer (same layout as big in _reduce_block: rows = nodes, columns = per-key shards concatenated)
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        for b in self.blocks:
            keys = self.owned_keys[b]
            if keys:
                total, pin = self.owned_total[b], self.layer_pin[b]
                for k in keys:                                                 # CPU fp32 → bf16 into pinned memory (the previous step's H2D was synchronized long ago)
                    off, Ls = self.owned_off[b][k], self.shard_len[k]
                    pin[off:off + Ls].copy_(self.masters[self._shard_index[k]].detach())
                mine = pin.to(self.device, non_blocking=True)
                if n == 1:
                    fullv = mine.view(1, total)
                else:
                    full = torch.empty(n * total, dtype=self.dtype, device=self.device)
                    if dist.get_backend(L.dp_group) == "nccl":
                        dist.all_gather_into_tensor(full, mine, group=L.dp_group)
                    else:
                        parts = list(full.split(total))
                        dist.all_gather(parts, mine, group=L.dp_group)
                        full = torch.cat(parts)
                    fullv = full.view(n, total)
                with torch.no_grad():
                    for k in keys:
                        off, Ls, t = self.owned_off[b][k], self.shard_len[k], self.tensors[k]
                        t.reshape(-1).copy_(fullv[:, off:off + Ls].reshape(-1)[:t.numel()])
                del fullv
            # shared experts: even GPU (optimizer owner) → odd GPU, the layer's 8 tensors flattened into one broadcast
            if self.shared_keys[b] and L.pair_group is not None:
                e = expert_id_of(self.shared_keys[b][0])
                src = L.node * L.local_world + L.owner[e][1]
                datas = [self.tensors[k].data for k in self.shared_keys[b]]
                flat = _flatten_dense_tensors(datas)
                dist.broadcast(flat, src=src, group=L.pair_group)
                with torch.no_grad():
                    for d, u in zip(datas, _unflatten_dense_tensors(flat, datas)):
                        d.copy_(u)
                del flat
        torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
        self.timing = {"clip": t1 - t0, "cpu_adam": t2 - t1, "gather": time.time() - t2}

    def save_optimizer(self, ds_dir):
        """Each GPU writes its shard's fp32 master weights + Adam m/v."""
        os.makedirs(ds_dir, exist_ok=True)
        state = {"shard_keys": self.shard_keys, "shard_len": self.shard_len, "node": self.layout.node, "n_nodes": self.layout.n_nodes,
                 "masters": [m.detach() for m in self.masters], "opt_state": self.opt.state_dict()}   # no clone (8 ranks cloning at once is a +80 GB transient)
        final = os.path.join(ds_dir, f"ep_opt_rank{self.layout.rank}.pt")
        torch.save(state, final + ".tmp")
        os.replace(final + ".tmp", final)

    def load_optimizer(self, ds_dir):
        state = torch.load(os.path.join(ds_dir, f"ep_opt_rank{self.layout.rank}.pt"), map_location="cpu")
        assert state["shard_keys"] == self.shard_keys and state["n_nodes"] == self.layout.n_nodes, "EP optimizer shard layout does not match the checkpoint"
        for m, saved in zip(self.masters, state["masters"]):
            m.data.copy_(saved)
        self.opt.load_state_dict(state["opt_state"])
        for m in self.masters:                                                  # load_state_dict may clear grad; re-attach a pinned grad
            if m.grad is None:
                m.grad = torch.zeros_like(m.data, pin_memory=(self.device.type == "cuda"))


# =========================================================================== #
#  EP MoE modules                                                             #
# =========================================================================== #
_EP = {"store": None, "layout": None, "ffn_ckpt": os.environ.get("EP_FFN_CKPT", "1") == "1"}


def install(store, layout):
    """Register the EP modules in the transformer file's MOE_IMPL switch (WanAttentionBlock picks its classes from it at construction)."""
    import wan_transformer3d_kiwi_moe_v3_2 as T
    _EP["store"], _EP["layout"] = store, layout
    T.MOE_IMPL["ffn"], T.MOE_IMPL["attn"] = EPMoEFFN, EPMoEKVSelfAttention


class _Router:
    """Expert-independent routing for one forward, shared by both module types: local top-k → pack → all_to_all → group locally by expert."""

    def __init__(self, tokens, topk_idx, num_experts, layout, device):
        self.layout = layout
        top_k = topk_idx.shape[1]
        # dispatch_by_expert inlined without counts.tolist() (the only sync left in routing is the one after the all_to_all of counts below); bitwise equivalent
        # to the original dispatch: positions p = token·k + slot stably sorted by expert id, p increasing within a group (= original nonzero order)
        flat = topk_idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts_t = torch.zeros(num_experts, dtype=torch.long, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))
        token_idx_all = torch.div(order, top_k, rounding_mode="floor")
        slot_all = order - token_idx_all * top_k
        self.usage = counts_t.float()
        Tk = token_idx_all.numel()
        eid = flat[order]                                                            # expert ids sorted by expert [Tk] (order = token·k + slot)
        # dest local_rank: the holder; shared experts alternate between the pair by in-group position parity
        base, shared, slot_tab = layout.route_tables(device)
        starts = torch.cumsum(counts_t, dim=0) - counts_t                            # start of each expert group (all on GPU, same as [0]+cumsum(counts[:-1]))
        pos_in_group = torch.arange(Tk, device=device) - starts[eid]
        dest = base[eid] + shared[eid] * (pos_in_group % 2)
        dslot = slot_tab[eid, dest]                                                   # slot on the dest GPU
        # fewer syncs: pack sorted by (dest GPU, slot) and send the per-(dest GPU, slot) counts in one all_to_all; from the counts the receiver knows each
        # slot's segment in every sender's chunk, so no expert ids and no second sync (previously 2 host syncs + 1 expert-id all_to_all per module).
        order2 = torch.argsort(dest * layout.n_slots + dslot, stable=True)
        self.token_idx, self.slot = token_idx_all[order2], slot_all[order2]
        W, S = layout.local_world, layout.n_slots
        send_mat = torch.zeros(W * S, dtype=torch.long, device=device).scatter_add_(0, dest * S + dslot, torch.ones_like(dest)).view(W, S)
        if layout.world == 1 or not dist.is_initialized():
            recv_mat = send_mat.clone()
        else:
            recv_mat = torch.empty_like(send_mat)
            dist.all_to_all_single(recv_mat, send_mat, group=layout.ep_group)      # recv_mat[j, s] = number of tokens sender j sends to my slot s
        both = torch.cat([send_mat.sum(1), recv_mat.reshape(-1)]).tolist()          # the only sync
        self.send_counts, recv_flat = both[:W], both[W:]
        self.recv_mat = [recv_flat[j * S:(j + 1) * S] for j in range(W)]
        self.recv_counts = [sum(r) for r in self.recv_mat]
        self.tokens = tokens

    def exchange(self, x):
        """[Tk, D] tensor packed by dest GPU → [R, D] received on each GPU (differentiable). Also exchanges expert ids (non-differentiable)."""
        L = self.layout
        if L.world == 1 or not dist.is_initialized():
            return x, None
        R = sum(self.recv_counts)
        out = torch.empty(R, x.shape[1], dtype=x.dtype, device=x.device)
        y = all_to_all_single_autograd(out, x.contiguous(), self.recv_counts, self.send_counts, group=L.ep_group)
        return y, None

    def send_back(self, y_recv):
        """[R, Dout] → back in the sender's packing order [Tk, Dout] (differentiable)."""
        L = self.layout
        if L.world == 1 or not dist.is_initialized():
            return y_recv
        Tk = sum(self.send_counts)
        out = torch.empty(Tk, y_recv.shape[1], dtype=y_recv.dtype, device=y_recv.device)
        return all_to_all_single_autograd(out, y_recv.contiguous(), self.send_counts, self.recv_counts, group=L.ep_group)

    def group_local(self, _unused, my_experts):
        """Group received tokens by slot (counts only, no sync): receive buffer = senders' chunks concatenated, ascending slot order within a chunk;
        returns (order3, inv, (start, n) per local expert)."""
        L = self.layout
        S = L.n_slots
        device = self.tokens.device
        chunk_start, pieces, spans, start = 0, [], {}, 0
        seg_start = []                                                              # seg_start[j][s] = start of sender j's slot-s segment in the receive buffer
        for j in range(L.local_world):
            row, s0 = self.recv_mat[j], chunk_start
            starts_j = []
            for c in row:
                starts_j.append(s0); s0 += c
            seg_start.append(starts_j); chunk_start = s0
        for si, e in enumerate(sorted(my_experts)):
            n_e = 0
            for j in range(L.local_world):
                c = self.recv_mat[j][si]
                if c:
                    pieces.append(torch.arange(seg_start[j][si], seg_start[j][si] + c, device=device))
                    n_e += c
            spans[e] = (start, n_e); start += n_e
        assert start == chunk_start == sum(self.recv_counts), (start, chunk_start, sum(self.recv_counts))
        order3 = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.long, device=device)
        inv = torch.argsort(order3) if order3.numel() else order3
        return order3, inv, spans


def _ffn_forward(x, w1, b1, w2, b2):
    h = F.linear(x, w1, b1)
    h = F.gelu(h, approximate="tanh")
    return F.linear(h, w2, b2)


class EPMoEFFN(nn.Module):
    """EP MoEFFN: same interface, gate, aux loss and stats as the original; expert weights live in ExpertStore, tokens go to the holders via all_to_all."""

    def __init__(self, dim, ffn_dim, num_experts, top_k, aux_loss_alpha, expert_dtype=None):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.dim, self.ffn_dim, self.num_experts, self.top_k, self.aux_loss_alpha = dim, ffn_dim, num_experts, top_k, aux_loss_alpha
        self.gate = Fp32Gate(dim, num_experts)
        self.block_index = None          # set by bind_block_indices
        self.last_aux_loss = None
        self.last_selected = None
        self.last_topk_idx = None

    def forward(self, u):
        store, layout = _EP["store"], _EP["layout"]
        batch, seq_len, dim = u.shape
        tokens = u.reshape(-1, dim)
        scores, topk_weight, topk_idx = route_tokens(self.gate, tokens, self.top_k)
        r = _Router(tokens, topk_idx, self.num_experts, layout, u.device)
        x_send = tokens[r.token_idx]                                                # [Tk, dim]
        x_recv, eid_recv = r.exchange(x_send)
        order3, inv, spans = r.group_local(eid_recv, layout.my_experts)
        x_sorted = x_recv[order3]
        outs = []
        for e in layout.my_experts:
            s, n = spans[e]
            w1, b1, w2, b2 = store.ffn_weights(self.block_index, e)
            if self.training and n > 0 and _EP.get("ffn_ckpt", True):
                # expert FFN wrapped in a non-reentrant checkpoint (in backward recompute a hot GPU otherwise holds FFN intermediates h/gelu(h) [tokens, 14336],
                # several GB each, at once → OOM): h is recomputed on demand, only the expert input is saved; same numerics, one extra FFN forward (≈ +0.3 s/step)
                outs.append(torch_checkpoint(_ffn_forward, x_sorted[s:s + n], w1, b1, w2, b2, use_reentrant=False))
            else:
                outs.append(_ffn_forward(x_sorted[s:s + n], w1, b1, w2, b2))
        y_sorted = torch.cat(outs, dim=0) if outs else x_sorted[:0]
        del outs, x_sorted, x_recv
        y_back = r.send_back(y_sorted[inv])                                         # [Tk, dim] packing order
        del y_sorted
        weight = topk_weight[r.token_idx, r.slot].unsqueeze(-1).to(y_back.dtype)    # cast to bf16 before multiplying, as in the original
        contrib = y_back.float() * weight
        del y_back
        y = torch.zeros(tokens.shape, dtype=torch.float32, device=u.device)
        T = tokens.shape[0]
        by_slot = torch.argsort(r.slot, stable=True)                                # each slot has exactly T entries; one stable sort replaces 3 nonzero calls (one sync each);
        for j in range(self.top_k):                                                 # ascending packing position within a slot = nonzero order → index_add_ bitwise unchanged, no duplicate indices
            sel = by_slot[j * T:(j + 1) * T]
            y.index_add_(0, r.token_idx[sel], contrib[sel])
        y = y.to(tokens.dtype)
        if self.training and self.aux_loss_alpha > 0:
            aux_loss = expert_balance_loss(scores, topk_idx, batch, seq_len, self.num_experts, self.top_k, self.aux_loss_alpha)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
            self.last_aux_loss = aux_loss.detach()
        else:
            self.last_aux_loss = None
        self.last_selected = r.usage / tokens.shape[0]
        self.last_topk_idx = topk_idx.detach()
        return y.reshape(batch, seq_len, dim)


class EPMoEKVSelfAttention(WanSelfAttention):
    """EP MoEKVSelfAttention (the original forward copied section by section; only the per-expert k/v computation is replaced by all_to_all to the holders)."""
    PAD = MoEKVSelfAttention.PAD

    def __init__(self, dim, num_heads, window_size, qk_norm, eps, num_experts, top_k, aux_loss_alpha, shared_kv=False, expert_dtype=None):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        assert tuple(window_size) == (-1, -1), "self-attention with k/v experts does not support windowed attention"
        assert 1 <= top_k <= num_experts
        assert not shared_kv, "the EP version only supports non-shared (k, v) (the videoeditor-56b default)"
        self.num_experts, self.top_k, self.aux_loss_alpha, self.shared_kv = num_experts, top_k, aux_loss_alpha, False
        del self.k
        del self.v
        self.gate = Fp32Gate(dim, num_experts)
        self.block_index = None
        self.last_aux_loss = None
        self.last_selected = None
        self.last_topk_idx = None

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
        store, layout = _EP["store"], _EP["layout"]
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        copies = self.top_k
        tokens = x.reshape(-1, self.dim)
        num_tokens = tokens.shape[0]
        # 1) gate
        scores, topk_weight, topk_idx = route_tokens(self.gate, tokens, self.top_k)
        # 2) `copies` k / v per token plus the log prior; k/v are computed by the holders
        r = _Router(tokens, topk_idx, self.num_experts, layout, x.device)
        x_recv, eid_recv = r.exchange(tokens[r.token_idx])
        order3, inv, spans = r.group_local(eid_recv, layout.my_experts)
        x_sorted = x_recv[order3]
        outs = []
        for e in layout.my_experts:
            s0, cnt = spans[e]
            wk, bk, wv, bv = store.kv_weights(self.block_index, e)
            sub = x_sorted[s0:s0 + cnt]
            outs.append(torch.cat([F.linear(sub, wk, bk), F.linear(sub, wv, bv)], dim=-1))    # [cnt, 2·dim]
        kv_sorted = torch.cat(outs, dim=0) if outs else x_sorted[:0].repeat(1, 2)
        del outs, x_sorted, x_recv
        kv_back = r.send_back(kv_sorted[inv])                                       # [Tk, 2·dim] packing order
        del kv_sorted
        K = tokens.new_zeros(num_tokens, copies, self.dim)
        V = tokens.new_zeros(num_tokens, copies, self.dim)
        log_prior = torch.zeros(num_tokens, copies, device=x.device, dtype=torch.float32)
        K = K.index_put((r.token_idx, r.slot), kv_back[:, :self.dim].to(K.dtype))
        V = V.index_put((r.token_idx, r.slot), kv_back[:, self.dim:].to(V.dtype))
        log_prior = log_prior.index_put((r.token_idx, r.slot), torch.log(topk_weight[r.token_idx, r.slot].clamp_min(1e-6)))
        # 3) as in the original: k goes through norm_k (over C, per copy), then split into heads; q unchanged
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        K = self.norm_k(K).view(b, s, copies, n, d)
        V = V.view(b, s, copies, n, d)
        # 4) RoPE
        q = rope_apply(q, grid_sizes, freqs)
        K = rope_apply(K.reshape(b, s, copies * n, d), grid_sizes, freqs).view(b, s, copies, n, d)
        # 5) concatenate into (copies·L)
        K = K.reshape(b, s * copies, n, d)
        V = V.reshape(b, s * copies, n, d)
        prior = log_prior.view(b, s * copies)
        # 6) put log ĝ into the PAD dims
        prior_hi = prior.to(dtype).float()
        prior_lo = prior - prior_hi
        pad_q = q.new_zeros(b, s, n, self.PAD)
        pad_q[..., 0] = math.sqrt(d)
        pad_q[..., 1] = math.sqrt(d)
        pad_k = K.new_zeros(b, s * copies, n, self.PAD)
        pad_k[..., 0] = prior_hi.view(b, s * copies, 1).expand(-1, -1, n).to(pad_k.dtype)
        pad_k[..., 1] = prior_lo.view(b, s * copies, 1).expand(-1, -1, n).to(pad_k.dtype)
        pad_v = V.new_zeros(b, s * copies, n, self.PAD)
        q_ = torch.cat([q, pad_q], dim=-1).to(dtype)
        K_ = torch.cat([K, pad_k], dim=-1).to(dtype)
        V_ = torch.cat([V, pad_v], dim=-1).to(dtype)
        k_lens = None if seq_lens is None or bool((seq_lens == s).all()) else seq_lens * copies
        out = attention_with_scale(q_, K_, V_, k_lens=k_lens, softmax_scale=1.0 / math.sqrt(d))
        out = out[..., :d]
        # 7) output projection; balance loss; stats
        x = self.o(out.flatten(2))
        if self.training and self.aux_loss_alpha > 0:
            aux_loss = expert_balance_loss(scores, topk_idx, b, s, self.num_experts, self.top_k, self.aux_loss_alpha)
            x = AddAuxiliaryLoss.apply(x, aux_loss)
            self.last_aux_loss = aux_loss.detach()
        else:
            self.last_aux_loss = None
        self.last_selected = r.usage / num_tokens
        self.last_topk_idx = topk_idx.detach()
        return x


def set_store(store):
    _EP["store"] = store


def load_dense_weights(dit, path):
    """EP: load only the dense (non-expert) keys from the full moe_v3_2_trainable.safetensors; expert keys go to ExpertStore. Reads per key, never the
    whole file into memory."""
    from safetensors import safe_open
    state = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if not is_expert_key(k):
                state[k] = f.get_tensor(k)
    missing, unexpected = dit.load_state_dict(state, strict=False)
    missing = [k for k in missing if not is_expert_key(k)]
    assert not missing and not unexpected, f"dense weight key mismatch: missing={missing[:3]} unexpected={list(unexpected)[:3]}"
    return len(state)


def bind_block_indices(dit):
    """Store each EP module's block index (forward uses it to fetch weights)."""
    for i in dit.moe_layers:
        dit.blocks[i].ffn.block_index = i
        dit.blocks[i].self_attn.block_index = i
    return dit


def expert_id_of(key):
    """Expert id in a key name: blocks.{b}.ffn.experts.{e}.… / blocks.{b}.self_attn.k_experts.{e}.… / v_experts."""
    m = re.search(r"\.(?:ffn\.experts|k_experts|v_experts)\.(\d+)\.", key)
    assert m, f"not an expert key: {key}"
    return int(m.group(1))


def is_expert_key(name):
    return (".ffn.experts." in name) or (".self_attn.k_experts." in name) or (".self_attn.v_experts." in name)
