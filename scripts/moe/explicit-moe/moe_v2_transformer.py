#!/usr/bin/env python
"""explicit-moe model definition.
All model structure lives here; the training loop is in moe_v2_train.py, inference in moe_v2_infer.py.

Structure in brief:
  The kiwi-edit DiT (30 blocks, fully frozen) runs 1 base stream + K expert streams in parallel:
    base stream     = LoRA off + the full complex instruction (K sub-instructions joined with " Then ")
    expert stream k = LoRA of expert k on + its own sub-instruction (no_grad, read-only)
  After each block in [fusion_start, fusion_end) a FusionAttention is attached (independent weights, trainable):
    Q = base, K/V = [base; expert1; expert2; ...], expert RoPE width coordinates shifted by (k+1)*w_lat,
    output added back to base as a residual. Experts see neither base nor each other (half attention).
  Text CFG: during training the base instruction is replaced by an empty string with prob base_text_drop_prob
    (expert instructions are always kept); inference pred = u + s*(c-u), u = base with empty instruction,
    c = base with instruction, same experts on both sides.
Only fusions.* and router_head.* are trainable.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLASSES = ["style", "add", "remove", "replace", "background"]   # output order of the 5 router-head classes
# Where each LoRA expert lives, relative to --lora_experts_dir; {name} = expert name (released layout: lora_experts/<name>/dit_query_router.safetensors).
DEFAULT_LORA_EXPERT_TEMPLATE = "{name}/dit_query_router.safetensors"


# =========================================================================== #
#  1. 3D RoPE with a width offset                                             #
# =========================================================================== #
_ROPE_TABLE = {}


def _rope_table(grid, freqs, w_off):
    """3D RoPE phase table [f*h*w, 1, c] (complex) for an (f,h,w) grid, width coordinates shifted by +w_off.
    Computed once per (resolution, offset) and reused afterwards."""
    key = (tuple(grid), int(w_off), freqs.device, freqs.data_ptr())
    t = _ROPE_TABLE.get(key)
    if t is None:
        f, h, w = grid
        c = freqs.size(1)
        fr = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        t = torch.cat([
            fr[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            fr[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            fr[2][w_off:w_off + w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).contiguous()
        if len(_ROPE_TABLE) > 64:
            _ROPE_TABLE.clear()
        _ROPE_TABLE[key] = t
    return t


def rope_apply_offset(x, grid_sizes, freqs, w_off=0):
    """Same as wan's rope_apply plus a width offset: x [B, L, heads, head_dim];
    w_off=0 is the original position (base); expert k uses (k+1)*w_lat -> placed to the right of the frame as seen by base."""
    dtype = x.dtype
    n = x.size(2)
    out = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        L = f * h * w
        xi = torch.view_as_complex(x[i, :L].to(torch.float32).reshape(L, n, -1, 2))
        xi = torch.view_as_real(xi * _rope_table((f, h, w), freqs, w_off)).flatten(2)
        out.append(torch.cat([xi, x[i, L:]]))      # padding passed through unchanged
    return torch.stack(out).to(dtype)


# =========================================================================== #
#  2. Fusion layer: base absorbs expert features                              #
# =========================================================================== #
class FusionAttention(nn.Module):
    """forward(base [B,L,D], experts [K,B,L,D], grid_sizes, freqs, w_lat) -> [B,L,D]
    Q = base (RoPE at original position); K/V = [base (original position); expert j (width shift (j+expert_off0)*w_lat)];
    no RoPE on V; SDPA in bf16; output goes through out_proj and is added back to base. With K=0 it reduces to base self-attention."""

    def __init__(self, dim, heads, init="normal", gain=0.05, expert_off0=1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.expert_off0 = expert_off0
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.bias)
        if init == "zero":
            nn.init.zeros_(self.out_proj.weight)
        elif init == "scaled":
            with torch.no_grad():
                self.out_proj.weight.mul_(gain)
        elif init != "normal":
            raise ValueError(f"unknown init: {init}")

    def forward(self, base, experts, grid_sizes, freqs, w_lat):
        B, L, D = base.shape
        K = experts.size(0)

        def split_heads(t):                              # [B,L,D] -> [B,L,n,d]
            return t.view(t.size(0), t.size(1), self.heads, self.head_dim)

        q = self.q_proj(self.ln_q(base))
        q = rope_apply_offset(split_heads(q), grid_sizes, freqs, 0).reshape(B, L, D)
        k_base = self.k_proj(self.ln_kv(base))
        k_parts = [rope_apply_offset(split_heads(k_base), grid_sizes, freqs, 0).reshape(B, L, D)]
        v_parts = [self.v_proj(self.ln_kv(base))]
        for j in range(K):
            kj = self.k_proj(self.ln_kv(experts[j]))
            k_parts.append(rope_apply_offset(split_heads(kj), grid_sizes, freqs,
                                             (j + self.expert_off0) * w_lat).reshape(B, L, D))
            v_parts.append(self.v_proj(self.ln_kv(experts[j])))
        k = torch.cat(k_parts, dim=1)                    # [B,(K+1)L,D]
        v = torch.cat(v_parts, dim=1)

        def to_sdpa(t):                                  # [B,*,D] -> [B,n,*,d] bf16
            return t.view(B, -1, self.heads, self.head_dim).transpose(1, 2).to(torch.bfloat16)

        o = F.scaled_dot_product_attention(to_sdpa(q), to_sdpa(k), to_sdpa(v))
        o = o.transpose(1, 2).reshape(B, -1, D).to(base.dtype)
        return base + self.out_proj(o)


# =========================================================================== #
#  3. Per-block LoRA switch (expert stream: one expert on; base stream: all off) #
# =========================================================================== #
def set_block_lora(lora_layers, name):
    """Switch the LoRA of the whole block to expert `name`. PEFT's set_adapter also turns on requires_grad
    for that adapter; experts are always frozen in explicit-moe, so turn it off again right away."""
    for m in lora_layers:
        m.set_adapter(name)
        for p in m.parameters():
            p.requires_grad_(False)


def set_block_lora_enabled(lora_layers, on):
    """Enable/disable all LoRA of the whole block (off = plain kiwi-edit, used by the base stream)."""
    for m in lora_layers:
        m.enable_adapters(on)
        for p in m.parameters():
            p.requires_grad_(False)


# =========================================================================== #
#  4. Full model: DiT + MLLM + router head + fusion layers                    #
# =========================================================================== #
class MoEV2Model(nn.Module):
    """Submodule names (dit / mllm / router_head / fusions) define the checkpoint key names; do not rename."""

    def __init__(self, dit, mllm, router_head, adapter_names, fusion_start, fusion_end,
                 fusion_heads=24, fusion_init="normal", fusion_init_gain=0.05, expert_off0=1):
        super().__init__()
        self.dit = dit
        self.mllm = mllm
        self.router_head = router_head
        self.adapter_names = list(adapter_names)          # expert name = PEFT adapter name
        n_blocks = len(dit.blocks)
        assert 0 <= fusion_start < fusion_end <= n_blocks, \
            f"invalid fusion range: [{fusion_start},{fusion_end}) blocks={n_blocks}"
        self.fusion_start = int(fusion_start)
        self.fusion_end = int(fusion_end)
        self.fusions = nn.ModuleList([
            FusionAttention(dit.dim, fusion_heads, init=fusion_init, gain=fusion_init_gain,
                            expert_off0=expert_off0)
            for _ in range(self.fusion_end - self.fusion_start)])
        from peft.tuners.tuners_utils import BaseTunerLayer
        self.block_lora_map = [[m for m in blk.modules() if isinstance(m, BaseTunerLayer)]
                               for blk in dit.blocks]

    @property
    def expert_off0(self):
        return self.fusions[0].expert_off0

    # ---------------- MLLM encoding ----------------
    @torch.no_grad()
    def encode_prompt(self, prompt, src_frames):
        """Instruction + source video frames -> MLLM last-layer hidden [L, 2048] (before the connector), plus the
        number of trailing query tokens. Replicates MLLMInContext.forward line by line up to prompt_embeds."""
        from videox_fun.models.qwen_vl_utils import process_vision_info
        m = self.mllm
        messages, input_query, num_queries, _ = m._build_messages(prompt, None, src_frames, None)
        text = m.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = m.tokenizer(text=[text], images=image_inputs, videos=video_inputs,
                             padding=True, return_tensors="pt")
        inputs = inputs.to(m.mllm_backbone.device, dtype=next(m.mllm_backbone.parameters()).dtype)
        outputs = m.mllm_backbone(
            **inputs, learnable_query=input_query.to(m.mllm_backbone.device),
            output_attentions=False, output_hidden_states=True, return_dict=True)
        prompt_embeds, _ = m._extract_text_and_queries_bs1(
            inputs["input_ids"], inputs["attention_mask"], outputs.hidden_states[-1])
        return prompt_embeds[0], num_queries

    @torch.no_grad()
    def text_ctx(self, prompt, src_frames):
        """ctx [L, 3072] for DiT cross-attention = connector(MLLM hidden)."""
        pe, _ = self.encode_prompt(prompt, src_frames)
        return self.mllm.connetor(pe.to(self.mllm.connetor[0].weight.dtype))

    def null_ctx(self, src_frames):
        """Unconditional branch for text CFG: empty instruction, the MLLM still sees the source video."""
        return self.text_ctx("", src_frames)

    # ---------------- Routing ----------------
    def route(self, sub_prompts, src_frames):
        """Per sub-instruction: MLLM encode -> ctx; text segment (query tokens removed) -> router_head -> 5-class logits -> argmax picks the expert.
        Returns ctxs (K x [L_i,3072]), expert_ids (K ints), logits (K x [1,5], with grad, for joint training), raw_cls (K class names)."""
        ctxs, eids, logits_list, raw_cls = [], [], [], []
        for sp in sub_prompts:
            pe, nq = self.encode_prompt(sp, src_frames)
            with torch.no_grad():
                seg = pe[:-nq].clone().float()
                ctxs.append(self.mllm.connetor(pe.to(self.mllm.connetor[0].weight.dtype)))
            logits = self.router_head(seg.unsqueeze(0))
            cls = CLASSES[int(logits.detach().argmax(1).item())]
            assert cls in self.adapter_names, f"routed to an expert that is not loaded: {cls}"
            raw_cls.append(cls)
            eids.append(self.adapter_names.index(cls))
            logits_list.append(logits)
        return ctxs, eids, logits_list, raw_cls

    # ---------------- DiT forward (shared by training/inference) ----------------
    def denoise(self, noisy_list, t, ctx_base, ctx_experts, adapter_seq, seq_len, src_latents, sigma):
        """A batch of B samples sharing the same expert set adapter_seq (K names).
        noisy_list: B x [C,f,h,w]; t: [B]; ctx_base: B x [Lc,D]; ctx_experts: K lists, each B x [Lc,D];
        src_latents: [B,C,f,h,w]; sigma: [B]. Returns pred [B,C,f,h,w]."""
        from torch.utils.checkpoint import checkpoint
        from videox_fun.models.wan_transformer3d_kiwi import sinusoidal_embedding_1d
        dit = self.dit
        assert dit.sp_world_size == 1
        B, K = len(noisy_list), len(adapter_seq)
        assert len(ctx_base) == B and len(ctx_experts) == K and all(len(c) == B for c in ctx_experts)
        assert src_latents.size(0) == B and sigma.numel() == B and t.dim() == 1
        device = dit.patch_embedding.weight.device
        dtype = noisy_list[0].dtype
        if dit.freqs.device != device:
            dit.freqs = dit.freqs.to(device)

        # ---- input: patch embedding + source-video condition (times sigma)
        x = [dit.patch_embedding(u.unsqueeze(0)) for u in noisy_list]
        src_cond = dit.vae_condition(src_latents)
        sigma_b = sigma.to(src_cond.dtype).reshape(-1)
        x = [u + src_cond[i:i + 1] * sigma_b[i] for i, u in enumerate(x)]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # ---- time embedding
        e = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t).float()).to(dtype)
        e0 = dit.time_projection(e).unflatten(1, (6, dit.dim))
        use_gc = bool(torch.is_grad_enabled() and getattr(dit, "gradient_checkpointing", False))

        def pad_ctx(ctx_list):                           # B x [Lc_i,D] -> [B,Lc_max,D] + lengths
            lens = [c.size(0) for c in ctx_list]
            lc = max(lens)
            ctx = torch.stack([torch.cat([c, c.new_zeros(lc - c.size(0), c.size(1))]) if c.size(0) < lc else c
                               for c in ctx_list])
            return ctx, torch.tensor(lens, dtype=torch.int32, device=ctx_list[0].device)

        def run_block(block, lora_layers, expert_name, inp, ctx_list):
            """Run one block. expert_name=None -> base stream (all LoRA off); otherwise expert stream (only that expert on)."""
            context, context_lens = pad_ctx(ctx_list)

            def fwd(*inputs):
                if expert_name is None:
                    set_block_lora_enabled(lora_layers, False)
                    try:
                        return block(*inputs)
                    finally:
                        set_block_lora_enabled(lora_layers, True)
                set_block_lora(lora_layers, expert_name)
                return block(*inputs)

            args = (inp, e0, seq_lens, grid_sizes, dit.freqs, context, context_lens, dtype, t)
            return checkpoint(fwd, *args, use_reentrant=False) if use_gc else fwd(*args)

        # ---- per block: base stream || K expert streams, fused after blocks in [fusion_start, fusion_end)
        base_state = x
        expert_states = [x.detach()] * K
        for bi, block in enumerate(dit.blocks):
            lora_layers = self.block_lora_map[bi]
            base_state = run_block(block, lora_layers, None, base_state, ctx_base)
            with torch.no_grad():
                expert_states = [run_block(block, lora_layers, name, expert_states[k], ctx_experts[k])
                                 for k, name in enumerate(adapter_seq)]
            if self.fusion_start <= bi < self.fusion_end:
                fusion = self.fusions[bi - self.fusion_start]
                xs = (torch.stack([s.detach() for s in expert_states], dim=0) if K > 0
                      else base_state.new_zeros((0,) + tuple(base_state.shape)))
                w_lat = int(grid_sizes[0, 2])
                if use_gc:
                    # bind _fu=fusion as a default arg: non-reentrant checkpoint calls the closure during backward,
                    # late binding would make every layer point to the last fusion -> gradients silently go to the wrong layer
                    base_state = checkpoint(lambda a, b, _fu=fusion: _fu(a, b, grid_sizes, dit.freqs, w_lat),
                                            base_state, xs, use_reentrant=False)
                else:
                    base_state = fusion(base_state, xs, grid_sizes, dit.freqs, w_lat)

        out = checkpoint(lambda a, b: dit.head(a, b), base_state, e, use_reentrant=False) if use_gc \
            else dit.head(base_state, e)
        return torch.stack(dit.unpatchify(out, grid_sizes))

    # ---------------- Training forward (must be called through the DDP wrapper, hence named forward) ----------------
    def forward(self, samples, noisy, t, src_latents, sigma, seq_len, joiner=" Then ", text_drop=None):
        """samples: B dataset samples of one group (same expert multiset); noisy [B,C,f,h,w]; t/sigma [B].
        Per sample: route -> turns with a Qwen label pick the expert by label instead (training-time GT) -> sub-instructions
        sorted by expert name to align stream positions -> base ctx = full instruction, or empty instruction if text_drop[i].
        Samples whose expert set differs from the first sample's are dropped (a dataset retry may swap in a sample of another group).
        Returns dict: pred [Bk,...], kept (indices of samples in the forward), router_logits [N,5]/router_labels [N] (for joint-training CE), routes (stats)."""
        text_drop = list(text_drop) if text_drop is not None else [False] * len(samples)
        per, all_logits, all_labels = [], [], []
        for i, s in enumerate(samples):
            plist = list(s["prompts_multi"])
            gt = list(s.get("turn_types_qwen") or s.get("turn_types") or [])
            ctxs, eids, logits_list, raw_cls = self.route(plist, s["src_video"])
            for j, qt in enumerate(gt[:len(eids)]):
                if qt in CLASSES:
                    all_logits.append(logits_list[j])
                    all_labels.append(CLASSES.index(qt))
                    if qt in self.adapter_names:
                        eids[j] = self.adapter_names.index(qt)       # during training, pick the expert by label
            order = sorted(range(len(eids)), key=lambda j: (self.adapter_names[eids[j]], j))
            ctx_base = self.null_ctx(s["src_video"]) if text_drop[i] else self.text_ctx(joiner.join(plist), s["src_video"])
            per.append({"key": tuple(self.adapter_names[eids[j]] for j in order),
                        "ctxs": [ctxs[j] for j in order], "ctx_base": ctx_base,
                        "route": {"raw": raw_cls, "used": [self.adapter_names[e] for e in eids],
                                  "gt": gt[:len(eids)], "text_drop": text_drop[i]}})
        key0 = per[0]["key"]
        kept = [i for i, p in enumerate(per) if p["key"] == key0]
        adapter_seq = list(key0)
        ctx_experts = [[per[i]["ctxs"][k] for i in kept] for k in range(len(adapter_seq))]
        pred = self.denoise([noisy[i] for i in kept], t[kept], [per[i]["ctx_base"] for i in kept],
                            ctx_experts, adapter_seq, seq_len, src_latents[kept], sigma[kept])
        return {"pred": pred, "kept": kept,
                "router_logits": torch.cat(all_logits, 0) if all_logits else None,
                "router_labels": torch.tensor(all_labels, device=all_logits[0].device) if all_logits else None,
                "routes": [per[i]["route"] for i in kept]}


# =========================================================================== #
#  5. Assembly and checkpoints                                                #
# =========================================================================== #
def build_model(args, dtype=torch.bfloat16):
    """Load kiwi-edit DiT + VAE + MLLM + each expert's LoRA + router head and assemble a MoEV2Model (all on CPU; caller does .to(device)).
    args needs: pretrained_wan_path, config_path, mllm_model, init_weights, router_head, lora_experts_dir, experts,
    (optional lora_expert_template, default DEFAULT_LORA_EXPERT_TEMPLATE), dit_lora_rank, dit_lora_target_modules, use_gradient_checkpointing,
    fusion_start, fusion_end (0 = up to and including the last block), fusion_heads, fusion_init, fusion_init_gain, rope_expert_offset0.
    Returns (model, vae)."""
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from peft import LoraConfig, inject_adapter_in_model
    from videox_fun.models import AutoencoderKLWan, AutoencoderKLWan3_8
    from videox_fun.models.wan_transformer3d_kiwi import KiwiWanTransformer3DModel
    from videox_fun.moe_common.mllm_encoder_moe import MLLMInContext, MLLMInContextConfig
    from videox_fun.moe_common.attn_head import AttnHead

    config = OmegaConf.load(args.config_path)

    # VAE (type chosen by config: wan_civitai_5b.yaml = AutoencoderKLWan3_8)
    vae_kwargs = OmegaConf.to_container(config["vae_kwargs"])
    vae_cls = {"AutoencoderKLWan": AutoencoderKLWan, "AutoencoderKLWan3_8": AutoencoderKLWan3_8}[
        vae_kwargs.get("vae_type", "AutoencoderKLWan3_8")]
    vae = vae_cls.from_pretrained(os.path.join(args.pretrained_wan_path, vae_kwargs.get("vae_subpath", "Wan2.2_VAE.pth")),
                                  additional_kwargs=vae_kwargs).to(dtype=dtype)
    vae.eval().requires_grad_(False)

    # DiT
    sub = config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "./")
    dit_dir = os.path.join(args.pretrained_wan_path, sub)
    if not os.path.isfile(os.path.join(dit_dir, "config.json")):
        dit_dir = args.pretrained_wan_path
    dit = KiwiWanTransformer3DModel.from_pretrained(
        dit_dir, transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"])
    ).to(dtype=dtype)
    if args.use_gradient_checkpointing:
        dit.enable_gradient_checkpointing()

    # MLLM (Qwen2.5-VL + connector)
    mllm = MLLMInContext(MLLMInContextConfig(
        mllm_id=args.mllm_model, num_image_queries=256, num_video_queries=512, num_ref_queries=768,
        max_object_token=768, max_frames=16, max_pixels_per_frame=262144, dit_dim=dit.config.dim,
        _gradient_checkpointing=False))

    # initial weights: connector / query / dit.vae_condition (source-video condition conv) come from moe_expert_init
    init_sd = load_file(args.init_weights)
    mllm.load_state_dict({k[len("mllm."):]: v for k, v in init_sd.items() if k.startswith("mllm.")}, strict=False)
    _, unexpected = dit.load_state_dict({k[len("dit."):]: v for k, v in init_sd.items() if k.startswith("dit.")},
                                        strict=False)
    assert not unexpected, f"init dit key mismatch: {unexpected[:3]}"

    # expert LoRA: one adapter per expert, weight file = <lora_experts_dir>/<lora_expert_template with {name}>
    names = [s.strip() for s in args.experts.split(",") if s.strip()]
    tmods = [m.strip() for m in args.dit_lora_target_modules.split(",")]
    expert_tmpl = getattr(args, "lora_expert_template", "") or DEFAULT_LORA_EXPERT_TEMPLATE
    for name in names:
        inject_adapter_in_model(LoraConfig(r=args.dit_lora_rank, lora_alpha=args.dit_lora_rank, target_modules=tmods),
                                dit, adapter_name=name)
        path = os.path.join(args.lora_experts_dir, expert_tmpl.format(name=name))
        assert os.path.isfile(path), f"weights for expert {name} not found: {path}"
        remap = {k[len("dit."):].replace(".default.", f".{name}."): v
                 for k, v in load_file(path).items() if ".lora_" in k}
        _, unexp = dit.load_state_dict(remap, strict=False)
        assert len(remap) == 600 and not unexp, f"expert {name}: LoRA keys {len(remap)}, not applied {len(unexp)} (expected 600/0)"
        print(f"[moe-v2] expert '{name}' LoRA loaded: {len(remap)} tensors", flush=True)

    # router head
    ck = torch.load(args.router_head, map_location="cpu")
    if ck.get("classes") is not None:
        assert list(ck["classes"]) == CLASSES, f"router head class order mismatch: {ck['classes']}"
    router = AttnHead(2048, ck.get("dmodel", 512), ck.get("heads", 8), 0.0)
    router.load_state_dict(ck["state_dict"])
    router.eval().float().requires_grad_(False)
    print(f"[moe-v2] router head loaded (val_agree={ck.get('val_agree')})", flush=True)

    fusion_end = int(args.fusion_end) if int(args.fusion_end) > 0 else len(dit.blocks)
    model = MoEV2Model(dit, mllm, router, names, args.fusion_start, fusion_end,
                       fusion_heads=args.fusion_heads, fusion_init=args.fusion_init,
                       fusion_init_gain=args.fusion_init_gain, expert_off0=args.rope_expert_offset0)
    return model, vae


def load_checkpoint(model, state_path):
    """Load fusions.* and router_head.* from a training checkpoint (moe_v2_state.pt) into model, checking variant/fusion range/layer count.
    Returns the ck dict (includes step / opt etc., used for training resume)."""
    ck = torch.load(state_path, map_location="cpu")
    assert ck.get("variant") == "moe_v2", f"not a moe_v2 checkpoint: {ck.get('variant')}"
    assert ck["fusion_start"] == model.fusion_start and ck["fusion_end"] == model.fusion_end, \
        f"fusion range mismatch: ckpt=[{ck['fusion_start']},{ck['fusion_end']}) model=[{model.fusion_start},{model.fusion_end})"
    assert ck["n_fusion_layers"] == len(model.fusions), f"fusion layer count mismatch: ckpt={ck['n_fusion_layers']} model={len(model.fusions)}"
    assert ck["rope_expert_offset0"] == model.expert_off0, \
        f"rope_expert_offset0 mismatch: ckpt={ck['rope_expert_offset0']} model={model.expert_off0}"
    missing, unexpected = model.load_state_dict(ck["trainable"], strict=False)
    missing = [k for k in missing if k.startswith(("fusions.", "router_head."))]
    assert not missing and not unexpected, f"checkpoint key mismatch: missing={missing[:3]} unexpected={list(unexpected)[:3]}"
    return ck
