#!/usr/bin/env python
"""FAST: copy of moe_v3_2_model.py that builds the DiT from wan_transformer3d_kiwi_moe_v3_2_fast.py (only that import differs). The original file is untouched.

Assembly file for videoeditor-56b: builds the MoE DiT (self-attention k/v and FFN both 35 experts, top-3) and loads its initial weights; MLLM text encoding; VAE. No DiT architecture code here.
videoeditor-56b = videoeditor-14b architecture unchanged, experts 7 → 35 (each of the 7 videoeditor-14b experts replicated 5 times, every copy slightly perturbed; still top-3).

  DiT     KiwiMoeTransformer3DModel from wan_transformer3d_kiwi_moe_v3_2.py (full copy of the original kiwi DiT + MoE self-attention k/v + MoE FFN)
  MLLM    Qwen2.5-VL + connector (frozen); produces the DiT cross-attention context (full instruction; empty instruction = text-CFG unconditional branch)
  VAE     Wan2.2 VAE (frozen)

Where the MoE-layer weights come from (build_dit); self-attention k_experts / v_experts and FFN experts share one ordering:
  Source experts (SOURCE_EXPERTS = the 7 videoeditor-14b experts, same order; weights still taken from experts_v3_1.safetensors made by merge_experts_v3_1.py):
      main                                         ← original kiwi (base file self_attn.k / self_attn.v / ffn.0 / ffn.2)
      style / add / remove / replace / background  ← original + merged category-expert LoRA (experts 1..5 in the videoeditor-14b file)
      direct_finetune                              ← kiwi-edit fully fine-tuned directly on goku (expert 6 in the videoeditor-14b file)
  Expert j (j = 0..N-1, N = len(sources) × copies):  source = sources[j // copies], copy number j % copies;
      each copy = source weight + independent Gaussian perturbation randn × (std of that weight tensor × expert_copy_noise) (biases not perturbed),
      generated with fixed seed 2026 in (layer, projection, j) order, so every rank builds exactly the same experts (same scheme as the noisy copies
      in mlp-moe; in videoeditor-56b all 35 copies are perturbed, so copies of one source differ from step 0 and diverge during training)
  Names EXPERT_NAMES = [f"{source}_{copy}"], e.g. main_0 .. main_4, style_0 .. , direct_finetune_4 (order = expert index; used by training logs / inference plots / meta.json)
  gate           one per site (self-attention / FFN), Linear(dim → 35), random init (std 0.02), trained from scratch
  self-attention q / o / norm ← original kiwi (loaded directly by from_pretrained); the original k / v projections are removed when there is no shared copy
Build memory: 35 experts = 52.4B parameters (55.9B total); expert linears are built directly in bf16 (moe_expert_dtype="bfloat16", ~120 GB peak per process;
building in fp32 needs 224 GB, and 8 processes would exhaust 2 TB of host memory).
Trainable (mark_trainable): the whole DiT, in full. Parameters form three learning-rate groups:
  base     original parameters (attention q/o, cross-attention, non-MoE layers, embedding, head, ...)   always learning_rate
  experts  MoE-layer k_experts / v_experts / ffn.experts                                  start at new_layers_lr, decay linearly to learning_rate
  gate     the two MoE-layer gates                                                        start at gate_lr, linear to gate_lr_end
"""
import os
import re
import sys
import time

import torch
import torch.nn as nn

_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)

V3_1_FILE_EXPERTS = ["main", "style", "add", "remove", "replace", "background", "direct_finetune"]   # indices 0..6 in experts_v3_1.safetensors (0 = base, not stored in the file)
SOURCE_EXPERTS = list(V3_1_FILE_EXPERTS)                                                                   # videoeditor-56b default: 7 source experts = all 7 videoeditor-14b experts
DEFAULT_EXPERT_COPIES = 5                                                                                  # 5 copies per source → 35 experts
NOISE_SEED = 2026                                                                                          # same seed as the noisy copies in mlp-moe


def expert_names(sources=None, copies=DEFAULT_EXPERT_COPIES):
    """Name of expert j = f"{sources[j // copies]}_{j % copies}"; order = expert index."""
    sources = list(sources) if sources else SOURCE_EXPERTS
    return [f"{name}_{c}" for name in sources for c in range(int(copies))]


def parse_sources(spec):
    """Comma-separated --expert_sources → list; empty = default SOURCE_EXPERTS. Every name must exist in experts_v3_1.safetensors (or be main)."""
    names = [x.strip() for x in str(spec or "").split(",") if x.strip()] or list(SOURCE_EXPERTS)
    bad = [n for n in names if n not in V3_1_FILE_EXPERTS]
    assert not bad, f"unknown source expert(s) {bad}; choices: {V3_1_FILE_EXPERTS}"
    return names


EXPERT_NAMES = expert_names()      # default 35 names (7 sources × 5 copies); training/inference recompute them from args (see dit.expert_names set by build_dit)
MOE_PROJS = ("self_attn.k", "self_attn.v", "ffn.0", "ffn.2")


class no_zero3_init:
    """Temporarily disable transformers' DeepSpeed ZeRO-3 detection while building the MLLM / VAE.
    Once accelerate installs a ZeRO-3 config, transformers' from_pretrained also ZeRO-3-shards every newly built HF model
    (parameters become 0-dim; the forward fails with 'weight' must be 2-D). The MLLM / VAE are frozen, one copy per GPU and not
    managed by DeepSpeed, so they must be built in full."""

    def __enter__(self):
        import transformers.integrations.deepspeed as hf_ds
        self.hf_ds = hf_ds
        self.saved = hf_ds._hf_deepspeed_config_weak_ref
        hf_ds._hf_deepspeed_config_weak_ref = None
        return self

    def __exit__(self, *exc):
        self.hf_ds._hf_deepspeed_config_weak_ref = self.saved
        return False


# =========================================================================== #
#  DiT                                                                         #
# =========================================================================== #
def build_dit(args, dtype=torch.bfloat16, verbose=True):
    """Build KiwiMoeTransformer3DModel and load its weights; returns the model on CPU (bf16).
    args needs: pretrained_wan_path, config_path, init_weights, experts_merged (experts_v3_1.safetensors produced by merge_experts_v3_1.py;
        "" = inference from a released checkpoint: the merged-expert file is not read, experts are left as constructed and the
        source-weight self-checks are skipped -- the caller must then fill every expert with load_trainable_weights(<ckpt>)),
    moe_layers (comma-separated), moe_num_experts (= len(expert_sources) × expert_copies, default 35), moe_top_k, moe_aux_loss_alpha, use_gradient_checkpointing;
    optional: expert_sources (comma-separated, default the 7 sources), expert_copies (default 5), expert_copy_noise (default 0.01; 0 = all copies identical).
    After building, dit.expert_names / dit.expert_sources / dit.expert_copies record the expert table for logs / meta.json."""
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from safetensors.torch import load_file
    from wan_transformer3d_kiwi_moe_v3_2_fast import KiwiMoeTransformer3DModel   # FAST: the inference-speed copy of the DiT

    t_build0 = time.time()
    sources = parse_sources(getattr(args, "expert_sources", ""))
    copies = int(getattr(args, "expert_copies", DEFAULT_EXPERT_COPIES))
    noise_scale = float(getattr(args, "expert_copy_noise", 0.01))
    names = expert_names(sources, copies)
    num_experts = int(args.moe_num_experts)
    assert num_experts == len(names), f"moe_num_experts={num_experts} must equal len(expert_sources) × expert_copies = {len(sources)} × {copies} = {len(names)}"
    assert copies >= 1 and noise_scale >= 0, (copies, noise_scale)

    config = OmegaConf.load(args.config_path)
    dit_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    dit_dir = os.path.join(args.pretrained_wan_path, dit_kwargs.get("transformer_low_noise_model_subpath", "./"))
    if not os.path.isfile(os.path.join(dit_dir, "config.json")):
        dit_dir = args.pretrained_wan_path
    moe_layers = [int(s) for s in str(args.moe_layers).split(",") if s.strip()]
    dit_kwargs.update(moe_layers=moe_layers, moe_num_experts=num_experts, moe_top_k=int(args.moe_top_k),
                      moe_aux_loss_alpha=float(args.moe_aux_loss_alpha), moe_expert_dtype=str(dtype).replace("torch.", ""))   # MOE-V3.2 ⑤: build experts directly in the target dtype

    # 1) Original weights: from_pretrained reports the MoE-layer experts / gate keys as missing (new modules), filled by hand below; q/o/norm load normally
    dit = KiwiMoeTransformer3DModel.from_pretrained(dit_dir, transformer_additional_kwargs=dit_kwargs)

    t_shell = time.time() - t_build0          # build the shell (default nn.Linear init of 52.4B params) + load base weights

    # 2) Source-video conditioning conv vae_condition (in moe_expert_init.safetensors, keys dit.vae_condition.*)
    init_state = load_file(args.init_weights)
    cond_state = {key[len("dit."):]: value for key, value in init_state.items() if key.startswith("dit.")}
    _, unexpected = dit.load_state_dict(cond_state, strict=False)
    assert not unexpected, f"init key mismatch: {unexpected[:3]}"

    # 3) MoE layers: expert j ← weights of source sources[j // copies] + independent perturbation. Source "main" is read from the base file, the rest
    #    from the videoeditor-14b merged file (keys blocks.{b}.experts.{i}.{proj}.{weight,bias}, i = videoeditor-14b expert index)
    if int(getattr(args, "ep", 0)):
        # EP (expert parallel): expert weights live outside the DiT (moe_v3_2_ep.ExpertStore reads this GPU's experts from the checkpoint);
        # nothing is filled or checked here, the DiT holds only the dense part + gates
        dit = dit.to(dtype=dtype)
        if args.use_gradient_checkpointing:
            dit.enable_gradient_checkpointing()
        dit.expert_names, dit.expert_sources, dit.expert_copies, dit.expert_copy_noise = names, sources, copies, noise_scale
        if verbose:
            print(f"[moe-v3.2] DiT(EP): moe_layers={moe_layers} experts={num_experts} top_k={args.moe_top_k} expert weights held by ExpertStore; "
                  f"DiT params {sum(p.numel() for p in dit.parameters()) / 1e9:.2f}B (dense + gates), shell+base {t_shell:.0f}s", flush=True)
        return dit
    if not str(getattr(args, "experts_merged", "") or ""):
        # Inference from a released checkpoint (experts_merged = ""): the merged-expert init file is not needed. The expert
        # linears keep their constructor values and MUST be overwritten by load_trainable_weights(<ckpt>) (the inference
        # script asserts that the checkpoint covers every parameter). The self-checks against the source weights are skipped.
        dit = dit.to(dtype=dtype)
        if args.use_gradient_checkpointing:
            dit.enable_gradient_checkpointing()
        dit.expert_names, dit.expert_sources, dit.expert_copies, dit.expert_copy_noise = names, sources, copies, noise_scale
        if verbose:
            print(f"[moe-v3.2] DiT: moe_layers={moe_layers} experts={num_experts} top_k={args.moe_top_k}; experts_merged='' -> expert weights "
                  f"must come from the checkpoint; total params {sum(p.numel() for p in dit.parameters()) / 1e9:.2f}B, "
                  f"shell+base {t_shell:.0f}s", flush=True)
        return dit
    base = safe_open(os.path.join(dit_dir, "diffusion_pytorch_model.safetensors"), "pt")
    merged = safe_open(args.experts_merged, "pt")

    def source_tensors(block_index, proj, name):
        i = V3_1_FILE_EXPERTS.index(name)
        if i == 0:
            return base.get_tensor(f"blocks.{block_index}.{proj}.weight"), base.get_tensor(f"blocks.{block_index}.{proj}.bias")
        return merged.get_tensor(f"blocks.{block_index}.experts.{i}.{proj}.weight"), merged.get_tensor(f"blocks.{block_index}.experts.{i}.{proj}.bias")

    def expert_module(block, proj, j):
        """nn.Linear of expert j on projection proj in MoE block `block`."""
        if proj == "self_attn.k":
            return block.self_attn.k_experts[j]
        if proj == "self_attn.v":
            return block.self_attn.v_experts[j]
        return block.ffn.experts[j][0 if proj == "ffn.0" else 2]

    generator = torch.Generator().manual_seed(NOISE_SEED)      # fixed seed + fixed loop order (layer → projection → expert j): every rank builds exactly the same perturbation
    rel_noise = {}
    t_read = t_noise = t_copy = 0.0
    with torch.no_grad():
        for block_index in moe_layers:
            block = dit.blocks[block_index]
            for proj in MOE_PROJS:
                cache = {}
                for j, name in enumerate(names):
                    src = sources[j // copies]
                    if src not in cache:                 # per (layer, projection) each source is read, cast to fp32 and its std computed only once
                        _t = time.time()                 #   (not recomputed for every copy)
                        W, bias = source_tensors(block_index, proj, src)
                        W32 = W.float() if noise_scale > 0 else None
                        cache[src] = (W, bias, W32, (float(W32.std()) if W32 is not None else 0.0))
                        t_read += time.time() - _t
                    W, bias, W32, w_std = cache[src]
                    linear = expert_module(block, proj, j)
                    if noise_scale > 0:
                        _t = time.time()
                        noise = torch.randn(W32.shape, generator=generator) * (w_std * noise_scale)
                        t_noise += time.time() - _t
                        _t = time.time()
                        linear.weight.copy_((W32 + noise).to(linear.weight.dtype))
                        t_copy += time.time() - _t
                        if block_index == moe_layers[0]:
                            rel_noise[f"{proj}/{name}"] = round((noise.norm() / W32.norm()).item(), 5)
                    else:
                        _t = time.time()
                        linear.weight.copy_(W)
                        t_copy += time.time() - _t
                    linear.bias.copy_(bias)
    dit = dit.to(dtype=dtype)
    if args.use_gradient_checkpointing:
        dit.enable_gradient_checkpointing()
    dit.expert_names, dit.expert_sources, dit.expert_copies, dit.expert_copy_noise = names, sources, copies, noise_scale

    # 4) Self-check (first MoE layer): each expert = source + small perturbation (relative difference ~ noise_scale and > 0), two copies of one source
    #    differ, different sources differ, bias = source bias, expert dtype = target dtype
    with torch.no_grad():
        block_index = moe_layers[0]
        block = dit.blocks[block_index]
        for proj in ("self_attn.k", "ffn.0"):
            for j in sorted({0, min(copies, num_experts - 1), num_experts - 1}):    # with a single source copies == num_experts, so clamp
                W_src, b_src = source_tensors(block_index, proj, sources[j // copies])
                lin = expert_module(block, proj, j)
                assert lin.weight.dtype == dtype, (proj, j, lin.weight.dtype)
                rel = ((lin.weight.float() - W_src.float()).norm() / W_src.float().norm()).item()
                if noise_scale > 0:
                    assert 0 < rel < 4 * noise_scale + 1e-2, f"expert {j} {proj}: relative difference to source {rel:.4g} is not at the perturbation scale ({noise_scale})"
                else:
                    assert rel < 1e-2, (proj, j, rel)            # bf16 rounding
                assert torch.equal(lin.bias, b_src.to(dtype)), (proj, j)
        if copies > 1 and noise_scale > 0:
            assert not torch.equal(block.ffn.experts[0][0].weight, block.ffn.experts[1][0].weight), "two copies of the same source must differ (perturbation not applied?)"
        if len(sources) > 1:
            assert not torch.equal(block.self_attn.v_experts[0].weight, block.self_attn.v_experts[copies].weight), "experts from different sources must differ"
    if verbose:
        n_moe = sum(p.numel() for i in moe_layers for p in list(dit.blocks[i].ffn.parameters()) + list(dit.blocks[i].self_attn.parameters()))
        print(f"[moe-v3.2] DiT: moe_layers={moe_layers} experts={num_experts} = {len(sources)} sources ({','.join(sources)}) × {copies} copies, "
              f"top_k={args.moe_top_k} aux_alpha={args.moe_aux_loss_alpha} copy_noise={noise_scale} MoE-layer (self-attn+FFN) params {n_moe / 1e9:.2f}B "
              f"total params {sum(p.numel() for p in dit.parameters()) / 1e9:.2f}B", flush=True)
        if rel_noise:
            vals = sorted(rel_noise.values())
            print(f"[moe-v3.2] layer {moe_layers[0]} perturbation |noise|/|W| range {vals[0]:.4g} ~ {vals[-1]:.4g} ({len(vals)} tensors)", flush=True)
        print(f"[moe-v3.2] build time: shell+base {t_shell:.0f}s, read source weights {t_read:.0f}s, noise {t_noise:.0f}s, write experts {t_copy:.0f}s, total {time.time() - t_build0:.0f}s", flush=True)
    return dit


def mark_trainable(dit):
    """Full training: every parameter is trainable. Returns three learning-rate groups (base_params, expert_params, gate_params):
    expert_params = MoE-layer experts (ffn.experts / self_attn.k_experts / self_attn.v_experts), gate_params = the two MoE-layer gates,
    base_params = everything else (incl. self-attention q/o/shared k/v). Only MoE layers have these name segments, so the parameter name is enough."""
    base_params, expert_params, gate_params = [], [], []
    for name, param in dit.named_parameters():
        param.requires_grad_(True)
        m = re.match(r"blocks\.(\d+)\.(ffn|self_attn)\.(experts|k_experts|v_experts|gate)\.", name)
        if m is None:
            base_params.append(param)                 # non-MoE layers, or q/o/shared k/v/cross-attention/norm etc. of MoE layers
        elif m.group(3) == "gate":
            gate_params.append(param)
        else:
            expert_params.append(param)
    assert len(gate_params) == 2 * len(dit.moe_layers), (len(gate_params), len(dit.moe_layers))
    count = lambda params: sum(getattr(p, 'ds_numel', p.numel()) for p in params)   # after ZeRO-3 sharding p.numel() is only this GPU's shard; use ds_numel
    print(f"[moe-v3.2] trainable(full): base {count(base_params) / 1e9:.2f}B + experts(attn k/v + ffn) {count(expert_params) / 1e9:.2f}B "
          f"+ gate {count(gate_params):,}  = total {(count(base_params) + count(expert_params) + count(gate_params)) / 1e9:.2f}B", flush=True)
    return base_params, expert_params, gate_params


def trainable_state_keys(dit):
    """Set of trainable parameter names (only these go into the ckpt; with full training that is every DiT parameter, no buffers)."""
    return {name for name, param in dit.named_parameters() if param.requires_grad}


# =========================================================================== #
#  MLLM text encoding                                                          #
# =========================================================================== #
class TextEncoder:
    """Encode instruction + source-video frames into the DiT cross-attention context. The MLLM is frozen and not managed by DeepSpeed."""

    def __init__(self, mllm):
        self.mllm = mllm

    @torch.no_grad()
    def encode(self, prompt, src_frames):
        """Instruction + source-video frames → "text + query" segments of the last Qwen2.5-VL hidden state → connector → [L, 3072].
        Reproduces MLLMInContext.forward line by line up to prompt_embeds."""
        from videox_fun.models.qwen_vl_utils import process_vision_info
        mllm = self.mllm
        messages, input_query, _, _ = mllm._build_messages(prompt, None, src_frames, None)
        text = mllm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = mllm.tokenizer(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(mllm.mllm_backbone.device, dtype=next(mllm.mllm_backbone.parameters()).dtype)
        outputs = mllm.mllm_backbone(**inputs, learnable_query=input_query.to(mllm.mllm_backbone.device),
                                     output_attentions=False, output_hidden_states=True, return_dict=True)
        prompt_embeds, _ = mllm._extract_text_and_queries_bs1(inputs["input_ids"], inputs["attention_mask"],
                                                              outputs.hidden_states[-1])
        hidden = prompt_embeds[0]                                                   # [L, 2048]
        return mllm.connetor(hidden.to(mllm.connetor[0].weight.dtype))              # [L, 3072]

    def encode_null(self, src_frames):
        """Unconditional branch of text CFG: empty instruction; the MLLM still sees the source video."""
        return self.encode("", src_frames)

    @torch.no_grad()
    def encode_batch(self, prompts, frames_list):
        """Encode a whole batch in one MLLM forward (per-sample encode needs 8 forwards per step, ~2.7 s → 1).
        prompts: list[str] (empty string = text-CFG unconditional branch, same as encode_null, encoded in the same batch); frames_list: list[list[PIL]] of equal length.
        Returns list[Tensor[L_b, 3072]], per sample equivalent to encode, via mllm_encoder_moe.MLLMInContext.forward_batch:
        it temporarily switches to right padding (real token positions 0..L_b-1 as with bs=1, so RoPE does not shift), expands learnable_query to [B, nq, dim] for masked_scatter,
        then per sample _extract_text_and_queries_bs1 takes [text + query] through the connector; the only difference is bf16 rounding of the batched Qwen forward."""
        assert len(prompts) == len(frames_list) and len(prompts) > 0, (len(prompts), len(frames_list))
        return self.mllm.forward_batch(list(prompts), list(frames_list), is_video=True)

    @torch.no_grad()
    def encode_batch_pre(self, seqs):
        """Sequences already preprocessed by the dataloader workers → list of ctx (bitwise identical to encode_batch; the main process skips chat template/resize/normalize/tokenize)."""
        assert len(seqs) > 0
        return self.mllm.forward_batch_pre(seqs)

    @torch.no_grad()
    def self_test(self, prompt, src_frames):
        """Check stage by stage where non-finite values first appear in one encoding: returns None (all finite) or
        "pixel_inputs" / "backbone_hidden" / "connector". Used at training start to tell whether NaN MLLM output on a rank
        comes from input preprocessing, the backbone or the connector."""
        from videox_fun.models.qwen_vl_utils import process_vision_info
        mllm = self.mllm
        messages, input_query, _, _ = mllm._build_messages(prompt, None, src_frames, None)
        text = mllm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = mllm.tokenizer(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(mllm.mllm_backbone.device, dtype=next(mllm.mllm_backbone.parameters()).dtype)
        for key, value in inputs.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                return f"pixel_inputs:{key}"
        outputs = mllm.mllm_backbone(**inputs, learnable_query=input_query.to(mllm.mllm_backbone.device),
                                     output_attentions=False, output_hidden_states=True, return_dict=True, logits_to_keep=1)
        if not torch.isfinite(outputs.hidden_states[-1]).all():
            return "backbone_hidden"
        prompt_embeds, _ = mllm._extract_text_and_queries_bs1(inputs["input_ids"], inputs["attention_mask"], outputs.hidden_states[-1])
        ctx = mllm.connetor(prompt_embeds[0].to(mllm.connetor[0].weight.dtype))
        if not torch.isfinite(ctx).all():
            return "connector"
        return None


def build_mllm(args, dit_dim=3072):
    """Qwen2.5-VL + connector; loads connector / query from moe_expert_init.safetensors. Returns the module on CPU (frozen, eval)."""
    from safetensors.torch import load_file
    from videox_fun.moe_common.mllm_encoder_moe import MLLMInContext, MLLMInContextConfig
    impl = getattr(args, "mllm_attn_impl", "sdpa")
    if impl and impl != "auto":
        os.environ["MLLM_ATTN_IMPL"] = impl        # the backend must be set at construction (set_attn_implementation is a silent no-op on the kiwi copy of Qwen)
    with no_zero3_init():
        mllm = MLLMInContext(MLLMInContextConfig(
            mllm_id=args.mllm_model, num_image_queries=256, num_video_queries=512, num_ref_queries=768,
            max_object_token=768, max_frames=16, max_pixels_per_frame=262144, dit_dim=dit_dim, _gradient_checkpointing=False))
    assert next(mllm.mllm_backbone.parameters()).dim() >= 1 and mllm.mllm_backbone.get_input_embeddings().weight.dim() == 2, \
        "MLLM was ZeRO-3-sharded (0-dim weights); check no_zero3_init"
    init_state = load_file(args.init_weights)
    mllm.load_state_dict({key[len("mllm."):]: value for key, value in init_state.items() if key.startswith("mllm.")}, strict=False)
    mllm.eval().requires_grad_(False)
    # The Qwen backbone defaults to flash_attention_2, which gave random NaNs in ~3-5% of calls in our setup (same weights and inputs;
    # SDPA on the same GPU is fine). After building the 55.9B videoeditor-56b model the rate is higher and rebuilding in-process does not help.
    # The MLLM is a frozen encoder, so switching backend only changes bf16 rounding; default sdpa, pass "auto" to keep the original behavior.
    if impl and impl != "auto":
        got = getattr(mllm.mllm_backbone.config, "_attn_implementation", None)
        assert got == impl, f"MLLM attention backend should be {impl}, got {got} (env var MLLM_ATTN_IMPL not picked up by mllm_encoder_moe?)"
    # Non-persistent buffers are not in the state_dict, so weight scans miss them; assert here that RoPE inv_freq was recomputed correctly (see mllm_encoder_moe)
    bad = [n for n, b in mllm.named_buffers() if n.endswith("inv_freq") and (not torch.isfinite(b).all() or float(b.abs().max()) > 1.0 + 1e-6)]
    assert not bad, f"MLLM RoPE inv_freq still holds garbage values: {bad}"
    return mllm


def build_vae(args, dtype=torch.bfloat16):
    """Wan2.2 VAE (class chosen by config), frozen, eval."""
    from omegaconf import OmegaConf
    from videox_fun.models import AutoencoderKLWan, AutoencoderKLWan3_8
    config = OmegaConf.load(args.config_path)
    vae_kwargs = OmegaConf.to_container(config["vae_kwargs"])
    vae_cls = {"AutoencoderKLWan": AutoencoderKLWan, "AutoencoderKLWan3_8": AutoencoderKLWan3_8}[vae_kwargs.get("vae_type", "AutoencoderKLWan3_8")]
    with no_zero3_init():
        vae = vae_cls.from_pretrained(os.path.join(args.pretrained_wan_path, vae_kwargs.get("vae_subpath", "Wan2.2_VAE.pth")),
                                      additional_kwargs=vae_kwargs).to(dtype=dtype)
    vae.eval().requires_grad_(False)
    return vae


def load_trainable_weights(dit, path):
    """Load the moe_v3_trainable.safetensors saved by training into dit (for inference / weights-only resume).
    Full training saves the whole DiT; an earlier run (MoE ffn only) saved just the MoE-layer ffn parameters. Both load
    (missing keys are allowed only for the MoE-layer ffn)."""
    # Stream-load key by key (safe_open) instead of reading the whole 104 GB file into memory: every rank calling load_file at once
    # would add ~900 GB of host memory per node. Same semantics as load_state_dict(strict=False): extra keys in the file are an error;
    # missing MoE-layer ffn/self_attn keys are an error; other missing keys (buffers etc.) are tolerated.
    from safetensors import safe_open
    targets = dict(dit.named_parameters())
    targets.update(dict(dit.named_buffers()))
    loaded, unexpected = set(), []
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = targets.get(k)
            if t is None:
                unexpected.append(k)
                continue
            with torch.no_grad():
                t.copy_(f.get_tensor(k).to(t.dtype))
            loaded.add(k)
    missing = [k for k in targets if k not in loaded and (".ffn." in k or ".self_attn." in k) and any(k.startswith(f"blocks.{i}.") for i in dit.moe_layers)]
    assert not missing and not unexpected, f"weight key mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
    return loaded
