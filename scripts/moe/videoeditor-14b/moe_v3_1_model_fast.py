#!/usr/bin/env python
"""FAST: copy of moe_v3_1_model.py that builds the DiT from wan_transformer3d_kiwi_moe_v3_1_fast.py (only that import differs). The original file is untouched.

Model assembly for videoeditor-14b: builds the MoE DiT (self-attention k/v and FFN both 7 experts, top-3) and loads its initial
weights; MLLM text encoder; VAE. Contains no DiT architecture code.

  DiT     KiwiMoeTransformer3DModel from wan_transformer3d_kiwi_moe_v3_1.py (full copy of the original kiwi DiT + MoE self-attention k/v + MoE FFN)
  MLLM    Qwen2.5-VL + connector (frozen), produces the DiT cross-attention context (full instruction; empty instruction = text-CFG unconditional branch)
  VAE     Wan2.2 VAE (frozen)

Where the MoE-layer weights come from (build_dit); self-attention k_experts / v_experts and FFN experts use the same order:
  experts[0]     main            ← original kiwi self_attn.k / self_attn.v / ffn.0 / ffn.2 (weight + bias)
  experts[1..5]  style / add / remove / replace / background ← original + merged per-category expert LoRA (merge_experts_v3_1.py)
  experts[6]     direct_finetune ← kiwi-edit fully fine-tuned directly on the goku data (step-3581)
  (training start point only; at inference experts_merged is empty, this step and the self-checks are skipped, and all expert
   weights come from load_trainable_weights(ckpt))
  gate           one for self-attention and one for the FFN, random init (std 0.02), trained from scratch
  self-attention q / o / norm ← original kiwi (loaded directly by from_pretrained); the original k / v projections are removed
  when there is no shared part (expert 0 replaces them)
Trainable (mark_trainable): the whole DiT, full fine-tuning. Parameters fall into three learning-rate groups:
  base     pre-existing params (attention q/o/shared k/v, cross-attention, non-MoE layers, embedding, head, ...)   constant learning_rate
  experts  k_experts / v_experts / ffn.experts of the MoE layers                                                   starts at new_layers_lr, linear decay to learning_rate
  gate     the two gates of each MoE layer                                                                         starts at gate_lr, linear to gate_lr_end
"""
import os
import re
import sys

import torch
import torch.nn as nn

_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXPERT_NAMES = ["main", "style", "add", "remove", "replace", "background", "direct_finetune"]     # experts 0..6, same order as merge_experts_v3_1.py
MOE_PROJS = ("self_attn.k", "self_attn.v", "ffn.0", "ffn.2")


class no_zero3_init:
    """Temporarily disable the DeepSpeed ZeRO-3 detection of transformers while building the MLLM / VAE.
    Once accelerate has registered a ZeRO-3 config, from_pretrained in transformers ZeRO-3-shards every newly built HF model too
    (parameters become 0-dim and the forward fails with 'weight' must be 2-D). The MLLM / VAE are frozen, replicated on every
    GPU and not managed by DeepSpeed, so they must be built in full."""

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
    """Build KiwiMoeTransformer3DModel and load its weights; returns the model on CPU.
    args needs: pretrained_wan_path, config_path, init_weights, experts_merged (experts_v3_1.safetensors written by merge_experts_v3_1.py),
    moe_layers (comma-separated), moe_num_experts (=7), moe_top_k, moe_aux_loss_alpha, use_gradient_checkpointing.
    experts_merged == "" (inference, only the released checkpoint): the merged file is not read and the expert self-checks are
    skipped; MoE experts keep their constructor values and the caller must then overwrite all of them with
    load_trainable_weights(ckpt) (build_dit_fast in the inference script asserts that the checkpoint covers every parameter)."""
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from safetensors.torch import load_file
    from wan_transformer3d_kiwi_moe_v3_1_fast import KiwiMoeTransformer3DModel   # FAST: the inference-speed copy of the DiT

    config = OmegaConf.load(args.config_path)
    dit_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    dit_dir = os.path.join(args.pretrained_wan_path, dit_kwargs.get("transformer_low_noise_model_subpath", "./"))
    if not os.path.isfile(os.path.join(dit_dir, "config.json")):
        dit_dir = args.pretrained_wan_path
    moe_layers = [int(s) for s in str(args.moe_layers).split(",") if s.strip()]
    dit_kwargs.update(moe_layers=moe_layers, moe_num_experts=int(args.moe_num_experts),
                      moe_top_k=int(args.moe_top_k), moe_aux_loss_alpha=float(args.moe_aux_loss_alpha))

    # 1) Original weights: from_pretrained reports the MoE-layer experts / gate keys as missing (new modules), filled by hand below; q/o/shared k/v/norm load normally
    dit = KiwiMoeTransformer3DModel.from_pretrained(dit_dir, transformer_additional_kwargs=dit_kwargs)

    # 2) Source-video condition conv vae_condition (in moe_expert_init.safetensors, keys dit.vae_condition.*)
    init_state = load_file(args.init_weights)
    cond_state = {key[len("dit."):]: value for key, value in init_state.items() if key.startswith("dit.")}
    _, unexpected = dit.load_state_dict(cond_state, strict=False)
    assert not unexpected, f"init key mismatch: {unexpected[:3]}"

    # 3) MoE layers: experts[0] ← original kiwi; experts[1..6] ← merged file (keys blocks.{b}.experts.{i}.{proj}.{weight,bias})
    assert int(args.moe_num_experts) == len(EXPERT_NAMES), f"this model is designed for {len(EXPERT_NAMES)} experts: {EXPERT_NAMES}"
    use_merged = bool(args.experts_merged)
    if use_merged:
        base = safe_open(os.path.join(dit_dir, "diffusion_pytorch_model.safetensors"), "pt")
        merged = safe_open(args.experts_merged, "pt")
    elif verbose:
        print("[moe-v3.1] experts_merged is empty: skipping merged-expert initialisation and its self-checks; "
              "MoE expert weights must come from load_trainable_weights(<ckpt>)", flush=True)

    def expert_module(block, proj, i):
        """The nn.Linear of expert i for projection proj in MoE-layer block."""
        if proj == "self_attn.k":
            return block.self_attn.k_experts[i]
        if proj == "self_attn.v":
            return block.self_attn.v_experts[i]
        return block.ffn.experts[i][0 if proj == "ffn.0" else 2]

    with torch.no_grad():
        for block_index in (moe_layers if use_merged else []):
            block = dit.blocks[block_index]
            for proj in MOE_PROJS:
                for i in range(len(EXPERT_NAMES)):
                    if i == 0:
                        W = base.get_tensor(f"blocks.{block_index}.{proj}.weight")
                        bias = base.get_tensor(f"blocks.{block_index}.{proj}.bias")
                    else:
                        W = merged.get_tensor(f"blocks.{block_index}.experts.{i}.{proj}.weight")
                        bias = merged.get_tensor(f"blocks.{block_index}.experts.{i}.{proj}.bias")
                    linear = expert_module(block, proj, i)
                    linear.weight.copy_(W)
                    linear.bias.copy_(bias)
    dit = dit.to(dtype=dtype)
    if args.use_gradient_checkpointing:
        dit.enable_gradient_checkpointing()

    # 4) Self-check (only when initialised from the merged file): expert 0 matches the original (k projection equals the shared k),
    #    expert 1 matches the merged file, expert 6 differs from the original
    if use_merged:
        with torch.no_grad():
            block_index = moe_layers[0]
            block = dit.blocks[block_index]
            assert torch.equal(block.self_attn.k_experts[0].weight, base.get_tensor(f"blocks.{block_index}.self_attn.k.weight").to(dtype))
            assert torch.equal(block.ffn.experts[0][2].weight, base.get_tensor(f"blocks.{block_index}.ffn.2.weight").to(dtype))
            assert torch.equal(block.ffn.experts[1][0].weight, merged.get_tensor(f"blocks.{block_index}.experts.1.ffn.0.weight").to(dtype))
            assert not torch.equal(block.self_attn.v_experts[6].weight, block.self_attn.v_experts[0].weight)
    if verbose:
        n_moe = sum(p.numel() for i in moe_layers for p in list(dit.blocks[i].ffn.parameters()) + list(dit.blocks[i].self_attn.parameters()))
        print(f"[moe-v3.1] DiT: moe_layers={moe_layers} experts={args.moe_num_experts}({','.join(EXPERT_NAMES)}) top_k={args.moe_top_k} "
              f"aux_alpha={args.moe_aux_loss_alpha} MoE-layer (self-attn+FFN) params {n_moe / 1e9:.2f}B "
              f"total params {sum(p.numel() for p in dit.parameters()) / 1e9:.2f}B", flush=True)
    return dit


def mark_trainable(dit):
    """Full training: every parameter is trainable. Returns three learning-rate groups (base_params, expert_params, gate_params):
    expert_params = MoE-layer experts (ffn.experts / self_attn.k_experts / self_attn.v_experts), gate_params = the two gates of each MoE layer,
    base_params = everything else (including self-attention q/o/shared k/v). These name segments exist only in MoE layers, so the
    parameter name alone decides the group."""
    base_params, expert_params, gate_params = [], [], []
    for name, param in dit.named_parameters():
        param.requires_grad_(True)
        m = re.match(r"blocks\.(\d+)\.(ffn|self_attn)\.(experts|k_experts|v_experts|gate)\.", name)
        if m is None:
            base_params.append(param)                 # non-MoE layers, or q/o/shared k/v/cross-attn/norm etc. of MoE layers
        elif m.group(3) == "gate":
            gate_params.append(param)
        else:
            expert_params.append(param)
    assert len(gate_params) == 2 * len(dit.moe_layers), (len(gate_params), len(dit.moe_layers))
    count = lambda params: sum(p.numel() for p in params)
    print(f"[moe-v3.1] trainable (full): base {count(base_params) / 1e9:.2f}B + experts(attn k/v + ffn) {count(expert_params) / 1e9:.2f}B "
          f"+ gate {count(gate_params):,}  = total {(count(base_params) + count(expert_params) + count(gate_params)) / 1e9:.2f}B", flush=True)
    return base_params, expert_params, gate_params


def trainable_state_keys(dit):
    """Names of the trainable parameters (only these are saved to the ckpt; with full training that is every DiT parameter, no buffers)."""
    return {name for name, param in dit.named_parameters() if param.requires_grad}


# =========================================================================== #
#  MLLM text encoder                                                           #
# =========================================================================== #
class TextEncoder:
    """Encodes instruction + source-video frames into the DiT cross-attention context. The MLLM is frozen and not managed by DeepSpeed."""

    def __init__(self, mllm):
        self.mllm = mllm

    @torch.no_grad()
    def encode(self, prompt, src_frames):
        """Instruction + source-video frames → the "text + query" segments of the last Qwen2.5-VL hidden state → connector → [L, 3072].
        Line-by-line replica of MLLMInContext.forward up to prompt_embeds."""
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
        """Text-CFG unconditional branch: empty instruction, the MLLM still sees the source video."""
        return self.encode("", src_frames)

    @torch.no_grad()
    def encode_batch(self, prompts, frames_list):
        """Speed-up: run a whole batch through the MLLM in one pass (per-sample encode costs 8 forward passes per step, ~2.7 s → 1 pass).
        prompts: list[str] ("" = text-CFG unconditional branch, same as encode_null, encoded in the same batch as the others);
        frames_list: list[list[PIL]] of the same length.
        Returns list[Tensor[L_b, 3072]], per sample equivalent to encode, via mllm_encoder_moe.MLLMInContext.forward_batch:
        it temporarily switches to right padding (real token positions 0..L_b-1 match bs=1, so RoPE does not shift), expands
        learnable_query to [B, nq, dim] for masked_scatter,
        then per sample extracts [text + query] with _extract_text_and_queries_bs1 and applies the connector; the only difference is
        bf16 rounding in the batched Qwen forward."""
        assert len(prompts) == len(frames_list) and len(prompts) > 0, (len(prompts), len(frames_list))
        return self.mllm.forward_batch(list(prompts), list(frames_list), is_video=True)

    @torch.no_grad()
    def self_test(self, prompt, src_frames):
        """Find the earliest stage of one encode that yields non-finite values: returns None (all finite) or "pixel_inputs" / "backbone_hidden" / "connector".
        Used at training start to tell whether a NaN MLLM output on some rank comes from input preprocessing, the backbone or the connector."""
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
                                     output_attentions=False, output_hidden_states=True, return_dict=True)
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
        "MLLM was ZeRO-3 sharded (0-dim weights); check no_zero3_init"
    init_state = load_file(args.init_weights)
    mllm.load_state_dict({key[len("mllm."):]: value for key, value in init_state.items() if key.startswith("mllm.")}, strict=False)
    mllm.eval().requires_grad_(False)
    # The Qwen backbone defaults to flash_attention_2, which gave random NaNs in ~3-5% of calls in our setup (same weights and inputs; SDPA is fine).
    # The MLLM is a frozen encoder, so switching backend only changes bf16 rounding; default sdpa, pass "auto" to keep the original behaviour.
    if impl and impl != "auto":
        got = getattr(mllm.mllm_backbone.config, "_attn_implementation", None)
        assert got == impl, f"MLLM attention backend should be {impl}, got {got} (env var MLLM_ATTN_IMPL not picked up by mllm_encoder_moe?)"
    # Non-persistent buffers are not in the state_dict, so weight scans miss them; assert here that every RoPE inv_freq was recomputed correctly (see mllm_encoder_moe)
    bad = [n for n, b in mllm.named_buffers() if n.endswith("inv_freq") and (not torch.isfinite(b).all() or float(b.abs().max()) > 1.0 + 1e-6)]
    assert not bad, f"MLLM RoPE inv_freq still holds garbage values: {bad}"
    return mllm


def build_vae(args, dtype=torch.bfloat16):
    """Wan2.2 VAE (type chosen by the config), frozen, eval."""
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
    """Load the moe_v3_1_trainable.safetensors saved by training into dit (for inference / weights-only resume).
    Full training saves the whole DiT; a partial checkpoint also loads as long as every ffn / self_attn key of the MoE layers is
    present (keys outside them may be missing)."""
    from safetensors.torch import load_file
    state = load_file(path)
    missing, unexpected = dit.load_state_dict(state, strict=False)
    missing = [k for k in missing if (".ffn." in k or ".self_attn." in k) and any(k.startswith(f"blocks.{i}.") for i in dit.moe_layers)]
    assert not missing and not unexpected, f"weight key mismatch: missing={missing[:3]} unexpected={list(unexpected)[:3]}"
    return state
