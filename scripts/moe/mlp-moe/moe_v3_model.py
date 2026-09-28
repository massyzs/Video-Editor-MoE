#!/usr/bin/env python
"""Assembly file for mlp-moe: builds the MoE DiT and loads its initial weights; MLLM text encoding; VAE. No DiT architecture code here.

  DiT     KiwiMoeTransformer3DModel from wan_transformer3d_kiwi_moe.py (full copy of the original kiwi DiT + MoE FFN)
  MLLM    Qwen2.5-VL + connector (frozen); produces the DiT cross-attention context (full instruction; empty instruction = text-CFG unconditional branch)
  VAE     Wan2.2 VAE (frozen)

Where the MoE-layer weights come from (build_dit; no shared expert, Mixtral style):
  experts[0..5]  ← the 5 category experts (style/add/remove/replace/background) merged by merge_experts.py + the GOKU direct fine-tune (direct_finetune);
                    biases: kiwi's for the LoRA experts, its own for direct_finetune
  experts[6..11] ← a copy of each of experts[0..5] with 1% random noise (relative to each tensor's std), fixed seed, identical on all ranks
  gate          ← random init (std 0.02), trained from scratch
Trainable (mark_trainable): the whole DiT is trained in full (an earlier run trained only the MoE-layer ffn). Parameters form three LR groups:
  base     all original parameters (attention, non-MoE FFN, embedding, head, ...)  always learning_rate
  experts  routed experts added in the MoE layers (ffn.experts.*)                  start at new_layers_lr, decay linearly to learning_rate
  gate     MoE-layer gates                                                         same as experts
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

EXPERT_NAMES = ["style", "add", "remove", "replace", "background", "direct_finetune"]   # 6 base experts; each is duplicated with noise for training = 12 experts


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
    """Build KiwiMoeTransformer3DModel and load its weights; returns the model on CPU.
    args needs: pretrained_wan_path, config_path, init_weights, experts_merged (output of merge_experts.py),
    moe_layers (comma-separated), moe_num_experts, moe_top_k, moe_aux_loss_alpha, expert_copy_noise, use_gradient_checkpointing.
    experts_merged == "" (inference with only the released checkpoint): skips step 3 (read merged experts + perturb the second copy)
    and step 4 (self-check of experts vs merged weights); the MoE-layer experts/gates keep their constructor values and the caller
    must then overwrite them all with load_trainable_weights(<ckpt>/moe_v3_trainable.safetensors)."""
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from safetensors.torch import load_file
    from wan_transformer3d_kiwi_moe import KiwiMoeTransformer3DModel

    config = OmegaConf.load(args.config_path)
    dit_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    dit_dir = os.path.join(args.pretrained_wan_path, dit_kwargs.get("transformer_low_noise_model_subpath", "./"))
    if not os.path.isfile(os.path.join(dit_dir, "config.json")):
        dit_dir = args.pretrained_wan_path
    moe_layers = [int(s) for s in str(args.moe_layers).split(",") if s.strip()]
    dit_kwargs.update(moe_layers=moe_layers, moe_num_experts=int(args.moe_num_experts),
                      moe_top_k=int(args.moe_top_k), moe_aux_loss_alpha=float(args.moe_aux_loss_alpha))

    # 1) Original weights: from_pretrained reports the MoE-layer ffn keys as missing (module names changed); loaded by hand below
    dit = KiwiMoeTransformer3DModel.from_pretrained(dit_dir, transformer_additional_kwargs=dit_kwargs)

    # 2) Source-video conditioning conv vae_condition (in moe_expert_init.safetensors, keys dit.vae_condition.*)
    init_state = load_file(args.init_weights)
    cond_state = {key[len("dit."):]: value for key, value in init_state.items() if key.startswith("dit.")}
    _, unexpected = dit.load_state_dict(cond_state, strict=False)
    assert not unexpected, f"init key mismatch: {unexpected[:3]}"

    # 3) MoE layers: experts ← merged weights (+ perturbed second copy); biases from kiwi (no shared expert)
    #    empty experts_merged = inference from the released checkpoint only: this step and the step-4 self-check are skipped,
    #    the experts are filled by load_trainable_weights
    use_merged = bool(getattr(args, "experts_merged", ""))
    num_base = len(EXPERT_NAMES)
    noise_scale = float(args.expert_copy_noise)
    if use_merged:
        base = safe_open(os.path.join(dit_dir, "diffusion_pytorch_model.safetensors"), "pt")
        merged = safe_open(args.experts_merged, "pt")
        assert int(args.moe_num_experts) == 2 * num_base, f"this version is designed for {num_base} base experts x 2 copies = {2 * num_base} experts"
        generator = torch.Generator().manual_seed(2026)          # fixed seed: every rank builds exactly the same perturbation
        with torch.no_grad():
            for block in moe_layers:
                ffn = dit.blocks[block].ffn
                for proj, layer_index in (("ffn.0", 0), ("ffn.2", 2)):
                    base_bias = base.get_tensor(f"blocks.{block}.{proj}.bias")
                    for i, expert_name in enumerate(EXPERT_NAMES):
                        W_expert = merged.get_tensor(f"blocks.{block}.{expert_name}.{proj}.weight")
                        bias_key = f"blocks.{block}.{expert_name}.{proj}.bias"
                        b = merged.get_tensor(bias_key) if bias_key in merged.keys() else base_bias     # direct_finetune carries its own bias
                        ffn.experts[i][layer_index].weight.copy_(W_expert)
                        ffn.experts[i][layer_index].bias.copy_(b)
                        # second copy: same weights plus a small perturbation, so the copies differ from step 0 and diverge in training
                        noise = torch.randn(W_expert.shape, generator=generator) * (W_expert.float().std() * noise_scale)
                        ffn.experts[num_base + i][layer_index].weight.copy_((W_expert.float() + noise).to(W_expert.dtype))
                        ffn.experts[num_base + i][layer_index].bias.copy_(b)
    dit = dit.to(dtype=dtype)
    if args.use_gradient_checkpointing:
        dit.enable_gradient_checkpointing()

    # 4) Self-check: experts match the merged weights and the second copy really differs (only when merged weights are used)
    if use_merged:
        with torch.no_grad():
            block = moe_layers[0]
            ffn = dit.blocks[block].ffn
            assert torch.equal(ffn.experts[0][2].weight, merged.get_tensor(f"blocks.{block}.{EXPERT_NAMES[0]}.ffn.2.weight").to(dtype))
            assert not torch.equal(ffn.experts[0][0].weight, ffn.experts[num_base][0].weight)
    if verbose:
        n_moe = sum(p.numel() for i in moe_layers for p in dit.blocks[i].ffn.parameters())
        print(f"[moe-v3] DiT: moe_layers={moe_layers} experts={args.moe_num_experts} top_k={args.moe_top_k} "
              f"aux_alpha={args.moe_aux_loss_alpha} copy_noise={noise_scale} MoE params {n_moe / 1e9:.2f}B "
              f"total params {sum(p.numel() for p in dit.parameters()) / 1e9:.2f}B", flush=True)
    return dit


def mark_trainable(dit):
    """Full training: every parameter is trainable. Returns three learning-rate groups (base_params, expert_params, gate_params):
    expert_params = MoE-layer experts, gate_params = MoE-layer gates, base_params = everything else.
    Only the MoE-layer ffn has the name segments experts. / gate., so the parameter name is enough to split them."""
    base_params, expert_params, gate_params = [], [], []
    for name, param in dit.named_parameters():
        param.requires_grad_(True)
        m = re.match(r"blocks\.(\d+)\.ffn\.(experts|gate)\.", name)
        if m is None:
            base_params.append(param)                 # non-MoE layers, or attention etc. of MoE layers
        elif m.group(2) == "experts":
            expert_params.append(param)
        else:
            gate_params.append(param)
    assert len(gate_params) == len(dit.moe_layers), (len(gate_params), len(dit.moe_layers))
    count = lambda params: sum(p.numel() for p in params)
    print(f"[moe-v3] trainable(full): base {count(base_params) / 1e9:.2f}B + experts(new) {count(expert_params) / 1e9:.2f}B "
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
    with no_zero3_init():
        mllm = MLLMInContext(MLLMInContextConfig(
            mllm_id=args.mllm_model, num_image_queries=256, num_video_queries=512, num_ref_queries=768,
            max_object_token=768, max_frames=16, max_pixels_per_frame=262144, dit_dim=dit_dim, _gradient_checkpointing=False))
    assert next(mllm.mllm_backbone.parameters()).dim() >= 1 and mllm.mllm_backbone.get_input_embeddings().weight.dim() == 2, \
        "MLLM was ZeRO-3 sharded (0-dim weights); check no_zero3_init"
    init_state = load_file(args.init_weights)
    mllm.load_state_dict({key[len("mllm."):]: value for key, value in init_state.items() if key.startswith("mllm.")}, strict=False)
    mllm.eval().requires_grad_(False)
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
    from safetensors.torch import load_file
    state = load_file(path)
    # Older ckpts contain ffn.shared.* (cancelled out in the former difference formulation, never affected output or got updated); drop on load
    state = {k: v for k, v in state.items() if ".ffn.shared." not in k}
    missing, unexpected = dit.load_state_dict(state, strict=False)
    missing = [k for k in missing if ".ffn." in k and any(k.startswith(f"blocks.{i}.") for i in dit.moe_layers)]
    assert not missing and not unexpected, f"weight key mismatch: missing={missing[:3]} unexpected={list(unexpected)[:3]}"
    return state
