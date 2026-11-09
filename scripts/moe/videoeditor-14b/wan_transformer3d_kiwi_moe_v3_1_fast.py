# =============================================================================
# FAST: inference-speed copy of wan_transformer3d_kiwi_moe_v3_1.py (that file is untouched; loaded by moe_v3_1_model_fast.py). Every difference is marked "# FAST".
#   The MoE module region (make_ffn .. MoEKVSelfAttention) is the same code as the 56B FAST copy (identical semantics for the 14B: expert dtype defaults to None):
#   FAST ① fuse_for_inference() stacks expert weights into contiguous tensors and aliases the expert parameters to views; FAST ② fused k/v GEMM;
#   FAST ③ impl="grouped" = one torch._grouped_mm per linear, no host syncs (bias added after the bf16 GEMM rounding is the only numerical difference to the loop);
#   FAST ④ KiwiMoeTransformer3DModel.fuse_experts_for_inference(impl, fuse_kv); FAST ⑥ sinusoidal_embedding_1d builds its arange on the position's device.
# =============================================================================
# videoeditor-14b DiT (moe_v3_1): on top of mlp-moe, both the FFN and the self-attention k/v are MoE; 7 full-weight experts, no shared FFN
# This file = a complete copy of videox_fun/models/wan_transformer3d_kiwi.py; every change is marked "MOE-V3.1":
#   MOE-V3.1 ①  MoE modules (before WanAttentionBlock):
#       route_tokens / expert_balance_loss / AddAuxiliaryLoss / Fp32Gate   shared pieces: gating and the DeepSeek expert-level balance loss
#       MoEFFN          y = Σ_{i∈top_k} ĝ_i · expert_i(u)                 Mixtral form, no shared expert (each expert is a complete set of weights)
#       MoEKVSelfAttention   q / o shared; each token carries top_k expert (k_i, v_i) copies (no shared pair, Mixtral form; shared_kv=True adds 1 shared pair back),
#                            all concatenated into the key/value sequence and normalized by one softmax; the gate is a multiplicative prior:  y_s = Σ_t Σ_i ĝ_{t,i} e^{q_s·k_t^i} v_t^i / Z_s
#                            At step 0, with every expert ≈ base, this equals the original attention exactly. log ĝ enters the dot product via 8 extra head dims; kernel unchanged.
#   MOE-V3.1 ②  moe argument of WanAttentionBlock: if given, self.self_attn becomes MoEKVSelfAttention and self.ffn becomes MoEFFN
#   MOE-V3.1 ③  moe_layers / moe_num_experts(7) / moe_top_k(3) / moe_aux_loss_alpha(0.001) in KiwiMoeTransformer3DModel.__init__,
#               moe_attns() / moe_ffns() for the training script's logging
#   MOE-V3.1 ④  in forward, the time_embedding input is cast .to(dtype) (same as mlp-moe; ZeRO-3 linears ignore autocast)
# Cross-attention / modulation / forward are otherwise verbatim the original. Sources of the 7 experts: see merge_experts_v3_1.py / moe_v3_1_model.py:
#   0 = original kiwi, 1..5 = original + category-expert LoRA merged (style/add/remove/replace/background), 6 = kiwi directly full-finetuned on goku data (direct_finetune)
# =============================================================================
# Modified from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

"""Kiwi-Edit DiT: a FULLY SELF-CONTAINED Wan2.2 transformer (no longer inherits Wan2_2Transformer3DModel).

Everything needed from wan_transformer3d.py is INLINED here: RoPE/sinusoidal helpers, WanRMSNorm/
WanLayerNorm, WanSelfAttention/WanCrossAttention (+T2V/I2V variants), WanAttentionBlock, Head, MLPProj,
and the whole base model class (__init__ / enable_* / from_pretrained / _convert / unpatchify / init_weights).
KiwiWanTransformer3DModel inherits directly from (ModelMixin, ConfigMixin, FromOriginalModelMixin).

Three KIWI changes relative to the original Wan2.2:
  1. Source-video condition self.vae_condition (zero-init Conv3d, added to the noisy latent scaled by sigma);
  2. Reference-image condition self.ref_vae_condition (zero-init Conv3d, tokens prepended/appended to the sequence);
  3. MLLM context replaces text_embedding (bypassed in forward; padded + context_lens for varlen attention).

* explicit-moe extension point: to add a grouped-query cross-attention after the regular kiwi-edit query
  cross-attention, edit WanAttentionBlock.cross_attn_ffn / WanCrossAttention in this file; other wan models are unaffected.
"""

import glob
import json
import math
import os
import types
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import time   # FAST
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import is_torch_version, logging
from torch import nn

from videox_fun.dist import (get_sequence_parallel_rank,
                    get_sequence_parallel_world_size, get_sp_group,
                    usp_attn_forward, xFuserLongContextAttention)
from videox_fun.utils import cfg_skip
from videox_fun.models.attention_utils import attention

# Attention backend switch (to enable FA3), env var MOE_ATTN_BACKEND ∈ {fa2 (default, bit-identical to before), fa3, cudnn, auto}.
#   fa2 / fa3: use the library attention() with a forced fa_version (the library prefers FA3 when flash_attn_interface is importable; pinning it here removes the environment dependence);
#   cudnn: cuDNN backend of torch's built-in SDPA (no extra package; benchmark: head_dim 136 unsupported, so ruled out); equal-length keys only (k_lens=None), variable length falls back to the library;
#   auto: library default (FA4 > FA3 > FA2). Single-GPU benchmark (real shapes): FA3 accuracy on par with FA2 (2.3e-3 relative to fp32), ~2x faster -> the launcher exports MOE_ATTN_BACKEND=fa3.
_ATTN_BACKEND = os.environ.get("MOE_ATTN_BACKEND", "fa2").lower()
_FA_VERSION = {"fa2": 2, "fa3": 3}.get(_ATTN_BACKEND)


_CAPTURE_SAFE = False      # FAST ⑨: set by set_capture_safe(True) when the DiT forward is to be captured in a CUDA graph


def set_capture_safe(flag):
    """FAST ⑨: route equal-length attention through _flash2_equal_len (cu_seqlens built on the device). The library's flash_attention() builds q_lens/k_lens as
    CPU tensors and copies them to the GPU in every call, a pageable host->device copy that CUDA-graph capture does not allow. Same kernel, same arguments, same output."""
    global _CAPTURE_SAFE
    _CAPTURE_SAFE = bool(flag)


def _flash2_equal_len(q, k, v, softmax_scale=None, window_size=(-1, -1), dtype=torch.bfloat16):
    """FAST ⑨: videox_fun.models.attention_utils.flash_attention() for q_lens = k_lens = None and FA2, line by line, except that cu_seqlens come from a device-side
    arange (identical values to cat([0], [l]*b).cumsum). q/k/v: [B, L, H, D]."""
    import flash_attn
    half_dtypes = (torch.float16, torch.bfloat16)
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
    half = lambda x: x if x.dtype in half_dtypes else x.to(dtype)
    q = half(q.flatten(0, 1)); k = half(k.flatten(0, 1)); v = half(v.flatten(0, 1))
    q = q.to(v.dtype); k = k.to(v.dtype)
    cu_q = torch.arange(0, (b + 1) * lq, lq, dtype=torch.int32, device=q.device)
    cu_k = torch.arange(0, (b + 1) * lk, lk, dtype=torch.int32, device=q.device)
    x = flash_attn.flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=lq, max_seqlen_k=lk, dropout_p=0.0,
                                          softmax_scale=softmax_scale, causal=False, window_size=window_size, deterministic=False).unflatten(0, (b, lq))
    return x.type(out_dtype)


def _attn(q, k, v, k_lens=None, softmax_scale=None, window_size=(-1, -1)):
    """Single entry point for all self- / cross- / MoE attention (see _ATTN_BACKEND). q/k/v: [B, L, H, D]."""
    if _ATTN_BACKEND == "cudnn" and q.device.type == "cuda" and k_lens is None and tuple(window_size) == (-1, -1):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale)
        return out.transpose(1, 2).contiguous()
    if _CAPTURE_SAFE and k_lens is None and _FA_VERSION == 2 and q.device.type == "cuda":   # FAST ⑨
        return _flash2_equal_len(q, k, v, softmax_scale=softmax_scale, window_size=window_size)
    return attention(q, k, v, k_lens=k_lens, softmax_scale=softmax_scale, window_size=window_size, fa_version=_FA_VERSION)
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from videox_fun.models.kiwi_embedders import ConditionalEmbedder
from videox_fun.models.cache_utils import TeaCache
from videox_fun.models.wan_camera_adapter import SimpleAdapter


def sinusoidal_embedding_1d(dim, position):
    """
    Generate 1D sinusoidal positional embeddings.
    
    Args:
        dim: Embedding dimension (must be even)
        position: Position indices tensor
    
    Returns:
        Sinusoidal embedding tensor with shape (len(position), dim)
    """
    # Validate dimension is even and split into half
    assert dim % 2 == 0
    half = dim // 2
    # Convert position to float64 for numerical precision
    position = position.type(torch.float64)

    # Compute sinusoidal frequencies: position * 10000^(-i/dim/2)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half, device=position.device, dtype=position.dtype).div(half)))   # FAST ⑥: no synchronous host->device copy
    # Concatenate cosine and sine components
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """
    Compute RoPE frequency parameters for a given sequence length.
    
    Args:
        max_seq_len: Maximum sequence length
        dim: Dimension of the embedding (must be even)
        theta: Base frequency scaling factor
    
    Returns:
        Complex frequency tensor with shape (max_seq_len, dim/2)
    """
    assert dim % 2 == 0
    # Compute inverse frequency scaling: 1/theta^(2i/dim)
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    # Convert to complex form: exp(i * freq)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# Modified from https://github.com/thu-ml/RIFLEx/blob/main/riflex_utils.py
@amp.autocast(enabled=False)
def get_1d_rotary_pos_embed_riflex(
    pos: Union[np.ndarray, int],
    dim: int,
    theta: float = 10000.0,
    use_real=False,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
    L_test_scale: Optional[int] = None,
):
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    # Convert position to tensor if needed
    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    # Compute base frequencies: 1/theta^(2i/dim)
    freqs = 1.0 / torch.pow(theta,
        torch.arange(0, dim, 2).to(torch.float64).div(dim))

    # === Riflex modification start ===
    # Reduce the intrinsic frequency to stay within a single period after extrapolation (see Eq. (8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply by 0.9 to keep the extrapolated length below 90% of a single period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===
    if L_test_scale is not None:
        freqs[k-1] = freqs[k-1] / L_test_scale

    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # Lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Complex64     # [S, D/2]
        return freqs_cis


@amp.autocast(enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    """
    Apply rotary positional embedding to input tensor.
    
    Args:
        x: Input tensor with shape (batch, seq_len, n_channels, c*2)
        grid_sizes: Grid dimensions (f, h, w) for each sample
        freqs: Precomputed frequency parameters
    
    Returns:
        Tensor with RoPE applied
    """
    dtype = x.dtype
    n, c = x.size(2), x.size(3) // 2

    # Split frequencies into temporal, height, and width components
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # Process each sample in the batch
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # Reshape and convert to complex numbers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        # Broadcast frequencies to match spatial dimensions
        freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
        dim=-1).reshape(seq_len, 1, -1)

        # Apply rotation: x * exp(i*freq)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # Concatenate with padding tokens (if any)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # Append to collection
        output.append(x_i)
    return torch.stack(output).to(dtype)


def rope_apply_qk(q, k, grid_sizes, freqs):
    """
    Apply RoPE to both query and key tensors.
    
    Args:
        q: Query tensor
        k: Key tensor
        grid_sizes: Grid dimensions for spatial positioning
        freqs: Precomputed frequency parameters
    
    Returns:
        Tuple of (q, k) with RoPE applied
    """
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)
    return q, k


class WanRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """Layer Normalization with dtype preservation."""

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x).type_as(x)

    def _norm(self, x):
        """Compute Layer Normalization with dtype handling."""
        if self.weight is not None:
            x = super().forward(x.to(self.weight.dtype))
            return x
        else:
            return super().forward(x)


class WanSelfAttention(nn.Module):
    """Wan self-attention mechanism with RoPE and optional windowed attention."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]; None when the model forward finds all lengths equal, so attention() uses a flatten view instead of per-sample slice copies of k/v
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # Query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q, k = rope_apply_qk(q, k, grid_sizes, freqs)

        x = _attn(
            q.to(dtype), 
            k.to(dtype), 
            v=v.to(dtype),
            k_lens=seq_lens,
            window_size=self.window_size
        )

        # Output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    """Text-to-video cross-attention layer."""

    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        
        # Compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        
        # Compute attention
        x = _attn(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )

        # Output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    """Image-to-video cross-attention layer with separate image context processing."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        img_x = _attn(
            q.to(dtype), 
            k_img.to(dtype), 
            v_img.to(dtype), 
            k_lens=None
        )
        # Compute text cross-attention
        x = _attn(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )

        # Output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    """Generic cross-attention layer."""
    
    def forward(self, x, context, context_lens, dtype=torch.bfloat16, t=0):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        
        # Compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        
        # Compute attention
        x = _attn(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )

        # Output
        x = x.flatten(2)
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    'cross_attn': WanCrossAttention,
}


# =============================================================================
# MOE-V3.1 ①  MoE modules
# =============================================================================
class AddAuxiliaryLoss(torch.autograd.Function):
    """Attach the load-balancing auxiliary loss to the main output's graph (as in DeepSeek).
    Forward: returns y unchanged. Backward: gives aux_loss a gradient of 1.
    So the training script calls backward on the main loss only, and each MoE layer's aux-loss gradient reaches its gate automatically,
    with no per-layer collection; gradient checkpointing rebuilds this node on recompute, with identical results."""

    @staticmethod
    def forward(ctx, y, aux_loss):
        assert aux_loss.numel() == 1
        ctx.dtype = aux_loss.dtype
        ctx.required_aux_loss = aux_loss.requires_grad
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class Fp32Gate(nn.Module):
    """Gate: Linear(dim → num_experts, no bias), computing logits in fp32 inside its own forward.
    It must be a module rather than using .weight from outside: ZeRO-3 only gathers a module's sharded parameters when its forward is called,
    so direct .weight access returns the shard placeholder (the mlp-moe smoke test failed with "'weight' must be 2-D")."""

    def __init__(self, dim, num_experts):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, tokens):                      # [T, dim] -> [T, num_experts] fp32
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            return F.linear(tokens.float(), self.weight.float())


def route_tokens(gate, tokens, top_k):
    """Gate scores -> each token picks top_k experts, weights renormalized within top_k to sum to 1.
    Returns scores [T, N] fp32 (full softmax, for the balance loss), topk_weight [T, k] fp32 (ĝ), topk_idx [T, k]."""
    scores = F.softmax(gate(tokens), dim=-1)
    topk_weight, topk_idx = torch.topk(scores, top_k, dim=-1)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    return scores, topk_weight, topk_idx


def dispatch_by_expert(topk_idx, num_experts):
    """Speed-up: group the [T, k] top-k choices by expert with one sort instead of one (topk_idx == i).nonzero() per expert.
    Why: nonzero() must know the nonzero count to allocate its output, so each call is a host-device sync: 7 experts = 7 per module; here only 1 per module.
    How: flatten position p = token·k + slot, stable-sort by expert id -> each expert's p are contiguous and increasing, identical element by element to the row-major
    order of (topk_idx == i).nonzero() (token ascending, slot ascending), so every expert gets exactly the same (token_idx, slot) as before and the later expert forward / index_put_ /
    index_add_ order is unchanged (bitwise equivalent, not just mathematically).
    Counting deliberately avoids torch.bincount: its CUDA implementation calls max().item() to size the bins, another sync; scatter_add_ integer accumulation is exact.
    Returns token_idx_all [T·k], slot_all [T·k] (sorted by expert, sliced in turn by counts), counts: python list[int] (the only sync), usage: [N] fp32 tensor (for logging)."""
    top_k = topk_idx.shape[1]
    flat = topk_idx.reshape(-1)                                                       # [T·k], position p = token·k + slot
    order = torch.argsort(flat, stable=True)                                          # group by expert id, p increasing within a group (= original nonzero order)
    counts_t = torch.zeros(num_experts, dtype=torch.long, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))
    counts = counts_t.tolist()                                                        # the only host-device sync
    token_idx_all = torch.div(order, top_k, rounding_mode="floor")
    slot_all = order - token_idx_all * top_k
    return token_idx_all, slot_all, counts, counts_t.float()


def expert_balance_loss(scores, topk_idx, batch, seq_len, num_experts, top_k, alpha):
    """DeepSeek expert-level load-balancing loss, computed per sample, keeps experts from dying:
        f_i = times expert i is selected / (top_k·L) · N      P_i = mean softmax score of expert i over the sample's tokens
        aux = alpha · mean_samples Σ_i f_i · P_i           (= alpha when perfectly balanced)"""
    counts = F.one_hot(topk_idx, num_experts).sum(dim=1)                          # [T, N] which experts each token chose
    counts = counts.view(batch, seq_len, num_experts).sum(dim=1).float()          # [B, N] selection count of each expert per sample
    load = counts * num_experts / (top_k * seq_len)                               # f_i
    prob = scores.view(batch, seq_len, num_experts).mean(dim=1)                   # P_i
    return (load * prob).sum(dim=-1).mean() * alpha


def make_ffn(dim, ffn_dim, dtype=None):
    """Original FFN structure: Linear(dim→ffn_dim) → GELU(tanh) → Linear(ffn_dim→dim); every expert is one. MOE-V3.2 ⑤ (56B copy; dtype None here): dtype=None uses the default dtype."""
    return nn.Sequential(nn.Linear(dim, ffn_dim, dtype=dtype), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim, dtype=dtype))


class MoEFFN(nn.Module):
    """MoE FFN (replaces self.ffn in WanAttentionBlock, same input/output shape), Mixtral style, no shared expert:

        y = Σ_{i ∈ top_k(u)} ĝ_i · expert_i(u),      ĝ = softmax weights renormalised within top_k (Σĝ = 1)

    experts   N full FFNs (original structure), each a full set of weights: 0 = original kiwi, 1..5 = category-expert LoRA merges, 6 = goku direct fine-tune
    gate      Fp32Gate(dim → N), scores every token
    Each expert only reads the tokens that selected it; every expert is called once per step (possibly on an empty tensor), since ZeRO-3 needs every parameter in the forward every step.
    The balance loss is attached to y via AddAuxiliaryLoss; last_aux_loss / last_selected / last_topk_idx are only for logging and inference plots.
    (mlp-moe's residual form shared + Σĝ(e_i − shared) cancels shared when Σĝ = 1, so shared gets no gradient; hence videoeditor-14b/56b drop the shared expert.)
    """

    def __init__(self, dim, ffn_dim, num_experts, top_k, aux_loss_alpha, expert_dtype=None):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_alpha = aux_loss_alpha
        self.experts = nn.ModuleList([make_ffn(dim, ffn_dim, expert_dtype) for _ in range(num_experts)])   # MOE-V3.2 ⑤ (56B copy; dtype None here)
        self.gate = Fp32Gate(dim, num_experts)
        self.last_aux_loss = None       # aux loss of the latest forward (detached), for logging
        self.last_selected = None       # per-expert fraction of tokens that selected it in the latest forward [N] (mean of 1 = selected / 0 = not, Σ = top_k), for logging
        self.last_topk_idx = None       # expert indices chosen by each token in the latest forward [T, k], for inference plots
        self.impl = "loop"              # FAST: "loop" (original per-expert loop) | "grouped" (one torch._grouped_mm per linear; needs fuse_for_inference)
        self._fused = None              # FAST: (W1 [E,F,D], b1 [E,F], W2 [E,D,F], b2 [E,D]) stacked expert weights, set by fuse_for_inference

    @torch.no_grad()
    def fuse_for_inference(self, impl="loop"):
        """FAST ①: stack the E experts' Linear weights into contiguous tensors on this module's device and re-point every expert parameter to a view of the
        stack (same values, same state_dict keys; the per-expert loop keeps running on the views). Call after the module is on its final device."""
        assert impl in ("loop", "grouped"), impl
        if impl == "grouped":
            assert hasattr(torch, "_grouped_mm"), "impl='grouped' needs torch._grouped_mm (torch >= 2.8)"
        E = self.num_experts
        lin1, lin2 = self.experts[0][0], self.experts[0][2]
        dev, dt = lin1.weight.device, lin1.weight.dtype
        D, Fd = lin1.in_features, lin1.out_features
        W1 = torch.empty(E, Fd, D, device=dev, dtype=dt); b1 = torch.empty(E, Fd, device=dev, dtype=dt)
        for i, ex in enumerate(self.experts):                      # copy, then alias (the old per-expert tensors are freed as they are dereferenced)
            W1[i].copy_(ex[0].weight); b1[i].copy_(ex[0].bias)
            ex[0].weight = nn.Parameter(W1[i], requires_grad=False); ex[0].bias = nn.Parameter(b1[i], requires_grad=False)
        W2 = torch.empty(E, D, Fd, device=dev, dtype=dt); b2 = torch.empty(E, D, device=dev, dtype=dt)
        for i, ex in enumerate(self.experts):
            W2[i].copy_(ex[2].weight); b2[i].copy_(ex[2].bias)
            ex[2].weight = nn.Parameter(W2[i], requires_grad=False); ex[2].bias = nn.Parameter(b2[i], requires_grad=False)
        self._fused = (W1, b1, W2, b2)
        self.impl = impl

    def _forward_grouped(self, tokens, topk_weight, topk_idx):
        """FAST ③: rows (token, slot) sorted by expert id → one grouped GEMM per linear (int32 cumulative offsets, zero-size groups allowed) → bias gathered per row →
        GELU → second grouped GEMM → weighted fp32 index_add_ into y. Same (token, slot) order as dispatch_by_expert, no host sync. Returns y fp32 [T, D], usage [E]."""
        E, k = self.num_experts, self.top_k
        W1, b1, W2, b2 = self._fused
        assert W1.device == tokens.device and W1.dtype == self.experts[0][0].weight.dtype, "stacked expert weights are stale: call fuse_for_inference again after moving the module"
        flat = topk_idx.reshape(-1)                                                       # [T·k], position p = token·k + slot
        order = torch.argsort(flat, stable=True)
        counts_t = torch.zeros(E, dtype=torch.long, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))
        offs = torch.cumsum(counts_t, dim=0).to(torch.int32)                              # group ends, on the GPU
        eid = flat[order]                                                                 # expert id of every sorted row
        token_idx_all = torch.div(order, k, rounding_mode="floor")
        slot_all = order - token_idx_all * k
        xs = tokens[token_idx_all].to(W1.dtype)                                           # [T·k, D] gathered input rows, grouped by expert
        h = torch._grouped_mm(xs, W1.transpose(1, 2), offs=offs)                          # [T·k, F]  (mat_b given as a transposed view: no copy)
        h.add_(b1[eid])
        h = F.gelu(h, approximate="tanh")
        ys = torch._grouped_mm(h, W2.transpose(1, 2), offs=offs)                          # [T·k, D]
        ys.add_(b2[eid])
        weight = topk_weight[token_idx_all, slot_all].unsqueeze(-1).to(ys.dtype)          # [T·k, 1]; rounded to the expert dtype exactly as the loop does (review 11/07)
        # Deterministic accumulation in the loop's order: the loop adds a token's experts in ascending expert id (fp32 index_add_ per expert). An index_add_ with
        # duplicate token indices would use atomics in arbitrary order, so write each contribution to its own cell [token, rank] (rank = position of the expert
        # id among the token's k experts) and add the k cells sequentially.
        rank = topk_idx.argsort(dim=1).argsort(dim=1)                                      # [T, k] rank of each slot by expert id
        buf = torch.zeros(tokens.shape[0], k, tokens.shape[1], dtype=torch.float32, device=tokens.device)
        buf[token_idx_all, rank[token_idx_all, slot_all]] = ys.float() * weight            # unique (token, rank) cells
        y = buf[:, 0]
        for j in range(1, k):
            y = y + buf[:, j]
        return y, counts_t.float()

    def forward(self, u):
        batch, seq_len, dim = u.shape
        tokens = u.reshape(-1, dim)                                   # [T, dim], T = batch × seq_len
        scores, topk_weight, topk_idx = route_tokens(self.gate, tokens, self.top_k)
        if self._fused is not None and self.impl == "grouped":       # FAST ③
            y, usage = self._forward_grouped(tokens, topk_weight, topk_idx)
        else:                                                         # original per-expert loop, unchanged (the experts' parameters may be views of the stacks)
            y = torch.zeros(tokens.shape, dtype=torch.float32, device=u.device)   # accumulate the 3 experts' outputs in fp32, cast to bf16 once at the end (same number of roundings as the original dense FFN)
            # One (topk_idx == i).nonzero() per expert would cost one sync per expert per module; dispatch_by_expert sorts once and we cut by counts.
            # Each expert gets element-wise the same token_idx / slot (see that function's docstring); expert forward and index_add_ order are unchanged
            token_idx_all, slot_all, counts, usage = dispatch_by_expert(topk_idx, self.num_experts)
            start = 0
            for i, expert in enumerate(self.experts):
                n_i = counts[i]                                                                  # number of tokens that selected expert i (python int)
                token_idx, slot = token_idx_all[start:start + n_i], slot_all[start:start + n_i]  # indices of the tokens that selected expert i, and its slot in their top_k
                start += n_i
                expert_out = expert(tokens[token_idx])                                           # [n_i, dim]
                weight = topk_weight[token_idx, slot].unsqueeze(-1).to(expert_out.dtype)         # [n_i, 1]
                y.index_add_(0, token_idx, (expert_out.float() * weight))
        y = y.to(tokens.dtype)
        if self.training and self.aux_loss_alpha > 0:
            aux_loss = expert_balance_loss(scores, topk_idx, batch, seq_len, self.num_experts, self.top_k, self.aux_loss_alpha)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
            self.last_aux_loss = aux_loss.detach()
        else:
            self.last_aux_loss = None
        self.last_selected = usage / tokens.shape[0]
        self.last_topk_idx = topk_idx.detach()
        return y.reshape(batch, seq_len, dim)


def attention_with_scale(q, k, v, k_lens, softmax_scale):
    """Thin wrapper around attention() that passes softmax_scale explicitly. MoEKVSelfAttention pads the head dim by 8 to carry log ĝ, so the scale must still use the original head_dim:
    on CUDA with flash available, use attention_utils' flash_attention (supports softmax_scale and variable-length keys via k_lens);
    otherwise (CPU parity checks / VIDEOX_ATTENTION_TYPE=SDPA) call SDPA directly with scale (attention_utils' SDPA branch does not pass scale, so it cannot be used)."""
    from videox_fun.models import attention_utils as AU
    flash_ok = getattr(AU, "FLASH_ATTN_2_AVAILABLE", False) or getattr(AU, "FLASH_ATTN_3_AVAILABLE", False) or getattr(AU, "FLASH_ATTN_4_AVAILABLE", False)
    attention_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION")
    if q.device.type == "cuda" and attention_type == "FLASH_ATTENTION" and flash_ok:
        return _attn(q, k, v, k_lens=k_lens, softmax_scale=softmax_scale)      # via the backend switch (default fa2 = previous behaviour)
    if k_lens is not None:
        assert bool((k_lens == k.shape[1]).all()), "SDPA path does not support variable-length keys: all samples must have the same sequence length"
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


class MoEKVSelfAttention(WanSelfAttention):
    """Self-attention with per-expert k / v (replaces self.self_attn in WanAttentionBlock, same call signature). q, o, norm_q, norm_k are the original ones (shared).

    Each token t carries top_k expert (key, value) copies (Mixtral style, no shared pair):
        copy j   k_t^i = norm_k(W_k^i x_t), v_t^i = W_v^i x_t, i = the j-th selected expert        prior weight ĝ_{t,i}
    (shared_kv=True adds a shared pair as copy 0, k_t = norm_k(W_k x_t), v_t = W_v x_t, prior 1; off by default, in which case the base class's self.k / self.v are deleted, otherwise they would be 264M trainable parameters never used in forward)
    All go into the key/value sequence (length copies·L, interleaved per token: position copies·t + j); q has one copy, one attention call, softmax normalised over all copies together:
        y_s = Σ_t Σ_i ĝ_{t,i} e^{q_s·k_t^i/√d} v_t^i / Z_s
    At step 0, with every expert ≈ the base, a token's copies are identical, so numerator and denominator scale alike and cancel (Σĝ = 1): the output equals the original model exactly; the gate gets main-loss gradients directly through log ĝ.
    Injecting log ĝ: flash-attn has no additive bias, so the head dim is padded by PAD=8 at the end (flash needs multiples of 8):
        q padded with [√d, √d, 0, …],  k with [hi, lo, 0, …] (hi = bf16(log ĝ), lo = log ĝ − hi, ~2^-16 precision combined),  v with 0,
        softmax_scale stays 1/√d  ⇒  (q·k + √d·(hi + lo)) / √d = q·k/√d + log ĝ. The shared pair has log 1 = 0.
    RoPE is applied to q and to every key copy (all copies of a token use the same position). Variable-length keys use k_lens = (1+top_k)·seq_lens (valid keys are contiguous at the front).
    Every expert is called every step (even on an empty tensor), as ZeRO-3 requires. The balance loss is attached to the output via AddAuxiliaryLoss.
    """
    PAD = 8

    def __init__(self, dim, num_heads, window_size, qk_norm, eps, num_experts, top_k, aux_loss_alpha, shared_kv=False, expert_dtype=None):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        assert tuple(window_size) == (-1, -1), "per-expert k/v self-attention does not support windowed attention"
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_alpha = aux_loss_alpha
        self.shared_kv = shared_kv
        if not shared_kv:                       # the original k / v projections are unused (expert 0's weights are read from the base file at assembly); delete them to save parameters and optimizer state
            del self.k
            del self.v
        self.k_experts = nn.ModuleList([nn.Linear(dim, dim, dtype=expert_dtype) for _ in range(num_experts)])   # MOE-V3.2 ⑤ (56B copy; dtype None here)
        self.v_experts = nn.ModuleList([nn.Linear(dim, dim, dtype=expert_dtype) for _ in range(num_experts)])
        self.gate = Fp32Gate(dim, num_experts)
        self.last_aux_loss = None
        self.last_selected = None
        self.last_topk_idx = None
        self.impl = "loop"              # FAST: "loop" | "grouped" (see MoEFFN)
        self._fused = None              # FAST: (Wkv [E, 2·dim, dim], bkv [E, 2·dim]); rows [:dim] = k expert, [dim:] = v expert
        self._fused_kv = False          # FAST ②: in the loop path, run k and v of an expert as one GEMM on the stacked rows

    @torch.no_grad()
    def fuse_for_inference(self, impl="loop", fuse_kv=True):
        """FAST ①②: stack k_experts / v_experts weights into Wkv [E, 2·dim, dim] (+ bkv) on this module's device and alias the expert parameters to views."""
        assert impl in ("loop", "grouped"), impl
        if impl == "grouped":
            assert hasattr(torch, "_grouped_mm"), "impl='grouped' needs torch._grouped_mm (torch >= 2.8)"
        E, D = self.num_experts, self.dim
        w0 = self.k_experts[0].weight
        dev, dt = w0.device, w0.dtype
        Wkv = torch.empty(E, 2 * D, D, device=dev, dtype=dt); bkv = torch.empty(E, 2 * D, device=dev, dtype=dt)
        for i, (ke, ve) in enumerate(zip(self.k_experts, self.v_experts)):
            Wkv[i, :D].copy_(ke.weight); Wkv[i, D:].copy_(ve.weight); bkv[i, :D].copy_(ke.bias); bkv[i, D:].copy_(ve.bias)
            ke.weight = nn.Parameter(Wkv[i, :D], requires_grad=False); ke.bias = nn.Parameter(bkv[i, :D], requires_grad=False)
            ve.weight = nn.Parameter(Wkv[i, D:], requires_grad=False); ve.bias = nn.Parameter(bkv[i, D:], requires_grad=False)
        self._fused = (Wkv, bkv)
        self._fused_kv = bool(fuse_kv)
        self.impl = impl

    def _fill_kv_grouped(self, tokens, topk_weight, topk_idx, K, V, log_prior, shared):
        """FAST ③: one grouped GEMM (N = 2·dim) over all experts fills the expert k / v copies of every (token, slot) and the log prior; no host sync. Returns usage [E]."""
        E, k, D = self.num_experts, self.top_k, self.dim
        Wkv, bkv = self._fused
        assert Wkv.device == tokens.device and Wkv.dtype == self.k_experts[0].weight.dtype, "stacked k/v expert weights are stale: call fuse_for_inference again after moving the module"
        flat = topk_idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts_t = torch.zeros(E, dtype=torch.long, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))
        offs = torch.cumsum(counts_t, dim=0).to(torch.int32)
        eid = flat[order]
        token_idx_all = torch.div(order, k, rounding_mode="floor")
        slot_all = order - token_idx_all * k
        xs = tokens[token_idx_all].to(Wkv.dtype)
        kv = torch._grouped_mm(xs, Wkv.transpose(1, 2), offs=offs)                          # [T·k, 2D]
        kv.add_(bkv[eid])
        copies = K.shape[1]
        pos = token_idx_all * copies + (shared + slot_all)                                  # flat index into [T, copies, D]: same cells as K[token_idx, shared + slot]
        K.view(-1, D)[pos] = kv[:, :D].to(K.dtype)
        V.view(-1, D)[pos] = kv[:, D:].to(V.dtype)
        log_prior.view(-1)[pos] = torch.log(topk_weight[token_idx_all, slot_all].clamp_min(1e-6))
        return counts_t.float()

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        shared = 1 if self.shared_kv else 0
        copies = shared + self.top_k                                  # (k, v) copies per token: (1 shared +) top_k expert copies
        tokens = x.reshape(-1, self.dim)                               # [T, C]
        num_tokens = tokens.shape[0]

        # 1) gating
        scores, topk_weight, topk_idx = route_tokens(self.gate, tokens, self.top_k)

        # 2) per-token copies of k / v and the log prior: (copy 0 shared,) copy shared+j = the j-th selected expert
        K = tokens.new_zeros(num_tokens, copies, self.dim)
        V = tokens.new_zeros(num_tokens, copies, self.dim)
        log_prior = torch.zeros(num_tokens, copies, device=x.device, dtype=torch.float32)
        if self.shared_kv:
            K[:, 0] = self.k(tokens)
            V[:, 0] = self.v(tokens)
        # As in MoEFFN, one nonzero sync per expert → a single dispatch_by_expert; (token_idx, slot) pairs never overlap across experts, so index_put_ is order-independent
        if self._fused is not None and self.impl == "grouped":                           # FAST ③
            usage = self._fill_kv_grouped(tokens, topk_weight, topk_idx, K, V, log_prior, shared)
        else:                                                                             # original loop (FAST ②: optionally one GEMM for k and v)
            token_idx_all, slot_all, counts, usage = dispatch_by_expert(topk_idx, self.num_experts)
            start = 0
            for i, (k_expert, v_expert) in enumerate(zip(self.k_experts, self.v_experts)):
                n_i = counts[i]
                token_idx, slot = token_idx_all[start:start + n_i], slot_all[start:start + n_i]
                start += n_i
                sub = tokens[token_idx]                                                       # gather the tokens once, shared by the k / v experts (same values as gathering twice)
                if self._fused is not None and self._fused_kv:                                # FAST ②
                    kv = F.linear(sub, self._fused[0][i], self._fused[1][i])                  # [n_i, 2·dim]: rows [:dim] = k_expert(sub), [dim:] = v_expert(sub)
                    K[token_idx, shared + slot] = kv[:, :self.dim].to(K.dtype)
                    V[token_idx, shared + slot] = kv[:, self.dim:].to(V.dtype)
                else:
                    K[token_idx, shared + slot] = k_expert(sub).to(K.dtype)                   # index_put_ needs matching dtypes (Linear outputs bf16 under autocast)
                    V[token_idx, shared + slot] = v_expert(sub).to(V.dtype)
                log_prior[token_idx, shared + slot] = torch.log(topk_weight[token_idx, slot].clamp_min(1e-6))

        # 3) as in the original: k goes through norm_k (over C, per copy), then is split into heads; q unchanged
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        K = self.norm_k(K).view(b, s, copies, n, d)
        V = V.view(b, s, copies, n, d)

        # 4) RoPE: fold copies into the head dim; all copies of a token use the same position
        q = rope_apply(q, grid_sizes, freqs)
        K = rope_apply(K.reshape(b, s, copies * n, d), grid_sizes, freqs).view(b, s, copies, n, d)

        # 5) interleave per token into a (copies·L)-long key/value sequence; valid keys are each sample's first copies·seq_len, contiguous
        K = K.reshape(b, s * copies, n, d)
        V = V.reshape(b, s * copies, n, d)
        prior = log_prior.view(b, s * copies)

        # 6) pad the head dim by PAD to carry log ĝ (see class docstring); softmax_scale stays 1/√d
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
        # Pass k_lens only for variable-length keys (flash then slices by length, two extra copies); in training latents share a shape and seq_lens all equal s, so pass None and use a view
        # The equal-length check now runs once at the top of the model forward (equal → seq_lens arrives as None and short-circuits here);
        # the second clause is only a fallback for probe / inference code calling this module with a tensor directly; training never reaches it
        k_lens = None if seq_lens is None or bool((seq_lens == s).all()) else seq_lens * copies
        out = attention_with_scale(q_, K_, V_, k_lens=k_lens, softmax_scale=1.0 / math.sqrt(d))
        out = out[..., :d]                                             # drop the padded dims

        # 7) output projection; attach the balance loss to the output; stats
        x = self.o(out.flatten(2))
        if self.training and self.aux_loss_alpha > 0:
            aux_loss = expert_balance_loss(scores, topk_idx, b, s, self.num_experts, self.top_k, self.aux_loss_alpha)
            x = AddAuxiliaryLoss.apply(x, aux_loss)
            self.last_aux_loss = aux_loss.detach()
        else:
            self.last_aux_loss = None
        self.last_selected = usage / num_tokens
        self.last_topk_idx = topk_idx.detach()
        return x


class WanAttentionBlock(nn.Module):
    """Wan transformer block with self-attention, cross-attention, and FFN."""

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 moe=None):          # MOE-V3.1 ②: None = original; dict(num_experts, top_k, aux_loss_alpha) = both self-attention k/v and FFN become MoE
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Layers
        self.norm1 = WanLayerNorm(dim, eps)
        if moe is None:
            self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        else:                                       # MOE-V3.1 ②: this layer's self-attention k/v are per-expert, same call signature
            self.self_attn = MoEKVSelfAttention(dim, num_heads, window_size, qk_norm, eps, **moe)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        if moe is None:
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, dim))
        else:                                       # MOE-V3.1 ②: this layer's FFN becomes MoE, same call signature (self.ffn(temp_x))
            self.ffn = MoEFFN(dim, ffn_dim, **moe)

        # Modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.bfloat16,
        t=0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C] or [B, L, 6, C] for modulation
            seq_lens(Tensor): Shape [B], length of each sequence in batch; None when the top-level forward finds all lengths equal
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Shape [B, L_context, C]
            context_lens(Tensor): Shape [B] (CPU int32); None when the top-level forward finds all lengths equal
        """
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:        
            e = (self.modulation + e).chunk(6, dim=1)

        # Self-attention with modulation
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype, t=t)
        x = x + y * e[2]

        # Cross-attention and FFN with modulation
        def cross_attn_ffn(x, context, context_lens, e):
            # Cross-attention: attend to text context
            x = x + self.cross_attn(self.norm3(x).to(x.dtype), context, context_lens, dtype, t=t)

            # FFN with modulation
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)
            
            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    """Output head for reconstructing video from transformer features."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # Layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # Modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        if e.dim() > 2:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        
        x = self.head(
            (self.norm(x) * (1 + e[1]) + e[0]).to(x.dtype)
        )
        return x


class MLPProj(torch.nn.Module):
    """MLP projection network for embedding transformation."""

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens



class KiwiMoeTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    # ignore_for_config = [
    #     'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    # ]
    # _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type=None,
        add_ref_condition=False,   # KIWI: build the reference-image Conv3d (instruction-only weights have no ref -> off by default)
        # ---- MOE-V3.1 ③: which blocks use MoE (self-attention k/v + FFN), and the MoE config (passed by the train/infer scripts via transformer_additional_kwargs) ----
        moe_layers=(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27),   # 0-based block indices = layers 2,4,...,28 counted from 1
        moe_num_experts=7,         # number of experts: 0 original / 1..5 category experts / 6 goku direct finetune
        moe_top_k=3,               # experts activated per token (3 for both self-attention k/v and FFN)
        moe_aux_loss_alpha=0.001,  # coefficient of the load-balancing aux loss (one per MoE site), 0 = off
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            in_channels (`int`, *optional*, defaults to 16):
                Alias for in_dim (diffusers compatibility)
            hidden_size (`int`, *optional*, defaults to 2048):
                Alias for dim (diffusers compatibility)
            add_control_adapter (`bool`, *optional*, defaults to False):
                Enable camera control adapter
            in_dim_control_adapter (`int`, *optional*, defaults to 24):
                Input channels for control adapter
            downscale_factor_control_adapter (`int`, *optional*, defaults to 8):
                Downscale factor for control adapter
            add_ref_conv (`bool`, *optional*, defaults to False):
                Enable reference frame convolution
            in_dim_ref_conv (`int`, *optional*, defaults to 16):
                Input channels for reference convolution
            cross_attn_type (`str`, *optional*, defaults to None):
                Cross-attention type, auto-determined from model_type if None
        """

        super().__init__()

        # assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Blocks
        # KIWI: Wan2.2-5B always uses the generic cross_attn (as in the original Wan2_2Transformer3DModel).
        # explicit-moe adds a second grouped-query cross-attention via WanAttentionBlock / WanCrossAttention.
        cross_attn_type = "cross_attn"
        # ---- MOE-V3.1 ③: blocks in moe_layers use MoE (self-attention k/v + FFN), the rest are original ----
        self.moe_layers = sorted(int(i) for i in (moe_layers or ()))
        assert all(0 <= i < num_layers for i in self.moe_layers), f"moe_layers out of range: {self.moe_layers}"
        moe_config = dict(num_experts=moe_num_experts, top_k=moe_top_k, aux_loss_alpha=moe_aux_loss_alpha)
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              moe=(moe_config if i in self.moe_layers else None))
            for i in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # Head
        self.head = Head(dim, out_dim, patch_size, eps)

        # Buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d
        self.dim = dim
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1
        )

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:], downscale_factor=downscale_factor_control_adapter)
        else:
            self.control_adapter = None

        if add_ref_conv:
            self.ref_conv = nn.Conv2d(in_dim_ref_conv, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.ref_conv = None

        self.teacache = None
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None
        self.gradient_checkpointing = False
        self.all_gather = None
        self.sp_world_size = 1
        self.sp_world_rank = 0
        self.init_weights()

        # ============ KIWI addition: source/reference condition Conv3d (+ drop the i2v img_emb) ============
        # The base __init__ above builds blocks/head from the INLINED WanAttentionBlock/Head (same names,
        # so from_pretrained loads as usual). Add two zero-init Conv3d that project source-video / ref-image
        # latents into conditions. Zero init => no-op at step 0 (output bit-identical to Wan2.2), learned later.
        if hasattr(self, "img_emb"):
            del self.img_emb   # Wan2.2-5B does not use the i2v image CLIP projection
        # Source-video condition (instruction-only weights contain vae_condition; it gets loaded)
        self.vae_condition = ConditionalEmbedder(in_dim, dim, patch_size, zero_init=True)
        # Reference-image condition: instruction-only weights have NO ref keys (none of the 825+8 keys),
        # so the module is not built by default (=None); forward / set_ref_pad_first skip the ref branch.
        # Interface kept: pass add_ref_condition=True to enable ref training.
        self.ref_vae_condition = (
            ConditionalEmbedder(in_dim, dim, patch_size, zero_init=True)
            if add_ref_condition else None
        )

    def moe_ffns(self):
        """MOE-V3.1: MoEFFN modules of all MoE layers (in block order); the training script reads last_aux_loss / last_selected from them."""
        return [self.blocks[i].ffn for i in self.moe_layers]

    def moe_attns(self):
        """MOE-V3.1: MoEKVSelfAttention modules of all MoE layers (in block order)."""
        return [self.blocks[i].self_attn for i in self.moe_layers]

    @torch.no_grad()
    def fuse_experts_for_inference(self, impl="loop", fuse_kv=True):
        """FAST ④: inference only. Stack every MoE module's expert weights (FAST ①), optionally fuse k/v (FAST ②) and select the expert implementation
        ("loop" = original per-expert loop, "grouped" = torch._grouped_mm, FAST ③). Call once, after the model is on its final device(s) (place_dit);
        calling .to()/.cuda() afterwards would turn the parameter views back into copies (correct but doubles expert memory)."""
        assert not self.training, "fuse_experts_for_inference is for inference (eval mode) only"
        t0 = time.time()
        for m in self.moe_ffns():
            m.fuse_for_inference(impl)
        for m in self.moe_attns():
            m.fuse_for_inference(impl, fuse_kv)
        return dict(impl=impl, fuse_kv=bool(fuse_kv), ffn_modules=len(self.moe_ffns()), attn_modules=len(self.moe_attns()), seconds=round(time.time() - t0, 1))

    def _set_gradient_checkpointing(self, *args, **kwargs):
        if "value" in kwargs:
            self.gradient_checkpointing = kwargs["value"]
            if hasattr(self, "motioner") and hasattr(self.motioner, "gradient_checkpointing"):
                self.motioner.gradient_checkpointing = kwargs["value"]
        elif "enable" in kwargs:
            self.gradient_checkpointing = kwargs["enable"]
            if hasattr(self, "motioner") and hasattr(self.motioner, "gradient_checkpointing"):
                self.motioner.gradient_checkpointing = kwargs["enable"]
        else:
            raise ValueError("Invalid set gradient checkpointing")

    def enable_teacache(
        self,
        coefficients,
        num_steps: int,
        rel_l1_thresh: float,
        num_skip_start_steps: int = 0,
        offload: bool = True,
    ):
        self.teacache = TeaCache(
            coefficients, num_steps, rel_l1_thresh=rel_l1_thresh, num_skip_start_steps=num_skip_start_steps, offload=offload
        )

    def share_teacache(
        self,
        transformer = None,
    ):
        self.teacache = transformer.teacache

    def disable_teacache(self):
        self.teacache = None

    def enable_cfg_skip(self, cfg_skip_ratio, num_steps):
        if cfg_skip_ratio != 0:
            self.cfg_skip_ratio = cfg_skip_ratio
            self.current_steps = 0
            self.num_inference_steps = num_steps
        else:
            self.cfg_skip_ratio = None
            self.current_steps = 0
            self.num_inference_steps = None

    def share_cfg_skip(
        self,
        transformer = None,
    ):
        self.cfg_skip_ratio = transformer.cfg_skip_ratio
        self.current_steps = transformer.current_steps
        self.num_inference_steps = transformer.num_inference_steps

    def disable_cfg_skip(self):
        self.cfg_skip_ratio = None
        self.current_steps = 0
        self.num_inference_steps = None

    def enable_riflex(
        self,
        k = 6,
        L_test = 66,
        L_test_scale = 4.886,
    ):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                get_1d_rotary_pos_embed_riflex(1024, self.d - 4 * (self.d // 6), use_real=False, k=k, L_test=L_test, L_test_scale=L_test_scale),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def disable_riflex(self):
        device = self.freqs.device
        self.freqs = torch.cat(
            [
                rope_params(1024, self.d - 4 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6)),
                rope_params(1024, 2 * (self.d // 6))
            ],
            dim=1
        ).to(device)

    def enable_multi_gpus_inference(self,):
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_world_rank = get_sequence_parallel_rank()
        self.all_gather = get_sp_group().all_gather

        # For normal model.
        for block in self.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn)

        # For vace model.
        if hasattr(self, 'vace_blocks'):
            for block in self.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)

    def set_ref_pad_first(self, ref_pad_first: bool) -> None:
        """Prepend (True) vs append (False) the ref-image tokens. Set from the CLI."""
        if self.ref_vae_condition is not None:    # None for instruction-only (no ref module): skip
            self.ref_vae_condition.ref_pad_first = bool(ref_pad_first)

    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        mllm_context: List[torch.Tensor],
        seq_len: int,
        vae_source_input: Optional[torch.Tensor] = None,
        vae_ref_image: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ):
        r"""Forward with Kiwi-Edit conditioning.

        Args:
            x (List[Tensor]): noisy video latents, each [C_in, F, H, W].
            t (Tensor): timesteps, [B] or [B, seq_len].
            mllm_context (List[Tensor]): per-sample MLLM context, each [L_i, dim].
            seq_len (int): max video token length for padding.
            vae_source_input (Tensor, optional): source-video latent for cond.
            vae_ref_image (Tensor, optional): reference-image latent for cond.
            sigma (Tensor, optional): flow-match sigma scaling the source cond.
        """
        # ---- Device / dtype / freqs ----
        device = self.patch_embedding.weight.device
        dtype = x[0].dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        # ---- Patch embedding (as in original Wan2.2) ----
        # 3D patch convolution on each video latent -> feature map before tokenization
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]      # shape=List[[1, dim, F_p, H_p, W_p]]

        # ============ Addition #1 over Wan2.2: source-video (pre-edit) condition ============
        # The original adds control_adapter(y_camera) here; kiwi instead projects the source-video latent
        # with a zero-init Conv3d, scales it by the flow-match sigma and adds it to the noisy latent, so the
        # model explicitly sees the pre-edit content. Skipped entirely in stage1 (vae_source_input=None).
        if vae_source_input is not None:
            if sigma is None:
                raise ValueError("vae_source_input requires `sigma` (scalar tensor).")
            # Project the source-video VAE latent into condition features via zero-init Conv3d
            src_cond = self.vae_condition(vae_source_input)        # shape=[B, dim, F_p, H_p, W_p]
            # Flow-match sigma as the scale factor (scalar or one per sample).
            # * Required for real batching: each u is a SINGLE sample [1,dim,...] and must be scaled by
            # ITS OWN sigma scalar on src_cond[i]. Multiplying by the whole-batch sigma.view(-1,1,1,1,1)
            # = [B,1,1,1,1] would broadcast src_cond[i:i+1]=[1,dim,...] to [B,dim,...] and mix
            # samples across the batch (silent at B=1). Hence a per-sample scalar.
            sigma_b = sigma.to(src_cond.dtype).reshape(-1)         # shape=[1] or [B]
            x = [
                u + (src_cond[i:i + 1] if src_cond.size(0) > 1 else src_cond[:1])
                    * (sigma_b[i] if sigma_b.numel() > 1 else sigma_b)   # this sample's sigma scalar
                for i, u in enumerate(x)                           # each u shape=[1, dim, F_p, H_p, W_p]
            ]

        # ---- Grid sizes (as in original Wan2.2) ----
        # Per-sample patch grid (F_p, H_p, W_p), used by RoPE / unpatchify
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])  # shape=[B, 3]
        # Addition over Wan2.2: keep the base grid to restore unpatchify after stripping ref tokens
        base_grid_sizes = grid_sizes.clone()                          # shape=[B, 3]

        # ---- Flatten (as in original Wan2.2) ----
        # Flatten each feature map over its spatial dims into a token sequence
        x = [u.flatten(2).transpose(1, 2) for u in x]                 # shape=List[[1, L_i, dim]] (L_i=F_p*H_p*W_p)

        # ============ Addition #2 over Wan2.2: reference-image token concatenation ============
        # The original concatenates full_ref / subject_ref here; kiwi projects the ref-image latent into
        # tokens with ref_vae_condition (Conv3d) and prepends/appends them per ref_pad_first, extending
        # grid_sizes / seq_len / per-token timestep t accordingly. Only in ref mode, otherwise skipped.
        ref_token_count = 0
        # Concatenation side: True=prepend (like original full_ref), False=append (like original subject_ref)
        ref_pad_first = (
            getattr(self.ref_vae_condition, "ref_pad_first", False)
            if self.ref_vae_condition is not None else False
        )
        if vae_ref_image is not None and self.ref_vae_condition is not None:
            # Project the ref-image VAE latent into a feature map via zero-init Conv3d
            ref_emb = self.ref_vae_condition(vae_ref_image)   # shape=[B, dim, F_r, H_r, W_r]
            ref_F = ref_emb.shape[2]                          # F_r (number of ref frame patches)
            # Flatten into a token sequence
            ref_emb = ref_emb.flatten(2).transpose(1, 2)      # shape=[B, F_r*H_r*W_r, dim]
            ref_token_count = ref_emb.size(1)                 # ref token count = F_r*H_r*W_r
            # Fold the ref frames into the grid's temporal dim (consistent at unpatchify)
            grid_sizes = torch.stack(
                [torch.tensor([gs[0] + ref_F, gs[1], gs[2]]) for gs in grid_sizes]
            ).to(grid_sizes.device)                           # shape=[B, 3]
            seq_len += ref_token_count                        # extend the sequence length accordingly
            if ref_pad_first:
                # Prepend ref tokens to each sample's video sequence
                x = [torch.concat([_r.unsqueeze(0), u], dim=1) for _r, u in zip(ref_emb, x)]  # shape=List[[1, ref+L_i, dim]]
                # Pad the per-token timesteps at the FRONT to match (ref tokens reuse the last token's t)
                if t.dim() != 1 and t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    padding = t[:, -1].unsqueeze(1).repeat(1, pad_size)
                    t = torch.cat([padding, t], dim=1)            # shape=[B, seq_len]
            else:
                # Append ref tokens to each sample's video sequence
                x = [torch.concat([u, _r.unsqueeze(0)], dim=1) for u, _r in zip(x, ref_emb)]  # shape=List[[1, L_i+ref, dim]]
                # Pad the per-token timesteps at the END to match
                if t.dim() != 1 and t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    padding = t[:, -1].unsqueeze(1).repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)            # shape=[B, seq_len]

        # ---- Pad video tokens to seq_len ----
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len, (seq_lens.tolist(), seq_len)
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])
        # Speed-up: whether all sequences have equal length is decided only here (seq_lens is a CPU tensor, no GPU work); if equal, pass None to every block:
        #   non-MoE attention()/flash_attention then use a flatten view for k_lens=None instead of per-layer torch.cat([u[:v] ...]) slice copies of k and v
        #   plus building q_lens / cu_seqlens on the GPU; MoE layers also skip recomputing bool((seq_lens == s).all()) per layer.
        # In training the latents share a shape, so lengths are always equal; unequal lengths (only when inference batches mixed lengths) still pass the tensor and take the original variable-length key path, same math.
        if bool((seq_lens == seq_len).all()):
            seq_lens = None

        # ---- Time embeddings ----
        if t.dim() != 1:
            if t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                padding = t[:, -1].unsqueeze(1).repeat(1, pad_size)
                t = torch.cat([t, padding], dim=1)
            bt = t.size(0)
            ft = t.flatten()
            # MOE-V3.1 ④: the original feeds fp32 into the Linear (autocast converts to bf16); DeepSpeed ZeRO-3 replaces F.linear with its own autograd
            # Function where autocast has no effect (smoke test: "mat1 and mat2 must have the same dtype"), so cast to x's dtype before the Linear
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, ft).unflatten(0, (bt, seq_len)).to(dtype)
            ).to(dtype)
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        else:
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).to(dtype)        # MOE-V3.1 ④, as above
            ).to(dtype)
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # ============ Addition #3 over Wan2.2: MLLM context replaces the text embedding ============
        # The original calls self.text_embedding(...) here: projects 4096-d T5/UMT5 tokens to dim and
        # pads to a fixed text_len (512). The kiwi MLLM connector already outputs dim-d tokens, so we
        # BYPASS text_embedding, pad to the batch max length and compute context_lens so cross-attn
        # can use FlashAttention varlen to skip padding without wasted compute.
        # Input mllm_context: List[[L_i, dim]] (per-sample MLLM context, variable length L_i)
        # Record each sample's true length before padding, for cross-attn varlen
        # Speed-up: context_lens used to be a CUDA tensor, and u[:v] in attention() sliced with a CUDA scalar -> one D2H sync per layer per sample for each of k and v.
        # Lengths are python ints (u.size(0)) anyway, so decide on the host: all equal (same MLLM token count within the per-GPU batch, or a single sample) -> None,
        # and cross-attention uses a flatten view (cu_seqlens identical to equal-length varlen); unequal -> CPU int32 tensor, so the library's u[:v] reads host scalars and
        # cu_seqlens is computed on CPU then moved to GPU, with no D2H sync anywhere; the sliced key/value and cu_seqlens are identical, same math.
        ctx_lens = [u.size(0) for u in mllm_context]               # list of python ints, no GPU tensor
        # Max context length in this batch
        max_ctx = max(ctx_lens)                                    # scalar
        context_lens = None if all(l == max_ctx for l in ctx_lens) else torch.tensor(ctx_lens, dtype=torch.int32)   # shape=[B] (CPU) or None
        # Zero-pad samples shorter than max_ctx and stack into a dense tensor for cross-attn
        context = torch.stack([
            torch.cat([u, u.new_zeros(max_ctx - u.size(0), u.size(1))])
            if u.size(0) < max_ctx else u
            for u in mllm_context
        ])                                                         # shape=[B, max_ctx, dim]

        # ---- Sequence parallelism ----
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]

        # ---- Block loop (training; teacache is inference-only) ----
        use_gc = bool(torch.is_grad_enabled() and getattr(self, "gradient_checkpointing", False))
        if use_gc:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

        for block in self.blocks:
            if use_gc:
                x = torch_checkpoint(
                    create_custom_forward(block),
                    x, e0, seq_lens, grid_sizes, self.freqs,
                    context, context_lens, dtype, t,
                    use_reentrant=False,
                )
            else:
                kwargs: Dict[str, Any] = dict(
                    e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs,
                    context=context, context_lens=context_lens, dtype=dtype, t=t,
                )
                x = block(x, **kwargs)

        # ---- Head ----
        if use_gc:
            x = torch_checkpoint(create_custom_forward(self.head), x, e, use_reentrant=False)
        else:
            x = self.head(x, e)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # ============ Addition #4 over Wan2.2: strip ref tokens, restore grid ============
        # Ref tokens are conditioning only, not part of the output video: cut them off on the side they
        # were concatenated before unpatchify, and restore grid_sizes so the output has the target shape.
        if ref_token_count > 0:
            if ref_pad_first:
                # Prepended -> drop ref tokens at the start of the sequence
                x = x[:, ref_token_count:]                        # shape=[B, L_video, dim]
            else:
                # Appended -> drop ref tokens at the end of the sequence
                x = x[:, :-ref_token_count]                       # shape=[B, L_video, dim]
            grid_sizes = base_grid_sizes                          # restore shape=[B, 3]

        # ---- Unpatchify ----
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        return x

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # Basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # Init output layer
        nn.init.zeros_(self.head.head.weight)

    @staticmethod
    def _convert_from_wan_model_config(config: dict) -> dict:
        """
        Convert WanModel config to WanTransformer3DModel config.
        This enables loading WanModel checkpoints with WanTransformer3DModel.
        """
        new_config = config.copy()
        
        # Convert has_image_input -> model_type
        if 'has_image_input' in new_config:
            new_config['model_type'] = 'i2v' if new_config.pop('has_image_input') else 't2v'
        
        # Convert has_ref_conv -> add_ref_conv
        if 'has_ref_conv' in new_config:
            new_config['add_ref_conv'] = new_config.pop('has_ref_conv')
        
        # Set WanTransformer3DModel specific defaults
        new_config.setdefault('cross_attn_type', 'cross_attn')
        new_config.setdefault('qk_norm', True)
        new_config.setdefault('cross_attn_norm', True)
        new_config.setdefault('text_len', 512)
        new_config.setdefault('window_size', (-1, -1))
        
        # Compatibility fields
        if 'in_dim' in new_config:
            new_config['in_channels'] = new_config['in_dim']
        if 'dim' in new_config:
            new_config['hidden_size'] = new_config['dim']
        
        # Remove WanModel-specific keys that WanTransformer3DModel doesn't use
        keys_to_remove = [
            'has_image_pos_emb', 'require_clip_embedding', 'require_vae_embedding',
            'seperated_timestep', 'fuse_vae_embedding_in_latents', '_class_name',
            '_diffusers_version', '_name_or_path'
        ]
        for key in keys_to_remove:
            new_config.pop(key, None)
        
        return new_config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        # Auto-convert WanModel config to WanTransformer3DModel config
        if 'has_image_input' in config:
            print("Detected diffsynth config, converting to WanTransformer3DModel config...")
            config = cls._convert_from_wan_model_config(config)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                if diffusers_version >= "0.33.0":
                    from diffusers.models.model_loading_utils import \
                        load_model_dict_into_meta
                else:
                    from diffusers.models.modeling_utils import \
                        load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
                    tmp_state_dict = torch.zeros(model.state_dict()['patch_embedding.weight'].size(), dtype=torch_dtype, device=param_device)
                    tmp_state_dict[:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight'][:, :model.state_dict()['patch_embedding.weight'].size()[1], :, :]
                    state_dict['patch_embedding.weight'] = tmp_state_dict

                filtered_state_dict = {}
                for key in state_dict:
                    if key in model.state_dict() and model.state_dict()[key].size() == state_dict[key].size():
                        filtered_state_dict[key] = state_dict[key]
                    else:
                        print(f"Skipping key '{key}' due to size mismatch or absence in model.")
                        
                model_keys = set(model.state_dict().keys())
                loaded_keys = set(filtered_state_dict.keys())
                missing_keys = model_keys - loaded_keys

                def initialize_missing_parameters(missing_keys, model_state_dict, torch_dtype=None):
                    initialized_dict = {}
                    
                    with torch.no_grad():
                        for key in missing_keys:
                            param_shape = model_state_dict[key].shape
                            param_dtype = torch_dtype if torch_dtype is not None else model_state_dict[key].dtype
                            if 'weight' in key:
                                if any(norm_type in key for norm_type in ['norm', 'ln_', 'layer_norm', 'group_norm', 'batch_norm']):
                                    initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                                elif 'embedding' in key or 'embed' in key:
                                    initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                                elif 'head' in key or 'output' in key or 'proj_out' in key:
                                    initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                                elif len(param_shape) >= 2:
                                    initialized_dict[key] = torch.empty(param_shape, dtype=param_dtype)
                                    nn.init.xavier_uniform_(initialized_dict[key])
                                else:
                                    initialized_dict[key] = torch.randn(param_shape, dtype=param_dtype) * 0.02
                            elif 'bias' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            elif 'running_mean' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            elif 'running_var' in key:
                                initialized_dict[key] = torch.ones(param_shape, dtype=param_dtype)
                            elif 'num_batches_tracked' in key:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=torch.long)
                            else:
                                initialized_dict[key] = torch.zeros(param_shape, dtype=param_dtype)
                            
                    return initialized_dict

                if missing_keys:
                    print(f"Missing keys will be initialized: {sorted(missing_keys)}")
                    initialized_params = initialize_missing_parameters(
                        missing_keys, 
                        model.state_dict(), 
                        torch_dtype
                    )
                    filtered_state_dict.update(initialized_params)

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(filtered_state_dict)
                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        filtered_state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )
        
        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
            model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight'][:, :model.state_dict()['patch_embedding.weight'].size()[1], :, :]
            model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
            state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']
        
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        model = model.to(torch_dtype)
        return model


