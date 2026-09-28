# =============================================================================
# mlp-moe DiT: a DeepSeekMoE-style MoE on the kiwi-edit FFN
# This file = a full copy of videox_fun/models/wan_transformer3d_kiwi.py plus the additions below, all marked "MOE-V3":
#   MOE-V3 ①  MoEFFN: N experts + a gate, each token picks its own top_k experts (no shared expert, Mixtral style; the earlier residual form cancelled shared when Σĝ=1)
#              y = Σ_{i∈top_k} ĝ_i · expert_i(u),  ĝ = softmax weights renormalised within top_k (Σĝ = 1)
#              plus DeepSeek's expert-level load-balancing loss (keeps experts from dying), attached to the graph via AddAuxiliaryLoss
#   MOE-V3 ②  WanAttentionBlock gets a moe argument: if given, self.ffn becomes MoEFFN, otherwise identical to the original
#   MOE-V3 ③  KiwiMoeTransformer3DModel.__init__ gets moe_layers / moe_num_experts / moe_top_k / moe_aux_loss_alpha,
#   MOE-V3 ④  in forward, the time_embedding input goes from .float() to .to(dtype) (ZeRO-3's linear ignores autocast, see the comment there)
#              (③) blocks listed in moe_layers (0-based indices) use the MoE FFN; 1-based layers 2,4,...,28 = indices 1,3,...,27
# attention / cross-attention / modulation / forward are otherwise verbatim (relative imports made absolute so the file can live in this directory).
# Weight source (see moe_v3_model.py): routed experts = the 5 category experts' LoRA merged into the kiwi FFN (merge_experts.py),
# 2 copies each, the second with a 1% random perturbation; the gate is trained from scratch.
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
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
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
            seq_lens(Tensor): Shape [B]
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

        x = attention(
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
        x = attention(
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

        img_x = attention(
            q.to(dtype), 
            k_img.to(dtype), 
            v_img.to(dtype), 
            k_lens=None
        )
        # Compute text cross-attention
        x = attention(
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
        x = attention(
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
# MOE-V3 ①  MoE FFN
# =============================================================================
class AddAuxiliaryLoss(torch.autograd.Function):
    """Attach the load-balancing aux loss to the main output's autograd graph (as in the original DeepSeek code).
    Forward: return y unchanged. Backward: give aux_loss a gradient of 1.
    The training script then calls backward on the main loss only, and every MoE layer's aux-loss gradient reaches its gate
    without per-layer collection; gradient-checkpointing recompute rebuilds this node with identical results."""

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
    """Gate: Linear(dim → num_experts, no bias) that computes the logits in fp32 in its own forward.
    It must be a module rather than MoEFFN using the weight directly: ZeRO-3 gathers a module's parameters from their shards
    only when that module's forward is called, so reading self.gate.weight directly yields a shard placeholder ("'weight' must be 2-D" in a smoke test)."""

    def __init__(self, dim, num_experts):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, tokens):                      # [T, dim] -> [T, num_experts] fp32
        # The training forward runs under bf16 autocast; disable it here so the gate really computes in fp32
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            return F.linear(tokens.float(), self.weight.float())


def make_ffn(dim, ffn_dim):
    """Original FFN structure: Linear(dim→ffn_dim) → GELU(tanh) → Linear(ffn_dim→dim). Every expert uses it."""
    return nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))


class MoEFFN(nn.Module):
    """MoE FFN (replaces self.ffn in WanAttentionBlock, same input/output shapes), Mixtral style, no shared expert:

        y = Σ_{i ∈ top_k(u)} ĝ_i · expert_i(u),   ĝ_i = g_i / Σ_{top_k} g

    experts   N experts (full-width FFNs), each a complete weight set initialised from "kiwi FFN + category-expert LoRA" merged weights
    (the earlier residual form shared + Σĝ·(expert − shared) cancels shared algebraically when Σĝ = 1, so shared got no gradient; hence no shared expert)
    gate      Linear(dim → N) scores every token; softmax, then top_k, weights renormalised within top_k to sum to 1
              (in the former residual form: output = shared + weighted mean of selected-expert deltas ≈ category-expert output at step 0). Earlier runs without
              renormalisation (Σg ≈ k/N) fed expert deltas into the output at only ~10% strength; the gate's main-loss gradient was too small (no specialisation in 1950 steps)
    Each expert only sees the tokens that selected it; every expert is still called once per step (possibly on an empty tensor), as ZeRO-3 needs every parameter in every forward.

    Load-balancing aux loss (DeepSeek expert-level, per sample, keeps experts from dying):
        f_i = (times expert i is selected) / (top_k·L) · N      P_i = mean softmax score for expert i over the sample's tokens
        aux = alpha · mean_sample Σ_i f_i · P_i           (= alpha when perfectly balanced)
    Attached to y via AddAuxiliaryLoss; the training script need not handle it. last_aux_loss / last_usage / last_selected are for logging only.
    """

    def __init__(self, dim, ffn_dim, num_experts, top_k, aux_loss_alpha):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_alpha = aux_loss_alpha
        self.experts = nn.ModuleList([make_ffn(dim, ffn_dim) for _ in range(num_experts)])
        self.gate = Fp32Gate(dim, num_experts)          # parameter name is still gate.weight
        self.last_aux_loss = None       # aux loss of the latest forward (detached), for logging
        self.last_usage = None          # tokens routed to each expert in the latest forward [N], for logging
        self.last_selected = None       # per-expert "fraction of tokens selecting it" in the latest forward [N]: 1 if a token picks it else 0, averaged over tokens (Σ_i = top_k), for logging
        self.last_topk_idx = None       # expert indices picked by each token in the latest forward [T, k], used at inference to plot "which token uses which expert"

    def forward(self, u):
        batch, seq_len, dim = u.shape
        tokens = u.reshape(-1, dim)                                   # [T, dim], T = batch × seq_len

        # 1) Gate: fp32 scores → softmax → top_k per token
        scores = F.softmax(self.gate(tokens), dim=-1)                                   # [T, N] fp32
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1)                   # [T, k]
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)                # renormalise to sum to 1

        # 2) Experts: each expert only computes the tokens that picked it and adds ĝ_i · expert_i(u); accumulate in fp32, cast to bf16 once at the end
        y = torch.zeros(tokens.shape, dtype=torch.float32, device=u.device)
        usage = torch.zeros(self.num_experts, device=u.device)
        for i, expert in enumerate(self.experts):
            token_idx, slot = (topk_idx == i).nonzero(as_tuple=True)  # indices of tokens that picked expert i, and its slot in their top_k
            usage[i] = token_idx.numel()
            expert_out = expert(tokens[token_idx])                                       # [n_i, dim]
            weight = topk_weight[token_idx, slot].unsqueeze(-1).to(expert_out.dtype)     # [n_i, 1]
            y.index_add_(0, token_idx, expert_out.float() * weight)
        y = y.to(tokens.dtype)

        # 3) Load-balancing aux loss (training only), attached to y's graph
        if self.training and self.aux_loss_alpha > 0:
            counts = F.one_hot(topk_idx, self.num_experts).sum(dim=1)                      # [T, N] which experts each token picked
            counts = counts.view(batch, seq_len, self.num_experts).sum(dim=1).float()      # [B, N] per-sample selection count of each expert
            load = counts * self.num_experts / (self.top_k * seq_len)                     # f_i
            prob = scores.view(batch, seq_len, self.num_experts).mean(dim=1)              # P_i
            aux_loss = (load * prob).sum(dim=-1).mean() * self.aux_loss_alpha
            y = AddAuxiliaryLoss.apply(y, aux_loss)
            self.last_aux_loss = aux_loss.detach()
        else:
            self.last_aux_loss = None
        self.last_usage = usage
        self.last_selected = usage / tokens.shape[0]
        self.last_topk_idx = topk_idx.detach()
        return y.reshape(batch, seq_len, dim)


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
                 moe=None):          # MOE-V3 ②: None = original FFN; dict(num_experts, top_k, aux_loss_alpha) = MoE FFN
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
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
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
        else:                                       # MOE-V3 ②: this layer's FFN becomes MoE, called the same way (self.ffn(temp_x))
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
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Shape [B, L_context, C]
            context_lens(Tensor): Shape [B]
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
        # ---- MOE-V3 ③: which blocks use the MoE FFN, and the MoE config (passed by the train/infer scripts via transformer_additional_kwargs) ----
        moe_layers=(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27),   # 0-based block indices = 1-based layers 2,4,...,28
        moe_num_experts=10,        # number of routed experts (5 category experts × 2 copies)
        moe_top_k=3,               # routed experts activated per token (an earlier run used 2)
        moe_aux_loss_alpha=0.001,  # load-balancing aux loss coefficient, 0 = off (0.001 as in DeepSeekMoE 16B; 0.01 pinned the gate at the uniform point)
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
        # ---- MOE-V3 ③: blocks in moe_layers use the MoE FFN, the rest are original ----
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
        """MOE-V3: MoEFFN modules of all MoE layers (in block order); the training script reads last_aux_loss / last_usage from them."""
        return [self.blocks[i].ffn for i in self.moe_layers]

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

        # ---- Time embeddings ----
        if t.dim() != 1:
            if t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                padding = t[:, -1].unsqueeze(1).repeat(1, pad_size)
                t = torch.cat([t, padding], dim=1)
            bt = t.size(0)
            ft = t.flatten()
            # MOE-V3 ④: the original feeds fp32 to the Linear here (autocast casts to bf16); DeepSpeed ZeRO-3 swaps F.linear for its own autograd
            # Function where autocast has no effect (smoke test: "mat1 and mat2 must have the same dtype"), so cast to x's dtype before the Linear
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, ft).unflatten(0, (bt, seq_len)).to(dtype)
            ).to(dtype)
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        else:
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).to(dtype)        # MOE-V3 ④, as above
            ).to(dtype)
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # ============ Addition #3 over Wan2.2: MLLM context replaces the text embedding ============
        # The original calls self.text_embedding(...) here: projects 4096-d T5/UMT5 tokens to dim and
        # pads to a fixed text_len (512). The kiwi MLLM connector already outputs dim-d tokens, so we
        # BYPASS text_embedding, pad to the batch max length and compute context_lens so cross-attn
        # can use FlashAttention varlen to skip padding without wasted compute.
        # Input mllm_context: List[[L_i, dim]] (per-sample MLLM context, variable length L_i)
        # Record each sample's true length before padding, for cross-attn varlen
        context_lens = torch.tensor(
            [u.size(0) for u in mllm_context],
            dtype=torch.int32, device=mllm_context[0].device,
        )                                                          # shape=[B]
        # Max context length in this batch
        max_ctx = max(u.size(0) for u in mllm_context)             # scalar
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


