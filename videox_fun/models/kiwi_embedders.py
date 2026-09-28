"""Zero-init Conv3d patch-embedder used as the source/ref VAE conditioning module.

Ported verbatim from diffsynth/models/wan_video_dit.py (`ConditionalEmbedder`, lines
~775-786). Two instances are used in Kiwi-Edit:

  * `vae_condition`: takes the source (pre-edit) video latent, projects it via
    a zero-init Conv3d with the same patch_size as the DiT, and the result is
    added to the patchified noisy latents (scaled by sigma).
  * `ref_vae_condition`: takes the reference image latent, also Conv3d patch-
    embeds it, and the resulting tokens are CONCATENATED to the DiT input
    sequence (with extended RoPE freqs).
"""
from __future__ import annotations

import torch.nn as nn


class ConditionalEmbedder(nn.Module):
    def __init__(self, in_dim: int, dim: int, patch_size, zero_init: bool = False):
        super().__init__()
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size,
        )
        if zero_init:
            nn.init.zeros_(self.patch_embedding.weight)
            nn.init.zeros_(self.patch_embedding.bias)
        # Whether the ref-image tokens are prepended (True) or appended (False)
        # to the DiT input sequence. Diffsynth defaults to False ("pad last").
        # We set it on the module so the forward override can read it.
        self.ref_pad_first = False

    def forward(self, x):
        return self.patch_embedding(x)
