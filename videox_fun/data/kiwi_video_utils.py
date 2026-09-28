"""Helpers shared by training script and smoke tests."""
from __future__ import annotations

from typing import List

import torch
from PIL import Image


def frames_to_uint8_tensor(frames: List[Image.Image]) -> torch.Tensor:
    """Stack PIL frames into a COMPACT uint8 CPU tensor of shape [C=3, F, H, W].

    This is the buffer-pool representation produced inside the dataset
    workers (`KiwiEditDataset.__getitem__` when `return_pixels=True`). It is:

      - **compact**: uint8 is 4x smaller than the float32 the model needs, so
        the prefetch queue (num_workers x prefetch_factor x batch_size samples
        held in host RAM) stays bounded — see the memory estimate in
        `README_stage4_8gpu_muon_zh.md`.
      - **pin-able**: it is a real tensor, so `DataLoader(pin_memory=True)`
        actually pins it (PIL images are NOT tensors, so the old pipeline's
        `pin_memory=True` was a silent no-op).

    The expensive part (PIL -> ndarray copy) runs in the worker PROCESSES,
    overlapping the GPU step, instead of on the main process between steps
    (which idled the GPU). Normalization to [-1, 1] + dtype cast is deferred
    to `uint8_video_to_pixels`, done on the GPU after the (cheap, async) H2D
    copy of the small uint8 tensor.
    """
    if not frames:
        raise ValueError("frames_to_uint8_tensor: empty frame list")
    arrs = []
    for fr in frames:
        # [H, W, 3] uint8 -> [3, H, W] uint8
        arrs.append(torch.from_numpy(_pil_to_array(fr)).permute(2, 0, 1))
    return torch.stack(arrs, dim=1)  # [C, F, H, W] uint8


def uint8_video_to_pixels(
    u8: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = True,
) -> torch.Tensor:
    """uint8 [C,F,H,W] (or already-batched [1,C,F,H,W]) -> normalized pixels.

    Returns a [1, C, F, H, W] tensor in [-1, 1] on `device`/`dtype`, byte-for-
    byte equivalent to `frames_to_video_tensor` (same /255 then *2-1 mapping,
    same channel/frame layout). The uint8 -> float normalization is done on
    the GPU (cheap elementwise) so only the small uint8 buffer crosses PCIe.

    `non_blocking=True` lets the H2D copy overlap when `u8` is pinned (it is,
    when it came through `DataLoader(pin_memory=True)`).
    """
    x = u8.to(device=device, non_blocking=non_blocking)
    # Normalize in fp32 then cast, so the result is byte-identical to the old
    # `frames_to_video_tensor(dtype=float32).to(dtype)` path (which did the
    # /255*2-1 in fp32 and only then rounded to the model dtype).
    x = x.float().div_(255.0).mul_(2.0).sub_(1.0).to(dtype)
    if x.dim() == 4:
        x = x.unsqueeze(0)  # [C,F,H,W] -> [1,C,F,H,W]
    return x


def frames_to_video_tensor(frames: List[Image.Image], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Stack a list of PIL frames into a [1, C=3, F, H, W] tensor in [-1, 1].

    Output is what the Wan2.2 VAE expects (`vae.encode(x).sample()` etc.).

    This is the CPU/main-process fallback used when the dataset did NOT
    pre-build uint8 pixels (`return_pixels=False`, e.g. the smoke test). The
    real training path builds uint8 pixels in the workers and normalizes via
    `uint8_video_to_pixels`; this function and that pair produce identical
    tensors.
    """
    u8 = frames_to_uint8_tensor(frames)  # [C, F, H, W] uint8
    video = u8.to(dtype).div_(255.0).mul_(2.0).sub_(1.0)
    return video.unsqueeze(0)  # [1, C, F, H, W]


def _pil_to_array(img: Image.Image):
    import numpy as np

    # np.array (not np.asarray) returns a writable COPY rather than a
    # read-only view onto PIL's internal buffer. Without the copy,
    # torch.from_numpy warns "given NumPy array is not writable ...".
    # The downstream normalization already allocates a fresh tensor so the
    # warning was harmless, but the copy silences it cleanly.
    return np.array(img.convert("RGB"))
