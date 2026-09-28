"""kiwi_dataset_moe.py — Kiwi-Edit dataset/mix dataloader, **dedicated copy for the MoE expert models**.

(Full copy of videox_fun/data/kiwi_dataset.py, following the same "self-contained copy" convention as mllm_encoder_moe.
It adds only the MoE deltas below, all executed inside the worker processes:
  1. Periodic malloc_trim at the top of __getitem__ — prevents glibc heap retention in long-running persistent workers
     (N is overridden by MOE_WORKER_TRIM_EVERY, 0 = off).
  2. The return_pixels branch additionally emits **src_pixels** uint8 — the original only pre-converts tgt (tgt is only
     consumed by the VAE, so its PIL can be dropped, zero RAM growth); src PIL is kept because the MLLM needs it, so an
     extra uint8 copy is made here for the VAE. Both src/tgt PIL->tensor conversions thus leave the main process and
     overlap the GPU step (a key speed-up).
     Cost: ~59MB extra worker-queue RAM per sample (bs12 x 8 workers x prefetch2 ~= +11GB/rank).)
  3. (Speed-up) With constructor arg src_pil=False the return_pixels branch no longer returns the 49 src PIL frames (the main
     process would receive ~470 MB of pickled PIL per step, taking 3.5 s); only src_pixels uint8 is returned (via shared memory)
     and the main process rebuilds byte-identical frames for the MLLM with Image.fromarray (moe_v3_1_train.rebuild_src_frames).
     Default True = old behaviour.
  4. (Speed-up) _load_video_impl no longer calls count_frames() (which decodes the whole clip again just to count frames);
     it reads the first num_frames frames sequentially. Frames are byte-identical.

======= Original file description below =======
Kiwi-Edit dataset and mix dataloader.

Mirrors diffsynth's `UnifiedDataset` + `MixDataloader` (from
diffsynth/trainers/unified_dataset.py and diffsynth/trainers/utils.py).

CSV / JSON / JSONL format (see DATASET.md in the upstream repo):
    Stage 1 (image edit):     src_video, tgt_video, prompt
    Stage 2 (image + video):  src_video, tgt_video, prompt
    Stage 3 (ref-video):      src_video, tgt_video, ref_image, prompt

`src_video` and `tgt_video` may be either an image (jpg/png/...) or a video
(mp4/avi/...). For images, `tgt_video[0].size == src_video[0].size` is
enforced (matches diffsynth's `check_paired_size`).

`KiwiEditDataset.__getitem__` returns a dict:
    {
        "prompt":    str,
        "src_video": list[PIL.Image]  (1 frame for images, N for videos)
        "tgt_video": list[PIL.Image]  (matched length & size to src_video)
        "ref_image": list[PIL.Image]  (only when the row has ref_image)
    }

The training loop is responsible for VAE-encoding the tensors and feeding
the prompt + src_video/ref_image through the MLLM encoder.

`KiwiEditMixDataLoader` round-robins across (image, video, ref-video)
sub-loaders so each step trains the same set of stage-2/3 modules — same
behaviour as diffsynth's `MixDataloader`.
"""
from __future__ import annotations

import csv
import json
import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import imageio
import torch
import torchvision

# --------------------------------------------------------------------------- #
#  ★ MoE delta 1: periodic malloc_trim inside workers (prevents heap          #
#    retention in long-running persistent workers)                            #
# --------------------------------------------------------------------------- #
_worker_item_count = 0
_WORKER_TRIM_EVERY = int(os.environ.get("MOE_WORKER_TRIM_EVERY", "64") or "0")


def _worker_periodic_trim():
    global _worker_item_count
    _worker_item_count += 1
    if _WORKER_TRIM_EVERY <= 0 or _worker_item_count % _WORKER_TRIM_EVERY:
        return
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
from PIL import Image, ImageOps
from torch.utils.data import Dataset

IMAGE_EXTS = ("jpg", "jpeg", "png", "webp", "bmp")
VIDEO_EXTS = ("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm")


def _ext(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


def shrink_and_pad_centered(
    item: Image.Image,
    target_size: Tuple[int, int],
    scale: float = 0.75,
    color: str = "white",
) -> Image.Image:
    """Fit `item` inside `target_size`, shrink by `scale`, center on a canvas.

    Verbatim port of `shrink_and_pad_centered` from
    `Kiwi-Edit-main/diffsynth/pipelines/wan_video_mllm.py:278`.

    `target_size` is (width, height) — matching PIL ordering.
    """
    target_w, target_h = target_size
    fitted = ImageOps.contain(item, target_size)
    w, h = fitted.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = fitted.resize((new_w, new_h), Image.BICUBIC)
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas = Image.new(item.mode, target_size, color)
    canvas.paste(resized, (left, top))
    return canvas


def _crop_and_resize(image: Image.Image, target_h: int, target_w: int) -> Image.Image:
    width, height = image.size
    scale = max(target_w / width, target_h / height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    image = torchvision.transforms.functional.center_crop(image, (target_h, target_w))
    return image


def _resolve_hw(
    image: Image.Image,
    fixed_h: Optional[int],
    fixed_w: Optional[int],
    max_pixels: int,
    h_div: int,
    w_div: int,
) -> Tuple[int, int]:
    """Return (h, w) divisible by the patch-friendly factors."""
    if fixed_h is not None and fixed_w is not None:
        # Even when the launcher pins an explicit --height/--width, they
        # MUST be divisible by the division factor (= VAE spatial
        # compression × DiT patch_size, e.g. 16 × 2 = 32). Otherwise the
        # VAE latent gets an odd spatial dim, the DiT patchify//unpatchify
        # truncates a row/col, and `pred` ends up one latent-cell smaller
        # than `target` -> "size of tensor a (44) must match b (45)" at
        # the MSE. e.g. 720 -> 704 (704/16=44, even). 1280 stays 1280.
        h = max((fixed_h // h_div) * h_div, h_div)
        w = max((fixed_w // w_div) * w_div, w_div)
        return h, w
    width, height = image.size
    if width * height > max_pixels:
        scale = (width * height / max_pixels) ** 0.5
        height = int(height / scale)
        width = int(width / scale)
    height = (height // h_div) * h_div
    width = (width // w_div) * w_div
    height = max(height, h_div)
    width = max(width, w_div)
    return height, width


def _load_image(path: str) -> List[Image.Image]:
    return [Image.open(path).convert("RGB")]


# Video read timeout: a video read on network storage can hang forever -> the other ranks wait for the
# 10-minute NCCL timeout and all SIGABRT. imageio/ffmpeg hangs raise no exception, so the 30 retries do not help;
# func_timeout turns the hang into an exception -> retry with another sample. Seconds follow VIDEO_READER_TIMEOUT.
_VR_TIMEOUT = int(os.environ.get("VIDEO_READER_TIMEOUT", "120"))


def _vis_warn(msg: str) -> None:
    """The launcher sets PYTHONWARNINGS=ignore, which swallows warnings.warn -> print to stderr directly (with rank)."""
    import sys
    print(f"[dataset rank={os.environ.get('RANK', '?')}] {msg}", file=sys.stderr, flush=True)


def _load_video(path: str, num_frames: int, time_div: int, time_rem: int) -> List[Image.Image]:
    """Read up to `num_frames` from a video file (timeout after _VR_TIMEOUT s -> RuntimeError)."""
    from func_timeout import FunctionTimedOut, func_timeout
    try:
        return func_timeout(_VR_TIMEOUT, _load_video_impl,
                            args=(path, num_frames, time_div, time_rem))
    except FunctionTimedOut:
        _vis_warn(f"video read timeout (>{_VR_TIMEOUT}s), switching sample: {path}")
        raise RuntimeError(f"video read timeout(>{_VR_TIMEOUT}s): {path}")


def _load_video_impl(path: str, num_frames: int, time_div: int, time_rem: int) -> List[Image.Image]:
    """Read up to `num_frames` from a video file. Trim so frame count
    satisfies `(n % time_div == time_rem)` (Wan2.2 VAE wants 4n+1).
    """
    # Speed-up: the old code first called reader.count_frames() (imageio re-decodes the whole video just to count frames, one extra ffmpeg per video), then took the first n frames.
    # Now read the first num_frames frames sequentially, IndexError = end of video; frame content/order is byte-identical (both are get_data(0..n-1)),
    # then trim trailing frames until (n % time_div == time_rem) (same condition as the old while loop). Saves a full decode when the video is longer than num_frames.
    reader = imageio.get_reader(path)
    frames = []
    try:
        for i in range(num_frames):
            frames.append(Image.fromarray(reader.get_data(i)))
    except (IndexError, StopIteration):
        pass
    finally:
        reader.close()
    n = len(frames)
    while n > 1 and n % time_div != time_rem:
        n -= 1
    return frames[:n]


def _load_path(path: str, num_frames: int, time_div: int, time_rem: int) -> List[Image.Image]:
    ext = _ext(path)
    if ext in IMAGE_EXTS:
        return _load_image(path)
    if ext in VIDEO_EXTS:
        return _load_video(path, num_frames, time_div, time_rem)
    raise ValueError(f"Unsupported file extension {ext!r} for path {path}")


class KiwiEditDataset(Dataset):
    """Paired source/target dataset for Kiwi-Edit instruction-edit training."""

    def __init__(
        self,
        metadata_path: str,
        base_path: Optional[str] = None,
        num_frames: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: int = 1024 * 1024,
        height_division_factor: int = 32,
        width_division_factor: int = 32,
        time_division_factor: int = 4,
        time_division_remainder: int = 1,
        repeat: int = 1,
        require_ref_image: bool = False,
        ref_aug_training: bool = True,
        ref_aug_scale_min: float = 0.7,
        ref_aug_scale_max: float = 1.0,
        ref_aug_color: str = "white",
        return_pixels: bool = False,
        pad_frames_to_num: bool = False,
        src_pil: bool = True,                     # speed-up: False = return_pixels branch does not return src PIL (see header item 3)
    ):
        self.metadata_path = metadata_path
        self.base_path = base_path or ""
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.repeat = repeat
        self.require_ref_image = require_ref_image
        # Reference-image augmentation: matches upstream Kiwi-Edit
        # `WanVideoUnit_InputVideoEmbedder.process` (wan_video_mllm.py:326),
        # which during training shrinks the ref image by a random factor in
        # [0.7, 1.0] and pastes it on a white canvas. At inference time the
        # ref is simply pad-fitted (ImageOps.pad) to target_size.
        self.ref_aug_training = ref_aug_training
        self.ref_aug_scale_min = ref_aug_scale_min
        self.ref_aug_scale_max = ref_aug_scale_max
        self.ref_aug_color = ref_aug_color
        # Buffer-pool mode: when True, __getitem__ ALSO returns compact uint8
        # pixel tensors (src/tgt/ref) built here in the worker process, so the
        # expensive PIL->tensor conversion overlaps the GPU step instead of
        # running on the main process between steps. The tgt PIL frames are
        # then dropped (only the VAE consumes them, now via tgt_pixels), while
        # src/ref PIL frames are kept (the Qwen MLLM needs PIL). See
        # videox_fun/data/kiwi_video_utils.py.
        self.return_pixels = return_pixels
        # Needed for true batching: all clips in a batch must have the same frame count (-> latent time dim F), otherwise
        # they cannot be stacked into [B,C,F,H,W] (the source-condition Conv3d also needs a regular batch). When enabled,
        # videos shorter than num_frames are padded by repeating the last frame (matches videoxfun train.py fixed-length training).
        # Default False — the per-sample loop handles variable lengths, so the original behaviour is unchanged.
        self.pad_frames_to_num = pad_frames_to_num
        self.src_pil = bool(src_pil)
        if not self.src_pil and not self.return_pixels:
            # No src_pixels for the main process to rebuild from -> keep returning PIL; do not raise or change other trainers
            _vis_warn("src_pil=False but return_pixels=False (no src_pixels to rebuild from), forcing src_pil=True")
            self.src_pil = True

        self.data: List[Dict] = self._load_metadata(metadata_path)
        if not self.data:
            raise ValueError(f"Metadata file {metadata_path} is empty.")

    @staticmethod
    def _load_metadata(metadata_path: str) -> List[Dict]:
        if metadata_path.endswith(".json"):
            with open(metadata_path) as f:
                return list(json.load(f))
        if metadata_path.endswith(".jsonl"):
            rows = []
            with open(metadata_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        # CSV (with header)
        with open(metadata_path, newline="") as f:
            return list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.data) * self.repeat

    def _resolve(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.join(self.base_path, p)

    def _check_paired_size(
        self, a: List[Image.Image], b: List[Image.Image]
    ) -> Optional[str]:
        # Match upstream Kiwi-Edit ordering: this runs AFTER
        # `main_data_operator` (resize+crop), so we only enforce equality on
        # frame count and on the post-process spatial size — which is always
        # equal because both src and tgt are resized/cropped to the same
        # (h, w). Mixed-resolution raw pairs are accepted as long as they
        # normalize to the same target size, matching upstream
        # `Kiwi-Edit-main/diffsynth/trainers/unified_dataset.py:328-338`.
        if len(a) != len(b):
            return f"frame-count mismatch: src={len(a)} tgt={len(b)}"
        if a[0].size != b[0].size:
            return f"size mismatch after process: src={a[0].size} tgt={b[0].size}"
        return None

    def _process_pair(
        self,
        src: List[Image.Image],
        tgt: List[Image.Image],
        ref: Optional[Image.Image],
    ) -> Tuple[List[Image.Image], List[Image.Image], Optional[List[Image.Image]]]:
        # Determine output size from the source frame.
        h, w = _resolve_hw(
            src[0],
            self.height,
            self.width,
            self.max_pixels,
            self.height_division_factor,
            self.width_division_factor,
        )
        src_proc = [_crop_and_resize(im, h, w) for im in src]
        tgt_proc = [_crop_and_resize(im, h, w) for im in tgt]
        ref_proc = None
        if ref is not None:
            # Match upstream training-time augmentation (wan_video_mllm.py:326):
            # shrink the ref image by a random factor in [scale_min, scale_max]
            # and paste it on a `color` canvas of size (w, h). Inference uses
            # plain `ImageOps.pad`, which is the scale=1.0 limit of the same
            # operation; we do that when ref_aug_training is False.
            target_size = (w, h)  # PIL order
            if self.ref_aug_training:
                scale = random.uniform(self.ref_aug_scale_min, self.ref_aug_scale_max)
                ref_canvas = shrink_and_pad_centered(
                    ref, target_size, scale=scale, color=self.ref_aug_color,
                )
            else:
                ref_canvas = ImageOps.pad(
                    ref, target_size, color=self.ref_aug_color, centering=(0.5, 0.5),
                )
            ref_proc = [ref_canvas]
        return src_proc, tgt_proc, ref_proc

    def __getitem__(self, idx: int) -> Dict:
        _worker_periodic_trim()   # ★ MoE delta 1: prevent glibc heap retention (ms-level cost in worker)
        max_retry = 30
        for retry in range(max_retry):
            try:
                row = self.data[idx % len(self.data)]
                prompt = row.get("prompt") or row.get("text") or ""
                src_path = self._resolve(row["src_video"])
                tgt_path = self._resolve(row["tgt_video"])

                src_frames = _load_path(
                    src_path,
                    num_frames=self.num_frames,
                    time_div=self.time_division_factor,
                    time_rem=self.time_division_remainder,
                )
                tgt_frames = _load_path(
                    tgt_path,
                    num_frames=self.num_frames,
                    time_div=self.time_division_factor,
                    time_rem=self.time_division_remainder,
                )

                # Pre-process: only enforce frame-count equality. Spatial
                # mismatches are accepted because `_process_pair` normalizes
                # both to the same (h, w) — matches upstream Kiwi-Edit which
                # validates AFTER its `main_data_operator`.
                if len(src_frames) != len(tgt_frames):
                    raise ValueError(
                        f"frame-count mismatch: src={len(src_frames)} tgt={len(tgt_frames)}"
                    )

                # Fixed length for true batching: pad videos shorter than num_frames to exactly num_frames by REPEATING the last frame,
                # src/tgt padded identically (paired, so the "edit" on padded frames stays consistent). Videos only (num_frames>1).
                if self.pad_frames_to_num and self.num_frames > 1 and len(src_frames) < self.num_frames:
                    pad_n = self.num_frames - len(src_frames)
                    src_frames = src_frames + [src_frames[-1]] * pad_n
                    tgt_frames = tgt_frames + [tgt_frames[-1]] * pad_n

                ref_image = None
                if "ref_image" in row and row["ref_image"]:
                    ref_image = Image.open(self._resolve(row["ref_image"])).convert("RGB")
                elif self.require_ref_image:
                    raise ValueError("require_ref_image=True but row has no ref_image column")

                src_frames, tgt_frames, ref_frames = self._process_pair(
                    src_frames, tgt_frames, ref_image,
                )

                # Post-process check: src/tgt must agree on (h, w) and frame
                # count after `_process_pair`. Always passes when the upstream
                # pipeline ran cleanly; left in as a defensive guard.
                err = self._check_paired_size(src_frames, tgt_frames)
                if err is not None:
                    raise ValueError(err)

                if not self.return_pixels:
                    return {
                        "prompt": prompt,
                        "src_video_path": row["src_video"],   # original relative path (feature-cache key)
                        "src_video": src_frames,
                        "tgt_video": tgt_frames,
                        "ref_image": ref_frames,
                    }

                # Buffer-pool mode (ZERO net RAM increase). Pre-convert ONLY the
                # target frames to a compact uint8 tensor here in the worker, so
                # that conversion overlaps the GPU step, and DROP the tgt PIL
                # (only the VAE consumed tgt — never the MLLM — so nothing else
                # needs the PIL). src/ref stay PIL because the Qwen MLLM needs
                # PIL; their VAE pixels are converted on the main process as
                # before. Memory accounting per sample is unchanged: we swap the
                # tgt PIL frames (~uint8-sized) for a tgt uint8 tensor of the
                # same size. This keeps the OOM envelope identical to the old
                # pipeline (important: bs*num_workers*prefetch_factor drives RAM;
                # see README memory section). See utils.frames_to_uint8_tensor.
                from videox_fun.data.kiwi_video_utils import frames_to_uint8_tensor
                return {
                    "prompt": prompt,
                    "src_video_path": row["src_video"],   # original relative path (feature-cache key)
                    "src_video": src_frames if self.src_pil else None,   # speed-up: src_pil=False skips PIL (main process rebuilds from src_pixels)
                    "tgt_video": None,
                    "ref_image": ref_frames,
                    "tgt_pixels": frames_to_uint8_tensor(tgt_frames),   # [C,F,H,W] u8
                    # ★ MoE delta 2: src VAE pixels are also pre-converted in the worker (PIL still kept for the MLLM)
                    "src_pixels": frames_to_uint8_tensor(src_frames),   # [C,F,H,W] u8
                }
            except Exception as e:
                warnings.warn(
                    f"KiwiEditDataset: error {retry}/{max_retry} loading idx {idx}: {e}",
                )
                _vis_warn(f"sample load failed {retry}/{max_retry} idx={idx}: {str(e)[:160]}")
                idx = random.randint(0, len(self.data) - 1)
        raise RuntimeError(
            f"KiwiEditDataset: failed to load any data after {max_retry} retries."
        )


def _identity_collate(batch):
    """Pass-through collate.

    When the dataloader has `batch_size=1`, returns the single sample
    directly (matches diffsynth's `collate_fn=lambda x: x[0]`). When
    `batch_size>1`, returns the full list of dict samples — the trainer
    iterates them sequentially inside `compute_step_loss` (the MLLM
    encoder is single-chat-message-per-call by design).
    """
    if len(batch) == 1:
        return batch[0]
    return batch


class KiwiEditMixDataLoader:
    """Round-robins across image / video / ref-video dataloaders.

    Mirrors `diffsynth.trainers.utils.MixDataloader`. Pass `None` for any
    sub-loader to disable that stage. Iteration re-cycles each sub-loader
    when it's exhausted (so the longest one drives the wall-clock).
    """

    def __init__(
        self,
        image_dataloader: Optional[torch.utils.data.DataLoader] = None,
        video_dataloader: Optional[torch.utils.data.DataLoader] = None,
        video_ref_dataloader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 1,
        image_ratio: Optional[float] = None,
        ratio_seed: int = 0,
        accum_steps: int = 1,
        strict_mix: bool = False,
    ):
        self.image_dataloader = image_dataloader
        self.video_dataloader = video_dataloader
        self.video_ref_dataloader = video_ref_dataloader
        self.num_epochs = num_epochs
        # accum_steps: each gradient-accumulation window uses a SINGLE data type (see __iter__)
        self.accum_steps = max(1, int(accum_steps))
        # image_ratio: when both image and video exist, each step draws an image batch with this probability (else video).
        #   None -> fall back to round-robin (backward compatible). A fixed ratio_seed makes ALL ranks/nodes draw the
        #   same type (img/video) each step; otherwise GPUs get batches of different shapes -> NCCL shape-mismatch hang.
        self.image_ratio = image_ratio
        self.ratio_seed = ratio_seed
        # strict_mix: True -> no random draw; instead a DETERMINISTIC strict interleave (image windows spread evenly
        #   by image_ratio, Bresenham/largest remainder). ratio=0.5 -> video,image,video,image,... (video window first).
        #   All ranks use the same formula -> consistent by construction, no ratio_seed needed.
        self.strict_mix = bool(strict_mix)

        active = []
        lengths = []
        if image_dataloader is not None:
            active.append("image")
            lengths.append(len(image_dataloader))
        if video_dataloader is not None:
            active.append("video")
            lengths.append(len(video_dataloader))
        if video_ref_dataloader is not None:
            active.append("ref")
            lengths.append(len(video_ref_dataloader))
        if not active:
            raise ValueError("At least one sub-dataloader must be provided.")
        self.active = active
        self.length = min(lengths) * len(active)

    def _iter_loader(self, kind: str):
        if kind == "image":
            loader = self.image_dataloader
        elif kind == "video":
            loader = self.video_dataloader
        else:
            loader = self.video_ref_dataloader
        while True:
            for batch in loader:
                yield batch

    def __iter__(self):
        gens = {kind: self._iter_loader(kind) for kind in self.active}
        total = self.length * self.num_epochs
        # ratio mode: only when both image and video exist and image_ratio is set; fixed seed -> consistent across ranks
        use_ratio = (self.image_ratio is not None
                     and "image" in self.active and "video" in self.active)
        rng = random.Random(self.ratio_seed) if use_ratio else None
        accum = self.accum_steps
        # ★ Each gradient-accumulation window (accum micro-steps) uses a SINGLE data type.
        #   Reason: with DDP + find_unused_parameters, gradients are all-reduced only on the LAST micro-step of the window;
        #   if a window mixed img/video, parameters used only in earlier micro-steps (e.g. image_queries vs
        #   video_queries) would not be synchronised -> gradients differ across ranks / training diverges. Hence switch per window.
        kind = None
        for step in range(total):
            if step % accum == 0:
                w = step // accum                          # index of the accum window
                if use_ratio and self.strict_mix:
                    # Strict interleave: #image windows ~= round(total windows x ratio), spread evenly (Bresenham/largest remainder).
                    # ratio=0.5 -> w=0->video, w=1->image, w=2->video, ... (one window = batch*accum samples).
                    kind = "image" if int((w + 1) * self.image_ratio) > int(w * self.image_ratio) else "video"
                elif use_ratio:
                    kind = "image" if rng.random() < self.image_ratio else "video"
                else:
                    kind = self.active[w % len(self.active)]
            yield next(gens[kind])

    def __len__(self) -> int:
        return self.length * self.num_epochs


def make_default_dataloader(
    dataset: KiwiEditDataset,
    num_workers: int = 0,
    shuffle: bool = True,
    batch_size: int = 1,
    prefetch_factor: int = 2,
    sampler: Optional[torch.utils.data.Sampler] = None,
) -> torch.utils.data.DataLoader:
    """Default loader.

    Default `batch_size=1` returns one dict per step (back-compat:
    matches diffsynth's collate behavior). `batch_size>1` returns a
    list of dicts; the trainer's `compute_step_loss` iterates them
    sequentially and averages the per-sample loss before backward,
    so per-step gradient is mathematically equivalent to
    `gradient_accumulation_steps=batch_size train_batch_size=1` but
    surfaced as a single CLI knob.

    With `num_workers > 0` the loader prefetches batches in background
    worker processes so video decode (~81 frames/clip) AND — when the
    dataset has `return_pixels=True` — the PIL->uint8 tensor conversion
    overlap the GPU step instead of stalling it between steps (the
    main-process synchronous decode/convert that leaves the GPU idle at 0%
    util / baseline memory). `persistent_workers=True` avoids re-spawning the
    pool every epoch; `prefetch_factor` keeps a few batches queued ahead.
    `pin_memory=True` pins the returned tensors (only meaningful with
    `return_pixels=True` — PIL images are not tensors and are left as-is),
    so the small uint8 H2D copy can be async/`non_blocking`. These three
    kwargs are only valid when num_workers>0, so they're set conditionally.

    Host-RAM cost is bounded by `num_workers * prefetch_factor * batch_size`
    samples held in the queue; with uint8 pixels each video sample is small
   . Lower
    `prefetch_factor` to shrink the queue if RAM is tight.
    """
    # A custom `sampler` (e.g. length-bucketing) is mutually exclusive with
    # `shuffle` — when one is given, shuffling is delegated to the sampler.
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_identity_collate,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["pin_memory"] = True
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)
