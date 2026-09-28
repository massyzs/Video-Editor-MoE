#!/usr/bin/env python
"""videoeditor-14b training script: full training of the whole DiT with DeepSpeed ZeRO-3 (self-attention k/v and FFN are both 7-expert top-3 MoE), 3 nodes / 24 GPUs.

The model lives elsewhere: DiT architecture in wan_transformer3d_kiwi_moe_v3_1.py, assembly / weight loading / MLLM / VAE in moe_v3_1_model.py.
This file only has: data (two stages) → batching → noising → forward → loss → update → checkpointing / resume; the loss follows videoxfun wan2.2 and is inlined in the loop.

Data schedule (--data_schedule, default mixed):
  mixed      1 copy of OpenVE + --multi_repeat (2) copies each of goku / CoinVE, shuffled into one dataset, run for stage_a_epochs epochs (same sample count as two_stage, only the order is mixed)
  two_stage  stage A --stage_a_data (OpenVE, single instruction) for stage_a_epochs epochs, then stage B --stage_b_data (goku + CoinVE, multi-instruction) for stage_b_epochs epochs (the mlp-moe schedule)
  --exclude_prompt_words drops samples containing the given words; the sub-instructions of a multi-instruction sample are joined into one full instruction with " Then " (CoinVE) or GOKU phrasing
Each step:
  1. take a batch of B samples (random, no grouping)
  2. VAE-encode source / target videos → per-sample sigma and noise;  noisy = (1-σ)·tgt + σ·noise;  target = noise - tgt
  3. MLLM encodes the full instruction → ctx; with prob text_drop_prob a sample gets the empty-instruction ctx instead (unconditional branch for text CFG)
     (default --mllm_batch_encode 1: the whole batch goes through the MLLM in one pass; source PIL frames are rebuilt from src_pixels in the main process, see rebuild_src_frames)
  4. DiT forward (the MoE load-balancing losses are already attached to the graph, see wan_transformer3d_kiwi_moe_v3_1.AddAuxiliaryLoss; one each for self-attention and FFN)
  5. loss = custom_mse_loss(pred, target) (+ motion difference term); on backward the aux-loss gradient flows into the gates automatically
  6. DeepSpeed step (gradient clipping is in the ds config); full update. Three learning-rate groups (see the "learning rate" section):
       original params        always learning_rate
       routed experts (new)   start at new_layers_lr, decay linearly to learning_rate within new_layers_lr_steps, then constant
       gates                  start at gate_lr, move linearly to gate_lr_end within the same steps, then constant
Feature cache (--multi_cache_dir, default empty = off; the launcher passes ${OUT}/cache_multi): in mixed data every goku / CoinVE sample is visited multi_repeat (e.g. 8) times per epoch,
  and video decoding + VAE encoding (5.2 s/step) + MLLM encoding (3.1 s/step) were recomputed each time. The first visit (miss) computes as usual and also writes the VAE distribution params (mean + logvar for src / tgt;
  same sampling formula, fresh noise on every visit) and both MLLM ctx branches (full / empty instruction) to <dir>/<key>.st (flat; background threads, write .tmp then atomic rename, failures only counted);
  on later visits the dataset worker (kiwi_dataset_multi_moe) reads the file directly, the sample carries a "cache" key and the main process skips decoding / VAE / MLLM.
  The text-CFG draw still calls torch.rand(()) once per sample in the original order; hits take ctx_null / ctx_cond according to the draw. Guards 1 / 1b apply to hits as well.
  rank0 clears the directory at start (not reused across launches) and deletes it at DONE; the launcher EXIT trap is the fallback (the cache must be removed when training ends or is killed).
  Logs cache/hit_ratio (hits / multi-sample visits), cache/files (approximate file count), etc.; the console step line ends with cache=xx%.
Loader memory guard: before building the DataLoader, estimate in-flight memory as workers × prefetch × batch × ~118 MB per sample (src + tgt uint8) × local processes; above 60% of min(/dev/shm free, MemAvailable) it raises with suggested values.
Logging (tensorboard + wandb, every 10 steps, averaged over 24 GPUs × 10 steps): loss/main, loss/motion, loss/aux, loss/total, lr/*, grad/norm, time/{vae,mllm,dit} (rank0 seconds per step),
  route_{ffn|attn}/{category}/expert{i} = P(expert i selected | sample category c) (category × expert conditional selection rate, the direct specialization metric), route/{kind}_spread = mean over experts of the max gap across categories,
  and the per-layer, per-expert "selected fraction": self-attention gate moe_attn_L{block}/expert{i}, FFN gate moe_ffn_L{block}/expert{i}
  (token mean of 1 if selected, 0 otherwise; one tensorboard folder per layer); loss/aux_attn and loss/aux_ffn are logged separately
Checkpoints:
  every save_steps     ckpt/step-N/moe_v3_1_trainable.safetensors  trainable params in bf16 (full training = whole DiT; for inference / weights-only resume) + meta.json
  every ds_save_steps  ckpt/ds_state_step-N/                     full DeepSpeed state (weights + optimizer, 24 shards, for resume), only the latest one is kept
resume: --resume_ds_state <ds_state dir> (needs the same 24 GPUs)  or  --resume_weights <safetensors> (weights only, fresh optimizer)
Multi-node launch: train_multinode.sh (single-node 8-GPU smoke test: train_smoke.sh).
"""
import argparse
import gc
import json
import math
import os
import re
import shutil
import sys
import time

import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from moe_v3_1_model import EXPERT_NAMES, TextEncoder, build_dit, build_mllm, build_vae, load_trainable_weights, mark_trainable, trainable_state_keys  # noqa: E402

TAG = "[moe-v3.1]"


# =========================================================================== #
#  Arguments                                                                  #
# =========================================================================== #
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    # Model and weights
    p.add_argument("--pretrained_wan_path", required=True)
    p.add_argument("--config_path", required=True)
    p.add_argument("--mllm_model", required=True)
    p.add_argument("--init_weights", required=True, help="moe_expert_init.safetensors(connector / query / vae_condition)")
    p.add_argument("--experts_merged", required=True, help="output of merge_experts_v3_1.py (experts_v3_1.safetensors)")
    p.add_argument("--use_gradient_checkpointing", type=int, default=1)
    # MoE
    p.add_argument("--moe_layers", default="1,3,5,7,9,11,13,15,17,19,21,23,25,27", help="0-based block indices = layers 2,4,...,28 counting from 1")
    p.add_argument("--moe_num_experts", type=int, default=7, help="0 original / 1..5 category experts / 6 goku direct fine-tune")
    p.add_argument("--moe_top_k", type=int, default=3, help="number of routed experts activated per token")
    p.add_argument("--moe_aux_loss_alpha", type=float, default=0.001, help="load-balancing auxiliary loss coefficient")
    # Data
    p.add_argument("--stage_a_data", required=True, help="stage A json/jsonl files, comma-separated")
    p.add_argument("--stage_b_data", required=True, help="stage B json/jsonl files, comma-separated")
    p.add_argument("--stage_a_epochs", type=float, default=1.0)
    p.add_argument("--stage_b_epochs", type=float, default=2.0)
    p.add_argument("--exclude_prompt_words", default="overlay", help="comma-separated; samples whose instruction contains any of these words (case-insensitive) are dropped")
    p.add_argument("--data_schedule", default="mixed", choices=["mixed", "two_stage"],
                   help="mixed: 1 copy of OpenVE + multi_repeat copies each of goku/CoinVE shuffled into one dataset, run for stage_a_epochs epochs; two_stage: A then B (the mlp-moe schedule)")
    p.add_argument("--multi_repeat", type=int, default=2, help="number of copies of goku / CoinVE each in mixed mode")
    p.add_argument("--dataset_base_path", default="", help="prefix for relative video paths in the data json (os.path.join); empty = use the json paths as-is (the prefix has no effect on absolute paths)")
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--max_pixels", type=int, default=399360)
    p.add_argument("--dataset_num_workers", type=int, default=4)
    p.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    p.add_argument("--compound_joiner", default=" Then ", help="CoinVE multi-instruction samples (no native full instruction): sub-instructions are joined into one sentence with this")
    p.add_argument("--vae_encode_group", type=int, default=4, help="videos per VAE encode call (source / target grouped separately); 1 = per sample (old behavior)")
    p.add_argument("--z3_leaf_modules", type=int, default=0, help="speed-up: mark MoEFFN/MoEKVSelfAttention as ZeRO-3 leaf modules (gather each whole module at once); off by default")
    p.add_argument("--async_log", type=int, default=1, help="speed-up: run rank0's accelerator.log (~300 scalars; wandb+TB measured to block ~4 s per 10 steps) in a background thread")
    p.add_argument("--mllm_batch_encode", type=int, default=1,
                   help="speed-up: 1 = encode the whole batch in one MLLM pass (TextEncoder.encode_batch; 8 per-sample forwards ~2.7 s → 1); 0 = per-sample encode (old behavior)")
    # Feature cache: VAE distribution params + MLLM ctx of multi-edit (goku / CoinVE) samples (see module docstring); off by default
    p.add_argument("--vae_compile", default="off", choices=["off", "encoder", "encoder_dyn", "encoder_autotune", "encode"],
                   help="speed-up: torch.compile the VAE encoder (mode chosen by benchmarking on real shapes; off = eager)")
    p.add_argument("--cache_keep_existing", type=int, default=0, help="keep the existing cache on restart (1 = do not clear the leftover dir at start, reuse it)")
    p.add_argument("--cache_delete_on_done", type=int, default=1, help="delete the cache automatically when training ends (0 = keep it and print a notice, delete manually)")
    p.add_argument("--multi_cache_dir", default="", help="feature cache dir for multi-edit samples (~15 MB per file); empty = off. Cleared at start, deleted by rank0 when training ends")
    p.add_argument("--cache_write_threads", type=int, default=4, help="cache writer threads per rank (a single stream to network storage is ~10 MB/s; up to 8 × 15 MB per step)")
    p.add_argument("--cache_queue", type=int, default=64, help="cache write queue capacity (samples); when full, items are dropped (counted in cache/write_drop) without blocking training")
    # Training
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4, help="per-GPU batch; effective batch = 24 × batch_size")
    p.add_argument("--max_train_steps", type=int, default=0, help="0 = derived from the stage epochs")
    p.add_argument("--learning_rate", type=float, default=1e-5, help="base learning rate of full training; original params (incl. the shared part of MoE layers) always use it")
    p.add_argument("--new_layers_lr", type=float, default=1e-4, help="initial learning rate of the new layers (routed experts + gates)")
    p.add_argument("--new_layers_lr_steps", type=int, default=2000, help="steps for the new layers to decay linearly from new_layers_lr to learning_rate, then same as other params; the gate transition uses the same steps")
    p.add_argument("--gate_lr", type=float, default=1e-3, help="initial gate learning rate")
    p.add_argument("--gate_lr_end", type=float, default=1e-4, help="gate learning rate after new_layers_lr_steps (linear transition to it, then constant)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--text_drop_prob", type=float, default=0.1, help="text CFG: per-sample probability of blanking the instruction")
    p.add_argument("--motion_sub_loss_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--save_steps", type=int, default=250, help="interval for saving bf16 trainable weights")
    p.add_argument("--ds_save_steps", type=int, default=1000, help="interval for saving the full DeepSpeed state (for resume)")
    p.add_argument("--resume_ds_state", default="", help="ckpt/ds_state_step-N directory")
    p.add_argument("--resume_weights", default="", help="load only moe_v3_1_trainable.safetensors, fresh optimizer")
    # Logging
    p.add_argument("--use_wandb", type=int, default=0)
    p.add_argument("--wandb_project", default="video-editor-moe")
    p.add_argument("--wandb_run_name", default="moe_v3_1")
    return p.parse_args(argv)



# =========================================================================== #
#  Sample categories / batched VAE encoding / timing                          #
# =========================================================================== #
CATEGORIES = ["add", "remove", "replace", "style", "background", "multi", "unknown"]     # rows of the "category × expert" matrix in the logs
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
OPENVE_DIR_TO_CATEGORY = {"local_add": "add", "local_remove": "remove", "local_change": "replace", "global_style": "style", "background_change": "background"}


def category_of(record):
    """Edit category of a sample (logging only, not fed to the model): original OpenVE records use the directory category cat, the five-class subset uses cls;
    goku / CoinVE use turn_types_qwen: if all sub-instructions share one type that is the category, otherwise multi."""
    if record.get("cls") in CATEGORY_INDEX:
        return record["cls"]
    if record.get("cat") in OPENVE_DIR_TO_CATEGORY:
        return OPENVE_DIR_TO_CATEGORY[record["cat"]]
    types = [t for t in (record.get("turn_types_qwen") or []) if t]
    if types:
        return types[0] if all(t == types[0] for t in types) else "multi"
    return "unknown"


def vae_encode_batched(vae, pixels_list, group_size, return_moments=False):
    """Concatenate several [1, 3, F, H, W] videos into one batch for the VAE encoder; returns sampled latents [B, C, f, h, w].
    AutoencoderKLWan3_8.encode loops over samples, but the inner vae.model.encode supports a batch dim (the chunk loop and feat_cache both carry b),
    so the inner one is called directly, group_size videos at a time to bound peak memory (encoding happens before DiT activations are allocated). Per-sample serial encoding was the largest per-step cost.
    Feature cache: with return_moments=True returns (sample, mean, logvar); mean / logvar are the two channel halves of the encoder output [B, 2C, f, h, w] (vae.model.encode's
    mu is already normalized by vae.scale, logvar is the raw unclamped value); sampling is still DiagonalGaussianDistribution(...).sample() (same formula), and cache hits resample with the same formula."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    group = max(1, group_size)
    if not hasattr(vae, "model") or not hasattr(vae.model, "encode") or not hasattr(vae, "scale"):
        dists = [vae.encode(x)[0] for x in pixels_list]                                  # other VAE classes: fall back to per-sample
        samples = torch.cat([d.sample() for d in dists], dim=0)
        if not return_moments:
            return samples
        mean, logvar = torch.cat([d.parameters for d in dists], dim=0).chunk(2, dim=1)
        return samples, mean, logvar
    outs, moments = [], []
    for i in range(0, len(pixels_list), group):
        x = torch.cat(pixels_list[i:i + group], dim=0)
        try:
            m = vae.model.encode(x, vae.scale)                                             # [g, 2C, f, h, w] = cat([mu, log_var], 1)
        except Exception as e:                                                             # a torch.compile'd encoder may raise an Inductor internal error on the first real compile / a new shape
            eager = getattr(vae.model, "_eager_encoder", None)                             #   (seen with encoder_dyn, crashing the run): fall back to eager and keep training, print once
            if eager is None or vae.model.encoder is eager:
                raise
            print(f"[moe-v3.1] Warning: compiled VAE encoder failed, falling back to eager: {type(e).__name__}: {str(e)[:160]}", flush=True)
            vae.model.encoder = eager
            m = vae.model.encode(x, vae.scale)
        outs.append(DiagonalGaussianDistribution(m).sample())
        moments.append(m)
    samples = torch.cat(outs, dim=0)
    if not return_moments:
        return samples
    mean, logvar = torch.cat(moments, dim=0).chunk(2, dim=1)
    return samples, mean, logvar


def rebuild_src_frames(sample):
    """Speed-up: the dataset is built with src_pil=False, so workers no longer pickle 49 source PIL frames back to the main process (470 MB / 3.5 s per step),
    only src_pixels ([C=3, F, H, W] uint8, RGB; layout of kiwi_video_utils.frames_to_uint8_tensor: np.array(PIL.convert("RGB")) gives [H, W, 3],
    permute(2, 0, 1) to [3, H, W], then stack(dim=1) to [3, F, H, W]). The MLLM still needs PIL frames, so the main process inverts the same layout
    (permute back to [F, H, W, 3], then Image.fromarray per frame): an exact inverse of frames_to_uint8_tensor (same mode "RGB", same size, byte-identical).
    Only samples with src_video None and src_pixels present are handled; samples that already have PIL frames (src_pil=True / return_pixels=False) are left as-is."""
    if sample.get("src_video") is not None or sample.get("src_pixels") is None:
        return sample
    from PIL import Image
    u8 = sample["src_pixels"]
    assert u8.dim() == 4 and u8.shape[0] == 3 and u8.dtype == torch.uint8, f"src_pixels must be [3, F, H, W] uint8, got {tuple(u8.shape)} {u8.dtype}"
    frames = u8.permute(1, 2, 3, 0).contiguous().numpy()                     # [F, H, W, 3] uint8, one copy; each frame is a C-contiguous slice
    sample["src_video"] = [Image.fromarray(frames[f]) for f in range(frames.shape[0])]   # [H, W, 3] uint8 → mode "RGB", byte-identical to the original PIL frame
    return sample


def sync_time():
    """For timing: wait for the GPU queue to drain before reading the clock, so the VAE / MLLM / DiT timings are real."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


# =========================================================================== #
#  Multi-edit sample feature cache (main-process side) / loader memory guard  #
# =========================================================================== #
CACHE_VERSION = "v1"
CACHE_TENSOR_KEYS = ("src_mean", "src_logvar", "tgt_mean", "tgt_logvar", "ctx_cond", "ctx_null")      # the 6 tensors in a cache file (bf16 CPU)
CACHE_STAT_NAMES = ("hit", "miss", "other", "write_ok", "write_fail", "write_drop")                  # slots appended to the stat vector (reduced / zeroed with stat every 10 steps)
CACHE_FILES_COUNT_EVERY = 100                                                                        # every how many steps rank0 counts the cache files in a background thread


def multi_cache_key(src_video, tgt_video, prompt, num_frames, height, width):
    """Cache key = sha1("src|tgt|full instruction|FxHxW") (src / tgt are the raw record strings); same formula as KiwiMultiEditDataset.cache_key_of on the dataset side (keep both in sync);
    files are flat under <cache_dir>. The main process only falls back to this when a sample has no cache_path / cache_key (the dataset normally provides them, and they take precedence)."""
    import hashlib
    return hashlib.sha1(f"{src_video}|{tgt_video}|{prompt}|{num_frames}x{height}x{width}".encode("utf-8")).hexdigest()


def prompt_of(sample):
    """Full instruction of a sample (prompts_multi[0]; a cache hit without prompts_multi falls back to the prompt field)."""
    plist = sample.get("prompts_multi")
    return plist[0] if plist else (sample.get("prompt") or "")


def is_cached_sample(sample):
    """A cache hit returned by a dataset worker carries a "cache" key (dict of the file contents); a flat top-level layout with cached=True is also accepted."""
    return sample.get("cache") is not None or bool(sample.get("cached"))


def cache_payload_of(sample, expect=None):
    """Cache hit → dict of the 6 CPU tensors; returns None for a non-hit.
    Also returns None for missing keys / wrong ndim / mismatch with the latent shape or ctx dim seen in this run (expect = {"latent": (C, f, h, w), "ctx": 3072}, None = unknown, not checked);
    callers use is_cached_sample to tell "miss" from "hit with a bad file" (the latter has no pixels to fall back on and must be dropped)."""
    payload = sample.get("cache")
    if not isinstance(payload, dict):
        payload = sample if sample.get("cached") else None
    if payload is None:
        return None
    try:
        out = {k: payload[k] for k in CACHE_TENSOR_KEYS}
    except (KeyError, TypeError):
        return None
    for k, t in out.items():
        if not torch.is_tensor(t) or t.dim() != (2 if k.startswith("ctx") else 4):
            return None
    if not (out["src_mean"].shape == out["src_logvar"].shape == out["tgt_mean"].shape == out["tgt_logvar"].shape):
        return None
    if out["ctx_cond"].shape[1] != out["ctx_null"].shape[1]:
        return None
    if expect:
        if expect.get("latent") is not None and tuple(out["src_mean"].shape) != tuple(expect["latent"]):
            return None
        if expect.get("ctx") is not None and int(out["ctx_cond"].shape[1]) != int(expect["ctx"]):
            return None
    return out


def multi_cache_path_of(sample, cache_dir, args):
    """Cache file a missed sample is written to after this step; None if it is not in a cache group (OpenVE).
    Prefers cache_path / cache_key computed by the dataset (guaranteed to match the path it reads); if the sample only has cache_group="multi", falls back to multi_cache_key."""
    if not cache_dir:
        return None
    path = sample.get("cache_path")
    if path:
        return path
    key = sample.get("cache_key")
    if not key:
        if sample.get("cache_group") != "multi":
            return None
        src, tgt = sample.get("src_video_path"), sample.get("tgt_video_path")
        if not src or not tgt:
            return None
        key = multi_cache_key(src, tgt, prompt_of(sample), args.num_frames, args.height, args.width)
    return os.path.join(cache_dir, key + ".st")     # safetensors layout, suffix .st


def split_batch_for_cache(batch, cache_dir, args, cache_expect, rank, step):
    """Split a batch into cache hits / misses. Returns (batch, payloads, pending, counts):
      payloads[i]  dict of the 6 tensors for a hit, None for a miss (downstream splits idx_c / idx_u on this)
      pending[i]   {"path", "sample"} for a miss in a cache group; after this step it collects the distribution params / both ctx and is written to the cache; None otherwise. Filtered in sync with guard drops
      counts       [hits, misses (first multi visit), other (OpenVE)] for the cache slots in stat
    Samples with a "cache" key but bad contents (corrupt file / missing keys / shape mismatch) have no pixels to fall back on and are dropped with a message. Empty cache_dir = off: all misses, pending all None."""
    payloads = [cache_payload_of(s, cache_expect) if cache_dir else None for s in batch]
    bad = [i for i, s in enumerate(batch) if payloads[i] is None and is_cached_sample(s)]
    for i in bad:
        print(f"{TAG} cache: rank{rank} step {step}: bad cache-hit contents, dropping src={batch[i].get('src_video_path', '?')} key={batch[i].get('cache_key', '?')}", flush=True)
    if bad:
        batch = [s for i, s in enumerate(batch) if i not in bad]
        payloads = [p for i, p in enumerate(payloads) if i not in bad]
    pending = [None] * len(batch)
    if cache_dir:
        for i, sample in enumerate(batch):
            if payloads[i] is None:
                path = multi_cache_path_of(sample, cache_dir, args)
                if path:
                    pending[i] = {"path": path, "sample": sample}
    n_hit = sum(p is not None for p in payloads)
    n_miss = sum(p is not None for p in pending)
    return batch, payloads, pending, [n_hit, n_miss, len(batch) - n_hit - n_miss]


def sample_cached_latents(payloads, mean_key, logvar_key, device, dtype):
    """Resample latents of cache hits from the cached VAE distribution params, via DiagonalGaussianDistribution(cat([mean, logvar], 1)).sample() as in vae_encode_batched
    (= mean + exp(0.5·clamp(logvar, -30, 20))·randn, randn drawn on device in dtype); resampled on every visit, same distribution as without the cache. Returns [n, C, f, h, w]."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    mean = torch.stack([p[mean_key] for p in payloads]).to(device=device, dtype=dtype)
    logvar = torch.stack([p[logvar_key] for p in payloads]).to(device=device, dtype=dtype)
    return DiagonalGaussianDistribution(torch.cat([mean, logvar], dim=1)).sample()


def encode_latents_with_cache(vae, batch, payloads, pending, args, device, dtype, pixels_of, cache_expect):
    """Run the VAE only on misses (also collecting mean / logvar into pending for the cache writer); hits are resampled from the cached params; results are reassembled in batch order.
    Returns (src_latents, latents), each [B, C, f, h, w]; for an empty batch returns two 0-row tensors (guard 1 downstream then treats the whole batch as bad)."""
    idx_u = [i for i, p in enumerate(payloads) if p is None]
    idx_c = [i for i, p in enumerate(payloads) if p is not None]
    src_rows, tgt_rows = [None] * len(batch), [None] * len(batch)
    if idx_u:
        src_u, src_mean, src_logvar = vae_encode_batched(vae, [pixels_of(batch[i], "src_pixels", "src_video") for i in idx_u], args.vae_encode_group, return_moments=True)
        tgt_u, tgt_mean, tgt_logvar = vae_encode_batched(vae, [pixels_of(batch[i], "tgt_pixels", "tgt_video") for i in idx_u], args.vae_encode_group, return_moments=True)
        for k, i in enumerate(idx_u):
            src_rows[i], tgt_rows[i] = src_u[k], tgt_u[k]
            if pending[i] is not None:
                pending[i].update(src_mean=src_mean[k], src_logvar=src_logvar[k], tgt_mean=tgt_mean[k], tgt_logvar=tgt_logvar[k])
        if cache_expect.get("latent") is None:
            cache_expect["latent"] = tuple(src_u.shape[1:])            # later cache hits must match this shape (checked in cache_payload_of)
    if idx_c:
        src_c = sample_cached_latents([payloads[i] for i in idx_c], "src_mean", "src_logvar", device, dtype)
        tgt_c = sample_cached_latents([payloads[i] for i in idx_c], "tgt_mean", "tgt_logvar", device, dtype)
        for k, i in enumerate(idx_c):
            src_rows[i], tgt_rows[i] = src_c[k], tgt_c[k]
    if not batch:
        empty = torch.zeros(0, device=device, dtype=dtype)
        return empty, empty
    return torch.stack(src_rows), torch.stack(tgt_rows)


def encode_batch_chunked(text_encoder, prompts, frames_list, chunk):
    """Chunked encode_batch (at most chunk samples per call = the MLLM batch size validated in training): missed multi samples encode one extra branch (empty / full instruction),
    so a batch has up to 2 × batch_size prompts, and without chunking the MLLM forward batch would double. forward_batch is per-sample equivalent, so chunking does not change results."""
    chunk = max(1, chunk)
    out = []
    for i in range(0, len(prompts), chunk):
        out.extend(text_encoder.encode_batch(prompts[i:i + chunk], frames_list[i:i + chunk]))
    return out


def encode_ctx_with_cache(text_encoder, batch, payloads, pending, args, device, dtype, cache_expect):
    """Text-CFG draw + MLLM encoding; returns ctx_list aligned with batch (each [L_i, 3072] bf16, on device).
      · the draw calls float(torch.rand(())) once per sample in batch order (hits included): RNG consumption and order are identical to the uncached path
      · hits: take the cached ctx_null / ctx_cond according to the draw, no MLLM
      · misses: encoded as before (a dropped prompt is set to "" = encode_null semantics); those in a cache group also encode the other branch (full instruction if the empty one was drawn, else empty),
        and both go into pending (ctx_cond / ctx_null); with mllm_batch_encode=1 they are appended to the main list and run through chunked encode_batch, with =0 per-sample encode / encode_null.
    batch must not contain samples dropped by guard 1 (the caller filters them)."""
    idx_u = [i for i, p in enumerate(payloads) if p is None]
    idx_c = [i for i, p in enumerate(payloads) if p is not None]
    for i in idx_u:
        assert len(batch[i]["prompts_multi"]) == 1              # load_records already reduced each sample to one full instruction
        rebuild_src_frames(batch[i])                            # with src_pil=False the source PIL frames are rebuilt from src_pixels here (the MLLM needs PIL)
    prompts = ["" if float(torch.rand(())) < args.text_drop_prob else prompt_of(sample) for sample in batch]
    ctx_list = [None] * len(batch)
    for i in idx_c:
        ctx_list[i] = (payloads[i]["ctx_null"] if prompts[i] == "" else payloads[i]["ctx_cond"]).to(device=device, dtype=dtype)
    extra_pos = [i for i in idx_u if pending[i] is not None]
    other_prompt = {i: (batch[i]["prompts_multi"][0] if prompts[i] == "" else "") for i in extra_pos}      # instruction of the other branch
    if args.mllm_batch_encode:
        # speed-up: the whole batch goes through the MLLM at once (TextEncoder.encode_batch → MLLMInContext.forward_batch); main encodes first, extra branches appended after
        enc_prompts = [prompts[i] for i in idx_u] + [other_prompt[i] for i in extra_pos]
        enc_frames = [batch[i]["src_video"] for i in idx_u + extra_pos]
        enc_out = encode_batch_chunked(text_encoder, enc_prompts, enc_frames, args.batch_size) if enc_prompts else []
        for k, i in enumerate(idx_u):
            ctx_list[i] = enc_out[k]
        other_ctx = {i: enc_out[len(idx_u) + k] for k, i in enumerate(extra_pos)}
    else:                                                       # old path: per-sample encode
        other_ctx = {}
        for i in idx_u:
            frames = batch[i]["src_video"]
            ctx_list[i] = text_encoder.encode_null(frames) if prompts[i] == "" else text_encoder.encode(batch[i]["prompts_multi"][0], frames)
            if i in other_prompt:
                other_ctx[i] = text_encoder.encode_null(frames) if other_prompt[i] == "" else text_encoder.encode(other_prompt[i], frames)
    for i in extra_pos:                                         # assign branches: if the empty instruction was drawn, the main ctx is the null branch and the extra one the full branch; otherwise the reverse
        if prompts[i] == "":
            pending[i]["ctx_null"], pending[i]["ctx_cond"] = ctx_list[i], other_ctx[i]
        else:
            pending[i]["ctx_cond"], pending[i]["ctx_null"] = ctx_list[i], other_ctx[i]
    if idx_u and cache_expect.get("ctx") is None:
        cache_expect["ctx"] = int(ctx_list[idx_u[0]].shape[1])
    return ctx_list


def build_cache_payload(pending, args, step, rank):
    """Cache file contents for one sample: 6 contiguous bf16 CPU tensors + meta."""
    sample = pending["sample"]
    # copy=True + contiguous_format guarantee exactly one copy whether the source is on GPU or CPU (unit tests), with its own storage holding only its elements.
    # A plain .to("cpu", bf16).contiguous() does not copy a tensor that is already CPU bf16, so src_mean etc. stay views of the large VAE moments storage and the whole storage gets written (37 MB vs 14 MB)
    out = {k: pending[k].detach().to("cpu", torch.bfloat16, copy=True, memory_format=torch.contiguous_format) for k in CACHE_TENSOR_KEYS}
    out["meta"] = {"key": os.path.basename(pending["path"])[:-3], "src_video": sample.get("src_video_path"), "tgt_video": sample.get("tgt_video_path"),
                   "prompt": prompt_of(sample), "num_frames": int(args.num_frames), "height": int(args.height), "width": int(args.width),
                   "latent_shape": list(out["src_mean"].shape), "version": CACHE_VERSION, "step": int(step), "rank": int(rank)}
    return out


def save_cache_file(path, payload):
    """Write a cache file in safetensors layout (8-byte little-endian header length + JSON header + raw tensor bytes), as one bytes blob with unbuffered f.write.
    Why: py-spy showed 24.6% of rank0 GIL time in the writer threads' torch.save (zip open/close holds the GIL throughout), stalling kernel launches on the main thread and slowing the DiT part by ~2 s/step;
    tobytes/join are just memcpy (milliseconds) and the write syscall releases the GIL. The reader (kiwi_dataset_multi_moe) uses safetensors.safe_open (mmap).
    numpy has no bf16 dtype, so bytes are taken through an int16 view (same bit pattern) and the header says BF16 so safetensors reads them back as bf16."""
    import json as _json
    import struct
    header = {"__metadata__": {str(k): str(v) for k, v in payload["meta"].items()}}
    blobs, off = [], 0
    for k in CACHE_TENSOR_KEYS:
        t = payload[k]
        assert t.dtype == torch.bfloat16 and t.device.type == "cpu" and t.is_contiguous(), (k, t.dtype, t.device, t.is_contiguous())
        b = t.view(torch.int16).numpy().tobytes()
        header[k] = {"dtype": "BF16", "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hb = _json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    buf = struct.pack("<Q", len(hb)) + hb + b"".join(blobs)
    with open(path, "wb", buffering=0) as f:
        n = 0
        while n < len(buf):
            n += f.write(buf[n:])


def enqueue_cache_payloads(cache_writer, pending, args, step, rank):
    """Enqueue missed multi samples that passed guards 1 + 1b for writing; returns the number enqueued. Guard 1b only checked the main ctx, not the extra branch:
    if any of the 6 tensors is non-finite the sample is not written (path printed; it misses and is recomputed next time); otherwise a non-finite ctx in the file would get the sample dropped by guard 1b on every later hit, never re-encoded."""
    n = 0
    for p in pending:
        if p is None:
            continue
        bad = [k for k in CACHE_TENSOR_KEYS if not torch.isfinite(p[k]).all()]
        if bad:
            sample = p["sample"]
            print(f"{TAG} rank{rank} step {step}: skipping cache write (non-finite tensors {bad}) src={sample.get('src_video_path', '?')} prompt={prompt_of(sample)[:80]!r}", flush=True)
            continue
        cache_writer.put(p["path"], build_cache_payload(p, args, step, rank))
        n += 1
    return n


class MultiCacheWriter:
    """Per-rank background cache writer: the main thread only puts (path, payload) on a bounded queue; `threads` daemon threads write to <path>.tmp.<rank>.<pid>.<tid>
    and os.replace it atomically (concurrent writes of one key from 24 ranks: last one wins, readers never see a partial file). A full queue drops the item (write_drop); write errors are swallowed and counted (write_fail,
    only the first 3 printed; a cache dir deleted by another node or storage hiccups must not crash training). Threads start on the first put (after DataLoader workers fork, avoiding a fork with live threads).
    rank0 also has count_files_async(): a background os.scandir counting .st files (flat top level + one subdirectory level), result in files (-1 = not counted yet)."""

    def __init__(self, cache_dir, rank, threads=4, maxsize=64):
        import queue
        import threading
        self.dir, self.rank = cache_dir, rank
        self.q = queue.Queue(maxsize=max(1, maxsize))
        self.lock = threading.Lock()
        self.ok = self.fail = self.drop = 0
        self.files = -1
        self.num_threads = max(1, threads)
        self.threads = []
        self._count_thread = None

    def _ensure_started(self):
        import threading
        if not self.threads:
            self.threads = [threading.Thread(target=self._run, name=f"cache-writer-{i}", daemon=True) for i in range(self.num_threads)]
            for t in self.threads:
                t.start()

    def put(self, path, payload):
        import queue
        self._ensure_started()
        try:
            self.q.put_nowait((path, payload))
        except queue.Full:
            with self.lock:
                self.drop += 1

    def _run(self):
        import threading
        pid, tid = os.getpid(), threading.get_ident()
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                break
            path, payload = item
            tmp = f"{path}.tmp.{self.rank}.{pid}.{tid}"
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                save_cache_file(tmp, payload)             # not torch.save (it holds the GIL)
                os.replace(tmp, path)
                with self.lock:
                    self.ok += 1
            except Exception as e:                       # write failures must not affect training
                with self.lock:
                    self.fail += 1
                    n = self.fail
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                if n <= 3:
                    print(f"{TAG} cache: rank{self.rank} cache write failed (#{n}, counted only, training unaffected) {path}: {e!r}", flush=True)
            finally:
                self.q.task_done()

    def drain_counts(self):
        """Return and reset the (written, failed, dropped-on-full-queue) counts; added to stat every step."""
        with self.lock:
            counts = (self.ok, self.fail, self.drop)
            self.ok = self.fail = self.drop = 0
        return counts

    def qsize(self):
        return self.q.qsize()

    def count_files_async(self):
        import threading
        if self._count_thread is not None and self._count_thread.is_alive():
            return

        def _count():
            n = 0
            try:
                with os.scandir(self.dir) as entries:
                    for e in entries:
                        if e.is_dir(follow_symlinks=False):
                            try:
                                with os.scandir(e.path) as files:
                                    n += sum(1 for f in files if f.name.endswith(".st"))
                            except OSError:
                                pass
                        elif e.name.endswith(".st"):
                            n += 1
            except OSError:
                pass
            self.files = n

        self._count_thread = threading.Thread(target=_count, name="cache-count", daemon=True)
        self._count_thread.start()

    def close(self, timeout=300):
        """Drain the queue (wait at most timeout seconds). No threads started = nothing written, return immediately."""
        import queue
        if not self.threads:
            return
        deadline = time.time() + timeout
        for _ in self.threads:
            try:
                self.q.put(None, timeout=max(0.0, deadline - time.time()))
            except queue.Full:
                break
        for t in self.threads:
            t.join(max(0.0, deadline - time.time()))


def remove_cache_dir(cache_dir, wait_seconds=0, threads=16):
    """Delete the cache directory (it must be removed automatically when training ends or is killed).
    (1) atomically rename <dir> to <dir>.del.<pid>.<secs> (instant; a new dir with the same name is unaffected), together with leftover <dir>.del.* from earlier runs;
    (2) a background thread scans the top-level entries (flat files or subdirectories; readdir of ~220k entries on network storage takes tens of seconds, so not on the main thread),
       `threads` daemon threads delete in parallel (unlink files, rmtree subdirs; deleting small files is metadata-bound, a serial rm takes tens of minutes, parallel is ~10x faster), then the root is removed;
    (3) with wait_seconds > 0 wait at most that long and return even if unfinished (daemon threads keep deleting; after exit the launcher's EXIT trap nohup rm cleans up the rest).
    Returns (list of dirs being deleted, whether deletion finished)."""
    import glob
    import threading
    targets = []
    if os.path.isdir(cache_dir):
        renamed = f"{cache_dir}.del.{os.getpid()}.{int(time.time())}"
        try:
            os.rename(cache_dir, renamed)
            targets.append(renamed)
        except OSError as e:                             # e.g. another node's trap renamed it first
            print(f"{TAG} cache: renaming {cache_dir} failed ({e!r}), skipping", flush=True)
    targets += [d for d in sorted(glob.glob(cache_dir + ".del.*")) if os.path.isdir(d) and d not in targets]
    if not targets:
        return [], True
    work, lock = [], threading.Lock()

    def _worker():
        while True:
            with lock:
                if not work:
                    return
                item, is_dir = work.pop()
            if is_dir:
                shutil.rmtree(item, ignore_errors=True)
            else:
                try:
                    os.remove(item)
                except OSError:
                    pass

    def _finish():
        for target in targets:                           # scan dirs (work is filled single-threaded before the workers start, no lock needed)
            try:
                with os.scandir(target) as entries:
                    for e in entries:
                        work.append((e.path, e.is_dir(follow_symlinks=False)))
            except OSError:
                pass
        workers = [threading.Thread(target=_worker, name=f"cache-rm-{i}", daemon=True) for i in range(max(1, threads))]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        for target in targets:
            shutil.rmtree(target, ignore_errors=True)

    finisher = threading.Thread(target=_finish, name="cache-rm-finish", daemon=True)
    finisher.start()
    if wait_seconds > 0:
        finisher.join(wait_seconds)
    return targets, not finisher.is_alive()


def dataset_cache_kwargs(dataset_cls, cache_dir):
    """Constructor kwarg through which the dataset (kiwi_dataset_multi_moe.KiwiMultiEditDataset) receives the cache dir, matched against its actual signature (multi_cache_dir / cache_dir both accepted).
    Empty cache_dir returns {} (old behavior); if the signature has no such parameter returns None (the caller disables the cache and warns rather than passing an unknown kwarg to the parent KiwiEditDataset and crashing)."""
    import inspect
    if not cache_dir:
        return {}
    params = inspect.signature(dataset_cls.__init__).parameters
    for name in ("multi_cache_dir", "cache_dir", "cache_multi_dir"):
        if name in params:
            return {name: cache_dir}
    return None


def check_loader_memory(args, is_local_main):
    """Check the in-flight sample memory budget before building the DataLoader. Each rank queues up to num_workers × prefetch_factor batches in /dev/shm (persistent workers return via shared memory),
    batch_size samples per batch; a miss = src + tgt each [3, F, H, W] uint8 ≈ 118 MB (49 × 480 × 832; hits are only ~15 MB of features, worst case assumed),
    summed over the LOCAL_WORLD_SIZE (8) local processes; compared with min(/dev/shm free (statvfs), MemAvailable (/proc/meminfo)), above 60% it raises with suggested values.
    Only local rank 0 prints the estimate; all ranks compute the same numbers and exit together on error. Returns the estimated bytes."""
    per_sample = 2 * 3 * int(args.num_frames) * int(args.height) * int(args.width)
    workers = max(int(args.dataset_num_workers), 0)
    prefetch = int(args.dataloader_prefetch_factor or 2) if workers > 0 else 1
    procs = int(os.environ.get("LOCAL_WORLD_SIZE", 0) or 0) or (torch.cuda.device_count() if torch.cuda.is_available() else 0) or 1
    in_flight = max(workers, 1) * prefetch * int(args.batch_size)
    need = in_flight * per_sample * procs
    st = os.statvfs("/dev/shm")
    shm_avail = st.f_bavail * st.f_frsize
    mem_avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if mem_avail is None:
        mem_avail = shm_avail
    budget = 0.6 * min(shm_avail, mem_avail)
    gb = 2 ** 30
    msg = (f"{TAG} worker memory estimate: workers={workers} × prefetch={prefetch} × batch={args.batch_size} = {in_flight} in-flight samples per rank × {per_sample / 2 ** 20:.0f} MB × {procs} local processes "
           f"= up to {need / gb:.0f} GB; /dev/shm free {shm_avail / gb:.0f} GB, MemAvailable {mem_avail / gb:.0f} GB, 60% of the smaller = {budget / gb:.0f} GB "
           f"→ {'OVER BUDGET' if need > budget else 'OK'}")
    if is_local_main:
        print(msg, flush=True)
    if need > budget:
        unit_w = prefetch * int(args.batch_size) * per_sample * procs
        unit_p = max(workers, 1) * int(args.batch_size) * per_sample * procs
        raise RuntimeError(f"in-flight samples {need / gb:.0f} GB exceed 60% of min(/dev/shm free, MemAvailable) ({budget / gb:.0f} GB): "
                           f"suggest --dataset_num_workers ≤ {int(budget // unit_w)} (keeping prefetch {prefetch}) or --dataloader_prefetch_factor ≤ {int(budget // unit_p)} (keeping workers {workers})")
    return need

# =========================================================================== #
#  Data                                                                       #
# =========================================================================== #
def join_goku_style(prompts):
    """Join sub-instructions into one sentence using the phrasing of GOKU's native total_instruction_en:
    2 steps "A, then b";  3+ steps "A, then b, then c, and finally d" (later steps lowercased, trailing periods removed)."""
    steps = [p.strip().rstrip(".") for p in prompts]
    if len(steps) == 1:
        return steps[0]
    rest = [step[0].lower() + step[1:] for step in steps[1:]]
    if len(rest) == 1:
        return f"{steps[0]}, then {rest[0]}"
    return f"{steps[0]}, then " + ", then ".join(rest[:-1]) + f", and finally {rest[-1]}"


def full_prompt_of(record, joiner):
    """One full instruction per sample (this is what the MLLM receives):
      OpenVE            the prompt is already a single instruction
      GOKU multi-turn   records cover only the first steps of a case (the target video is an intermediate result), while the native total_instruction_en describes all steps (incl. unused camera/style steps),
                        so it does not match the target video; the record's own sub-instructions are joined with GOKU phrasing instead (join_goku_style)
      CoinVE multi-turn the source data only has sub-instruction lists and no native full instruction; joined into one sentence with joiner"""
    prompts = record["prompt"] if isinstance(record["prompt"], list) else [record["prompt"]]
    if re.search(r"videos/case_\d+/", record["src_video"]):
        return join_goku_style(prompts)
    return joiner.join(prompts)


def load_records(meta_paths, exclude_words, joiner, repeats=None, cache_groups=None):
    """Several json / jsonl files → records in a unified schema (one full instruction each); samples whose instruction contains exclude_words are dropped.
    repeats: same length as meta_paths, number of copies of each file's records (goku / CoinVE get 2 in mixed mode); None = 1 each.
    cache_groups: same length as meta_paths; the file's records get a cache_group (paths_b, i.e. goku / CoinVE, are tagged "multi"; the dataset computes cache_key from it and tries a cache hit);
    None / "" = no tag (OpenVE records unchanged)."""
    repeats = repeats or [1] * len(meta_paths)
    cache_groups = cache_groups or [None] * len(meta_paths)
    pattern = re.compile("|".join(re.escape(w) for w in exclude_words), re.I) if exclude_words else None
    records, stats = [], []
    for meta_path, repeat, cache_group in zip(meta_paths, repeats, cache_groups):
        kept = dropped = 0
        with open(meta_path) as f:
            raw_records = json.load(f) if meta_path.endswith(".json") else [json.loads(line) for line in f if line.strip()]
        for record in raw_records:
            prompt = full_prompt_of(record, joiner)
            if pattern is not None and pattern.search(prompt):
                dropped += 1
                continue
            record_out = {"src_video": record["src_video"], "tgt_video": record["tgt_video"], "prompt": [prompt],
                            "n_turns": 1, "mask_mode": "global",
                            "turn_types_qwen": [category_of(record)]}      # category is only for the "category × expert" log matrix, passed through the dataset unchanged
            if cache_group:
                record_out["cache_group"] = cache_group                   # reaches the dataset unchanged via the temp jsonl (json.dumps writes extra fields)
            records.extend([record_out] * repeat)
            kept += 1
        stats.append(f"{os.path.basename(meta_path)}:{kept}" + (f"×{repeat}" if repeat != 1 else "") + f"(dropped {dropped})")
    return records, " + ".join(stats)


def build_dataset(meta_paths, exclude_words, args, repeats=None, cache_groups=None, cache_kwargs=None):
    """Records → temp jsonl → KiwiMultiEditDataset (reads videos / frames, no mask).
    cache_groups: see load_records; cache_kwargs = constructor kwargs from dataset_cache_kwargs(...) ({} = no cache, old behavior)."""
    import tempfile
    from videox_fun.moe_common.kiwi_dataset_multi_moe import KiwiMultiEditDataset
    records, stats = load_records(meta_paths, exclude_words, args.compound_joiner, repeats, cache_groups)
    if records:                                                   # path sanity check on the first record only; if missing, hint at --dataset_base_path (non-fatal)
        probe = os.path.join(args.dataset_base_path, records[0]["src_video"])
        if not os.path.exists(probe):
            print(f"{TAG} Warning: source video of the first sample does not exist: {probe!r}; relative paths in the json need a --dataset_base_path prefix", flush=True)
    fd, merged_path = tempfile.mkstemp(suffix=".jsonl", prefix="moe_v3_data_")
    with os.fdopen(fd, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    dataset = KiwiMultiEditDataset(
        metadata_path=merged_path, base_path=args.dataset_base_path, max_turns=0, mask_loss=False, subset_single_prob=0.0,
        num_frames=args.num_frames, height=args.height, width=args.width, max_pixels=args.max_pixels, repeat=1,
        require_ref_image=False, pad_frames_to_num=True, return_pixels=True,
        src_pil=False,      # speed-up: source PIL frames are not pickled back; the main process rebuilds them from src_pixels for the MLLM (rebuild_src_frames)
        **(cache_kwargs or {}))     # cache dir (kwarg name per the dataset signature), empty dict = old behavior
    os.unlink(merged_path)
    assert len(dataset) == len(records), f"dataset filtered out {len(records) - len(dataset)} records"
    return dataset, stats


class IndexTaggedDataset(torch.utils.data.Dataset):
    """Data-position self-check: adds `_req_index` = the index requested by the sampler to each returned sample dict (the *requested* index, even if an inner retry swapped the sample);
    the training loop uses it to verify that each epoch's first batch is the sampler's batch number skip (guards against silent replays such as DataLoader's double iter swallowing skip). Other attributes are forwarded to the inner dataset."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        s["_req_index"] = int(idx)
        return s

    def __getattr__(self, name):
        base = self.__dict__.get("base")           # __dict__ is still empty while pickle/copy rebuilds the object: must raise AttributeError, not KeyError
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)


class ShardedBatchSampler:
    """Each epoch shuffles all samples with seed+epoch, splits them evenly across ranks and cuts fixed-size batches (the incomplete tail is dropped).
    skip = batches already consumed in this epoch (persistent, set by the training loop at the start of each epoch; see __iter__). All ranks have the same number of batches per epoch."""

    def __init__(self, num_samples, batch_size, num_replicas, rank, seed):
        self.num_samples, self.batch_size = num_samples, batch_size
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def epoch_batches(self):
        """Full list of this rank's batches in the current epoch (index = step within the epoch)."""
        generator = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        order = torch.randperm(self.num_samples, generator=generator).tolist()
        mine = order[self.rank::self.num_replicas]
        return [mine[i:i + self.batch_size] for i in range(0, len(self) * self.batch_size, self.batch_size)]

    def __iter__(self):
        # Fix (data-position bug): the old code did `skip, self.skip = self.skip, 0` (skip applied once). But torch DataLoader calls iter(sampler) twice when building an iterator
        # (once in _BaseDataLoaderIter.__init__, once in _reset; with persistent_workers once more in every later _reset), so the first call reset skip to 0
        # while batches were taken from the second → every resume replayed the epoch from batch 0 (not triggered with num_workers=0).
        # Now skip is persistent and set explicitly by the training loop at each epoch start (resume = in_stage % steps_per_epoch, later epochs = 0); the loop also checks the position of each epoch's first batch.
        return iter(self.epoch_batches()[self.skip:])

    def __len__(self):
        return (self.num_samples // self.num_replicas) // self.batch_size


# =========================================================================== #
#  loss (original videoxfun wan2.2)                                           #
# =========================================================================== #
def custom_mse_loss(noise_pred, target, weighting=None, threshold=50.0):
    """Element-wise MSE, masking outliers with |residual| > threshold, times the sd3 weighting (always 1 for scheme=none), then the mean."""
    noise_pred = noise_pred.float()
    target = target.float()
    mse = F.mse_loss(noise_pred, target, reduction="none")
    mask = ((noise_pred - target).abs() <= threshold).float()
    masked = mse * mask
    if weighting is not None:
        masked = masked * weighting
    return masked.mean()


# =========================================================================== #
#  Learning rate                                                              #
# =========================================================================== #
GROUP_BASE, GROUP_EXPERTS, GROUP_GATE = 0, 1, 2      # indices of the optimizer's three param groups (created in this order)


def linear_then_constant(step, lr_start, lr_end, num_steps):
    """Linear from lr_start to lr_end over the first num_steps steps, then constant lr_end. Based on the global step, so it continues seamlessly after resume."""
    if step >= num_steps:
        return lr_end
    return lr_start + (lr_end - lr_start) * step / num_steps


def lr_of_new_layers(step, args):
    """Routed experts (new layers): new_layers_lr → learning_rate over new_layers_lr_steps steps. All original params stay at learning_rate."""
    return linear_then_constant(step, args.new_layers_lr, args.learning_rate, args.new_layers_lr_steps)


def lr_of_gate(step, args):
    """Gates: gate_lr → gate_lr_end, over the same number of steps as the routed experts (new_layers_lr_steps)."""
    return linear_then_constant(step, args.gate_lr, args.gate_lr_end, args.new_layers_lr_steps)


def set_learning_rates(opt, step, args):
    """Write this step's learning rates into the optimizer's three param groups. After DeepSpeed wrapping opt.param_groups are the underlying AdamW groups, so setting lr takes effect directly.
    Returns (routed-expert lr, gate lr) for logging."""
    lr_new, lr_gate = lr_of_new_layers(step, args), lr_of_gate(step, args)
    opt.param_groups[GROUP_BASE]["lr"] = args.learning_rate
    opt.param_groups[GROUP_EXPERTS]["lr"] = lr_new
    opt.param_groups[GROUP_GATE]["lr"] = lr_gate
    return lr_new, lr_gate


# =========================================================================== #
#  Checkpointing                                                              #
# =========================================================================== #
def save_trainable_weights(accelerator, dit, keys, step, args):
    """Under ZeRO-3, gather the sharded weights to rank 0 (bf16) and save only trainable params (full training = whole DiT) as safetensors. Must be called on all ranks (collective)."""
    from safetensors.torch import save_file
    state = accelerator.get_state_dict(dit)                      # rank 0 gets the full dict, the others None
    if accelerator.is_main_process:
        out_dir = os.path.join(args.output_dir, "ckpt", f"step-{step}")
        os.makedirs(out_dir, exist_ok=True)
        picked = {}
        for name, value in state.items():
            name = name[len("module."):] if name.startswith("module.") else name
            if name in keys:
                picked[name] = value.to(torch.bfloat16).cpu().contiguous()
        assert len(picked) == len(keys), f"gathered trainable params {len(picked)} != {len(keys)}"
        save_file(picked, os.path.join(out_dir, "moe_v3_1_trainable.safetensors"))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"step": step, "variant": "moe_v3_1", "experts": EXPERT_NAMES, "moe_layers": args.moe_layers, "moe_num_experts": args.moe_num_experts,
                       "moe_top_k": args.moe_top_k, "moe_aux_loss_alpha": args.moe_aux_loss_alpha,
                       "trainable": "full_dit",
                       "learning_rate": args.learning_rate, "new_layers_lr": args.new_layers_lr,
                       "new_layers_lr_steps": args.new_layers_lr_steps, "gate_lr": args.gate_lr, "gate_lr_end": args.gate_lr_end,
                       "norm_topk_weights": True}, f, indent=1)
        print(f"{TAG} saved step-{step} ({len(picked)} tensors, {sum(v.numel() for v in picked.values()) * 2 / 2**30:.1f} GB)", flush=True)
        rotate_weight_ckpts(os.path.join(args.output_dir, "ckpt"))
    del state


def rotate_weight_ckpts(ckpt_root, keep_every=1000, keep_last=2):
    """step-N dirs: those with N % 1000 == 0 are kept forever, of the rest only the latest keep_last are kept."""
    steps = [int(m.group(1)) for name in os.listdir(ckpt_root)
             for m in [re.fullmatch(r"step-(\d+)", name)] if m and os.path.isdir(os.path.join(ckpt_root, name))]
    for step in sorted(s for s in steps if s % keep_every != 0)[:-keep_last]:
        shutil.rmtree(os.path.join(ckpt_root, f"step-{step}"), ignore_errors=True)
        print(f"{TAG} rotated out step-{step}", flush=True)


def save_ds_state(accelerator, step, args):
    """Full DeepSpeed state (each of the 24 ranks writes its own shard) to ckpt/ds_state_step-N; older ones are deleted afterwards."""
    out_dir = os.path.join(args.output_dir, "ckpt", f"ds_state_step-{step}")
    accelerator.save_state(out_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(os.path.join(out_dir, "step.json"), "w") as f:
            json.dump({"step": step}, f)
        ckpt_root = os.path.join(args.output_dir, "ckpt")
        for name in os.listdir(ckpt_root):
            m = re.fullmatch(r"ds_state_step-(\d+)", name)
            if m and int(m.group(1)) < step:
                shutil.rmtree(os.path.join(ckpt_root, name), ignore_errors=True)
                print(f"{TAG} rotated out {name}", flush=True)
        print(f"{TAG} saved DeepSpeed state @ step {step}", flush=True)


def build_checked(build_fn, name, rank, tries=3):
    """After building, check that all params and buffers are finite. With 24 processes reading weights from shared storage at once, one process occasionally gets a corrupt copy
    (seen in practice as a rank whose loss was nan from step 0); if so, discard and rebuild, up to tries times."""
    for attempt in range(1, tries + 1):
        module = build_fn()
        bad = [n for n, p in module.named_parameters() if not torch.isfinite(p).all()]
        bad += [n for n, b in module.named_buffers() if b.is_floating_point() and not torch.isfinite(b).all()]
        if not bad:
            return module
        print(f"{TAG} rank{rank}: {name} has {len(bad)} non-finite tensors after loading (e.g. {bad[:2]}), rebuild attempt {attempt}", flush=True)
        del module
        gc.collect()
    raise RuntimeError(f"{name} had non-finite weights in {tries} consecutive loads, rank{rank} exiting")


def check_zero3_gather(inner, rank, world, device):
    """ZeRO-3 gather self-check: gather each block's sharded params, compute an fp32 checksum per rank, all_gather and check that the 24 values agree.
    Motivation: some ranks had nan from step 0 although their own shard weight scans were clean → suspected corrupt all-gather data on individual ranks, verified directly here."""
    import deepspeed
    import torch.distributed as dist
    bad = []
    for i, block in enumerate(inner.blocks):
        params = list(block.parameters())
        with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
            local = torch.stack([p.detach().float().sum() for p in params]).sum().reshape(1)
            finite = float(all(torch.isfinite(p).all() for p in params))
        gathered = [torch.zeros_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        values = torch.cat(gathered)
        finite_all = torch.tensor([finite], device=device)
        dist.all_reduce(finite_all, op=dist.ReduceOp.MIN)
        if finite_all.item() == 0 or (values.max() - values.min()).abs().item() > 1e-2 * (values.abs().max().item() + 1):
            bad.append((i, round(local.item(), 3), round(values.min().item(), 3), round(values.max().item(), 3), finite))
    if bad:
        print(f"{TAG} rank{rank}: ZeRO-3 gather self-check failed (block, local checksum, global min, global max, local finite): {bad[:5]}", flush=True)
        raise RuntimeError(f"rank{rank}: ZeRO-3 all-gather result inconsistent across ranks or non-finite, see previous line")
    if rank == 0:
        print(f"{TAG} ZeRO-3 gather self-check passed: {len(inner.blocks)} blocks have matching checksums on {world} ranks", flush=True)


# =========================================================================== #
#  main                                                                        #
# =========================================================================== #
class AsyncTrackerLogger:
    """Speed-up: rank0's accelerator.log (~300 scalars, wandb + tensorboard) every 10 steps blocked the main thread ~4 s each time (py-spy),
    while the other 23 GPUs waited for it at the next all_reduce → 0.4 s/step amortized. Writes are queued to a background thread, the main thread only enqueues; close() flushes the queue at the end.
    Only changes when logs are written; training numerics are unaffected."""

    def __init__(self, accelerator):
        import queue
        import threading
        self.acc, self.q, self.err = accelerator, queue.Queue(), 0
        self.thread = threading.Thread(target=self._run, name="async-tracker-log", daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                break
            scalars, step = item
            try:
                self.acc.log(scalars, step=step)
            except Exception as e:                       # logging failures must not affect training
                self.err += 1
                if self.err <= 3:
                    print(f"{TAG} async log write failed (#{self.err}): {e!r}", flush=True)
            self.q.task_done()

    def log(self, scalars, step):
        self.q.put((dict(scalars), int(step)))

    def close(self, timeout=180):
        self.q.put(None)
        self.thread.join(timeout)


def main(argv=None):
    args = parse_args(argv)
    from datetime import timedelta
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs, set_seed
    from torch.utils.data import DataLoader
    from diffusers.training_utils import compute_loss_weighting_for_sd3
    from videox_fun.data import FlowMatchScheduler
    from videox_fun.data.kiwi_video_utils import frames_to_video_tensor, uint8_video_to_pixels

    accelerator = Accelerator(mixed_precision="bf16", log_with=(["tensorboard", "wandb"] if args.use_wandb else ["tensorboard"]),
                              project_dir=args.output_dir,
                              kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60))])
    set_seed(args.seed + accelerator.process_index)
    device, dtype = accelerator.device, torch.bfloat16
    is_main = accelerator.is_main_process
    world = accelerator.num_processes
    rank = accelerator.process_index
    # Create the CUDA context on this GPU first (so the GPU shows as in use), then wait for free memory (another process may still be holding GPU memory);
    # wait at most 10 minutes, then fail fast instead of hitting OOM only after a ~13-minute model build
    torch.zeros(1, device=device)
    for _wait in range(60):
        _free, _total = torch.cuda.mem_get_info(device)
        if _free >= 60 * 2 ** 30:
            break
        if _wait % 6 == 0:
            print(f"{TAG} rank{rank}: only {_free / 2 ** 30:.1f} GiB free on this GPU (<60), waiting for another process to release GPU memory... ({_wait * 10}s)", flush=True)
        time.sleep(10)
    else:
        raise RuntimeError(f"rank{rank}: still only {_free / 2 ** 30:.1f} GiB free on this GPU after 10 minutes; another process may be holding GPU memory")

    # ---- Model: DiT goes into DeepSpeed (ZeRO-3 sharded); MLLM / VAE are frozen, one copy per GPU, outside DeepSpeed
    # Building briefly takes ~120 GB of host memory per process (fp32, then cast to bf16); the 8 local ranks start 15 s apart so the peaks do not stack
    time.sleep(int(os.environ.get("LOCAL_RANK", 0)) * 15)
    dit = build_checked(lambda: build_dit(args, dtype, verbose=is_main), "DiT", rank)
    if args.resume_weights:
        load_trainable_weights(dit, args.resume_weights)
        if is_main:
            print(f"{TAG} loaded weights {args.resume_weights}", flush=True)
    base_params, expert_params, gate_params = mark_trainable(dit)
    keys = trainable_state_keys(dit)
    # Three param groups, order = GROUP_BASE / GROUP_EXPERTS / GROUP_GATE; per-step learning rates are written by set_learning_rates
    opt = torch.optim.AdamW([{"params": base_params, "lr": args.learning_rate},
                             {"params": expert_params, "lr": args.new_layers_lr},
                             {"params": gate_params, "lr": args.gate_lr}], weight_decay=args.weight_decay)
    # Speed-up: mark each MoE layer's MoEFFN / MoEKVSelfAttention as a ZeRO-3 leaf module. By default ZeRO-3 attaches a pair of forward hooks to every submodule (7 experts × several Linear + gate)
    # and gathers / releases params one by one, thousands of small gathers per step; as a leaf, all its params are gathered once on entry (DeepSpeed's recommended usage for MoE).
    # The math is unchanged (only when params are gathered); must be called before accelerator.prepare (DeepSpeed installs its hooks based on this flag at prepare time).
    if args.z3_leaf_modules:      # off by default
        try:
            from deepspeed.utils import set_z3_leaf_modules
            from wan_transformer3d_kiwi_moe_v3_1 import MoEFFN, MoEKVSelfAttention
            leaf_modules = set_z3_leaf_modules(dit, [MoEFFN, MoEKVSelfAttention])
            if is_main:
                print(f"{TAG} ZeRO-3 leaf modules: {len(leaf_modules)} (MoEFFN + MoEKVSelfAttention, one each per MoE layer)", flush=True)
        except ImportError as e:
            if is_main:
                print(f"{TAG} Warning: deepspeed.utils.set_z3_leaf_modules unavailable ({e}); MoE modules keep per-submodule ZeRO-3 hooks", flush=True)
    dit, opt = accelerator.prepare(dit, opt)                     # DeepSpeed takes over here: weights / grads / optimizer states sharded over 24 GPUs
    inner = accelerator.unwrap_model(dit)
    check_zero3_gather(inner, rank, world, device)
    # MLLM: build → move to GPU → scan weights → self-test output finiteness with a fixed instruction + blank frames. One rank once passed the weight scan but produced nan
    # (the other 23 ranks were fine on the same input), i.e. a one-off bad state of that process, so the whole MLLM is rebuilt and retried, up to 3 times.
    from PIL import Image
    probe_frames = [Image.new("RGB", (args.width, args.height))] * args.num_frames
    for attempt in range(1, 4):
        mllm = build_checked(lambda: build_mllm(args).to(device), "MLLM(GPU)", rank)
        text_encoder = TextEncoder(mllm)
        where = text_encoder.self_test("Add a red ball.", probe_frames)
        if where is None:
            print(f"{TAG} rank{rank}: MLLM self-test passed (attempt {attempt})", flush=True)
            break
        print(f"{TAG} rank{rank}: MLLM self-test failed, first non-finite at [{where}], attempt {attempt}, rebuilding", flush=True)
        del mllm, text_encoder
        gc.collect()  # use the module-level gc (imported at the top); a local import gc here would make gc a local of main(), and the normal path (self-test passes first time) would raise UnboundLocalError at gc.freeze() below
        torch.cuda.empty_cache()
    else:
        raise RuntimeError(f"rank{rank}: MLLM self-test failed 3 times in a row, exiting")
    vae = build_checked(lambda: build_vae(args, dtype).to(device), "VAE(GPU)", rank)
    if args.vae_compile != "off":      # speed-up: torch.compile the VAE encoder (frozen encoder only, same math; the first step spends tens of seconds compiling; falls back to eager on failure)
        vae.model._eager_encoder = vae.model.encoder      # fallback if compilation fails at runtime (see vae_encode_batched)
        try:
            if args.vae_compile == "encoder":
                vae.model.encoder = torch.compile(vae.model.encoder, dynamic=False)
            elif args.vae_compile == "encoder_dyn":      # dynamic shapes: the VAE batch size varies (guard drops / only misses encoded with the cache), avoids recompiling per batch size
                vae.model.encoder = torch.compile(vae.model.encoder, dynamic=True)
            elif args.vae_compile == "encoder_autotune":
                vae.model.encoder = torch.compile(vae.model.encoder, dynamic=False, mode="max-autotune-no-cudagraphs")
            elif args.vae_compile == "encode":
                vae.model.encode = torch.compile(vae.model.encode, dynamic=False)
            if is_main:
                print(f"{TAG} VAE torch.compile: {args.vae_compile}", flush=True)
        except Exception as e:
            if is_main:
                print(f"{TAG} Warning: VAE torch.compile({args.vae_compile}) failed, using eager: {e!r}", flush=True)
    scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=1000, training=True)

    # ---- Feature cache dir for multi-edit samples (see module docstring). The dataset (kiwi_dataset_multi_moe) takes it via a constructor kwarg; if its signature lacks one → disable and warn;
    #      rank0 first renames + deletes in the background any dir left by a previous SIGKILL (the cache is not reused across launches), then creates an empty dir; the other ranks wait for it
    from videox_fun.moe_common.kiwi_dataset_multi_moe import KiwiMultiEditDataset
    cache_dir = args.multi_cache_dir.strip()
    cache_kwargs = dataset_cache_kwargs(KiwiMultiEditDataset, cache_dir)
    if cache_dir and cache_kwargs is None:
        if is_main:
            print(f"{TAG} cache: Warning: KiwiMultiEditDataset constructor has no multi_cache_dir / cache_dir kwarg (no cache support in the dataset), --multi_cache_dir ignored, no caching in this run", flush=True)
        cache_dir, cache_kwargs = "", {}
    cache_writer = None
    if cache_dir:
        if is_main:
            if args.cache_keep_existing:                # keep the cache on restart, do not clear leftovers (the hit ratio continues from the last run)
                os.makedirs(cache_dir, exist_ok=True)
                print(f"{TAG} cache: keeping existing cache dir {cache_dir} (--cache_keep_existing 1), the files count accumulates from this launch", flush=True)
            else:
                stale, _ = remove_cache_dir(cache_dir, wait_seconds=0)
                if stale:
                    print(f"{TAG} cache: leftover cache dirs renamed to {stale}, deleting in the background", flush=True)
                os.makedirs(cache_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        cache_writer = MultiCacheWriter(cache_dir, rank, threads=args.cache_write_threads, maxsize=args.cache_queue)
        if is_main:
            print(f"{TAG} cache: dir {cache_dir}  writer threads {args.cache_write_threads}/rank  queue {args.cache_queue}  "
                  f"loader workers={args.dataset_num_workers} prefetch={args.dataloader_prefetch_factor}", flush=True)

    # ---- Data: one dataset / sampler / loader per stage
    exclude_words = [w.strip() for w in args.exclude_prompt_words.split(",") if w.strip()]
    stages = []
    paths_a = [s for s in args.stage_a_data.split(",") if s]
    paths_b = [s for s in args.stage_b_data.split(",") if s]
    if args.data_schedule == "mixed":      # one stage: 1 copy of OpenVE + multi_repeat copies each of goku / CoinVE, shuffled together
        plan = [("MIX", paths_a + paths_b, [1] * len(paths_a) + [args.multi_repeat] * len(paths_b), args.stage_a_epochs,
                 [None] * len(paths_a) + ["multi"] * len(paths_b))]      # records from paths_b (goku / CoinVE) get cache_group="multi"
    else:                                  # two stages: A then B (the mlp-moe schedule; "exactly one epoch of OpenVE, then one of multi" = stage_b_epochs 1)
        plan = [("A", paths_a, None, args.stage_a_epochs, None), ("B", paths_b, None, args.stage_b_epochs, ["multi"] * len(paths_b))]
    built = []
    for name, paths, repeats, epochs, cache_groups in plan:
        dataset, stats = build_dataset(paths, exclude_words, args, repeats, cache_groups, cache_kwargs)
        built.append((name, dataset, stats, epochs))
    # Before building the DataLoader, check the in-flight memory / /dev/shm budget (workers × prefetch × batch × ~118 MB per sample × local processes); raise above 60%
    check_loader_memory(args, accelerator.is_local_main_process)
    # Speed-up: after building the datasets, run a full GC and then freeze all live objects (remove them from the generational GC scans).
    # Why: 3.58M records (mixed mode incl. copies) are ~11M Python dicts/lists/strs that stay alive until training ends;
    # the main process ran a full GC roughly every 48 steps, each traversing these 11M objects (measured 4 s), and they never become garbage during training.
    # After freezing, GC only scans objects created during training; done before the DataLoader forks its workers, so the workers skip these objects too (fewer copy-on-write page touches).
    gc.collect()
    gc.freeze()
    for name, dataset, stats, epochs in built:
        sampler = ShardedBatchSampler(len(dataset), args.batch_size, world, rank, args.seed)
        loader = DataLoader(IndexTaggedDataset(dataset), batch_sampler=sampler, num_workers=args.dataset_num_workers,
                            prefetch_factor=args.dataloader_prefetch_factor or None, collate_fn=lambda b: b,
                            persistent_workers=bool(args.dataset_num_workers))
        steps = int(math.ceil(len(sampler) * epochs))
        stages.append({"name": name, "sampler": sampler, "loader": loader, "steps": steps, "stats": stats, "size": len(dataset)})
        if is_main:
            print(f"{TAG} stage {name}: {stats} total={len(dataset)} {len(sampler)} steps/epoch × {epochs} epochs = {steps} steps", flush=True)
    stage_start = [0, stages[0]["steps"]]                          # global start step of each stage (with a single stage stage_start[1] = total steps, so always stage 0)
    max_steps = args.max_train_steps or sum(st["steps"] for st in stages)

    # ---- resume
    step0 = 0
    if args.resume_ds_state:
        accelerator.load_state(args.resume_ds_state)
        with open(os.path.join(args.resume_ds_state, "step.json")) as f:
            step0 = int(json.load(f)["step"])
        if is_main:
            print(f"{TAG} resume DeepSpeed state @ step {step0}", flush=True)

    if is_main:
        os.makedirs(os.path.join(args.output_dir, "ckpt"), exist_ok=True)
        print(f"{TAG} start: world={world} batch={args.batch_size}(effective {world * args.batch_size}) steps={max_steps} "
              f"moe_layers={args.moe_layers} experts={args.moe_num_experts} top_k={args.moe_top_k} aux={args.moe_aux_loss_alpha} "
              f"lr base={args.learning_rate} experts={args.new_layers_lr}→{args.learning_rate} gate={args.gate_lr}→{args.gate_lr_end}(linear over the first {args.new_layers_lr_steps} steps) "
              f"text_drop={args.text_drop_prob} motion={args.motion_sub_loss_ratio} "
              f"cache_dir={cache_dir or 'off'} workers={args.dataset_num_workers} prefetch={args.dataloader_prefetch_factor}",
              flush=True)
    accelerator.init_trackers(args.wandb_project, config=vars(args),
                              init_kwargs={"wandb": {"name": args.wandb_run_name, "id": args.wandb_run_name, "resume": "allow", "dir": args.output_dir}})
    async_logger = AsyncTrackerLogger(accelerator) if (is_main and args.async_log) else None   # speed-up: only rank0 actually writes logs
    pbar = None
    if is_main:
        from tqdm import tqdm
        pbar = tqdm(total=max_steps, initial=step0, desc="moe-v3.1", dynamic_ncols=True)

    def pixels_of(sample, pixel_key, frames_key):
        if sample.get(pixel_key) is not None:
            return uint8_video_to_pixels(sample[pixel_key], device, dtype)
        return frames_to_video_tensor(sample[frames_key], dtype=torch.float32).to(device, dtype)

    moe_ffns, moe_attns = inner.moe_ffns(), inner.moe_attns()
    moe_blocks = [int(b) for b in args.moe_layers.split(",")]          # aligned with moe_ffns / moe_attns, used for log labels
    num_experts = args.moe_num_experts
    # Stat vector reduced across ranks every 10 steps: [sum loss_main, sum loss_motion, sum aux_attn, sum aux_ffn, sum total loss (weighted main/motion + aux), count]
    # followed by the summed per-layer per-expert "selected fraction" of the self-attention gates (layer × expert), then the same block for the FFN gates
    NUM_LOSS_SLOTS = 6
    per_moe = len(moe_ffns) * num_experts
    cache_off = NUM_LOSS_SLOTS + 2 * per_moe                                  # cache counter slots start here (appended at the end; NUM_LOSS_SLOTS and the MoE offsets are unchanged)
    stat = torch.zeros(cache_off + len(CACHE_STAT_NAMES), device=device)
    cache_expect = {"latent": None, "ctx": None}                              # latent shape / ctx dim recorded at the first VAE / MLLM encode of this run; cache hits must match
    # "category × expert" selection counts (summed over all MoE layers): [kind=2 (attn, ffn), category, expert]; every 10 steps reduced and row-normalized to P(expert i selected | sample category c)
    moe_kinds = [("attn", moe_attns), ("ffn", moe_ffns)]
    cat_stat = torch.zeros(len(moe_kinds), len(CATEGORIES), num_experts, device=device)
    timing = {"vae": 0.0, "mllm": 0.0, "dit": 0.0, "n": 0}      # this rank's time per stage (s), averaged and logged every 10 steps
    skipped_steps = 0                   # steps skipped due to bad samples / non-finite loss (not counted in step)
    consecutive_bad = 0                 # consecutive steps with non-finite loss on this rank; 5 in a row means this process's weights are bad, exit for inspection
    step, t0 = step0, time.time()
    dit.train()

    while step < max_steps:
        # ---- current stage, epoch within the stage and consumed steps (resume goes through here too, so the data order continues exactly)
        stage_index = 1 if (len(stages) > 1 and step >= stage_start[1]) else 0
        stage = stages[stage_index]
        in_stage = step - stage_start[stage_index]
        steps_per_epoch = len(stage["sampler"])
        stage["sampler"].set_epoch(in_stage // steps_per_epoch)
        stage["sampler"].skip = in_stage % steps_per_epoch
        stage_end = min(max_steps, stage_start[stage_index] + stage["steps"])
        if is_main:
            print(f"{TAG} stage {stage['name']} epoch {in_stage // steps_per_epoch} resuming at step {in_stage % steps_per_epoch}, until global step {stage_end}", flush=True)

        check_pos = in_stage % steps_per_epoch      # data-position self-check: the epoch's first batch must be the sampler's batch check_pos (each rank checks its own)
        for batch in stage["loader"]:
            if check_pos is not None:
                expect = stage["sampler"].epoch_batches()[check_pos]
                got = [s.get("_req_index") for s in batch]
                if got != expect:
                    raise RuntimeError(f"{TAG} rank{rank} data-position self-check failed: stage {stage['name']} epoch {stage['sampler'].epoch} expected batch {check_pos} of this epoch "
                                       f"{expect[:4]}..., got {got[:4]}... (sampler skip swallowed? see the ShardedBatchSampler.__iter__ comment)")
                if is_main:
                    print(f"{TAG} data-position self-check OK: stage {stage['name']} epoch {stage['sampler'].epoch} first batch = batch {check_pos} of this epoch (sample indices {expect[:3]}...)", flush=True)
                check_pos = None
            # ---- 0. Cache split. Hits (the dataset worker read a cache file, "cache" key present) have no pixels, only VAE distribution params + both ctx;
            #      payloads[i] = hit tensors / None, pending[i] = miss in a cache group (written after this step) / None; both filtered in sync with guard drops
            batch, payloads, pending, cache_counts = split_batch_for_cache(batch, cache_dir if cache_writer is not None else "", args, cache_expect, rank, step)
            # ---- 1. VAE encode: source / target videos go to the encoder in groups of vae_encode_group (per-sample serial encoding was the main per-step cost, see vae_encode_batched)
            #      cache: only misses are encoded (collecting mean / logvar into pending); hits are resampled from the cached params with the same formula; reassembled in batch order
            t_mark = sync_time()
            with torch.no_grad():
                src_latents, latents = encode_latents_with_cache(vae, batch, payloads, pending, args, device, dtype, pixels_of, cache_expect)   # each [B, C, F, H, W]
            timing["vae"] += sync_time() - t_mark

            # ---- Guard 1 (local part): input finiteness. Videos occasionally decode to nan/inf, which would permanently corrupt the weights through the gradients (DeepSpeed bf16 has no fp16-style overflow step skipping).
            #      Checked per sample; bad samples are dropped from the batch with their paths printed; if no good sample remains on this rank, all ranks skip this step together
            #      (collectives require all 24 ranks in lockstep; a single rank cannot skip backward alone).
            #      Speed-up: guard 1 (here) and guard 1b (after the MLLM) used to each do an accelerator.reduce().item(), each a full 24-GPU sync
            #      plus waiting for the slowest rank, ~2 s for both. Now only the local flag bad_latent is computed and samples are dropped locally here, then MLLM encoding and a local bad_ctx,
            #      and finally both flags are reduced together in one vector (see "guards 1 + 1b merged" below). The decision semantics are unchanged:
            #      any rank with an all-bad latent batch → all skip (consecutive_bad untouched); otherwise any rank with an all-bad ctx batch → all skip (consecutive_bad as before).
            #      Only difference: on a step where some rank's latent batch is all bad, the other ranks have already done noise sampling and MLLM encoding before learning to skip (wasted work, very rare).
            keep = []
            for i, sample in enumerate(batch):
                if torch.isfinite(latents[i]).all() and torch.isfinite(src_latents[i]).all():
                    keep.append(i)
                else:
                    print(f"{TAG} rank{rank} step {step}: dropping bad sample (non-finite latent) src={sample.get('src_video_path', '?')} "
                          f"tgt={sample.get('tgt_video_path', '?')} prompt={sample['prompts_multi'][0][:80]!r}", flush=True)
            bad_latent = len(keep) == 0                 # whole latent batch bad on this rank (guard 1 local flag; the global decision is deferred to the merged reduce)
            if len(keep) < len(batch):
                batch = [batch[i] for i in keep]
                src_latents, latents = src_latents[keep], latents[keep]
                payloads, pending = [payloads[i] for i in keep], [pending[i] for i in keep]      # cache: filter the per-sample state in sync
            bsz = latents.size(0)                       # 0 if the whole latent batch is bad: noising below yields empty tensors, MLLM is skipped, all ranks continue after the merged reduce

            # ---- 2. flow-match noising (independent timestep / sigma per sample)
            noise = torch.randn_like(latents)
            timestep_index = torch.randint(0, scheduler.num_train_timesteps, (bsz,))
            sigmas = scheduler.sigmas[timestep_index].to(device=device, dtype=latents.dtype).view(bsz, 1, 1, 1, 1)
            timesteps = scheduler.timesteps[timestep_index].to(device=device, dtype=dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            target = noise - latents


            # ---- 3. full instruction → MLLM → ctx; text-CFG draw
            # (under DeepSpeed accelerate does not wrap forward in autocast, so bf16 autocast is applied explicitly here:
            #  the original kiwi forward feeds fp32 inputs to bf16 weights in places such as time_embedding and needs autocast; the inference script does the same)
            #  cache: the draw / encoding / cached ctx for hits / extra branch for missed multi samples are all in encode_ctx_with_cache (see its docstring):
            #  the draw still calls torch.rand(()) once per sample in batch order (identical RNG consumption to both the batched path and the old per-sample path);
            #  batch has already been filtered locally by guard 1, and ctx_list is aligned with batch.
            t_mark = sync_time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx_list = encode_ctx_with_cache(text_encoder, batch, payloads, pending, args, device, dtype, cache_expect) if bsz > 0 else []
            timing["mllm"] += sync_time() - t_mark
            # ---- Guard 1b (local part): ctx finiteness (a nan MLLM output makes everything nan from block0's cross-attn on)
            keep = [i for i, ctx in enumerate(ctx_list) if torch.isfinite(ctx).all()]
            for i, sample in enumerate(batch):
                if i not in keep:
                    print(f"{TAG} rank{rank} step {step}: dropping bad sample (non-finite MLLM ctx) src={sample.get('src_video_path', '?')} "
                          f"prompt={sample['prompts_multi'][0][:80]!r}", flush=True)
            bad_ctx = len(keep) == 0 and not bad_latent   # good latents but the whole ctx batch is bad (guard 1b local flag; with an all-bad latent batch ctx_list is empty and guard 1 handles it)
            # ---- Guards 1 + 1b merged into one reduce: vector [some rank has all-bad latents, some rank has all-bad ctx], one all-reduce + one sync
            bad_flags = accelerator.reduce(torch.tensor([float(bad_latent), float(bad_ctx)], device=device), reduction="sum")
            any_bad_latent, any_bad_ctx = (bad_flags > 0).tolist()
            if any_bad_latent:                              # = guard 1's global branch (consecutive_bad untouched)
                skipped_steps += 1
                if is_main:
                    print(f"{TAG} step {step}: a rank has an all-bad batch, all ranks skip this step (skipped so far: {skipped_steps})", flush=True)
                continue
            if any_bad_ctx:                                 # = guard 1b's global branch
                skipped_steps += 1
                consecutive_bad = consecutive_bad + 1 if len(keep) == 0 else 0
                if consecutive_bad >= 5:
                    raise RuntimeError(f"rank{rank}: MLLM ctx all non-finite for {consecutive_bad} consecutive steps, this GPU is faulty, exiting")
                if is_main:
                    print(f"{TAG} step {step}: a rank has an all non-finite ctx batch, all ranks skip this step (skipped so far: {skipped_steps})", flush=True)
                continue
            if len(keep) < len(batch):
                batch = [batch[i] for i in keep]
                ctx_list = [ctx_list[i] for i in keep]
                src_latents, latents = src_latents[keep], latents[keep]
                noise, noisy_latents, target = noise[keep], noisy_latents[keep], target[keep]
                sigmas, timesteps = sigmas[keep], timesteps[keep]
                pending = [pending[i] for i in keep]                  # cache: only samples passing guards 1 + 1b are written
                bsz = latents.size(0)
            # ---- Cache: for missed multi samples, move the distribution params + both ctx to CPU (bf16) and hand them to the writer threads (dropped if the queue is full, failures only counted, never blocking training);
            #      the 6 tensors are re-checked for finiteness before enqueueing (guard 1b did not check the extra ctx branch; non-finite ones are not written), see enqueue_cache_payloads
            if cache_writer is not None:
                enqueue_cache_payloads(cache_writer, pending, args, step, rank)

            # ---- 4. DiT forward (load-balancing losses are attached to the graph inside the MoE layers), under bf16 autocast
            t_mark = sync_time()
            patch = inner.patch_size
            f_lat, h_lat, w_lat = latents.shape[2:]
            seq_len = (f_lat // patch[0]) * (h_lat // patch[1]) * (w_lat // patch[2])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = dit(x=[noisy_latents[i] for i in range(bsz)], t=timesteps, mllm_context=ctx_list, seq_len=seq_len,
                           vae_source_input=src_latents, sigma=sigmas.reshape(bsz))

            # ---- 5. loss: videoxfun wan2.2 loss + motion difference term (fp32)
            weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
            loss_main = custom_mse_loss(pred, target, weighting.float())
            loss = loss_main
            loss_motion = None
            if args.motion_sub_loss_ratio > 0 and pred.size(2) > 2:
                loss_motion = custom_mse_loss(pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:] - target[:, :, :-1], weighting.float())
                loss = loss * (1 - args.motion_sub_loss_ratio) + loss_motion * args.motion_sub_loss_ratio
            aux_attn_values = [m.last_aux_loss for m in moe_attns if m.last_aux_loss is not None]
            aux_ffn_values = [m.last_aux_loss for m in moe_ffns if m.last_aux_loss is not None]
            loss_aux_attn = torch.stack(aux_attn_values).sum() if aux_attn_values else torch.zeros((), device=device)
            loss_aux_ffn = torch.stack(aux_ffn_values).sum() if aux_ffn_values else torch.zeros((), device=device)
            loss_aux = loss_aux_attn + loss_aux_ffn

            # ---- Guard 2: loss finiteness (any rank non-finite → all skip, no backward, weights untouched)
            bad_local = torch.tensor(float(not torch.isfinite(loss.detach())), device=device)
            consecutive_bad = consecutive_bad + 1 if bad_local.item() > 0 else 0
            if consecutive_bad >= 5:
                raise RuntimeError(f"rank{rank}: non-finite loss for {consecutive_bad} consecutive steps (with different samples), this process's weights/environment are faulty, exiting")
            if accelerator.reduce(bad_local, reduction="sum").item() > 0:
                skipped_steps += 1
                if bad_local.item() > 0:
                    print(f"{TAG} rank{rank} step {step}: non-finite loss main={float(loss_main):.4g} aux={float(loss_aux):.4g} "
                          f"batch samples src={[s_.get('src_video_path', '?') for s_ in batch]} tgt={[s_.get('tgt_video_path', '?') for s_ in batch]}", flush=True)
                if is_main:
                    print(f"{TAG} step {step}: a rank has non-finite loss, all ranks skip this step (skipped so far: {skipped_steps})", flush=True)
                del pred, loss
                continue

            # ---- 6. backward + update (accelerate's DeepSpeed wrapper also steps inside backward; clipping is in the ds config)
            lr_new, lr_gate = set_learning_rates(opt, step, args)   # this step's learning rates (routed experts / gates transition linearly with step, the rest constant)
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
            timing["dit"] += sync_time() - t_mark
            timing["n"] += 1
            step += 1

            # ---- 7. stats and logging
            stat[0] += float(loss_main.detach()); stat[1] += float(loss_motion.detach()) if loss_motion is not None else 0.0
            stat[2] += float(loss_aux_attn); stat[3] += float(loss_aux_ffn); stat[4] += float(loss.detach()) + float(loss_aux); stat[5] += 1
            if cache_writer is not None:                                    # cache: hit / miss / other + writer counts, appended at the end of stat
                stat[cache_off:cache_off + len(CACHE_STAT_NAMES)] += torch.tensor(cache_counts + list(cache_writer.drain_counts()), device=device, dtype=stat.dtype)
                # count_files_async() is no longer run every 100 steps: scandir over ~70k entries on network storage takes 100+ s, and the per-entry Python loop competes with the main thread for the GIL,
                # causing 110-140 s spikes every 100 steps, worse as the dir grows (~5 s/step at 220k entries). The file count is approximated by the cumulative write count of all ranks instead.
            for layer_i, (attn, ffn) in enumerate(zip(moe_attns, moe_ffns)):   # per layer per expert: token mean of 1 if selected / 0 otherwise
                start = NUM_LOSS_SLOTS + layer_i * num_experts
                stat[start:start + num_experts] += attn.last_selected
                stat[per_moe + start:per_moe + start + num_experts] += ffn.last_selected
            # category × expert: per-sample selection counts of each expert (all tokens, all slots) accumulated by sample category
            cat_idx = torch.tensor([CATEGORY_INDEX.get((s_.get("turn_types_qwen") or ["unknown"])[0], CATEGORY_INDEX["unknown"]) for s_ in batch], device=device)
            for kind_i, (_, modules) in enumerate(moe_kinds):
                for m in modules:
                    per_sample = F.one_hot(m.last_topk_idx.view(bsz, -1), num_experts).sum(dim=1).float()     # [B, N]
                    cat_stat[kind_i].index_add_(0, cat_idx, per_sample)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{float(loss.detach()):.4f}", aux=f"{float(loss_aux):.4f}", refresh=False)
            if step % 10 == 0:
                totals = accelerator.reduce(stat.clone(), reduction="sum").tolist()
                cat_totals = accelerator.reduce(cat_stat.clone(), reduction="sum")            # [kind, category, expert]
                if is_main:
                    count = max(totals[5], 1)                                       # = 10 steps × 24 GPUs
                    scalars = {"loss/main": totals[0] / count, "loss/motion": totals[1] / count,
                               "loss/aux_attn": totals[2] / count, "loss/aux_ffn": totals[3] / count, "loss/aux": (totals[2] + totals[3]) / count,
                               "loss/total": totals[4] / count,
                               "lr/base": args.learning_rate, "lr/new_layers": lr_new, "lr/gate": lr_gate,
                               "skipped_steps": skipped_steps, "stage": stage_index}
                    # one folder per MoE layer (moe_attn_L{block} / moe_ffn_L{block}) with one curve per expert: fraction of tokens selected
                    summary = {}
                    for kind, offset in (("attn", NUM_LOSS_SLOTS), ("ffn", NUM_LOSS_SLOTS + per_moe)):
                        selected_by_layer = [totals[offset + layer_i * num_experts: offset + (layer_i + 1) * num_experts]
                                             for layer_i in range(len(moe_ffns))]
                        for block, per_expert in zip(moe_blocks, selected_by_layer):
                            for i, value in enumerate(per_expert):
                                scalars[f"moe_{kind}_L{block:02d}/expert{i:02d}"] = value / count
                        all_selected = [v / count for per_expert in selected_by_layer for v in per_expert]
                        scalars[f"moe/{kind}_selected_min"] = min(all_selected)
                        scalars[f"moe/{kind}_selected_max"] = max(all_selected)
                        summary[kind] = (min(all_selected), max(all_selected),
                                         [sum(per_expert[i] for per_expert in selected_by_layer) / (count * len(selected_by_layer)) for i in range(num_experts)])
                    # category × expert conditional selection rate P(expert i | category c) = selections of expert i in category c / all selections in c; specialization index = max gap across categories per expert
                    spread_note = []
                    for kind_i, (kind, _) in enumerate(moe_kinds):
                        rows = cat_totals[kind_i]                                             # [category, expert]
                        present = rows.sum(dim=1) > 0
                        probs = rows / rows.sum(dim=1, keepdim=True).clamp_min(1)
                        for c_i, cat_name in enumerate(CATEGORIES):
                            if present[c_i]:
                                for i in range(num_experts):
                                    scalars[f"route_{kind}/{cat_name}/expert{i:02d}"] = float(probs[c_i, i])
                        if int(present.sum()) >= 2:
                            spread = float((probs[present].max(dim=0).values - probs[present].min(dim=0).values).mean())
                            scalars[f"route/{kind}_spread"] = spread
                            spread_note.append(f"{kind}_spread={spread:.3f}")
                    if timing["n"] > 0:
                        for key in ("vae", "mllm", "dit"):
                            scalars[f"time/{key}"] = timing[key] / timing["n"]
                    grad_norm = dit.get_global_grad_norm() if hasattr(dit, "get_global_grad_norm") else None
                    if grad_norm is not None:
                        scalars["grad/norm"] = float(grad_norm)
                    cache_note = ""
                    if cache_writer is not None:                            # cache: hit ratio (hits / multi-sample visits), fraction of all samples, writes / drops / failures, file count, this rank's queue
                        c_hit, c_miss, c_other, c_ok, c_fail, c_drop = totals[cache_off:cache_off + len(CACHE_STAT_NAMES)]
                        cache_writer.files = int(getattr(cache_writer, "files_est", 0) + c_ok); cache_writer.files_est = cache_writer.files   # cumulative writes ≈ file count (no directory scan)
                        scalars["cache/hit_ratio"] = c_hit / max(c_hit + c_miss, 1)
                        scalars["cache/hit_frac"] = c_hit / max(c_hit + c_miss + c_other, 1)
                        scalars["cache/hits"], scalars["cache/misses"] = c_hit, c_miss
                        scalars["cache/writes"], scalars["cache/write_fail"], scalars["cache/write_drop"] = c_ok, c_fail, c_drop
                        scalars["cache/queue"] = cache_writer.qsize()
                        if cache_writer.files >= 0:
                            scalars["cache/files"] = cache_writer.files
                        cache_note = (f"cache={100 * c_hit / max(c_hit + c_miss, 1):.0f}% (h/m/o={c_hit:.0f}/{c_miss:.0f}/{c_other:.0f}) "
                                      f"w={c_ok:.0f} fail={c_fail:.0f} drop={c_drop:.0f} files={cache_writer.files} q={cache_writer.qsize()}")
                    if async_logger is not None:
                        async_logger.log(scalars, step)          # speed-up: written by a background thread, the main thread does not block
                    else:
                        accelerator.log(scalars, step=step)
                    rate = (step - step0) * world * args.batch_size / max(time.time() - t0, 1e-6)
                    print(f"{TAG} step {step}/{max_steps} stage={stage['name']} loss={totals[0] / count:.4f} aux_attn={totals[2] / count:.4f} aux_ffn={totals[3] / count:.4f} "
                          f"lr_new={lr_new:.2e} lr_gate={lr_gate:.2e} "
                          f"attn sel {summary['attn'][0]:.2f}~{summary['attn'][1]:.2f} per-expert {'/'.join(f'{v:.2f}' for v in summary['attn'][2])} | "
                          f"ffn sel {summary['ffn'][0]:.2f}~{summary['ffn'][1]:.2f} per-expert {'/'.join(f'{v:.2f}' for v in summary['ffn'][2])} "
                          f"{rate:.1f}rec/s eta={(max_steps - step) / max(rate, 1e-6) * world * args.batch_size / 3600:.1f}h "
                          f"t(vae/mllm/dit)={timing['vae'] / max(timing['n'], 1):.1f}/{timing['mllm'] / max(timing['n'], 1):.1f}/{timing['dit'] / max(timing['n'], 1):.1f}s "
                          + " ".join(spread_note) + (" " + cache_note if cache_note else ""), flush=True)
                stat.zero_()
                cat_stat.zero_()
                timing.update(vae=0.0, mllm=0.0, dit=0.0, n=0)
            if step % args.save_steps == 0:
                save_trainable_weights(accelerator, dit, keys, step, args)
            if step % args.ds_save_steps == 0:
                save_ds_state(accelerator, step, args)
            if step >= stage_end:
                break

    save_trainable_weights(accelerator, dit, keys, step, args)
    if step % args.ds_save_steps != 0:
        save_ds_state(accelerator, step, args)
    if is_main:
        final_dir = os.path.join(args.output_dir, "ckpt", "final_components")
        shutil.rmtree(final_dir, ignore_errors=True)
        shutil.copytree(os.path.join(args.output_dir, "ckpt", f"step-{step}"), final_dir)
        if pbar is not None:
            pbar.close()
    if cache_writer is not None:                         # cache: all ranks drain the write queue → barrier → rank0 deletes the cache dir (the cache must be removed when training ends)
        cache_writer.close()
        accelerator.wait_for_everyone()
        if is_main and not args.cache_delete_on_done:   # no auto-delete at the end; delete manually
            with open(os.path.join(args.output_dir, "flags", "CACHE_KEPT_ASK_USER"), "w") as f:
                f.write(f"{time.strftime('%F %T')} training finished, cache dir {cache_dir} kept; decide manually whether to delete it\n")
            print(f"{TAG} cache: training finished, cache dir {cache_dir} kept (--cache_delete_on_done 0); decide manually whether to delete it", flush=True)
        elif is_main:
            t_rm = time.time()
            targets, done = remove_cache_dir(cache_dir, wait_seconds=900)
            print(f"{TAG} cache: deleting cache dirs {targets} {'done' if done else 'not finished after 15 min, the launcher EXIT trap deletes the rest in the background'}, took {time.time() - t_rm:.0f}s", flush=True)
    if async_logger is not None:
        async_logger.close()                             # flush the remaining queued logs before shutdown
    accelerator.end_training()
    if is_main:
        print(f"{TAG} DONE @ step {step}", flush=True)


if __name__ == "__main__":
    main()
