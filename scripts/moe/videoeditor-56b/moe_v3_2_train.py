#!/usr/bin/env python
"""moe_v3_2 training script: DeepSpeed ZeRO-3 full training of the whole DiT (self-attention k/v and FFN are both 35-expert top-3 MoE), 3 nodes x 24 GPUs.

The model is not defined here: DiT structure in wan_transformer3d_kiwi_moe_v3_2.py, assembly / weight loading / MLLM / VAE in moe_v3_2_model.py.
This file only has: data (two stages) -> batching -> noising -> forward -> loss -> update -> save / resume; the loss is inlined in the loop as in videoxfun wan2.2.

Data schedule (--data_schedule, default mixed):
  mixed      1 copy of OpenVE + --multi_repeat (2) copies each of goku / CoinVE, shuffled into one dataset, run for stage_a_epochs epochs (same total samples as two_stage, only the order is mixed)
  two_stage  stage A --stage_a_data (OpenVE, single instruction) for stage_a_epochs epochs, then stage B --stage_b_data (goku + CoinVE, multi-instruction) for stage_b_epochs epochs (the old mlp-moe schedule)
  --exclude_prompt_words drops samples containing the given words; the sub-instructions of multi-instruction samples are joined into one full instruction with " Then " (CoinVE) or GOKU wording
Each step:
  1. take a batch of B samples (random, no grouping)
  2. VAE-encode source / target videos -> per-sample sigma and noise;  noisy = (1-sigma)*tgt + sigma*noise;  target = noise - tgt
  3. MLLM-encode the full instruction -> ctx; each sample is replaced by the empty-instruction ctx with probability text_drop_prob (unconditional branch of text CFG)
     (default --mllm_batch_encode 1: the whole batch goes through the MLLM at once; source PIL frames are rebuilt from src_pixels in the main process, see rebuild_src_frames)
  4. DiT forward (the MoE load-balancing loss is already attached to the graph, see wan_transformer3d_kiwi_moe_v3_2.AddAuxiliaryLoss; one for self-attention and one for FFN)
  5. loss = custom_mse_loss(pred, target) (+ motion difference term); backward sends the aux-loss gradient into the gates automatically
  6. DeepSpeed step (gradient clipping in the ds config); full update. Three learning-rate groups (see the learning-rate section):
       original params        always learning_rate
       routed experts (new)   start at new_layers_lr, linearly down to learning_rate within new_layers_lr_steps steps, then constant
       gates                  start at gate_lr, linearly to gate_lr_end within the same number of steps, then constant
Cache (--multi_cache_dir, default empty = off; the launcher can pass ${OUT}/cache_multi): in mixed data each goku / CoinVE sample is visited multi_repeat times per epoch,
  and video decoding + VAE encoding (5.2 s/step) + MLLM encoding (3.1 s/step) are recomputed every time. The first visit (miss) computes as usual and also writes the VAE
  distribution params (src / tgt mean + logvar; sampling formula unchanged, noise resampled on every visit) and both MLLM ctx branches (full / empty instruction)
  to <dir>/<key>.st (flat, one level; background threads, write .tmp then atomic rename, write failures are only counted);
  on later visits the dataset worker (kiwi_dataset_multi_moe) reads the file directly, the sample carries a "cache" key, and the main process skips decode / VAE / MLLM.
  The text-CFG draw is still one torch.rand(()) per sample in the original order; hit samples take ctx_null / ctx_cond by the draw. Guards 1 / 1b apply to hit samples too.
  rank0 empties the directory at startup (not reused across launches) and deletes it at DONE; the launcher's EXIT trap is the fallback (the cache must be removed when training ends or is killed).
  Logs cache/hit_ratio (hits / multi-sample visits), cache/files (file count), etc.; the console step line ends with cache=xx%.
Loader memory guard: before building the DataLoader, in-flight memory is estimated as workers x prefetch x batch x ~118 MB per sample (src + tgt uint8) x processes per node; above 60% of min(/dev/shm available, MemAvailable) it raises with suggested values.
Logging (tensorboard + wandb, every 10 steps, averaged over 24 GPUs x 10 steps): loss/main, loss/motion, loss/aux, loss/total, lr/*, grad/norm, time/{vae,mllm,dit} (rank0 seconds per step),
  route_{ffn|attn}/{category}/expert{i} = P(expert i selected | sample category c) (category x expert conditional selection rate, the direct specialization metric), route/{kind}_spread = mean over experts of the max gap across categories,
  plus the "selection fraction" of every expert in every MoE layer: self-attention gate moe_attn_L{block}/expert{i}, FFN gate moe_ffn_L{block}/expert{i}
  (token mean of 1 if selected else 0; one folder per layer in tensorboard); loss/aux_attn and loss/aux_ffn are logged separately
Checkpoints:
  every save_steps steps     ckpt/step-N/moe_v3_2_trainable.safetensors  trainable params in bf16 (full training = whole DiT; for inference / weights-only resume) + meta.json
  every ds_save_steps steps  ckpt/ds_state_step-N/                     full DeepSpeed state (weights + optimizer, 24 shards, for resume), only the latest one is kept
resume: --resume_ds_state <ds_state dir> (needs the same 24 GPUs)  or  --resume_weights <safetensors> (weights only, fresh optimizer)
Multi-node launch: train_multinode.sh (smoke test: train_smoke.sh).
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

# Speed-up: bind CPU cores to the socket of the GPU (nvidia-smi topo -m on the training hosts: GPU0-3 -> NUMA0 = CPU 0-47,96-143; GPU4-7 -> NUMA1 = CPU 48-95,144-191).
# Must happen before `import torch` so that OMP/torch threads, pinned optimizer state from cudaHostAlloc (first touch lands on the local NUMA node) and dataloader workers all stay on this socket.
# Measured: without binding, the 69 GB optimizer state of 8 ranks lands randomly on both sockets and half of the CPU Adam accesses cross Infinity Fabric.
# The core lists below describe one specific host layout (2 sockets x 96 threads); binding is therefore opt-in: MOE_NUMA_BIND=1 enables it.
if os.environ.get("LOCAL_RANK") is not None and os.environ.get("MOE_NUMA_BIND", "0") == "1":
    try:
        _lr = int(os.environ["LOCAL_RANK"])
        _gpu = _lr                                     # a node may use only physical GPUs 4-7 (CUDA_VISIBLE_DEVICES=4,5,6,7): local ranks 0-3 then map to NUMA 1
        _cvd = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
        if _cvd and all(x.isdigit() for x in _cvd) and _lr < len(_cvd):
            _gpu = int(_cvd[_lr])
        _cpus = set(range(0, 48)) | set(range(96, 144)) if _gpu < 4 else set(range(48, 96)) | set(range(144, 192))
        _cpus &= os.sched_getaffinity(0)
        if len(_cpus) >= 8:
            os.sched_setaffinity(0, _cpus)
            print(f"[moe-v3.2] rank(local {_lr}, physical GPU {_gpu}) NUMA binding: {len(_cpus)} logical cores ({min(_cpus)}..)", flush=True)
    except Exception as _e:
        print(f"[moe-v3.2] NUMA binding failed (ignored): {_e}", flush=True)

import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from moe_v3_2_model import EXPERT_NAMES, TextEncoder, build_dit, build_mllm, build_vae, expert_names, load_trainable_weights, mark_trainable, parse_sources, trainable_state_keys  # noqa: E402

TAG = "[moe-v3.2]"


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
    p.add_argument("--experts_merged", required=True, help="output experts_v3_1.safetensors of merge_experts_v3_1.py (videoeditor-14b directory); \"\" with --resume_weights = experts come from the checkpoint")
    p.add_argument("--fast_init", type=int, default=1, help="with --resume_weights (not EP): skip torch.nn.init during construction (single-threaded random init of 55.9B parameters is most of the "
                        "~10 min build); the checkpoint must then cover every parameter (asserted) and the finiteness scan runs after loading")
    p.add_argument("--use_gradient_checkpointing", type=int, default=1)
    # MoE
    p.add_argument("--moe_layers", default="1,3,5,7,9,11,13,15,17,19,21,23,25,27", help="0-based block indices = layers 2,4,...,28 in 1-based counting")
    p.add_argument("--moe_num_experts", type=int, default=35, help="= len(expert_sources) x expert_copies, default 7 x 5 = 35")
    p.add_argument("--expert_sources", default="", help="source expert names (comma-separated, from main/style/add/remove/replace/background/direct_finetune in experts_v3_1.safetensors); empty = all 7")
    p.add_argument("--expert_copies", type=int, default=5, help="number of copies of each source expert")
    p.add_argument("--expert_copy_noise", type=float, default=0.01, help="Gaussian perturbation of each expert copy = weight std x this factor (0.01, same as the noisy copies in mlp-moe); 0 = no perturbation")
    p.add_argument("--moe_top_k", type=int, default=3, help="number of routed experts activated per token")
    p.add_argument("--moe_aux_loss_alpha", type=float, default=0.001, help="load-balancing auxiliary loss coefficient")
    # Data
    p.add_argument("--stage_a_data", required=True, help="stage A json/jsonl files, comma-separated")
    p.add_argument("--stage_b_data", required=True, help="stage B json/jsonl files, comma-separated")
    p.add_argument("--stage_a_epochs", type=float, default=1.0)
    p.add_argument("--stage_b_epochs", type=float, default=2.0)
    p.add_argument("--exclude_prompt_words", default="overlay", help="comma-separated; samples whose instruction contains any of these words (case-insensitive) are dropped")
    p.add_argument("--data_schedule", default="mixed", choices=["mixed", "two_stage"],
                   help="mixed: 1 copy of OpenVE + multi_repeat copies each of goku/CoinVE shuffled into one dataset, run for stage_a_epochs epochs; two_stage: A then B (the old mlp-moe schedule)")
    p.add_argument("--multi_repeat", type=int, default=2, help="number of copies of goku / CoinVE in mixed mode")
    p.add_argument("--dataset_base_path", default="", help="prefix joined to relative video paths in the dataset json files (absolute paths are used as is); \"\" = none")
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--max_pixels", type=int, default=399360)
    p.add_argument("--dataset_num_workers", type=int, default=4)
    p.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    p.add_argument("--compound_joiner", default=" Then ", help="CoinVE multi-instruction samples (no native full instruction): sub-instructions are joined into one sentence with this")
    p.add_argument("--vae_encode_group", type=int, default=4, help="videos per VAE encoder call (source and target grouped separately); 1 = per sample (old behavior)")
    p.add_argument("--mllm_attn_impl", default="sdpa", help="attention implementation of the Qwen backbone; flash_attention_2 produced random NaNs in our environment (see the build_mllm comment), default sdpa; auto = module default (flash_attention_2)")
    p.add_argument("--ds_config", default="",
                   help="path of the DeepSpeed config json (the launcher passes ${DS_CONFIG}). If set, deepspeed.zero.Init is called before accelerator.prepare "
                        "to shard the model in place - when parameters have no ds_id the DeepSpeed engine first calls module.to(device), moving the WHOLE model to one GPU "
                        "(engine.py:1853); 55.93B params in bf16 are 104 GiB, an immediate OOM on 80 GB GPUs (videoeditor-14b is only 26 GiB, so it never showed up). "
                        "zero.Init moves parameters to the GPU one at a time and shards them (largest single parameter 88 MB), so the peak is only the 4.7 GB of shards + one parameter.")
    p.add_argument("--final_save", type=int, default=1,
                   help="0 = do not save final weights / DS state at the end (for smoke tests). A final save of the 55.9B model = 112 GB bf16 weights + ~670 GB DS state, not needed for a 30-step smoke test")
    p.add_argument("--cpu_adam", type=int, default=0,
                   help="1 = use DeepSpeedCPUAdam (together with offload_optimizer.device=cpu in the ds config). Without offload, ZeRO-3 on the 55.9B model needs "
                        "37 GB per GPU for optimizer state alone (fp32 master weights + m + v + bf16 weights/grads), capping batch at 2; with offload only 9.3 GB per GPU remain and batch goes back to 8. "
                        "DeepSpeedCPUAdam adamw_mode=True matches the torch.optim.AdamW formula; measured 4.5G params/s, i.e. ~0.5 s/step for 2.33G params per GPU.")
    p.add_argument("--z3_leaf_modules", type=int, default=0, help="speed-up A5: mark MoEFFN/MoEKVSelfAttention as ZeRO-3 leaf modules (gather the whole module at once); off by default")
    p.add_argument("--async_log", type=int, default=1, help="speed-up: run rank0's accelerator.log (~300 scalars; wandb+TB measured blocking ~4 s per 10 steps) in a background thread")
    p.add_argument("--mllm_batch_encode", type=int, default=1,
                   help="speed-up: 1 = encode the whole batch in one MLLM pass (TextEncoder.encode_batch, 8 per-sample forwards ~2.7 s -> 1); 0 = per-sample encode (old behavior)")
    # Cache: VAE distribution params + MLLM ctx features of multi-edit (goku / CoinVE) samples (see file header); off by default
    p.add_argument("--vae_compile", default="off", choices=["off", "encoder", "encoder_dyn", "encoder_autotune", "encode"],
                   help="speed-up: torch.compile the VAE encoder (mode chosen by benchmarking on real shapes; off = eager)")
    p.add_argument("--cache_keep_existing", type=int, default=0, help="1 = keep the existing cache on restart (the leftover directory is not cleared at startup and is reused)")
    p.add_argument("--cache_delete_on_done", type=int, default=1, help="delete the cache automatically when training ends (0 = keep it and print a notice; the user decides)")
    p.add_argument("--multi_cache_dir", default="", help="feature cache directory for multi-edit samples (shared filesystem, ~15 MB per file); empty = off. Cleared at startup, deleted by rank0 when training ends")
    p.add_argument("--cache_write_threads", type=int, default=4, help="background cache-writer threads per rank (shared filesystem ~10 MB/s per stream, up to 8 x 15 MB per step)")
    p.add_argument("--cache_queue", type=int, default=64, help="cache write queue limit (samples); when full, items are dropped (counted in cache/write_drop) without blocking training")
    # Training
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4, help="per-GPU batch; effective batch = 24 x batch_size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="micro-batches per optimizer update (must equal gradient_accumulation_steps in the DeepSpeed config). One counted step is still one loader batch of "
                        "world x batch_size samples (data position, checkpoint names, LR schedule and save_steps stay in these units); the optimizer updates only on steps "
                        "that are multiples of this value, so saves at multiples of save_steps always fall on an update boundary. Not supported together with --ep")
    p.add_argument("--max_train_steps", type=int, default=0, help="0 = derived from the stage epochs")
    p.add_argument("--learning_rate", type=float, default=1e-5, help="base learning rate of full training; original params (including the shared part of MoE layers) always use it")
    p.add_argument("--new_layers_lr", type=float, default=1e-4, help="initial learning rate of new layers (routed experts + gates)")
    p.add_argument("--new_layers_lr_steps", type=int, default=2000, help="steps for new layers to go linearly from new_layers_lr to learning_rate, same as other params afterwards; the gate transition uses the same number of steps")
    p.add_argument("--gate_lr", type=float, default=1e-3, help="initial gate learning rate")
    p.add_argument("--gate_lr_end", type=float, default=1e-4, help="gate learning rate after new_layers_lr_steps steps (linear transition to it, then constant)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--opt_probe_wd0_at", type=int, default=0, help="diagnostic: if >0, set weight_decay of all optimizer param groups to 0 in place after this step (observe wd=0.01 and wd=0 segments from one init)")
    p.add_argument("--opt_probe", type=int, default=0, help="diagnostic: if >0, every N steps rank0 prints the Adam momentum/weight correlation and ratio, |w| median and norm of the ZeRO-3 fp32 shards (read-only, no checkpoint needed; coupled decay drives the ratio to ~weight_decay)")
    p.add_argument("--text_drop_prob", type=float, default=0.1, help="text CFG: per-sample probability of blanking the instruction")
    p.add_argument("--motion_sub_loss_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--save_steps", type=int, default=250, help="interval for saving bf16 trainable weights")
    p.add_argument("--ds_save_steps", type=int, default=1000, help="interval for saving the full DeepSpeed state (for resume)")
    p.add_argument("--resume_ds_state", default="", help="ckpt/ds_state_step-N directory")
    p.add_argument("--mllm_worker_pre", type=int, default=1, help="1 = run the MLLM CPU preprocessing (frame sampling/template/resize/normalize/tokenize; pixels computed once for both branches) in the dataloader workers, "
                   "the main process only runs the batched forward (bitwise equal to main-process preprocessing); 0 = old path")
    p.add_argument("--vae_prefetch", type=int, default=1, help="1 = before the CPU part of the optimizer (DeepSpeed step / expert CPU Adam), fetch the next batch and launch its VAE encoding kernels, "
                   "so the GPU finishes encoding while the CPU runs Adam (data order and RNG consumption order identical to off); 0 = original order")
    p.add_argument("--ep", type=int, default=0, help="1 = expert parallelism (moe_v3_2_ep.py: 8-GPU EP within a node + expert DP across 3 nodes + sharded CPU Adam); expert weights are not handed to ZeRO-3")
    p.add_argument("--ep_local_world", type=int, default=8, help="EP: GPUs per node (EP group size)")
    p.add_argument("--resume_step_from", default="", help="with --resume_weights: read the checkpoint step and global_batch from step.json in this directory (continues the data position when changing batch / switching to EP)")
    p.add_argument("--resume_ckpt_global_batch", type=int, default=0, help="on resume, if step.json has no global_batch (old checkpoint), use this as the effective batch (world x batch) at save time; 0 = same as now. If the batch changed, the global step is converted by sample count (samples consumed per rank = step x per-GPU batch; the per-rank sampler sequence does not depend on batch, so it resumes exactly)")
    p.add_argument("--resume_weights", default="", help="load only moe_v3_2_trainable.safetensors, fresh optimizer")
    # Logging
    p.add_argument("--use_wandb", type=int, default=0)
    p.add_argument("--wandb_project", default="video-editor-moe")
    p.add_argument("--wandb_run_name", default="moe_v3_2")
    return p.parse_args(argv)



# =========================================================================== #
#  Sample categories / batched VAE encoding / timing                          #
# =========================================================================== #
CATEGORIES = ["add", "remove", "replace", "style", "background", "multi", "unknown"]     # rows of the "category x expert" matrix in the logs
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
OPENVE_DIR_TO_CATEGORY = {"local_add": "add", "local_remove": "remove", "local_change": "replace", "global_style": "style", "background_change": "background"}


def category_of(record):
    """Edit category of a sample (only for logging, not fed to the model): original OpenVE records use the directory category cat, the five-class subset uses cls;
    goku / CoinVE use turn_types_qwen - if all sub-instructions share one type that is the category, otherwise multi."""
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
    The outer AutoencoderKLWan3_8.encode loops over samples, but the inner vae.model.encode supports a batch dim (the chunk loop and feat_cache both carry b),
    so the inner one is called directly; groups of group_size bound peak memory (encoding runs before DiT activations are allocated). Serial per-sample encoding was the largest per-step cost.
    Cache: with return_moments=True returns (sample, mean, logvar) - mean / logvar are the channel halves of the encoder output [B, 2C, f, h, w] (mu from vae.model.encode is
    already normalized by vae.scale, logvar is raw, not clamped); sampling is still DiagonalGaussianDistribution(...).sample() (same formula); cache hits resample with the same formula."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    group = max(1, group_size)
    if not hasattr(vae, "model") or not hasattr(vae.model, "encode") or not hasattr(vae, "scale"):
        dists = [vae.encode(x)[0] for x in pixels_list]                                  # other VAE classes: fall back to per sample
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
        except Exception as e:                                                             # a torch.compile'd encoder may raise an internal Inductor error on its first real compile / a new shape
            eager = getattr(vae.model, "_eager_encoder", None)                             #   (encoder_dyn once raised InductorError and crashed a short test); fall back to eager and keep training, print once
            if eager is None or vae.model.encoder is eager:
                raise
            print(f"[moe-v3.2] warning: compiled VAE encoder failed, falling back to eager: {type(e).__name__}: {str(e)[:160]}", flush=True)
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
    """Speed-up (A1): the dataset is built with src_pil=False, so workers no longer pickle 49 source PIL frames back to the main process (470 MB / 3.5 s per step went there)
    and only send src_pixels ([C=3, F, H, W] uint8, RGB; layout of kiwi_video_utils.frames_to_uint8_tensor: np.array(PIL.convert("RGB")) gives [H, W, 3],
    permute(2, 0, 1) gives [3, H, W], then stack(dim=1) gives [3, F, H, W]). The MLLM still needs PIL frames, so the main process inverts the same layout
    (permute back to [F, H, W, 3], then Image.fromarray per frame); exact inverse of frames_to_uint8_tensor: same mode "RGB", same size, byte-identical.
    Only samples with src_video None and src_pixels present are handled; samples that already have PIL frames (src_pil=True / return_pixels=False) are left untouched."""
    if sample.get("src_video") is not None or sample.get("src_pixels") is None:
        return sample
    from PIL import Image
    u8 = sample["src_pixels"]
    assert u8.dim() == 4 and u8.shape[0] == 3 and u8.dtype == torch.uint8, f"src_pixels must be [3, F, H, W] uint8, got {tuple(u8.shape)} {u8.dtype}"
    frames = u8.permute(1, 2, 3, 0).contiguous().numpy()                     # [F, H, W, 3] uint8, one copy; each frame is a C-contiguous slice
    sample["src_video"] = [Image.fromarray(frames[f]) for f in range(frames.shape[0])]   # [H, W, 3] uint8 -> mode "RGB", byte-identical to the original PIL frames
    return sample


EP_STORE = None            # EP: expert store (used for checkpoints / optimizer step), None = no EP


def opt_probe(engine, step, tag):
    """Diagnostic (read-only, rank-local, no collectives): for each fp32 shard sub_group of the ZeRO-3 optimizer print corr(w, exp_avg), median |exp_avg|/|w|, |w| median / norm.
    Healthy values with torch AdamW (videoeditor-14b): corr ~0, ratio ~1e-6; the pathology it detects: corr 1.0000, ratio ~ weight_decay (0.01) -> wd*w was added to the gradient (coupled L2) and the weights shrink toward zero by ~lr every step."""
    z = getattr(engine, "optimizer", None)
    flats = getattr(z, "fp32_partitioned_groups_flat", None)
    if flats is None:
        print(f"{tag} opt_probe step {step}: not a ZeRO-3 optimizer ({type(z).__name__})", flush=True)
        return
    out = []
    for i, flat in enumerate(flats):
        st = z.optimizer.state.get(flat, {})
        w = flat.detach().float()
        m = st.get("exp_avg")
        if m is None:
            out.append(f"sg{i}: no state |w|med={w.abs().median():.2e}")
            continue
        m = m.float()
        n = min(w.numel(), m.numel()); w = w[:n]; m = m[:n]
        if n > 50_000_000:                       # subsample 50M elements of large shards, accurate enough
            idx = torch.randint(0, n, (50_000_000,)); w = w[idx]; m = m[idx]
        corr = float(((w - w.mean()) * (m - m.mean())).mean() / (w.std() * m.std() + 1e-20))
        ratio = float(m.abs().median() / (w.abs().median() + 1e-20))
        out.append(f"sg{i}(lr={z.optimizer.param_groups[z.sub_group_to_group_id[i]]['lr']:.2e} wd={z.optimizer.param_groups[z.sub_group_to_group_id[i]]['weight_decay']}): "
                   f"corr={corr:.4f} m/w={ratio:.2e} |w|med={w.abs().median():.2e} |w|norm={float(flat.detach().float().norm()):.3f} step={st.get('step')}")
    print(f"{tag} opt_probe step {step}: " + " | ".join(out), flush=True)


def apply_ds_step_patches(engine, rank):
    """Speed-up (numerically identical): two patches on the DeepSpeed ZeRO-3 optimizer instance (only this instance's methods; library files untouched).
    (1) unscale_and_clip_grads: while the grad norm stays < 1 (0.19~0.82 at steps 80~190) combined_scale is always 1.0, yet the original still runs mul_(1.0) over 9.3 GB of fp32 grads
       - x*1.0 == x (bitwise, including NaN/Inf), so skipping it saves ~0.6 s/step; unchanged when the factor != 1. The check must be a strict == 1.0.
    (2) _reassign_or_swap_out_partitioned_parameters: when copying updated params from CPU fp32 back to GPU bf16, torch first converts to a pageable bf16 temporary on the CPU
       and then does a synchronous memcpy, measured at only 3.9 GB/s (1.2 s/step at 0 GPU utilization). Now: copy_ into a pinned bf16 buffer (same CPU fp32->bf16 conversion kernel, bitwise identical),
       then non_blocking H2D; two rotating buffers + CUDA events ensure the DMA finished before reuse. MOE_DS_STEP_PATCH=0 disables it."""
    import types
    opt_obj = getattr(engine, "optimizer", None)
    if opt_obj is None or "Stage3" not in type(opt_obj).__name__:
        print(f"{TAG} rank{rank}: not a ZeRO-3 optimizer ({type(opt_obj).__name__}), step patches not applied", flush=True)
        return
    need = ("unscale_and_clip_grads", "_reassign_or_swap_out_partitioned_parameters", "fp32_partitioned_groups_flat",
            "fp16_partitioned_groups_flat", "_unflatten_partitioned_parameters", "_partitioned_params_swap_out", "clip_grad")
    missing = [n for n in need if not hasattr(opt_obj, n)]
    if missing:
        print(f"{TAG} rank{rank}: ZeRO-3 optimizer lacks {missing}, step patches not applied", flush=True)
        return

    def unscale_and_clip_grads(self, sub_group_id, total_norm):
        ep_sq = float(getattr(self, "_moe_ep_norm_sq", 0.0))      # EP: global sum of squared expert grads, merged with the dense norm into one global norm
        if ep_sq > 0.0:
            total_norm = torch.sqrt(total_norm.double() ** 2 + ep_sq).to(total_norm.dtype) if torch.is_tensor(total_norm) else math.sqrt(float(total_norm) ** 2 + ep_sq)
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            clip = torch.clamp(clip, min=1.0) if torch.is_tensor(clip) else max(clip, 1.0)
            combined_scale = clip * self.loss_scale
        self._moe_ep_clip_coef = float(1.0 / combined_scale)       # EP: the same factor is applied to the expert grads
        self._moe_global_norm = float(total_norm)
        if float(combined_scale) == 1.0:          # x*1.0 == x: skip mul_ over the whole fp32 grad buffer
            self._moe_clip_skipped = getattr(self, "_moe_clip_skipped", 0) + 1
            return
        self.fp32_partitioned_groups_flat[sub_group_id].grad.mul_(1. / combined_scale)

    def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
        if self.fp16_partitioned_groups_flat[sub_group_id] is not None:
            dst = self.fp16_partitioned_groups_flat[sub_group_id].data
            src = self.fp32_partitioned_groups_flat[sub_group_id].data
            if dst.dtype != src.dtype and src.device.type == "cpu" and dst.device.type == "cuda":
                bufs = getattr(self, "_moe_pinned_bufs", None)
                if bufs is None:
                    bufs = self._moe_pinned_bufs = [None, None]
                    self._moe_pinned_evts = [None, None]
                    self._moe_pinned_idx = 0
                k = self._moe_pinned_idx; self._moe_pinned_idx = 1 - k
                n = src.numel()
                if self._moe_pinned_evts[k] is not None:
                    self._moe_pinned_evts[k].synchronize()   # wait until the DMA from two calls ago is done with this buffer
                if bufs[k] is None or bufs[k].numel() < n or bufs[k].dtype != dst.dtype:
                    bufs[k] = torch.empty(n, dtype=dst.dtype, device="cpu", pin_memory=True)
                buf = bufs[k][:n]
                buf.copy_(src)                               # fp32->bf16 on CPU (same conversion kernel as the original)
                dst.copy_(buf, non_blocking=True)            # pinned DMA
                evt = torch.cuda.Event(); evt.record(); self._moe_pinned_evts[k] = evt
            else:
                dst.copy_(src)
            self._unflatten_partitioned_parameters(sub_group_id)
        else:
            self._partitioned_params_swap_out(sub_group_id)

    opt_obj.unscale_and_clip_grads = types.MethodType(unscale_and_clip_grads, opt_obj)
    opt_obj._reassign_or_swap_out_partitioned_parameters = types.MethodType(_reassign_or_swap_out_partitioned_parameters, opt_obj)
    if rank == 0:
        print(f"{TAG} DeepSpeed step patches applied: skip mul_ when clip factor==1 / bf16 copy-back via pinned double buffer (clip_grad={opt_obj.clip_grad})", flush=True)


def sync_time():
    """For timing: wait for the GPU queue to drain before reading the clock, so the VAE / MLLM / DiT durations are real."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


# =========================================================================== #
#  Cache: multi-edit feature cache (main process) / loader memory guard       #
# =========================================================================== #
CACHE_VERSION = "v1"
CACHE_TENSOR_KEYS = ("src_mean", "src_logvar", "tgt_mean", "tgt_logvar", "ctx_cond", "ctx_null")      # the 6 tensors of a cache file (bf16, CPU)
CACHE_STAT_NAMES = ("hit", "miss", "other", "write_ok", "write_fail", "write_drop")                  # slots appended to the stat vector (reduced / cleared with stat every 10 steps)
CACHE_FILES_COUNT_EVERY = 100                                                                        # interval (steps) at which rank0 counts the cache directory files in a background thread


def multi_cache_key(src_video, tgt_video, prompt, num_frames, height, width):
    """Cache: key = sha1("src|tgt|full instruction|FxHxW") (src / tgt are the raw record strings), same formula as KiwiMultiEditDataset.cache_key_of on the dataset side (change both together);
    files <cache_dir>/<key>.st, flat. The main process only falls back to this when a sample carries no cache_path / cache_key (the dataset side normally provides them, and they take precedence)."""
    import hashlib
    return hashlib.sha1(f"{src_video}|{tgt_video}|{prompt}|{num_frames}x{height}x{width}".encode("utf-8")).hexdigest()


def prompt_of(sample):
    """Full instruction of a sample (prompts_multi[0]; hit samples without prompts_multi fall back to the prompt field)."""
    plist = sample.get("prompts_multi")
    return plist[0] if plist else (sample.get("prompt") or "")


def is_cached_sample(sample):
    """Cache: samples returned by a dataset worker on a cache hit carry a "cache" key (dict of file contents); the flat top-level + cached=True form is also accepted."""
    return sample.get("cache") is not None or bool(sample.get("cached"))


def cache_payload_of(sample, expect=None):
    """Cache: hit sample -> dict of 6 CPU tensors; None if not a hit.
    Also None if keys are missing / dims are wrong / the latent shape or ctx dim differs from what this run has seen (expect = {"latent": (C, f, h, w), "ctx": 3072}, None = unknown, not checked)
    - callers use is_cached_sample to tell "miss" from "hit with a bad file" (the latter has no pixels to fall back to and must be dropped)."""
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
        exp_lat = expect.get("latent")
        if isinstance(exp_lat, dict):                                   # per frame count (121 / 81 batches have different latent f)
            exp_lat = exp_lat.get(int(sample.get("num_frames_eff") or 0))
        if exp_lat is not None and tuple(out["src_mean"].shape) != tuple(exp_lat):
            return None
        if expect.get("ctx") is not None and int(out["ctx_cond"].shape[1]) != int(expect["ctx"]):
            return None
    return out


def multi_cache_path_of(sample, cache_dir, args):
    """Cache: the file a missed sample is written to after this step; None if not in a cache group (OpenVE).
    Prefers cache_path / cache_key computed by the dataset side (guaranteed to match the path it reads); if the sample only has cache_group="multi", falls back to multi_cache_key."""
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
        key = multi_cache_key(src, tgt, prompt_of(sample), int(sample.get("num_frames_eff") or args.num_frames), args.height, args.width)
    return os.path.join(cache_dir, key + ".st")     # safetensors layout, suffix .st


def split_batch_for_cache(batch, cache_dir, args, cache_expect, rank, step):
    """Cache: split a batch. Returns (batch, payloads, pending, counts):
      payloads[i]  dict of 6 tensors for hit samples, None for misses (downstream splits idx_c / idx_u by this)
      pending[i]   missed samples in a cache group {"path", "sample"}, written to the cache once this step has collected the distribution params / both ctx branches; None otherwise. Filtered together with guard drops
      counts       [hits, misses (first visit of multi samples), other (OpenVE)] for the cache slots of stat
    Samples with a "cache" key but bad contents (corrupt file / missing keys / shape mismatch) have no pixels to fall back to and are dropped with a message. Empty cache_dir = off: all misses, pending all None."""
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
    """Cache: resample latents of hit samples from the cached VAE distribution params - the same DiagonalGaussianDistribution(cat([mean, logvar], 1)).sample() as vae_encode_batched
    (= mean + exp(0.5*clamp(logvar, -30, 20))*randn, randn drawn on device in dtype); resampled on every visit, same distribution as without cache. Returns [n, C, f, h, w]."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    mean = torch.stack([p[mean_key] for p in payloads]).to(device=device, dtype=dtype)
    logvar = torch.stack([p[logvar_key] for p in payloads]).to(device=device, dtype=dtype)
    return DiagonalGaussianDistribution(torch.cat([mean, logvar], dim=1)).sample()


def encode_latents_with_cache(vae, batch, payloads, pending, args, device, dtype, pixels_of, cache_expect):
    """Cache: only missed samples go through the VAE (their mean / logvar are stored into pending for cache writing); hit samples are resampled from cached params; results are reassembled in batch order.
    Returns (src_latents, latents), each [B, C, f, h, w]; for an empty batch returns two 0-row tensors (guard 1 downstream then flags the whole batch as bad)."""
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
        if isinstance(cache_expect.get("latent"), dict):
            cache_expect["latent"].setdefault(int(batch[idx_u[0]].get("num_frames_eff") or args.num_frames), tuple(src_u.shape[1:]))   # per frame count
        elif cache_expect.get("latent") is None:
            cache_expect["latent"] = tuple(src_u.shape[1:])            # later hit samples must match this shape (checked in cache_payload_of)
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
    """Cache: chunked encode_batch (at most chunk samples per chunk = the MLLM batch size validated in training): missed multi samples encode one extra branch (empty / full instruction),
    so a batch has up to 2 x batch_size entries and would otherwise double the MLLM forward batch. forward_batch is per-sample equivalent, so chunking does not change semantics."""
    chunk = max(1, chunk)
    out = []
    for i in range(0, len(prompts), chunk):
        out.extend(text_encoder.encode_batch(prompts[i:i + chunk], frames_list[i:i + chunk]))
    return out


def encode_ctx_with_cache(text_encoder, batch, payloads, pending, args, device, dtype, cache_expect):
    """Cache: text-CFG draw + MLLM encoding; returns ctx_list aligned with batch (each [L_i, 3072] bf16, on device).
      - the draw is one float(torch.rand(())) per sample in batch order (hit samples draw too): RNG consumption count and order identical to no cache
      - hit samples: take ctx_null / ctx_cond from the cache according to the draw, no MLLM
      - missed samples: encoded as before (a dropped prompt becomes "" = encode_null semantics); those in a cache group also encode the other branch (full instruction if the empty one was drawn, else empty),
        and both go into pending (ctx_cond / ctx_null); with mllm_batch_encode=1 they are appended to the main list and chunked through encode_batch, with =0 per-sample encode / encode_null.
    batch must not contain samples dropped by guard 1 (the caller filters them)."""
    idx_u = [i for i, p in enumerate(payloads) if p is None]
    idx_c = [i for i, p in enumerate(payloads) if p is not None]
    use_pre = bool(idx_u) and all(isinstance(batch[i].get("mllm_pre"), dict) for i in idx_u)     # workers already did the MLLM preprocessing
    for i in idx_u:
        assert len(batch[i]["prompts_multi"]) == 1              # load_records already reduced every sample to one full instruction
        if not use_pre:
            rebuild_src_frames(batch[i])                        # speed-up (A1): with src_pil=False the source PIL frames are rebuilt here from src_pixels (the MLLM needs PIL)
    prompts = ["" if float(torch.rand(())) < args.text_drop_prob else prompt_of(sample) for sample in batch]
    ctx_list = [None] * len(batch)
    for i in idx_c:
        ctx_list[i] = (payloads[i]["ctx_null"] if prompts[i] == "" else payloads[i]["ctx_cond"]).to(device=device, dtype=dtype)
    extra_pos = [i for i in idx_u if pending[i] is not None]
    other_prompt = {i: (batch[i]["prompts_multi"][0] if prompts[i] == "" else "") for i in extra_pos}      # instruction of the other branch
    if use_pre:                                                 # same order as batch_encode below (main entries first, extra branches after); only the inputs come from the workers
        def _seq(i, prompt):
            pre = batch[i]["mllm_pre"]
            return {"input_ids": pre["null"] if prompt == "" else pre["cond"], "pixel_values_videos": pre["pixel_values_videos"], "video_grid_thw": pre["video_grid_thw"]}
        enc_seqs = [_seq(i, prompts[i]) for i in idx_u] + [_seq(i, other_prompt[i]) for i in extra_pos]
        chunk = max(1, args.batch_size)
        enc_out = []
        for k0 in range(0, len(enc_seqs), chunk):
            enc_out.extend(text_encoder.encode_batch_pre(enc_seqs[k0:k0 + chunk]))
        for k, i in enumerate(idx_u):
            ctx_list[i] = enc_out[k]
        other_ctx = {i: enc_out[len(idx_u) + k] for k, i in enumerate(extra_pos)}
    elif args.mllm_batch_encode:
        # speed-up (B1): the whole batch in one MLLM pass (TextEncoder.encode_batch -> MLLMInContext.forward_batch); main entries first, extra branches after
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
    for i in extra_pos:                                         # assign branches: if the empty instruction was drawn, the main ctx is the empty branch and the extra one is the full branch; otherwise the reverse
        if prompts[i] == "":
            pending[i]["ctx_null"], pending[i]["ctx_cond"] = ctx_list[i], other_ctx[i]
        else:
            pending[i]["ctx_cond"], pending[i]["ctx_null"] = ctx_list[i], other_ctx[i]
    if idx_u and cache_expect.get("ctx") is None:
        cache_expect["ctx"] = int(ctx_list[idx_u[0]].shape[1])
    return ctx_list


def build_cache_payload(pending, args, step, rank):
    """Cache: contents of one sample's cache file - 6 contiguous bf16 CPU tensors + meta (readable with torch.load(weights_only=True))."""
    sample = pending["sample"]
    # copy=True + contiguous_format guarantee exactly one copy whether the source is on GPU or CPU (unit tests), with its own storage holding only its elements.
    # .to("cpu", bf16).contiguous() does not copy a tensor that is already CPU bf16, so src_mean etc. would remain views of the large VAE moments storage and torch.save would write the whole storage (37 MB vs 14 MB)
    out = {k: pending[k].detach().to("cpu", torch.bfloat16, copy=True, memory_format=torch.contiguous_format) for k in CACHE_TENSOR_KEYS}
    out["meta"] = {"key": os.path.basename(pending["path"])[:-3], "src_video": sample.get("src_video_path"), "tgt_video": sample.get("tgt_video_path"),
                   "prompt": prompt_of(sample), "num_frames": int(sample.get("num_frames_eff") or args.num_frames), "height": int(args.height), "width": int(args.width),
                   "latent_shape": list(out["src_mean"].shape), "version": CACHE_VERSION, "step": int(step), "rank": int(rank)}
    return out


def save_cache_file(path, payload):
    """Write a cache file in safetensors layout (8-byte little-endian header length + JSON header + raw tensor bytes), the whole buffer in one unbuffered f.write.
    Reason: profiling showed 24.6% of rank0's GIL time in the writer threads' torch.save (holds the GIL through zip open/close), stalling kernel launches from the main thread and slowing the DiT stage by ~2 s/step;
    tobytes/join are just memcpy (milliseconds) and the write syscall releases the GIL. The reader kiwi_dataset_multi_moe uses safetensors.safe_open (mmap).
    numpy has no bf16 dtype, so the bytes are taken via an int16 view (same bit pattern) and the header says BF16 so safetensors reads them back as bf16."""
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
    """Cache: enqueue missed multi samples that passed guards 1 + 1b; returns the number enqueued. Guard 1b only checked the main ctx, not the extra branch:
    if any of the 6 tensors is non-finite the sample is not written (path printed; the next visit misses again and recomputes); otherwise a non-finite ctx in the file would get the sample dropped by guard 1b on every later hit, never re-encoded."""
    n = 0
    for p in pending:
        if p is None:
            continue
        bad = [k for k in CACHE_TENSOR_KEYS if not torch.isfinite(p[k]).all()]
        if bad:
            sample = p["sample"]
            print(f"{TAG} rank{rank} step {step}: cache not written (non-finite tensors {bad}) src={sample.get('src_video_path', '?')} prompt={prompt_of(sample)[:80]!r}", flush=True)
            continue
        cache_writer.put(p["path"], build_cache_payload(p, args, step, rank))
        n += 1
    return n


class MultiCacheWriter:
    """Cache: one background cache writer per rank - the main thread only puts (path, payload) into a bounded queue; `threads` daemon threads write to <path>.tmp.<rank>.<pid>.<tid>
    and then os.replace atomically (when 24 ranks on 3 nodes write the same key concurrently the last write wins; readers never see a partial file). A full queue drops the item (write_drop); write exceptions are swallowed and counted (write_fail,
    only the first 3 are printed; the cache dir being deleted by another node / filesystem hiccups must not crash training). Threads start on the first put (after the DataLoader workers fork, avoiding a fork with live threads).
    rank0 also has count_files_async(): a background thread counts the .st files with os.scandir (flat top level + one level of subdirectories), result in files (-1 = not counted yet)."""

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
                save_cache_file(tmp, payload)             # not torch.save (holds the GIL)
                os.replace(tmp, path)
                with self.lock:
                    self.ok += 1
            except Exception as e:                       # a failed write must not affect training
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
        """Take and reset the (written, failed, dropped-when-full) counters; added to stat every step."""
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
    """Cache: delete the cache directory (it must be removed automatically when training ends / is killed).
    (1) atomically rename <dir> to <dir>.del.<pid>.<seconds> (instant; a new directory with the same name is unaffected), together with leftover <dir>.del.* from earlier runs;
    (2) a background thread scans the top-level entries (flat files or subdirectories; readdir of 220k entries on a network filesystem takes tens of seconds, so not on the main thread),
       and `threads` daemon threads delete in parallel (unlink files, rmtree subdirs; deleting small files on a network filesystem is metadata-bound, serial rm takes tens of minutes, parallel is ~10x faster), then the root;
    (3) with wait_seconds > 0 wait at most that long and return even if unfinished (daemon threads keep deleting; after the process exits the nohup rm in the launcher's EXIT trap removes the rest).
    Returns (list of directories being deleted, whether deletion finished)."""
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
        for target in targets:                           # scan (work is filled single-threaded before the workers start, no lock needed)
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
    """Cache: the constructor kwarg through which the dataset side (kiwi_dataset_multi_moe.KiwiMultiEditDataset) receives the cache dir, passed per its actual signature (multi_cache_dir / cache_dir both accepted).
    Empty cache_dir returns {} (old behavior); if the signature has no such parameter returns None (the caller disables the cache with a warning instead of passing an unknown kwarg that would crash the parent KiwiEditDataset)."""
    import inspect
    if not cache_dir:
        return {}
    params = inspect.signature(dataset_cls.__init__).parameters
    for name in ("multi_cache_dir", "cache_dir", "cache_multi_dir"):
        if name in params:
            return {name: cache_dir}
    return None


def check_loader_memory(args, is_local_main):
    """Loader memory guard: check the memory budget of in-flight samples before building the DataLoader. Each rank has up to num_workers x prefetch_factor batches queued in /dev/shm (persistent workers return via shared memory),
    batch_size samples per batch, a missed sample = src + tgt each [3, F, H, W] uint8 ~ 118 MB (49 x 480 x 832; hit samples are only ~15 MB of features, worst case assumed),
    summed over LOCAL_WORLD_SIZE (8) processes per node; compared with min(/dev/shm available (statvfs), MemAvailable (/proc/meminfo)), above 60% it raises with suggested values.
    Only LOCAL_RANK 0 prints the estimate; all ranks compute the same numbers and exit together on error. Returns the estimated bytes."""
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
    msg = (f"{TAG} loader memory estimate: workers={workers} x prefetch={prefetch} x batch={args.batch_size} = {in_flight} in-flight samples per rank x {per_sample / 2 ** 20:.0f} MB x {procs} processes per node "
           f"= up to {need / gb:.0f} GB; /dev/shm available {shm_avail / gb:.0f} GB, MemAvailable {mem_avail / gb:.0f} GB, 60% of the smaller = {budget / gb:.0f} GB "
           f"-> {'OVER BUDGET' if need > budget else 'OK'}")
    if is_local_main:
        print(msg, flush=True)
    if need > budget:
        unit_w = prefetch * int(args.batch_size) * per_sample * procs
        unit_p = max(workers, 1) * int(args.batch_size) * per_sample * procs
        raise RuntimeError(f"in-flight samples {need / gb:.0f} GB exceed 60% of min(/dev/shm available, MemAvailable) ({budget / gb:.0f} GB): "
                           f"use --dataset_num_workers <= {int(budget // unit_w)} (prefetch {prefetch} unchanged) or --dataloader_prefetch_factor <= {int(budget // unit_p)} (workers {workers} unchanged)")
    return need

# =========================================================================== #
#  Data                                                                       #
# =========================================================================== #
def join_goku_style(prompts):
    """Join sub-instructions into one sentence using the wording of GOKU's native total_instruction_en:
    2 steps "A, then b";  3+ steps "A, then b, then c, and finally d" (later steps get a lower-case first letter, trailing periods removed)."""
    steps = [p.strip().rstrip(".") for p in prompts]
    if len(steps) == 1:
        return steps[0]
    rest = [step[0].lower() + step[1:] for step in steps[1:]]
    if len(rest) == 1:
        return f"{steps[0]}, then {rest[0]}"
    return f"{steps[0]}, then " + ", then ".join(rest[:-1]) + f", and finally {rest[-1]}"


def full_prompt_of(record, joiner):
    """One full instruction per sample (this is what the MLLM receives):
      OpenVE              the prompt is already a single instruction
      GOKU multi-step     a record covers only the first steps of a case (the target video is an intermediate result), while the native total_instruction_en describes all steps (including camera/style steps we do not use)
                          and does not match the target video, so the record's own sub-instructions are joined in GOKU's native wording (join_goku_style)
      CoinVE multi-step   the source data only has a list of sub-instructions and no native full instruction, so they are joined into one sentence with joiner"""
    prompts = record["prompt"] if isinstance(record["prompt"], list) else [record["prompt"]]
    if re.search(r"videos/case_\d+/", record["src_video"]):
        return join_goku_style(prompts)
    return joiner.join(prompts)


def load_records(meta_paths, exclude_words, joiner, repeats=None, cache_groups=None):
    """Several json / jsonl -> list of records with a unified schema (one full instruction each); samples whose instruction contains exclude_words are dropped.
    repeats: same length as meta_paths, number of copies of each file's records (goku / CoinVE get 2 in mixed mode); None = 1 each.
    cache_groups: same length as meta_paths; cache: adds cache_group to the file's records (paths_b, i.e. goku / CoinVE, get "multi"; the dataset side computes cache_key from it and tries a hit);
    None / "" = not added (OpenVE records unchanged)."""
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
                            "turn_types_qwen": [category_of(record)]}      # the category is only for the "category x expert" log matrix; carried unchanged through the dataset into the sample
            if cache_group:
                record_out["cache_group"] = cache_group                   # cache: passed to the dataset through the temporary jsonl (json.dumps writes extra fields)
            if record.get("num_frames_bucket"):
                record_out["num_frames_bucket"] = int(record["num_frames_bucket"])   # per-sample frame count from the frame census (121 / 81)
            records.extend([record_out] * repeat)
            kept += 1
        stats.append(f"{os.path.basename(meta_path)}:{kept}" + (f"×{repeat}" if repeat != 1 else "") + f"(dropped {dropped})")
    return records, " + ".join(stats)


def mllm_pre_config(args):
    """Constants for worker-side MLLM preprocessing - must exactly match the config build_mllm uses to construct MLLMInContext (max_frames/max_pixels/number of queries/system prompt)."""
    from videox_fun.moe_common.mllm_encoder_moe import MLLMInContextConfig
    cfg = MLLMInContextConfig(mllm_id=args.mllm_model, num_image_queries=256, num_video_queries=512, num_ref_queries=768,
                              max_object_token=768, max_frames=16, max_pixels_per_frame=262144, dit_dim=3072, _gradient_checkpointing=False)
    return dict(processor_path=args.mllm_model, system_prompt=cfg.system_prompt, max_frames=cfg.max_frames, max_pixels_per_frame=cfg.max_pixels_per_frame,
                max_pixels_image=cfg.max_pixels_image, num_queries=cfg.num_video_queries)


def build_dataset(meta_paths, exclude_words, args, repeats=None, cache_groups=None, cache_kwargs=None):
    """Records -> temporary jsonl -> KiwiMultiEditDataset (reads videos / frames, no mask).
    Cache: cache_groups see load_records; cache_kwargs = constructor kwargs from dataset_cache_kwargs(...) ({} = no cache, old behavior)."""
    import tempfile
    from videox_fun.moe_common.kiwi_dataset_multi_moe import KiwiMultiEditDataset
    records, stats = load_records(meta_paths, exclude_words, args.compound_joiner, repeats, cache_groups)
    fd, merged_path = tempfile.mkstemp(suffix=".jsonl", prefix="moe_v3_data_")
    with os.fdopen(fd, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    dataset = KiwiMultiEditDataset(
        metadata_path=merged_path, base_path=args.dataset_base_path, max_turns=0, mask_loss=False, subset_single_prob=0.0,
        num_frames=args.num_frames, height=args.height, width=args.width, max_pixels=args.max_pixels, repeat=1,
        require_ref_image=False, pad_frames_to_num=True, return_pixels=True,
        src_pil=False,      # speed-up (A1): source PIL frames are not pickled back; the main process rebuilds them from src_pixels for the MLLM (rebuild_src_frames)
        mllm_pre=(mllm_pre_config(args) if int(getattr(args, "mllm_worker_pre", 0)) else None),   # worker-side MLLM preprocessing
        **(cache_kwargs or {}))     # cache: cache directory (kwarg name per the dataset signature), empty dict = old behavior
    os.unlink(merged_path)
    assert len(dataset) == len(records), f"dataset filtered out {len(records) - len(dataset)} records"
    return dataset, stats



class BucketedShardedBatchSampler:
    """Variant of ShardedBatchSampler for per-sample frame counts (record field num_frames_bucket, e.g. 121 / 81): every batch holds samples of ONE
    frame count and all ranks process the SAME frame count in a given step (one global batch of world x batch_size samples is cut per step and
    sliced per rank), so tensors stack and the per-step cost is uniform across ranks. Per epoch: each bucket is shuffled with seed+epoch and cut into
    global batches (its incomplete tail is dropped), then the step order over all buckets is shuffled with the same generator. Same interface as
    ShardedBatchSampler (set_epoch / persistent skip / epoch_batches for the data-position self-check / __len__ identical on every rank)."""

    def __init__(self, num_samples, batch_size, num_replicas, rank, seed, buckets):
        assert len(buckets) == num_samples, (len(buckets), num_samples)
        self.num_samples, self.batch_size = num_samples, batch_size
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        self.epoch, self.skip = 0, 0
        self.bucket_values = sorted(set(int(b) for b in buckets))
        import numpy as _np
        _arr = _np.asarray(buckets)
        self._by_bucket = {b: torch.from_numpy(_np.flatnonzero(_arr == b)) for b in self.bucket_values}
        gb = self.batch_size * self.num_replicas
        self.steps_per_bucket = {b: int(len(v)) // gb for b, v in self._by_bucket.items()}
        self._len = sum(self.steps_per_bucket.values())
        self._cache_epoch, self._cache = None, None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self._len

    def epoch_global(self):
        """[(bucket, LongTensor[world x batch_size])] per step of the current epoch (deterministic in seed + epoch, identical on every rank)."""
        if self._cache_epoch == self.epoch:
            return self._cache
        generator = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        gb = self.batch_size * self.num_replicas
        steps = []
        for b in self.bucket_values:                                   # fixed bucket order, then one shared permutation of the steps
            idx = self._by_bucket[b]
            perm = idx[torch.randperm(len(idx), generator=generator)]
            steps += [(b, perm[i * gb:(i + 1) * gb]) for i in range(self.steps_per_bucket[b])]
        order = torch.randperm(len(steps), generator=generator).tolist()
        self._cache_epoch, self._cache = self.epoch, [steps[i] for i in order]
        return self._cache

    def epoch_batches(self):
        """This rank's batches of the current epoch (index = step within the epoch), each a list of dataset indices of one frame count."""
        r0 = self.rank * self.batch_size
        return [g[r0:r0 + self.batch_size].tolist() for _, g in self.epoch_global()]

    def epoch_buckets(self):
        return [b for b, _ in self.epoch_global()]

    def __iter__(self):
        return iter(self.epoch_batches()[self.skip:])


class IndexTaggedDataset(torch.utils.data.Dataset):
    """Data-position self-check: adds `_req_index` = the index requested by the sampler to every returned sample dict (the REQUESTED index even if an inner retry swapped the sample);
    the training loop uses it to verify that the first batch of each epoch is batch `skip` of the sampler (guards against silent replays such as DataLoader's double iter eating skip). Other attributes are forwarded to the inner dataset."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        s["_req_index"] = int(idx)
        return s

    def __getattr__(self, name):
        base = self.__dict__.get("base")           # __dict__ is still empty during pickle/copy reconstruction: must raise AttributeError, not KeyError
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)


class ShardedBatchSampler:
    """Each epoch shuffles all samples with seed+epoch, splits them evenly across ranks and cuts fixed-size batches (the incomplete tail is dropped).
    skip = batches of this epoch already consumed (persistent, set by the training loop at the start of each epoch; see __iter__). All ranks have the same number of batches per epoch."""

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
        # Data-position fix: the old form `skip, self.skip = self.skip, 0` applied skip only once. But torch DataLoader calls iter(sampler) twice when building an iterator
        # (once in _BaseDataLoaderIter.__init__, once in _reset; with persistent_workers each later epoch's _reset calls it again), so the first call reset skip to 0
        # and the batches came from the second -> every resume replayed the epoch from batch 0 (earlier mlp-moe / videoeditor-14b resumes were all affected; not triggered with num_workers=0).
        # Now skip is persistent and set explicitly by the training loop at the start of every epoch (resume = in_stage % steps_per_epoch, later epochs = 0); the loop also checks the position of each epoch's first batch.
        return iter(self.epoch_batches()[self.skip:])

    def __len__(self):
        return (self.num_samples // self.num_replicas) // self.batch_size


# =========================================================================== #
#  loss (original videoxfun wan2.2)                                           #
# =========================================================================== #
def custom_mse_loss(noise_pred, target, weighting=None, threshold=50.0):
    """Element-wise MSE with outliers |residual| > threshold masked out, times the sd3 weighting (always 1 for scheme=none), then the mean."""
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
GROUP_BASE, GROUP_EXPERTS, GROUP_GATE = 0, 1, 2      # indices of the three optimizer param groups (the optimizer is built in this order)


def linear_then_constant(step, lr_start, lr_end, num_steps):
    """Linear from lr_start to lr_end over the first num_steps steps, then constant lr_end. Uses the global step, so it continues across resume."""
    if step >= num_steps:
        return lr_end
    return lr_start + (lr_end - lr_start) * step / num_steps


def lr_of_new_layers(step, args):
    """Routed experts (new layers): new_layers_lr -> learning_rate over new_layers_lr_steps steps. All original params always use learning_rate."""
    return linear_then_constant(step, args.new_layers_lr, args.learning_rate, args.new_layers_lr_steps)


def lr_of_gate(step, args):
    """Gates: gate_lr -> gate_lr_end, same transition length as the routed experts (new_layers_lr_steps)."""
    return linear_then_constant(step, args.gate_lr, args.gate_lr_end, args.new_layers_lr_steps)


def set_learning_rates(opt, step, args):
    """Write this step's learning rates into the three optimizer param groups. After DeepSpeed wrapping, opt.param_groups are the underlying AdamW groups, so setting lr takes effect directly.
    Returns (routed-expert lr, gate lr) for logging."""
    lr_new, lr_gate = lr_of_new_layers(step, args), lr_of_gate(step, args)
    if len(opt.param_groups) == 2:          # EP: experts are not in the DeepSpeed optimizer, only the base / gate groups; the expert lr is used by ExpertStore.step
        opt.param_groups[0]["lr"] = args.learning_rate
        opt.param_groups[1]["lr"] = lr_gate
        return lr_new, lr_gate
    opt.param_groups[GROUP_BASE]["lr"] = args.learning_rate
    opt.param_groups[GROUP_EXPERTS]["lr"] = lr_new
    opt.param_groups[GROUP_GATE]["lr"] = lr_gate
    return lr_new, lr_gate


# =========================================================================== #
#  Checkpointing                                                              #
# =========================================================================== #
def save_trainable_weights(accelerator, dit, keys, step, args):
    """Under ZeRO-3, gather the sharded weights to rank 0 (bf16) and write only the trainable params (full training = whole DiT) as safetensors. Must be called on all ranks (collective)."""
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
        save_file(picked, os.path.join(out_dir, "moe_v3_2_trainable.safetensors" if EP_STORE is None else "moe_v3_2_dense.safetensors"))   # EP: the dense part is saved under another name; merge_ep_ckpt.py assembles the full model
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"step": step, "variant": "moe_v3_2", "experts": expert_names(parse_sources(args.expert_sources), args.expert_copies),
                       "expert_sources": parse_sources(args.expert_sources), "expert_copies": args.expert_copies, "expert_copy_noise": args.expert_copy_noise,
                       "moe_layers": args.moe_layers, "moe_num_experts": args.moe_num_experts,
                       "moe_top_k": args.moe_top_k, "moe_aux_loss_alpha": args.moe_aux_loss_alpha,
                       "trainable": "full_dit",
                       "learning_rate": args.learning_rate, "new_layers_lr": args.new_layers_lr,
                       "new_layers_lr_steps": args.new_layers_lr_steps, "gate_lr": args.gate_lr, "gate_lr_end": args.gate_lr_end,
                       "norm_topk_weights": True}, f, indent=1)
        print(f"{TAG} saved step-{step} ({len(picked)} tensors, {sum(v.numel() for v in picked.values()) * 2 / 2**30:.1f} GB)", flush=True)
        rotate_weight_ckpts(os.path.join(args.output_dir, "ckpt"))
    if EP_STORE is not None:                                     # EP: each of the 8 GPUs of node 0 writes the experts it holds (collective, all ranks enter)
        EP_STORE.save_shards(os.path.join(args.output_dir, "ckpt", f"step-{step}"))
    del state


def rotate_weight_ckpts(ckpt_root, keep_every=5000, keep_last=2):
    """step-N directories: keep N % keep_every == 0 forever, keep only the latest keep_last of the rest.
    keep_every is 5000 rather than 1000: one bf16 copy is 112 GB (28 GB for videoeditor-14b); over 47,340 steps, multiples of 1000 give 47 copies = 5.3 TB, multiples of 5000 give 9 ~ 1 TB."""
    steps = [int(m.group(1)) for name in os.listdir(ckpt_root)
             for m in [re.fullmatch(r"step-(\d+)", name)] if m and os.path.isdir(os.path.join(ckpt_root, name))]
    for step in sorted(s for s in steps if s % keep_every != 0)[:-keep_last]:
        shutil.rmtree(os.path.join(ckpt_root, f"step-{step}"), ignore_errors=True)
        print(f"{TAG} rotated out step-{step}", flush=True)


def save_ds_state(accelerator, step, args):
    """Full DeepSpeed state (each of the 24 ranks writes its own shard) in ckpt/ds_state_step-N; older ones are deleted afterwards."""
    out_dir = os.path.join(args.output_dir, "ckpt", f"ds_state_step-{step}")
    accelerator.save_state(out_dir)
    if EP_STORE is not None:                                     # EP: each GPU writes the fp32 master weights + Adam m/v of its shard
        EP_STORE.save_optimizer(out_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(os.path.join(out_dir, "step.json"), "w") as f:
            json.dump({"step": step, "batch_size": int(args.batch_size), "world": int(accelerator.num_processes),
                       "global_batch": int(args.batch_size) * int(accelerator.num_processes)}, f)     # record the batch, needed to resume with a different batch
        ckpt_root = os.path.join(args.output_dir, "ckpt")
        for name in os.listdir(ckpt_root):
            m = re.fullmatch(r"ds_state_step-(\d+)", name)
            if m and int(m.group(1)) < step:
                shutil.rmtree(os.path.join(ckpt_root, name), ignore_errors=True)
                print(f"{TAG} rotated out {name}", flush=True)
        print(f"{TAG} saved DeepSpeed state @ step {step}", flush=True)


def build_checked(build_fn, name, rank, tries=3):
    """After building, check all parameters and buffers for finiteness. 24 processes reading weights from a shared filesystem at once occasionally corrupt one process's copy
    (seen in explicit-moe inference, and as a rank whose loss was nan from step 0), so a bad copy is discarded and rebuilt, at most `tries` times."""
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
    """ZeRO-3 gather self-check: gather the sharded params block by block, each rank computes an fp32 checksum, all_gather and check that all 24 agree.
    In one run two ranks were nan from step 0 while their shard weights scanned fine -> suspected bad data received by all-gather on individual ranks; this checks it directly."""
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
        print(f"{TAG} rank{rank}: ZeRO-3 gather self-check failed (block, this-rank checksum, global min, global max, this-rank finite): {bad[:5]}", flush=True)
        raise RuntimeError(f"rank{rank}: ZeRO-3 all-gather results are inconsistent across ranks or non-finite, see the previous line")
    if rank == 0:
        print(f"{TAG} ZeRO-3 gather self-check passed: {len(inner.blocks)} blocks have matching checksums on {world} ranks", flush=True)


# =========================================================================== #
#  main                                                                        #
# =========================================================================== #
class AsyncTrackerLogger:
    """Speed-up: rank0's accelerator.log (~300 scalars, wandb + tensorboard) runs every 10 steps and was measured to block the main thread ~4 s each time,
    with the other 23 GPUs waiting at the next all_reduce -> 0.4 s/step amortized. Writes are queued to a background thread, the main thread only enqueues; close() flushes the queue at the end.
    Only the timing of log writes changes, not the training numerics."""

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
            except Exception as e:                       # a logging failure must not affect training
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
    # Create the CUDA context on this GPU first, then wait for free memory (another process may still be holding GPU memory);
    # wait at most 10 minutes and fail immediately rather than OOM after a 13-minute model build
    torch.zeros(1, device=device)
    for _wait in range(60):
        _free, _total = torch.cuda.mem_get_info(device)
        if _free >= 60 * 2 ** 30:
            break
        if _wait % 6 == 0:
            print(f"{TAG} rank{rank}: only {_free / 2 ** 30:.1f} GiB free on this GPU (<60), waiting for other processes to release memory... ({_wait * 10}s)", flush=True)
        time.sleep(10)
    else:
        raise RuntimeError(f"rank{rank}: after 10 minutes only {_free / 2 ** 30:.1f} GiB free on this GPU; another process is holding GPU memory, free the GPU first")

    # ---- Model: the DiT goes into DeepSpeed (ZeRO-3 sharded); MLLM / VAE are frozen, one copy per GPU, not in DeepSpeed
    # Model building peaks at ~126 GB host memory per process (the 35 experts are built directly in bf16, see moe_expert_dtype; measured on CPU: 1282 s / 125.7 GB).
    # The staggered start is only 105 s while building takes ~21 minutes, so the peaks of the 8 ranks of a node fully overlap (~1.0 TB of 2 TB host memory); watch free -g at startup.
    time.sleep(int(os.environ.get("LOCAL_RANK", 0)) * 15)
    if (args.resume_ds_state or args.resume_weights) and float(args.expert_copy_noise) != 0.0:
        # Speed-up: on resume the expert weights are fully overwritten by the checkpoint (full training, trainable=full_dit), so the initial perturbation is pointless.
        # Drawing Gaussians for 52.4B weights on CPU takes ~9 minutes of the 1282 s build on every restart, so it is turned off.
        if is_main:
            print(f"{TAG} resume: expert init perturbation changed from {args.expert_copy_noise} to 0 (the weights are overwritten by the checkpoint; saves ~9 minutes of 52.4B Gaussian samples)", flush=True)
        args.expert_copy_noise = 0.0
    ep_layout = ep_store = None
    if args.ep:
        # EP: register the EP MoE modules first (the class is chosen by MOE_IMPL when building the DiT); the expert store is built outside zero.Init (plain GPU tensors, invisible to DeepSpeed)
        import moe_v3_2_ep as EPM
        ep_layout = EPM.EPLayout(int(args.moe_num_experts), world, rank, local_world=int(args.ep_local_world))
        ep_layout.build_groups()
        EPM.install(None, ep_layout)
        if is_main:
            print(f"{TAG} {ep_layout.describe()}", flush=True)
    fast = bool(int(getattr(args, "fast_init", 1))) and bool(args.resume_weights) and not args.ep
    if fast:
        # Speed-up: no-op random init during construction (the checkpoint overwrites every parameter, asserted below); build_checked's finiteness scan is
        # meaningless on uninitialised memory, so the scan (and its rebuild-on-corruption retry) is applied after the weights are loaded instead.
        import torch.nn.init as _init
        _saved = {}
        for _fn in ("xavier_uniform_", "xavier_normal_", "normal_", "kaiming_uniform_", "kaiming_normal_", "uniform_", "trunc_normal_"):
            if hasattr(_init, _fn):
                _saved[_fn] = getattr(_init, _fn); setattr(_init, _fn, lambda t, *aa, **kk: t)
        t_fast = time.time()
        try:
            dit = build_dit(args, dtype, verbose=is_main)
        finally:
            for _fn, _f in _saved.items():
                setattr(_init, _fn, _f)
        from safetensors import safe_open as _safe_open
        with _safe_open(args.resume_weights, framework="pt", device="cpu") as _f:
            _ck_keys = set(_f.keys())
        _missing = sorted(n for n, _ in dit.named_parameters() if n not in _ck_keys)
        assert not _missing, f"--fast_init needs a checkpoint covering every parameter; {len(_missing)} missing, e.g. {_missing[:3]}"
        for _attempt in range(1, 4):
            load_trainable_weights(dit, args.resume_weights)
            _bad = [n for n, q in dit.named_parameters() if not torch.isfinite(q).all()]
            _bad += [n for n, b in dit.named_buffers() if b.is_floating_point() and not torch.isfinite(b).all()]
            if not _bad:
                break
            print(f"{TAG} rank{rank}: DiT has {len(_bad)} non-finite tensors after loading (e.g. {_bad[:2]}), reload attempt {_attempt}", flush=True)
            if _attempt == 3:
                raise RuntimeError(f"DiT had non-finite weights in 3 consecutive loads, rank{rank} exiting")
        if is_main:
            print(f"{TAG} fast_init: built without random init and loaded {args.resume_weights} ({len(_ck_keys)} tensors, every parameter covered) in {time.time() - t_fast:.0f}s", flush=True)
    else:
        dit = build_checked(lambda: build_dit(args, dtype, verbose=is_main), "DiT", rank)
    if args.ep:
        moe_layers_list = [int(s) for s in str(args.moe_layers).split(",") if s.strip()]
        blk = dit.blocks[moe_layers_list[0]]
        ep_store = EPM.ExpertStore(ep_layout, moe_layers_list, blk.dim, blk.ffn_dim, torch.device(device), dtype)
        EPM.set_store(ep_store)
        EPM.bind_block_indices(dit)
        global EP_STORE
        EP_STORE = ep_store
        if is_main:
            print(f"{TAG} EP expert store: {len(ep_store.tensors)} tensors on this GPU, {ep_store.numel / 1e9:.2f}B bf16 = {ep_store.numel * 2 / 2 ** 30:.1f} GiB", flush=True)
    if args.resume_weights:
        if args.ep:
            n_dense = EPM.load_dense_weights(dit, args.resume_weights)
            ep_store.load_from_safetensors(args.resume_weights, verbose=is_main)
            if is_main:
                print(f"{TAG} EP loaded weights {args.resume_weights}: {n_dense} dense keys + this GPU's experts", flush=True)
        elif not fast:                                                     # fast path already loaded (and scanned) above
            load_trainable_weights(dit, args.resume_weights)
        if is_main:
            print(f"{TAG} loaded weights {args.resume_weights}", flush=True)
    # The 55.93B model must be sharded before prepare (see the --ds_config help). zero.Init only handles parameters, not buffers,
    # and the skipped module.to(device) would also have moved the buffers (RoPE freqs etc.) to the GPU, so they are moved manually here.
    if args.ds_config:
        import deepspeed
        n_buf = 0
        for _, buf in dit.named_buffers():
            buf.data = buf.data.to(device)
            n_buf += 1
        t_init = time.time()
        deepspeed.zero.Init(module=dit, config_dict_or_path=args.ds_config, dtype=dtype,
                            remote_device=None, pin_memory=False, mem_efficient_linear=False)
        if is_main:
            print(f"{TAG} zero.Init in-place sharding done: {n_buf} buffers moved to GPU, took {time.time() - t_init:.0f}s "
                  f"(local shard per GPU {sum(p.ds_tensor.numel() for p in dit.parameters() if hasattr(p, 'ds_tensor')) * 2 / 2 ** 30:.1f} GiB)", flush=True)
    base_params, expert_params, gate_params = mark_trainable(dit)
    keys = trainable_state_keys(dit)
    # Three param groups in the order GROUP_BASE / GROUP_EXPERTS / GROUP_GATE; per-step learning rates are written by set_learning_rates
    groups = [{"params": base_params, "lr": args.learning_rate},
              {"params": expert_params, "lr": args.new_layers_lr},
              {"params": gate_params, "lr": args.gate_lr}]
    if args.ep:                                                   # EP: experts are not in DeepSpeed, keep only base / gate (an empty group breaks ZeRO-3 partitioning)
        assert len(expert_params) == 0, "with EP the DiT must not contain expert params"
        groups = [groups[0], groups[2]]
    if args.cpu_adam:            # with optimizer state offloaded to CPU, DeepSpeed's CPU Adam is required (DeepSpeed errors out otherwise)
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        opt = DeepSpeedCPUAdam(groups, weight_decay=args.weight_decay, adamw_mode=True)
        if is_main:
            print(f"{TAG} optimizer = DeepSpeedCPUAdam (state on CPU, adamw_mode)", flush=True)
    else:
        opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    # Speed-up (A5): mark each MoE layer's MoEFFN / MoEKVSelfAttention as a ZeRO-3 leaf module - by default ZeRO-3 hooks every submodule (35 experts x several Linear + gate)
    # and gathers / releases params one by one, thousands of small gathers per step; a leaf module gathers all its params at once on entry (DeepSpeed's recommended usage for MoE).
    # The math is unchanged (only when params are gathered); must be called before accelerator.prepare (DeepSpeed installs its hooks according to this flag during prepare).
    if args.z3_leaf_modules and not args.ep:      # off by default; not needed with EP (the MoE modules then only hold the gates)
        try:
            from deepspeed.utils import set_z3_leaf_modules
            from wan_transformer3d_kiwi_moe_v3_2 import MoEFFN, MoEKVSelfAttention
            leaf_modules = set_z3_leaf_modules(dit, [MoEFFN, MoEKVSelfAttention])
            if is_main:
                print(f"{TAG} ZeRO-3 leaf modules: {len(leaf_modules)} (MoEFFN + MoEKVSelfAttention, one each per MoE layer)", flush=True)
        except ImportError as e:
            if is_main:
                print(f"{TAG} warning: deepspeed.utils.set_z3_leaf_modules unavailable ({e}), MoE modules keep per-submodule ZeRO-3 hooks", flush=True)
    dit, opt = accelerator.prepare(dit, opt)                     # DeepSpeed takes over here: weights / grads / optimizer state are sharded over the 24 GPUs
    # Root-cause check: with offload_optimizer=cpu and no zero_force_ds_cpu_optimizer=false in the ds config, accelerate discards the given optimizer
    # and rebuilds it via map_pytorch_optim_to_deepspeed as DeepSpeedCPUAdam(adamw_mode=False) (True only for torch.optim.AdamW instances) -> Adam + coupled L2,
    # which shrinks the weights toward zero by lr every step. Assert that the optimizer actually running is DeepSpeedCPUAdam with adam_w_mode=True.
    if args.cpu_adam:
        _inner = getattr(getattr(dit, "optimizer", None), "optimizer", None)
        _mode = getattr(_inner, "adam_w_mode", None)
        assert _inner is not None and type(_inner).__name__ == "DeepSpeedCPUAdam" and _mode is True, \
            f"optimizer replaced / wrong mode: {type(_inner).__name__} adam_w_mode={_mode}; the ds config needs zero_force_ds_cpu_optimizer=false"
        if is_main:
            print(f"{TAG} optimizer check: {type(_inner).__name__} adam_w_mode={_mode} wd={[g['weight_decay'] for g in _inner.param_groups]} "
                  f"lr={[g['lr'] for g in _inner.param_groups]}", flush=True)
    accum = max(1, int(args.gradient_accumulation_steps))
    assert not (args.ep and accum > 1), "--gradient_accumulation_steps > 1 is not implemented for the EP path (ExpertStore reduces expert grads every backward)"
    if hasattr(dit, "gradient_accumulation_steps"):
        assert int(dit.gradient_accumulation_steps()) == accum, \
            f"DeepSpeed config gradient_accumulation_steps={dit.gradient_accumulation_steps()} != --gradient_accumulation_steps {accum} (engine.backward divides the loss by the config value)"
    if os.environ.get("MOE_DS_STEP_PATCH", "1") == "1" or args.ep:
        apply_ds_step_patches(dit, rank)                         # speed-up, see apply_ds_step_patches (required with EP: the clip factor must include the expert norm)
    if args.ep:
        assert args.resume_weights or args.resume_ds_state, "in EP mode the expert weights can only come from --resume_weights (full file) or --resume_ds_state (EP checkpoint)"
        ep_store.build_optimizer(lr=args.new_layers_lr, weight_decay=args.weight_decay)
        ep_store.install_grad_hooks()                             # expert grads are reduced + freed per tensor during backward (fixes the OOM at batch 5)
        if args.resume_weights:                                   # on the ds_state path the weights are loaded after load_state, so the self-check is done there
            assert ep_store.verify_replicas(), f"{TAG} rank{rank}: EP replica self-check failed (a shared expert differs between two GPUs or replicas differ across the 3 nodes)"
        if is_main:
            print(f"{TAG} EP expert optimizer: {ep_store.shard_numel / 1e9:.2f}B params in this GPU's shard (fp32 master weights + m/v in CPU pinned memory ~ {ep_store.shard_numel * 12 / 2 ** 30:.1f} GiB); replica self-check passed", flush=True)
    inner = accelerator.unwrap_model(dit)
    check_zero3_gather(inner, rank, world, device)
    # MLLM: build -> move to GPU -> scan weights -> self-test output finiteness with a fixed instruction + blank frames. In one run a rank passed the weight scan but its self-test output was nan
    # (the other 23 ranks were fine on the same input), i.e. a one-off bad state of that process, so the whole MLLM is rebuilt and retried, at most 3 times.
    from PIL import Image
    probe_frames = [Image.new("RGB", (args.width, args.height))] * args.num_frames
    for attempt in range(1, 4):
        mllm = build_checked(lambda: build_mllm(args).to(device), "MLLM(GPU)", rank)
        text_encoder = TextEncoder(mllm)
        where = text_encoder.self_test("Add a red ball.", probe_frames)
        if where is None:
            print(f"{TAG} rank{rank}: MLLM self-test passed (attempt {attempt}, attn={getattr(mllm.mllm_backbone.config, '_attn_implementation', '?')})", flush=True)
            break
        print(f"{TAG} rank{rank}: MLLM self-test failed, first non-finite at [{where}], attempt {attempt}, rebuilding", flush=True)
        del mllm, text_encoder
        gc.collect()  # use the module-level gc (imported at the top); a local `import gc` here would make gc a local of main(), and the normal path (self-test passes at once) would hit UnboundLocalError at gc.freeze() below
        torch.cuda.empty_cache()
    else:
        raise RuntimeError(f"rank{rank}: MLLM self-test failed 3 times in a row, exiting")
    vae = build_checked(lambda: build_vae(args, dtype).to(device), "VAE(GPU)", rank)
    if args.vae_compile != "off":      # speed-up: torch.compile the VAE encoder (only the frozen encoder, math unchanged; tens of seconds of compile on the first step; falls back to eager on failure)
        vae.model._eager_encoder = vae.model.encoder      # fallback if compilation fails at runtime (see vae_encode_batched)
        try:
            if args.vae_compile == "encoder":
                vae.model.encoder = torch.compile(vae.model.encoder, dynamic=False)
            elif args.vae_compile == "encoder_dyn":      # dynamic shapes: the VAE batch size varies (guard drops / only misses are encoded after cache hits), avoiding a recompile per batch size
                vae.model.encoder = torch.compile(vae.model.encoder, dynamic=True)
            elif args.vae_compile == "encoder_autotune":
                vae.model.encoder = torch.compile(vae.model.encoder, dynamic=False, mode="max-autotune-no-cudagraphs")
            elif args.vae_compile == "encode":
                vae.model.encode = torch.compile(vae.model.encode, dynamic=False)
            if is_main:
                print(f"{TAG} VAE torch.compile: {args.vae_compile}", flush=True)
        except Exception as e:
            if is_main:
                print(f"{TAG} warning: VAE torch.compile({args.vae_compile}) failed, using eager: {e!r}", flush=True)
    scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=1000, training=True)

    # ---- Cache: feature cache directory for multi-edit samples (see file header). The dataset side (kiwi_dataset_multi_moe) takes the directory as a constructor kwarg; if the signature lacks it, the cache is disabled with a warning;
    #      rank0 first renames + deletes in the background any directory left by a previous SIGKILL (the cache is not reused across launches), then creates an empty one; the other ranks wait for it
    from videox_fun.moe_common.kiwi_dataset_multi_moe import KiwiMultiEditDataset
    cache_dir = args.multi_cache_dir.strip()
    cache_kwargs = dataset_cache_kwargs(KiwiMultiEditDataset, cache_dir)
    if cache_dir and cache_kwargs is None:
        if is_main:
            print(f"{TAG} cache: warning: KiwiMultiEditDataset constructor has no multi_cache_dir / cache_dir parameter (no cache support on the dataset side), --multi_cache_dir ignored, caching disabled for this run", flush=True)
        cache_dir, cache_kwargs = "", {}
    cache_writer = None
    if cache_dir:
        if is_main:
            if args.cache_keep_existing:                # keep the cache on restart, do not clear leftovers (the hit ratio continues from the previous run)
                os.makedirs(cache_dir, exist_ok=True)
                print(f"{TAG} cache: keeping the existing cache directory {cache_dir} (--cache_keep_existing 1), the files count accumulates from this launch", flush=True)
            else:
                stale, _ = remove_cache_dir(cache_dir, wait_seconds=0)
                if stale:
                    print(f"{TAG} cache: leftover cache directories renamed {stale}, deleting in the background", flush=True)
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
                 [None] * len(paths_a) + ["multi"] * len(paths_b))]      # cache: records from paths_b (goku / CoinVE) get cache_group="multi"
    else:                                  # two stages: A then B (the old mlp-moe schedule; "exactly one epoch of OpenVE then one of multi" = stage_b_epochs 1)
        plan = [("A", paths_a, None, args.stage_a_epochs, None), ("B", paths_b, None, args.stage_b_epochs, ["multi"] * len(paths_b))]
    built = []
    for name, paths, repeats, epochs, cache_groups in plan:
        dataset, stats = build_dataset(paths, exclude_words, args, repeats, cache_groups, cache_kwargs)
        built.append((name, dataset, stats, epochs))
    # Check the in-flight memory / /dev/shm budget before building the DataLoader (workers x prefetch x batch x ~118 MB per sample x processes per node); raise above 60%
    check_loader_memory(args, accelerator.is_local_main_process)
    # Speed-up (A3): after building the datasets run one full GC, then freeze all surviving objects (removed from generational GC scans).
    # Reason: 4.54M records (mixed mode with copies: 2.31M single-instruction + 224k multi-instruction x 10) are ~14M dicts/lists/strs in Python that live until training ends;
    # the main process triggered a full GC about every 48 steps, each traversing these objects (measured 4 s), although they never become garbage in the training loop.
    # After freezing, GC only scans objects created during training; done before the DataLoader forks its workers, so the workers skip them too (fewer copy-on-write page touches).
    gc.collect()
    gc.freeze()
    for name, dataset, stats, epochs in built:
        _buckets = dataset.frame_buckets()                                 # per-sample frame counts (num_frames_bucket records), one value = old behaviour
        if len(set(_buckets)) > 1:
            sampler = BucketedShardedBatchSampler(len(dataset), args.batch_size, world, rank, args.seed, _buckets)
            if is_main:
                _bcount = {b: int(len(sampler._by_bucket[b])) for b in sampler.bucket_values}
                print(f"{TAG} stage {name}: frame buckets {_bcount} samples -> steps/epoch per bucket {sampler.steps_per_bucket} (batches homogeneous in frame count)", flush=True)
        else:
            sampler = ShardedBatchSampler(len(dataset), args.batch_size, world, rank, args.seed)
        loader = DataLoader(IndexTaggedDataset(dataset), batch_sampler=sampler, num_workers=args.dataset_num_workers,
                            prefetch_factor=args.dataloader_prefetch_factor or None, collate_fn=lambda b: b,
                            persistent_workers=bool(args.dataset_num_workers))
        steps = int(math.ceil(len(sampler) * epochs))
        stages.append({"name": name, "sampler": sampler, "loader": loader, "steps": steps, "stats": stats, "size": len(dataset)})
        if is_main:
            print(f"{TAG} stage {name}: {stats} total={len(dataset)} {len(sampler)} steps/epoch x {epochs} epochs = {steps} steps", flush=True)
    stage_start = [0, stages[0]["steps"]]                          # global start step of each stage (with a single stage stage_start[1] = total steps, so stage 0 is always used)
    max_steps = args.max_train_steps or sum(st["steps"] for st in stages)

    # ---- resume
    step0 = 0
    _sj = None
    if args.resume_ds_state:
        accelerator.load_state(args.resume_ds_state)
        with open(os.path.join(args.resume_ds_state, "step.json")) as f:
            _sj = json.load(f)
        if args.ep:                                               # EP: resume from an EP checkpoint: expert weights (shards written by node 0) + this GPU's expert optimizer state
            _step_dir = os.path.join(args.output_dir, "ckpt", f"step-{int(_sj['step'])}")
            ep_store.load_from_ep_shards(_step_dir)
            ep_store.load_optimizer(args.resume_ds_state)
            assert ep_store.verify_replicas(), "replicas inconsistent after loading the EP checkpoint"
            if is_main:
                print(f"{TAG} EP resume: expert weights {_step_dir}, optimizer state {args.resume_ds_state}", flush=True)
    elif args.resume_weights and args.resume_step_from:            # also continue the step / data position when loading weights only (batch change / switch to EP)
        with open(os.path.join(args.resume_step_from, "step.json")) as f:
            _sj = json.load(f)
        if is_main:
            print(f"{TAG} resume_weights + resume_step_from: step and data position continued from {args.resume_step_from} (fresh optimizer state)", flush=True)
    if _sj is not None:
        step0 = int(_sj["step"])
        # Resume with a different batch size, continuing from the data position reached. The per-rank sampler sequence mine = order[rank::world] depends only on
        # seed/epoch/rank/world, not on batch; at save time each rank consumed step x old_batch samples, i.e. (step x old_batch)/new_batch steps at the new batch. Requires the same world and exact division.
        _ckpt_gb = int(_sj.get("global_batch") or args.resume_ckpt_global_batch or 0)
        _now_gb = int(args.batch_size) * world
        if _ckpt_gb and _ckpt_gb != _now_gb:
            # A world-size change is also allowed. Each rank takes order[rank::world] of the global permutation (order depends only on seed/epoch); what was consumed at save time is
            # the prefix of step x old_effective_batch positions of the global permutation, independent of world; with the new world/batch take floor steps, the remainder (< new effective batch samples) is seen again.
            _ckpt_world = int(_sj.get("world") or world)
            _consumed = step0 * _ckpt_gb
            _new_step = _consumed // _now_gb
            _repeat = _consumed - _new_step * _now_gb
            if _ckpt_world == world:
                assert _repeat == 0, f"{step0 * (_ckpt_gb // world)} samples consumed per rank are not divisible by the new batch {args.batch_size}, cannot resume"
            if is_main:
                print(f"{TAG} resume with new batch/world: checkpoint global step {step0} x effective batch {_ckpt_gb} (world {_ckpt_world}) = {_consumed} samples consumed -> "
                      f"new effective batch {_now_gb} (world {world} x {args.batch_size}): global step {_new_step}, {_repeat} samples seen again "
                      f"(LR schedule / checkpoint counters follow the new step)", flush=True)
            step0 = _new_step
        if is_main:
            print(f"{TAG} resume DeepSpeed state @ step {step0}", flush=True)

    if is_main:
        os.makedirs(os.path.join(args.output_dir, "ckpt"), exist_ok=True)
        print(f"{TAG} start: world={world} batch={args.batch_size}(effective {world * args.batch_size}"
              f"{f' per step x accumulation {accum} = {world * args.batch_size * accum} per optimizer update' if accum > 1 else ''}) steps={max_steps} "
              f"moe_layers={args.moe_layers} experts={args.moe_num_experts}(={len(parse_sources(args.expert_sources))} sources x {args.expert_copies} copies, noise {args.expert_copy_noise}) top_k={args.moe_top_k} aux={args.moe_aux_loss_alpha} "
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
        pbar = tqdm(total=max_steps, initial=step0, desc="moe-v3.2", dynamic_ncols=True)

    def pixels_of(sample, pixel_key, frames_key):
        if sample.get(pixel_key) is not None:
            return uint8_video_to_pixels(sample[pixel_key], device, dtype)
        return frames_to_video_tensor(sample[frames_key], dtype=torch.float32).to(device, dtype)

    moe_ffns, moe_attns = inner.moe_ffns(), inner.moe_attns()
    moe_blocks = [int(b) for b in args.moe_layers.split(",")]          # one-to-one with moe_ffns / moe_attns, for log labels
    num_experts = args.moe_num_experts
    # Stat vector reduced across ranks every 10 steps: [sum loss_main, sum loss_motion, sum aux_attn, sum aux_ffn, sum total loss (weighted main/motion + aux), count]
    # followed by the summed "selection fraction" of each expert in each layer for the self-attention gate (layer x expert), then the same block for the FFN gate
    NUM_LOSS_SLOTS = 6
    per_moe = len(moe_ffns) * num_experts
    cache_off = NUM_LOSS_SLOTS + 2 * per_moe                                  # cache: cache counter slots start here (appended at the end; NUM_LOSS_SLOTS and the MoE offsets are unchanged)
    stat = torch.zeros(cache_off + len(CACHE_STAT_NAMES), device=device)
    cache_expect = {"latent": {}, "ctx": None}                              # cache: latent shape and ctx dim are recorded after the first VAE / MLLM encode of this run; hit samples must match
    # "category x expert" selection counts (summed over all MoE layers): [kind=2 (attn, ffn), category, expert]; every 10 steps reduced and row-normalized to P(expert i selected | sample category c)
    moe_kinds = [("attn", moe_attns), ("ffn", moe_ffns)]
    cat_stat = torch.zeros(len(moe_kinds), len(CATEGORIES), num_experts, device=device)
    timing = {"vae": 0.0, "mllm": 0.0, "dit": 0.0, "fwd": 0.0, "bwd": 0.0, "ep_red": 0.0, "ds_step": 0.0, "ep_opt": 0.0, "n": 0}      # per-stage time of this rank (s), averaged and logged every 10 steps; fwd (DiT forward) / bwd (backward + DeepSpeed step incl. CPU Adam) are timed with CUDA events, no extra syncs
    skipped_steps = 0                   # steps skipped due to bad samples / non-finite loss (not counted in step)
    consecutive_bad = 0                 # consecutive non-finite-loss steps on this rank; 5 in a row means this process's weights are bad, so exit for inspection
    step, t0 = step0, time.time()
    frames_sum, frames_n = 0.0, 0                                        # mean frame count of the steps in the logging window
    dit.train()

    while step < max_steps:
        # ---- current stage, epoch within the stage and steps consumed (resume goes through here too, so the data order continues exactly)
        stage_index = 1 if (len(stages) > 1 and step >= stage_start[1]) else 0
        stage = stages[stage_index]
        in_stage = step - stage_start[stage_index]
        steps_per_epoch = len(stage["sampler"])
        stage["sampler"].set_epoch(in_stage // steps_per_epoch)
        stage["sampler"].skip = in_stage % steps_per_epoch
        stage_end = min(max_steps, stage_start[stage_index] + stage["steps"])
        if is_main:
            print(f"{TAG} stage {stage['name']} epoch {in_stage // steps_per_epoch} resuming at step {in_stage % steps_per_epoch}, until global step {stage_end}", flush=True)

        check_pos = in_stage % steps_per_epoch      # data-position self-check: the first batch of this epoch must be batch check_pos of the sampler (each rank checks its own)
        loader_it = iter(stage["loader"])
        prefetched = None            # prefetch: next batch fetched in the previous step before the optimizer CPU part, with its VAE encoding kernels launched (batch, payloads, pending, cache_counts, src_latents, latents)

        def fetch_split_encode(step_for_log):
            """Fetch the next batch -> data-position self-check (first batch of the epoch) -> cache split -> launch the VAE encoding kernels (no host syncs in these stages, so they can be launched before CPU Adam and run on the GPU in parallel).
            Returns None when there is no next batch. Batch order and the GPU RNG consumption order of VAE sampling and noising are identical to no prefetch (VAE(k+1) still after noise(k) and before noise(k+1))."""
            nonlocal check_pos
            try:
                nb = next(loader_it)
            except StopIteration:
                return None
            if check_pos is not None:
                expect = stage["sampler"].epoch_batches()[check_pos]
                got = [s.get("_req_index") for s in nb]
                if got != expect:
                    raise RuntimeError(f"{TAG} rank{rank} data-position self-check failed: stage {stage['name']} epoch {stage['sampler'].epoch} expected batch {check_pos} of this epoch "
                                       f"{expect[:4]}..., got {got[:4]}... (sampler skip lost? see the comment in ShardedBatchSampler.__iter__)")
                if is_main:
                    print(f"{TAG} data-position self-check OK: stage {stage['name']} epoch {stage['sampler'].epoch} first batch = batch {check_pos} of this epoch (sample indices {expect[:3]}...)", flush=True)
                check_pos = None
            # ---- homogeneous frame count guard (defence in depth): the sampler cuts batches per frame bucket and the dataset replaces failed samples within the
            #      same bucket; should a sample with another frame count still appear, drop it instead of crashing in the stack (seen 10/21-10/23 before the dataset fix)
            if nb and len({int(s_.get("num_frames_eff") or 0) for s_ in nb}) > 1:
                _nf_major = max({int(s_.get("num_frames_eff") or 0) for s_ in nb}, key=lambda v: sum(1 for s_ in nb if int(s_.get("num_frames_eff") or 0) == v))
                _odd = [s_ for s_ in nb if int(s_.get("num_frames_eff") or 0) != _nf_major]
                print(f"{TAG} rank{rank}: dropping {len(_odd)} sample(s) whose frame count differs from the batch ({_nf_major}): {[s_.get('src_video_path') for s_ in _odd]}", flush=True)
                nb = [s_ for s_ in nb if int(s_.get("num_frames_eff") or 0) == _nf_major]
            # ---- 0. Cache split. Hit samples (the dataset worker read a cache file; they carry a "cache" key) have no pixels, only VAE distribution params + both ctx branches;
            #      payloads[i] = tensors of a hit sample / None, pending[i] = missed sample in a cache group (written to the cache after this step) / None; both are filtered together with guard drops
            nb, npl, npd, ncc = split_batch_for_cache(nb, cache_dir if cache_writer is not None else "", args, cache_expect, rank, step_for_log)
            # ---- 1. VAE encoding: source / target videos go to the encoder in groups of vae_encode_group (see vae_encode_batched)
            #      cache: only misses are encoded (mean / logvar stored into pending); hits are resampled from the cached params with the same formula; reassembled in batch order
            with torch.no_grad():
                nsrc, nlat = encode_latents_with_cache(vae, nb, npl, npd, args, device, dtype, pixels_of, cache_expect)   # each [B, C, F, H, W]
            return nb, npl, npd, ncc, nsrc, nlat

        while True:
            t_mark = sync_time()
            if prefetched is not None:                                                  # prefetch: the encoding kernels mostly finished during the previous step's CPU Adam; only the tail is waited for here
                batch, payloads, pending, cache_counts, src_latents, latents = prefetched
                prefetched = None
            else:
                fetched = fetch_split_encode(step)
                if fetched is None:
                    break
                batch, payloads, pending, cache_counts, src_latents, latents = fetched
            timing["vae"] += sync_time() - t_mark

            # ---- Guard 1 (local part): input finiteness. Videos occasionally decode to nan/inf, which permanently corrupts the weights once in the gradient (DeepSpeed bf16 has no fp16-style overflow skip).
            #      Checked per sample; bad samples are dropped from the batch with their paths printed; if no good sample is left on this rank, all ranks skip the step together
            #      (collectives require all 24 ranks in lockstep; one rank cannot skip backward alone).
            #      Speed-up (A4): guard 1 (here) and guard 1b (after the MLLM) each used to do an accelerator.reduce().item() - each a full 24-GPU sync
            #      plus waiting for the slowest rank, ~2 s for both. Now only the local flag bad_latent is computed here with local dropping, then MLLM encoding and a local bad_ctx,
            #      and both flags go into one vector for a single reduce (see "guards 1 + 1b merged" below). The decision semantics are unchanged:
            #      any rank with a fully bad latent batch -> all skip (consecutive_bad untouched); else any rank with a fully bad ctx batch -> all skip (consecutive_bad as before).
            #      Only difference: in a step where some rank has all latents bad, the other ranks have already drawn noise and run the MLLM before learning to skip (wasted once; such steps are very rare).
            keep = []
            for i, sample in enumerate(batch):
                if torch.isfinite(latents[i]).all() and torch.isfinite(src_latents[i]).all():
                    keep.append(i)
                else:
                    print(f"{TAG} rank{rank} step {step}: dropping bad sample (non-finite latent) src={sample.get('src_video_path', '?')} "
                          f"tgt={sample.get('tgt_video_path', '?')} prompt={sample['prompts_multi'][0][:80]!r}", flush=True)
            bad_latent = len(keep) == 0                 # this rank's whole latent batch is bad (local flag of guard 1; the global decision waits for the merged reduce)
            if len(keep) < len(batch):
                batch = [batch[i] for i in keep]
                src_latents, latents = src_latents[keep], latents[keep]
                payloads, pending = [payloads[i] for i in keep], [pending[i] for i in keep]      # cache: filter per-sample state in sync
            bsz = latents.size(0)                       # 0 when this rank's whole latent batch is bad: noising below yields empty tensors, the MLLM is skipped, all ranks continue after the merged reduce

            # ---- 2. flow-matching noise (independent timestep / sigma per sample)
            noise = torch.randn_like(latents)
            timestep_index = torch.randint(0, scheduler.num_train_timesteps, (bsz,))
            sigmas = scheduler.sigmas[timestep_index].to(device=device, dtype=latents.dtype).view(bsz, 1, 1, 1, 1)
            timesteps = scheduler.timesteps[timestep_index].to(device=device, dtype=dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            target = noise - latents


            # ---- 3. full instruction -> MLLM -> ctx; text-CFG draw
            # (under DeepSpeed, accelerate does not wrap forward in autocast, so bf16 autocast is applied explicitly:
            #  the original kiwi forward feeds fp32 inputs to bf16 weights in time_embedding etc. and only runs under autocast; the inference script does the same)
            #  cache: the draw / encoding / cached ctx for hits / extra branch for missed multi samples all live in encode_ctx_with_cache (see its docstring):
            #  the draw is still one torch.rand(()) per sample in batch order (RNG consumption identical to both the batched B1 path and the old per-sample path);
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
            bad_ctx = len(keep) == 0 and not bad_latent   # good latents but the whole ctx batch is bad (local flag of guard 1b; with all latents bad the ctx list is empty and guard 1 handles it)
            # ---- Guards 1 + 1b merged into one reduce (speed-up A4): vector [some rank has all latents bad, some rank has all ctx bad], one all-reduce + one sync
            bad_flags = accelerator.reduce(torch.tensor([float(bad_latent), float(bad_ctx)], device=device), reduction="sum")
            any_bad_latent, any_bad_ctx = (bad_flags > 0).tolist()
            if any_bad_latent:                              # = the global branch of guard 1 (consecutive_bad untouched)
                skipped_steps += 1
                if is_main:
                    print(f"{TAG} step {step}: a rank has only bad samples, all ranks skip this step (skipped so far {skipped_steps})", flush=True)
                continue
            if any_bad_ctx:                                 # = the global branch of guard 1b
                skipped_steps += 1
                consecutive_bad = consecutive_bad + 1 if len(keep) == 0 else 0
                if consecutive_bad >= 5:
                    raise RuntimeError(f"rank{rank}: MLLM ctx non-finite for {consecutive_bad} consecutive steps, this GPU is faulty, exiting")
                if is_main:
                    print(f"{TAG} step {step}: a rank has a fully non-finite ctx batch, all ranks skip this step (skipped so far {skipped_steps})", flush=True)
                continue
            if len(keep) < len(batch):
                batch = [batch[i] for i in keep]
                ctx_list = [ctx_list[i] for i in keep]
                src_latents, latents = src_latents[keep], latents[keep]
                noise, noisy_latents, target = noise[keep], noisy_latents[keep], target[keep]
                sigmas, timesteps = sigmas[keep], timesteps[keep]
                pending = [pending[i] for i in keep]                  # cache: only samples passing guards 1 + 1b are written
                bsz = latents.size(0)
            # ---- Cache: for missed multi samples, distribution params + both ctx branches are moved to CPU (bf16) and handed to the writer threads (dropped if the queue is full, failures only counted; never blocks training);
            #      the 6 tensors are checked for finiteness again before enqueueing (guard 1b did not check the extra ctx branch; non-finite ones are not written), see enqueue_cache_payloads
            if cache_writer is not None:
                enqueue_cache_payloads(cache_writer, pending, args, step, rank)

            # ---- 4. DiT forward (the load-balancing loss is attached to the graph inside the MoE layers), under bf16 autocast
            # Gradient accumulation: this micro-batch is an update boundary iff the step counter after this batch is a multiple of accum
            # (accum=1: always). DeepSpeed's engine-level override is mirrored into the ZeRO-3 optimizer, which then accumulates the partitioned
            # grads across micro-batches and copies them to the fp32 optimizer grads only on the boundary; engine.backward divides the loss by accum.
            # Skipped batches (guards) do not advance step, so a skipped boundary is simply retried with the next batch and all ranks stay in lockstep.
            is_boundary = ((step + 1) % accum == 0)
            dit.set_gradient_accumulation_boundary(is_boundary=is_boundary)
            t_mark = sync_time()
            patch = inner.patch_size
            f_lat, h_lat, w_lat = latents.shape[2:]
            seq_len = (f_lat // patch[0]) * (h_lat // patch[1]) * (w_lat // patch[2])
            ev_f0 = torch.cuda.Event(enable_timing=True); ev_f0.record()   # forward timing events (read at the logging step, no sync)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = dit(x=[noisy_latents[i] for i in range(bsz)], t=timesteps, mllm_context=ctx_list, seq_len=seq_len,
                           vae_source_input=src_latents, sigma=sigmas.reshape(bsz))
            ev_f1 = torch.cuda.Event(enable_timing=True); ev_f1.record()

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

            # ---- Guard 2: loss finiteness (non-finite on any rank -> all skip, no backward, weights untouched)
            bad_local = torch.tensor(float(not torch.isfinite(loss.detach())), device=device)
            consecutive_bad = consecutive_bad + 1 if bad_local.item() > 0 else 0
            if consecutive_bad >= 5:
                raise RuntimeError(f"rank{rank}: loss non-finite for {consecutive_bad} consecutive steps (different samples each time), this process's weights/environment are broken, exiting")
            if accelerator.reduce(bad_local, reduction="sum").item() > 0:
                skipped_steps += 1
                if bad_local.item() > 0:
                    print(f"{TAG} rank{rank} step {step}: non-finite loss main={float(loss_main):.4g} aux={float(loss_aux):.4g} "
                          f"batch samples src={[s_.get('src_video_path', '?') for s_ in batch]} tgt={[s_.get('tgt_video_path', '?') for s_ in batch]}", flush=True)
                if is_main:
                    print(f"{TAG} step {step}: a rank has a non-finite loss, all ranks skip this step (skipped so far {skipped_steps})", flush=True)
                del pred, loss
                continue

            # ---- 6. backward + update (accelerate's DeepSpeed wrapper also does the step inside backward; clipping is in the ds config)
            lr_new, lr_gate = set_learning_rates(opt, step, args)   # this step's learning rates (routed experts / gates transition linearly with step, the rest constant)
            ev_b0 = torch.cuda.Event(enable_timing=True); ev_b0.record()   # backward + DeepSpeed step (incl. reduce-scatter / CPU Adam / copy-back)
            if args.ep:
                # EP: split accelerate's backward wrapper (engine.backward + engine.step): after backward, compute the expert grads (paired all_reduce + DP shard / world)
                # and their squared sum, which DeepSpeed's clip patch merges into the global norm; engine.step updates the dense part; then the experts are updated with the same clip factor (sharded CPU Adam + all_gather)
                dit.set_gradient_accumulation_boundary(is_boundary=True)   # EP path: accum == 1 (asserted after prepare)
                dit.backward(loss)
                ev_bw = torch.cuda.Event(enable_timing=True); ev_bw.record()   # with EP, bwd = backward only (compute stream); reduction / dense step / expert Adam are timed separately
                t_ep = time.time()
                ep_sq = ep_store._reduce_grads()                            # per-layer bucket reduce (side stream) -> final sync -> fp32 shards are in pinned m.grad
                dit.optimizer._moe_ep_norm_sq = ep_store.expert_norm_sq(ep_sq)
                timing["ep_red"] += time.time() - t_ep
                if args.vae_prefetch and prefetched is None:
                    torch.cuda.empty_cache()                                 # after backward, return fragmented segments to the driver to free a contiguous block for the next batch's few-GB VAE encoding temporaries
                    prefetched = fetch_split_encode(step + 1)                # prefetch: launch the next batch's VAE encoding kernels now, overlapping with the two CPU Adam stages below
                t_ds = time.time()
                dit.step()
                timing["ds_step"] += time.time() - t_ds                      # DeepSpeed dense part (reduce-scatter tail / CPU Adam / copy-back)
                t_ep = time.time()
                ep_store.step(float(getattr(dit.optimizer, "_moe_ep_clip_coef", 1.0)), lr_new)
                timing["ep_opt"] += time.time() - t_ep
                for _k, _v in ep_store.timing.items():                       # clip / cpu_adam / gather stages
                    timing[f"ep_{_k}"] = timing.get(f"ep_{_k}", 0.0) + float(_v)
            else:
                # exactly equivalent to accelerate's DeepSpeed wrapper (engine.backward + engine.step), split only to launch the next batch's VAE encoding before CPU Adam;
                # with gradient accumulation the engine step (reduce tail / CPU Adam / copy-back) runs only on boundary micro-batches, backward on every one
                dit.set_gradient_accumulation_boundary(is_boundary=is_boundary)
                dit.backward(loss)
                if args.vae_prefetch and prefetched is None:
                    prefetched = fetch_split_encode(step + 1)                # prefetch
                if is_boundary:
                    dit.step()
                    opt.step()
                    opt.zero_grad()
                if args.opt_probe and is_main and is_boundary and (step + 1) % args.opt_probe == 0:
                    opt_probe(dit, step + 1, TAG)
                if args.opt_probe_wd0_at and (step + 1) == args.opt_probe_wd0_at:
                    for _g in opt.param_groups:
                        _g["weight_decay"] = 0.0
                    if is_main:
                        print(f"{TAG} opt_probe: after step {step + 1} weight_decay set to 0 in all param groups (now {[g['weight_decay'] for g in opt.param_groups]})", flush=True)
            ev_b1 = torch.cuda.Event(enable_timing=True); ev_b1.record()
            timing["dit"] += sync_time() - t_mark
            timing["fwd"] += ev_f0.elapsed_time(ev_f1) / 1000.0        # sync_time already synchronized, reading the events does not wait
            timing["bwd"] += ev_b0.elapsed_time(ev_bw if args.ep else ev_b1) / 1000.0   # EP: backward only; otherwise backward + step
            timing["n"] += 1
            frames_sum += float(batch[0].get("num_frames_eff") or args.num_frames) if batch else 0.0; frames_n += 1
            step += 1

            # ---- 7. statistics and logging
            stat[0] += float(loss_main.detach()); stat[1] += float(loss_motion.detach()) if loss_motion is not None else 0.0
            stat[2] += float(loss_aux_attn); stat[3] += float(loss_aux_ffn); stat[4] += float(loss.detach()) + float(loss_aux); stat[5] += 1
            if cache_writer is not None:                                    # cache: hit / miss / other + writer counters, appended at the end of stat
                stat[cache_off:cache_off + len(CACHE_STAT_NAMES)] += torch.tensor(cache_counts + list(cache_writer.drain_counts()), device=device, dtype=stat.dtype)
                # count_files_async() is no longer called every 100 steps: scandir over 70k entries on a network filesystem takes 100+ s, and the background thread running Python per entry competes with the main thread for the GIL,
                # causing a 110~140 s spike every 100 steps, worse as the directory grows (~5 s/step at 220k entries). The accumulated write count of all ranks approximates the file count instead.
            for layer_i, (attn, ffn) in enumerate(zip(moe_attns, moe_ffns)):   # per layer and expert: token mean of 1 if selected / 0 if not
                start = NUM_LOSS_SLOTS + layer_i * num_experts
                stat[start:start + num_experts] += attn.last_selected
                stat[per_moe + start:per_moe + start + num_experts] += ffn.last_selected
            # category x expert: per-sample selection counts of each expert (all tokens, all slots) accumulated by sample category
            cat_idx = torch.tensor([CATEGORY_INDEX.get((s_.get("turn_types_qwen") or ["unknown"])[0], CATEGORY_INDEX["unknown"]) for s_ in batch], device=device)
            for kind_i, (_, modules) in enumerate(moe_kinds):
                for m in modules:
                    per_sample = F.one_hot(m.last_topk_idx.view(bsz, -1), num_experts).sum(dim=1).float()     # [B, N]
                    cat_stat[kind_i].index_add_(0, cat_idx, per_sample)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{float(loss.detach()):.4f}", aux=f"{float(loss_aux):.4f}", refresh=False)
            if step % 10 == 0:
                # peak GPU memory (within this interval, max over 24 GPUs) - the 55.9B model is close to the 80 GB limit, so the headroom must be visible in the logs
                mem = accelerator.reduce(torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                                                      device=device, dtype=torch.float64), reduction="max").tolist()
                torch.cuda.reset_peak_memory_stats(device)
                totals = accelerator.reduce(stat.clone(), reduction="sum").tolist()
                cat_totals = accelerator.reduce(cat_stat.clone(), reduction="sum")            # [kind, category, expert]
                if is_main:
                    count = max(totals[5], 1)                                       # = 10 steps x 24 GPUs
                    scalars = {"loss/main": totals[0] / count, "loss/motion": totals[1] / count,
                               "loss/aux_attn": totals[2] / count, "loss/aux_ffn": totals[3] / count, "loss/aux": (totals[2] + totals[3]) / count,
                               "loss/total": totals[4] / count,
                               "lr/base": args.learning_rate, "lr/new_layers": lr_new, "lr/gate": lr_gate,
                               "skipped_steps": skipped_steps, "stage": stage_index, "data/frames_mean": frames_sum / max(frames_n, 1),
                               "mem/alloc_max_gb": mem[0] / 2 ** 30, "mem/reserved_max_gb": mem[1] / 2 ** 30}
                    # one folder per MoE layer (moe_attn_L{block} / moe_ffn_L{block}) with one curve per expert: fraction of tokens selecting it
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
                    # category x expert conditional selection rate P(expert i | category c) = selections of expert i in samples of that category / all selections in that category; specialization index = max gap of each expert across categories
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
                        for key, val in timing.items():
                            if key != "n":
                                scalars[f"time/{key}"] = val / timing["n"]
                    grad_norm = (getattr(dit.optimizer, "_moe_global_norm", None) if args.ep else      # EP: global norm of dense + experts combined
                                 (dit.get_global_grad_norm() if hasattr(dit, "get_global_grad_norm") else None))
                    if grad_norm is not None:
                        scalars["grad/norm"] = float(grad_norm)
                    cache_note = ""
                    if cache_writer is not None:                            # cache: hit ratio (hits / multi-sample visits), share of all samples, writes / drops / failures, file count, this rank's queue
                        c_hit, c_miss, c_other, c_ok, c_fail, c_drop = totals[cache_off:cache_off + len(CACHE_STAT_NAMES)]
                        cache_writer.files = int(getattr(cache_writer, "files_est", 0) + c_ok); cache_writer.files_est = cache_writer.files   # accumulated writes ~ file count (the directory is no longer scanned)
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
                          f"(fwd/{'bwd' if args.ep else 'bwd+step'}={timing['fwd'] / max(timing['n'], 1):.1f}/{timing['bwd'] / max(timing['n'], 1):.1f}s"
                          + (f", ep_red/ds_step/ep_opt={timing['ep_red'] / max(timing['n'], 1):.1f}/{timing['ds_step'] / max(timing['n'], 1):.1f}/"
                             f"{timing['ep_opt'] / max(timing['n'], 1):.1f}s(clip/adam/gather={timing.get('ep_clip', 0.0) / max(timing['n'], 1):.1f}/"
                             f"{timing.get('ep_cpu_adam', 0.0) / max(timing['n'], 1):.1f}/{timing.get('ep_gather', 0.0) / max(timing['n'], 1):.1f})" if args.ep else "") + ") "
                          f"mem={mem[0] / 2 ** 30:.1f}/{mem[1] / 2 ** 30:.1f}GiB frames={frames_sum / max(frames_n, 1):.0f} "
                          + " ".join(spread_note) + (" " + cache_note if cache_note else ""), flush=True)
                stat.zero_()
                cat_stat.zero_()
                frames_sum, frames_n = 0.0, 0
                for _k in timing:
                    timing[_k] = 0.0
                timing["n"] = 0
            if step % args.save_steps == 0:
                save_trainable_weights(accelerator, dit, keys, step, args)
            if step % args.ds_save_steps == 0:
                save_ds_state(accelerator, step, args)
            if step >= stage_end:
                break

    if args.final_save:                                  # smoke tests pass 0 to skip the ~780 GB final save
        save_trainable_weights(accelerator, dit, keys, step, args)
        if step % args.ds_save_steps != 0:
            save_ds_state(accelerator, step, args)
        if is_main:
            final_dir = os.path.join(args.output_dir, "ckpt", "final_components")
            shutil.rmtree(final_dir, ignore_errors=True)
            shutil.copytree(os.path.join(args.output_dir, "ckpt", f"step-{step}"), final_dir)
    if is_main and pbar is not None:
        pbar.close()
    if cache_writer is not None:                         # cache: all ranks drain their write queues -> barrier -> rank0 deletes the cache directory (the cache must be removed when training ends)
        cache_writer.close()
        accelerator.wait_for_everyone()
        if is_main and not args.cache_delete_on_done:   # do not delete automatically at the end; leave the decision to the user
            os.makedirs(os.path.join(args.output_dir, "flags"), exist_ok=True)
            with open(os.path.join(args.output_dir, "flags", "CACHE_KEPT_ASK_USER"), "w") as f:
                f.write(f"{time.strftime('%F %T')} training finished, cache directory {cache_dir} kept; the user decides whether to delete it\n")
            print(f"{TAG} cache: training finished, cache directory {cache_dir} kept (--cache_delete_on_done 0); delete it manually if no longer needed", flush=True)
        elif is_main:
            t_rm = time.time()
            targets, done = remove_cache_dir(cache_dir, wait_seconds=900)
            print(f"{TAG} cache: deleting cache directories {targets} {'done' if done else 'not finished within 15 minutes, the launcher EXIT trap deletes the rest in the background'}, took {time.time() - t_rm:.0f}s", flush=True)
    if async_logger is not None:
        async_logger.close()                             # flush the remaining queued logs before shutting down
    accelerator.end_training()
    if is_main:
        print(f"{TAG} DONE @ step {step}", flush=True)


if __name__ == "__main__":
    main()
