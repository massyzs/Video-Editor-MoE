#!/usr/bin/env python
"""mlp-moe training script: full training of the whole DiT (MoE variant) with DeepSpeed ZeRO-3, 3 nodes / 24 GPUs.

The model lives elsewhere: DiT architecture in wan_transformer3d_kiwi_moe.py; assembly / weight loading / MLLM / VAE in moe_v3_model.py.
This file only has: data (two stages) → batching → noising → forward → loss → update → checkpointing / resume; the loss follows videoxfun wan2.2, inlined in the loop.

Two-stage curriculum (data is switched inside the same process, no restart):
  Stage A  --stage_a_data  (OpenVE single instruction; --exclude_prompt_words drops samples containing given words)  for stage_a_epochs epochs
  Stage B  --stage_b_data  (goku + CoinVE multi-instruction, sub-instructions joined into one full instruction with " Then ")  for stage_b_epochs epochs
Each step:
  1. take a batch of B samples (random, no grouping)
  2. VAE-encode source / target videos → sample sigma and noise independently per sample;  noisy = (1-σ)·tgt + σ·noise;  target = noise - tgt
  3. MLLM encodes the full instruction → ctx; with probability text_drop_prob a sample gets the empty-instruction ctx (unconditional branch for text CFG)
  4. DiT forward (the MoE layers' load-balancing loss is already attached to the graph, see wan_transformer3d_kiwi_moe.AddAuxiliaryLoss)
  5. loss = custom_mse_loss(pred, target) (+ motion difference term); on backward the aux-loss gradient flows into the gates automatically
  6. DeepSpeed step (gradient clipping is in the ds config); full update. Three learning-rate tiers (see the learning-rate section):
       original params       always learning_rate
       routed experts (new)  start at new_layers_lr, decay linearly to learning_rate within new_layers_lr_steps steps, then constant
       gates                 start at gate_lr, linear to gate_lr_end within the same number of steps, then constant
Logging (tensorboard + wandb, every 10 steps, averaged over 24 GPUs × 10 steps): loss/main, loss/motion, loss/aux, loss/total, lr/*, grad/norm, time/{vae,mllm,dit} (rank0 seconds per step),
  route_{ffn|attn}/{category}/expert{i} = P(expert i selected | sample category c) (direct measure of specialization), route/{kind}_spread = mean over experts of the max difference across categories,
  and the per-expert "selection rate" of every MoE layer, moe_L{block}/expert{i} (token mean of 1 if selected, 0 otherwise; one tensorboard folder per layer)
Checkpoints:
  every save_steps steps     ckpt/step-N/moe_v3_trainable.safetensors  trainable params in bf16 (full training = the whole DiT; for inference / weights-only resume) + meta.json
  every ds_save_steps steps  ckpt/ds_state_step-N/                     full DeepSpeed state (weights + optimizer, 24 shards, for resume); only the latest one is kept
resume: --resume_ds_state <ds_state dir> (needs the same 24 GPUs)  or  --resume_weights <safetensors> (weights only, fresh optimizer)
Multi-node launch: train_multinode.sh (same directory); single-node smoke test: train_smoke.sh.
"""
import argparse
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

from moe_v3_model import TextEncoder, build_dit, build_mllm, build_vae, load_trainable_weights, mark_trainable, trainable_state_keys  # noqa: E402

TAG = "[moe-v3]"


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
    p.add_argument("--experts_merged", required=True, help="experts_merged.safetensors produced by merge_experts.py")
    p.add_argument("--use_gradient_checkpointing", type=int, default=1)
    # MoE
    p.add_argument("--moe_layers", default="1,3,5,7,9,11,13,15,17,19,21,23,25,27", help="0-based block indices = layers 2,4,...,28 in 1-based numbering")
    p.add_argument("--moe_num_experts", type=int, default=10)
    p.add_argument("--moe_top_k", type=int, default=3, help="routed experts activated per token (an earlier run used 2)")
    p.add_argument("--moe_aux_loss_alpha", type=float, default=0.001, help="load-balancing auxiliary loss coefficient (an earlier run used 0.01)")
    p.add_argument("--expert_copy_noise", type=float, default=0.01, help="perturbation magnitude of the second expert copy (relative to the weight std)")
    # Data
    p.add_argument("--stage_a_data", required=True, help="stage A json/jsonl files, comma-separated")
    p.add_argument("--stage_b_data", required=True, help="stage B json/jsonl files, comma-separated")
    p.add_argument("--stage_a_epochs", type=float, default=1.0)
    p.add_argument("--stage_b_epochs", type=float, default=2.0)
    p.add_argument("--exclude_prompt_words", default="overlay", help="comma-separated; samples whose instruction contains any of these words (case-insensitive) are dropped")
    p.add_argument("--dataset_base_path", default="", help="prefix for relative video paths in the data json; empty = use paths as-is (absolute paths are unaffected)")
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--max_pixels", type=int, default=399360)
    p.add_argument("--dataset_num_workers", type=int, default=4)
    p.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    p.add_argument("--compound_joiner", default=" Then ", help="CoinVE multi-instruction samples (no native full instruction): sub-instructions are joined into one sentence with this")
    p.add_argument("--vae_encode_group", type=int, default=4, help="videos per VAE encode call (source and target grouped separately); 1 = per sample (old behavior)")
    # Training
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4, help="per-GPU batch; effective batch = 24 × batch_size")
    p.add_argument("--max_train_steps", type=int, default=0, help="0 = derived from the epochs of the two stages")
    p.add_argument("--learning_rate", type=float, default=1e-5, help="base learning rate of full training; original params (including the shared part of MoE layers) always use it")
    p.add_argument("--new_layers_lr", type=float, default=1e-4, help="initial learning rate of the new layers (routed experts + gates)")
    p.add_argument("--new_layers_lr_steps", type=int, default=2000, help="steps for the new layers to go linearly from new_layers_lr to learning_rate (then same as other params); the gate transition uses the same steps")
    p.add_argument("--gate_lr", type=float, default=1e-3, help="initial gate learning rate")
    p.add_argument("--gate_lr_end", type=float, default=1e-4, help="gate learning rate after new_layers_lr_steps steps (linear transition to it, then constant)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--text_drop_prob", type=float, default=0.1, help="text CFG: per-sample probability of blanking the instruction")
    p.add_argument("--motion_sub_loss_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--save_steps", type=int, default=250, help="interval (steps) for saving the bf16 trainable weights")
    p.add_argument("--ds_save_steps", type=int, default=1000, help="interval (steps) for saving the full DeepSpeed state (for resume)")
    p.add_argument("--resume_ds_state", default="", help="ckpt/ds_state_step-N directory")
    p.add_argument("--resume_weights", default="", help="load moe_v3_trainable.safetensors only, fresh optimizer")
    # Logging
    p.add_argument("--use_wandb", type=int, default=0)
    p.add_argument("--wandb_project", default="video-editor-moe")
    p.add_argument("--wandb_run_name", default="moe_v3")
    return p.parse_args(argv)



# =========================================================================== #
#  Sample categories / batched VAE encoding / timing                          #
# =========================================================================== #
CATEGORIES = ["add", "remove", "replace", "style", "background", "multi", "unknown"]     # rows of the "category × expert" log matrix
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORIES)}
OPENVE_DIR_TO_CATEGORY = {"local_add": "add", "local_remove": "remove", "local_change": "replace", "global_style": "style", "background_change": "background"}


def category_of(record):
    """Edit category of a sample (log statistics only, never fed to the model): original OpenVE records use the directory category cat,
    the five-class subset uses cls; goku / CoinVE use turn_types_qwen: one shared sub-instruction type is the category, mixed types give multi."""
    if record.get("cls") in CATEGORY_INDEX:
        return record["cls"]
    if record.get("cat") in OPENVE_DIR_TO_CATEGORY:
        return OPENVE_DIR_TO_CATEGORY[record["cat"]]
    types = [t for t in (record.get("turn_types_qwen") or []) if t]
    if types:
        return types[0] if all(t == types[0] for t in types) else "multi"
    return "unknown"


def vae_encode_batched(vae, pixels_list, group_size):
    """Concatenate several [1, 3, F, H, W] videos into one batch for the VAE encoder; returns the sampled latents [B, C, f, h, w].
    AutoencoderKLWan3_8.encode loops over samples, but the inner vae.model.encode supports a batch dim (chunk loop and feat_cache both carry b),
    so the inner call is used directly, in groups of group_size to bound peak memory (encoding runs before DiT activations exist). Serial per-sample encoding dominated step time."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    if not hasattr(vae, "model") or not hasattr(vae.model, "encode") or not hasattr(vae, "scale"):
        return torch.cat([vae.encode(x)[0].sample() for x in pixels_list], dim=0)     # other VAE classes: fall back to per-sample
    outs = []
    for i in range(0, len(pixels_list), max(1, group_size)):
        x = torch.cat(pixels_list[i:i + max(1, group_size)], dim=0)
        outs.append(DiagonalGaussianDistribution(vae.model.encode(x, vae.scale)).sample())
    return torch.cat(outs, dim=0)


def sync_time():
    """For timing: drain the GPU queue before reading the clock, so the VAE / MLLM / DiT timings are real."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# =========================================================================== #
#  Data                                                                       #
# =========================================================================== #
def join_goku_style(prompts):
    """Join sub-instructions into one sentence in the wording of GOKU's native total_instruction_en:
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
      OpenVE            the prompt is already a single instruction
      GOKU multi        a record covers only the first steps of a case (the target video is intermediate), while the native total_instruction_en
                        describes all steps (incl. camera/style steps we do not use) and does not match the target video, so the record's own
                        sub-instructions are joined in GOKU's native wording (join_goku_style)
      CoinVE multi      the source data has only a list of sub-instructions, no native full instruction; they are joined with joiner"""
    prompts = record["prompt"] if isinstance(record["prompt"], list) else [record["prompt"]]
    if re.search(r"videos/case_\d+/", record["src_video"]):
        return join_goku_style(prompts)
    return joiner.join(prompts)


def load_records(meta_paths, exclude_words, joiner):
    """Several json / jsonl files → records in a unified schema (one full instruction each); samples whose instruction contains exclude_words are dropped."""
    pattern = re.compile("|".join(re.escape(w) for w in exclude_words), re.I) if exclude_words else None
    records, stats = [], []
    for meta_path in meta_paths:
        kept = dropped = 0
        with open(meta_path) as f:
            raw_records = json.load(f) if meta_path.endswith(".json") else [json.loads(line) for line in f if line.strip()]
        for record in raw_records:
            prompt = full_prompt_of(record, joiner)
            if pattern is not None and pattern.search(prompt):
                dropped += 1
                continue
            records.append({"src_video": record["src_video"], "tgt_video": record["tgt_video"], "prompt": [prompt],
                            "n_turns": 1, "mask_mode": "global",
                            "turn_types_qwen": [category_of(record)]})      # category is only for the "category × expert" log matrix; the dataset passes it through to the sample unchanged
            kept += 1
        stats.append(f"{os.path.basename(meta_path)}:{kept}(dropped {dropped})")
    return records, " + ".join(stats)


def build_dataset(meta_paths, exclude_words, args):
    """Records → temporary jsonl → KiwiMultiEditDataset (reads videos / frames, no mask)."""
    import tempfile
    from videox_fun.moe_common.kiwi_dataset_multi_moe import KiwiMultiEditDataset
    records, stats = load_records(meta_paths, exclude_words, args.compound_joiner)
    fd, merged_path = tempfile.mkstemp(suffix=".jsonl", prefix="moe_v3_data_")
    with os.fdopen(fd, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    dataset = KiwiMultiEditDataset(
        metadata_path=merged_path, base_path=args.dataset_base_path, max_turns=0, mask_loss=False, subset_single_prob=0.0,
        num_frames=args.num_frames, height=args.height, width=args.width, max_pixels=args.max_pixels, repeat=1,
        require_ref_image=False, pad_frames_to_num=True, return_pixels=True)
    os.unlink(merged_path)
    assert len(dataset) == len(records), f"dataset filtered out {len(records) - len(dataset)} records"
    return dataset, stats


class IndexTaggedDataset(torch.utils.data.Dataset):
    """Data-position self-check: adds `_req_index` = the index requested by the sampler to each returned sample dict (the *requested* index even if
    the inner retry swapped the sample). The training loop uses it to verify that each epoch's first batch is the sampler's batch number skip
    (guards against silent replays such as DataLoader's double iter() swallowing skip). Other attributes pass through to the inner dataset."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        s["_req_index"] = int(idx)
        return s

    def __getattr__(self, name):
        base = self.__dict__.get("base")           # during pickle/copy reconstruction __dict__ is still empty: must raise AttributeError, not KeyError
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)


class ShardedBatchSampler:
    """Each epoch shuffles all samples with seed+epoch, splits them evenly across ranks and cuts fixed-size batches (the incomplete tail is dropped).
    skip = batches already consumed in this epoch (persistent, set by the training loop at each epoch start; see __iter__). All ranks get the same batch count per epoch."""

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
        # Data-position fix: the old `skip, self.skip = self.skip, 0` applied skip only once. But torch DataLoader calls iter(sampler) twice when building
        # an iterator (once in _BaseDataLoaderIter.__init__, once in _reset; with persistent_workers once more per later _reset); the first call reset skip
        # to 0 and batches came from the second → every resume replayed the epoch from batch 0 (not triggered with num_workers=0).
        # Now skip is persistent and set explicitly by the training loop at each epoch start (resume = in_stage % steps_per_epoch, later epochs = 0);
        # the training loop also self-checks the position of each epoch's first batch.
        return iter(self.epoch_batches()[self.skip:])

    def __len__(self):
        return (self.num_samples // self.num_replicas) // self.batch_size


# =========================================================================== #
#  Loss (original videoxfun wan2.2)                                           #
# =========================================================================== #
def custom_mse_loss(noise_pred, target, weighting=None, threshold=50.0):
    """Element-wise MSE, drop outliers with |residual| > threshold, multiply by the sd3 weighting (always 1 for scheme=none), then take the mean."""
    noise_pred = noise_pred.float()
    target = target.float()
    mse = F.mse_loss(noise_pred, target, reduction="none")
    mask = ((noise_pred - target).abs() <= threshold).float()
    masked = mse * mask
    if weighting is not None:
        masked = masked * weighting
    return masked.mean()


# =========================================================================== #
#  Learning rates                                                             #
# =========================================================================== #
GROUP_BASE, GROUP_EXPERTS, GROUP_GATE = 0, 1, 2      # indices of the optimizer's three param groups (the optimizer is built in this order)


def linear_then_constant(step, lr_start, lr_end, num_steps):
    """Linear from lr_start to lr_end over the first num_steps steps, then constant lr_end. Based on the global step, so it continues across resume."""
    if step >= num_steps:
        return lr_end
    return lr_start + (lr_end - lr_start) * step / num_steps


def lr_of_new_layers(step, args):
    """Routed experts (new layers): new_layers_lr → learning_rate over new_layers_lr_steps steps. All other original params stay at learning_rate."""
    return linear_then_constant(step, args.new_layers_lr, args.learning_rate, args.new_layers_lr_steps)


def lr_of_gate(step, args):
    """Gates: gate_lr → gate_lr_end, same transition length as the routed experts (new_layers_lr_steps)."""
    return linear_then_constant(step, args.gate_lr, args.gate_lr_end, args.new_layers_lr_steps)


def set_learning_rates(opt, step, args):
    """Write this step's learning rates into the optimizer's three param groups. After DeepSpeed wrapping, opt.param_groups are the underlying
    AdamW groups, so setting lr directly takes effect. Returns (routed-expert lr, gate lr) for logging."""
    lr_new, lr_gate = lr_of_new_layers(step, args), lr_of_gate(step, args)
    opt.param_groups[GROUP_BASE]["lr"] = args.learning_rate
    opt.param_groups[GROUP_EXPERTS]["lr"] = lr_new
    opt.param_groups[GROUP_GATE]["lr"] = lr_gate
    return lr_new, lr_gate


# =========================================================================== #
#  Checkpointing                                                              #
# =========================================================================== #
def save_trainable_weights(accelerator, dit, keys, step, args):
    """Under ZeRO-3, gather the sharded weights to rank 0 (bf16) and write only the trainable params (full training = the whole DiT) as safetensors. Must be called on all ranks (collective)."""
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
        save_file(picked, os.path.join(out_dir, "moe_v3_trainable.safetensors"))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"step": step, "variant": "moe_v3", "moe_layers": args.moe_layers, "moe_num_experts": args.moe_num_experts,
                       "moe_top_k": args.moe_top_k, "moe_aux_loss_alpha": args.moe_aux_loss_alpha,
                       "expert_copy_noise": args.expert_copy_noise, "trainable": "full_dit",
                       "learning_rate": args.learning_rate, "new_layers_lr": args.new_layers_lr,
                       "new_layers_lr_steps": args.new_layers_lr_steps, "gate_lr": args.gate_lr, "gate_lr_end": args.gate_lr_end,
                       "norm_topk_weights": True}, f, indent=1)
        print(f"{TAG} saved step-{step} ({len(picked)} tensors, {sum(v.numel() for v in picked.values()) * 2 / 2**30:.1f} GB)", flush=True)
        rotate_weight_ckpts(os.path.join(args.output_dir, "ckpt"))
    del state


def rotate_weight_ckpts(ckpt_root, keep_every=1000, keep_last=2):
    """step-N dirs: those with N % 1000 == 0 are kept forever; of the rest only the latest keep_last are kept."""
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
    """After building, check all parameters and buffers for finiteness. With 24 processes reading the weights concurrently from a network filesystem,
    one process's copy is occasionally corrupted (e.g. a rank with nan loss from step 0); a bad copy is discarded and rebuilt, up to `tries` times."""
    import gc
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
    """ZeRO-3 gather self-check: gather the sharded params block by block, each rank computes an fp32 checksum, all_gather and compare the 24 copies.
    Some ranks once had nan loss from step 0 while their own shards scanned clean → suspected bad all-gather data on individual ranks; this checks it directly."""
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
        raise RuntimeError(f"rank{rank}: ZeRO-3 all-gather results inconsistent across ranks or non-finite, see previous line")
    if rank == 0:
        print(f"{TAG} ZeRO-3 gather self-check passed: {len(inner.blocks)} blocks have matching checksums on {world} ranks", flush=True)


# =========================================================================== #
#  main                                                                        #
# =========================================================================== #
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
    # Create the CUDA context on this GPU first, then wait for free memory (another process may be holding GPU memory);
    # wait at most 10 minutes, then fail fast instead of hitting OOM only after a 13-minute model build
    torch.zeros(1, device=device)
    for _wait in range(60):
        _free, _total = torch.cuda.mem_get_info(device)
        if _free >= 60 * 2 ** 30:
            break
        if _wait % 6 == 0:
            print(f"{TAG} rank{rank}: only {_free / 2 ** 30:.1f} GiB free on this GPU (<60), waiting for memory to be released... ({_wait * 10}s)", flush=True)
        time.sleep(10)
    else:
        raise RuntimeError(f"rank{rank}: after 10 minutes only {_free / 2 ** 30:.1f} GiB is free on this GPU; another process is holding GPU memory, free it first")

    # ---- Model: DiT goes into DeepSpeed (ZeRO-3 sharded); MLLM / VAE are frozen, one copy per GPU, outside DeepSpeed
    # Building briefly takes ~120 GB host RAM per process (built in fp32, then cast to bf16); the 8 local ranks start 15 s apart so peaks do not stack
    time.sleep(int(os.environ.get("LOCAL_RANK", 0)) * 15)
    dit = build_checked(lambda: build_dit(args, dtype, verbose=is_main), "DiT", rank)
    if args.resume_weights:
        load_trainable_weights(dit, args.resume_weights)
        if is_main:
            print(f"{TAG} loaded weights {args.resume_weights}", flush=True)
    base_params, expert_params, gate_params = mark_trainable(dit)
    keys = trainable_state_keys(dit)
    # three param groups, order = GROUP_BASE / GROUP_EXPERTS / GROUP_GATE; per-step learning rates are written by set_learning_rates
    opt = torch.optim.AdamW([{"params": base_params, "lr": args.learning_rate},
                             {"params": expert_params, "lr": args.new_layers_lr},
                             {"params": gate_params, "lr": args.gate_lr}], weight_decay=args.weight_decay)
    dit, opt = accelerator.prepare(dit, opt)                     # DeepSpeed takes over here: weights / grads / optimizer state are sharded across the 24 GPUs
    inner = accelerator.unwrap_model(dit)
    check_zero3_gather(inner, rank, world, device)
    # MLLM: build → move to GPU → scan weights → self-test output finiteness with a fixed instruction + blank frames. One rank once passed the weight
    # scan but gave nan in the self-test (other ranks fine on the same input), i.e. one-off per-process state corruption, so the whole MLLM is rebuilt, up to 3 times.
    from PIL import Image
    probe_frames = [Image.new("RGB", (args.width, args.height))] * args.num_frames
    for attempt in range(1, 4):
        mllm = build_checked(lambda: build_mllm(args).to(device), "MLLM(GPU)", rank)
        text_encoder = TextEncoder(mllm)
        where = text_encoder.self_test("Add a red ball.", probe_frames)
        if where is None:
            print(f"{TAG} rank{rank}: MLLM self-test passed (attempt {attempt})", flush=True)
            break
        print(f"{TAG} rank{rank}: MLLM self-test failed, first non-finite at [{where}], attempt {attempt}, rebuilding entirely", flush=True)
        del mllm, text_encoder
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    else:
        raise RuntimeError(f"rank{rank}: MLLM self-test failed 3 times in a row, exiting")
    vae = build_checked(lambda: build_vae(args, dtype).to(device), "VAE(GPU)", rank)
    scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=1000, training=True)

    # ---- Data: one dataset / sampler / loader per stage
    exclude_words = [w.strip() for w in args.exclude_prompt_words.split(",") if w.strip()]
    stages = []
    for name, paths, epochs in (("A", args.stage_a_data, args.stage_a_epochs), ("B", args.stage_b_data, args.stage_b_epochs)):
        dataset, stats = build_dataset([s for s in paths.split(",") if s], exclude_words, args)
        sampler = ShardedBatchSampler(len(dataset), args.batch_size, world, rank, args.seed)
        loader = DataLoader(IndexTaggedDataset(dataset), batch_sampler=sampler, num_workers=args.dataset_num_workers,
                            prefetch_factor=args.dataloader_prefetch_factor or None, collate_fn=lambda b: b,
                            persistent_workers=bool(args.dataset_num_workers))
        steps = int(math.ceil(len(sampler) * epochs))
        stages.append({"name": name, "sampler": sampler, "loader": loader, "steps": steps, "stats": stats, "size": len(dataset)})
        if is_main:
            print(f"{TAG} stage {name}: {stats} total={len(dataset)} {len(sampler)} steps/epoch × {epochs} epochs = {steps} steps", flush=True)
    stage_start = [0, stages[0]["steps"]]                          # global start step of each stage
    max_steps = args.max_train_steps or (stages[0]["steps"] + stages[1]["steps"])

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
              f"text_drop={args.text_drop_prob} motion={args.motion_sub_loss_ratio}",
              flush=True)
    accelerator.init_trackers(args.wandb_project, config=vars(args),
                              init_kwargs={"wandb": {"name": args.wandb_run_name, "id": args.wandb_run_name, "resume": "allow", "dir": args.output_dir}})
    pbar = None
    if is_main:
        from tqdm import tqdm
        pbar = tqdm(total=max_steps, initial=step0, desc="moe-v3", dynamic_ncols=True)

    def pixels_of(sample, pixel_key, frames_key):
        if sample.get(pixel_key) is not None:
            return uint8_video_to_pixels(sample[pixel_key], device, dtype)
        return frames_to_video_tensor(sample[frames_key], dtype=torch.float32).to(device, dtype)

    moe_ffns = inner.moe_ffns()
    moe_blocks = [int(b) for b in args.moe_layers.split(",")]          # one-to-one with moe_ffns, used for log labels
    num_experts = args.moe_num_experts
    # stats vector reduced across ranks every 10 steps: [sum loss_main, sum loss_motion, sum loss_aux, sum total loss (weighted main/motion + aux), count]
    # followed by the summed per-expert "selection rate" of every MoE layer (layers × experts)
    NUM_LOSS_SLOTS = 5
    stat = torch.zeros(NUM_LOSS_SLOTS + len(moe_ffns) * num_experts, device=device)
    # "category × expert" selection counts (summed over MoE layers): [kind=1 (ffn), category, expert]; reduced every 10 steps, row-normalized to P(expert i | category c)
    moe_kinds = [("ffn", moe_ffns)]
    cat_stat = torch.zeros(len(moe_kinds), len(CATEGORIES), num_experts, device=device)
    timing = {"vae": 0.0, "mllm": 0.0, "dit": 0.0, "n": 0}      # this rank's time in the three phases (s), averaged and logged every 10 steps
    skipped_steps = 0                   # steps skipped due to bad samples / non-finite loss (not counted in step)
    consecutive_bad = 0                 # consecutive non-finite-loss steps on this rank; 5 in a row means this process's weights are corrupt, so exit for inspection
    step, t0 = step0, time.time()
    dit.train()

    while step < max_steps:
        # ---- current stage, epoch within the stage and steps consumed (resume goes through here too, so the data order resumes exactly)
        stage_index = 1 if step >= stage_start[1] else 0
        stage = stages[stage_index]
        in_stage = step - stage_start[stage_index]
        steps_per_epoch = len(stage["sampler"])
        stage["sampler"].set_epoch(in_stage // steps_per_epoch)
        stage["sampler"].skip = in_stage % steps_per_epoch
        stage_end = min(max_steps, stage_start[stage_index] + stage["steps"])
        if is_main:
            print(f"{TAG} stage {stage['name']} epoch {in_stage // steps_per_epoch} resuming from step {in_stage % steps_per_epoch}, until global step {stage_end}", flush=True)

        check_pos = in_stage % steps_per_epoch      # data-position self-check: this epoch's first batch must be the sampler's batch check_pos (each rank checks its own)
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
            # ---- 1. VAE encoding: source / target videos go to the encoder in groups of vae_encode_group (serial per-sample encoding dominated step time, see vae_encode_batched)
            t_mark = sync_time()
            with torch.no_grad():
                src_latents = vae_encode_batched(vae, [pixels_of(s, "src_pixels", "src_video") for s in batch], args.vae_encode_group)   # [B, C, F, H, W]
                latents = vae_encode_batched(vae, [pixels_of(s, "tgt_pixels", "tgt_video") for s in batch], args.vae_encode_group)
            timing["vae"] += sync_time() - t_mark

            # ---- Guard 1: input finiteness. A video occasionally decodes to nan/inf; once in the gradients it corrupts the weights for good (DeepSpeed bf16 has no fp16-style overflow skip).
            #      Checked per sample; bad samples are dropped from the batch and their paths printed; if no good sample is left on this rank, all ranks skip the step together
            #      (collectives require the 24 ranks to stay in lockstep; one rank cannot skip backward alone).
            keep = []
            for i, sample in enumerate(batch):
                if torch.isfinite(latents[i]).all() and torch.isfinite(src_latents[i]).all():
                    keep.append(i)
                else:
                    print(f"{TAG} rank{rank} step {step}: dropped bad sample (non-finite latent) src={sample.get('src_video_path', '?')} "
                          f"tgt={sample.get('tgt_video_path', '?')} prompt={sample['prompts_multi'][0][:80]!r}", flush=True)
            no_good_sample = torch.tensor(float(len(keep) == 0), device=device)
            if accelerator.reduce(no_good_sample, reduction="sum").item() > 0:
                skipped_steps += 1
                if is_main:
                    print(f"{TAG} step {step}: some rank has an all-bad batch, all ranks skip this step (skipped so far: {skipped_steps})", flush=True)
                continue
            if len(keep) < len(batch):
                batch = [batch[i] for i in keep]
                src_latents, latents = src_latents[keep], latents[keep]
            bsz = latents.size(0)

            # ---- 2. flow-match noising (independent timestep / sigma per sample)
            noise = torch.randn_like(latents)
            timestep_index = torch.randint(0, scheduler.num_train_timesteps, (bsz,))
            sigmas = scheduler.sigmas[timestep_index].to(device=device, dtype=latents.dtype).view(bsz, 1, 1, 1, 1)
            timesteps = scheduler.timesteps[timestep_index].to(device=device, dtype=dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            target = noise - latents


            # ---- 3. full instruction → MLLM → ctx; text-CFG draw
            # (under DeepSpeed, accelerate does not wrap forward in autocast automatically, so bf16 autocast is applied explicitly:
            #  the original kiwi forward feeds fp32 inputs to bf16 weights, e.g. in time_embedding, and only runs under autocast; the inference script does the same)
            t_mark = sync_time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx_list = []
                for sample in batch:
                    assert len(sample["prompts_multi"]) == 1          # load_records already reduced each sample to one full instruction
                    if float(torch.rand(())) < args.text_drop_prob:
                        ctx_list.append(text_encoder.encode_null(sample["src_video"]))
                    else:
                        ctx_list.append(text_encoder.encode(sample["prompts_multi"][0], sample["src_video"]))
            timing["mllm"] += sync_time() - t_mark
            # ---- Guard 1b: ctx finiteness (nan MLLM output makes everything nan from the block0 cross-attn onward)
            keep = [i for i, ctx in enumerate(ctx_list) if torch.isfinite(ctx).all()]
            for i, sample in enumerate(batch):
                if i not in keep:
                    print(f"{TAG} rank{rank} step {step}: dropped bad sample (non-finite MLLM ctx) src={sample.get('src_video_path', '?')} "
                          f"prompt={sample['prompts_multi'][0][:80]!r}", flush=True)
            no_good_ctx = torch.tensor(float(len(keep) == 0), device=device)
            if accelerator.reduce(no_good_ctx, reduction="sum").item() > 0:
                skipped_steps += 1
                consecutive_bad = consecutive_bad + 1 if len(keep) == 0 else 0
                if consecutive_bad >= 5:
                    raise RuntimeError(f"rank{rank}: MLLM ctx fully non-finite for {consecutive_bad} consecutive steps, this GPU is faulty, exiting")
                if is_main:
                    print(f"{TAG} step {step}: some rank has an all-non-finite ctx batch, all ranks skip this step (skipped so far: {skipped_steps})", flush=True)
                continue
            if len(keep) < len(batch):
                batch = [batch[i] for i in keep]
                ctx_list = [ctx_list[i] for i in keep]
                src_latents, latents = src_latents[keep], latents[keep]
                noise, noisy_latents, target = noise[keep], noisy_latents[keep], target[keep]
                sigmas, timesteps = sigmas[keep], timesteps[keep]
                bsz = latents.size(0)

            # ---- 4. DiT forward (the load-balancing loss is attached to the graph inside the MoE layers), under bf16 autocast
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
            aux_values = [ffn.last_aux_loss for ffn in moe_ffns if ffn.last_aux_loss is not None]
            loss_aux = torch.stack(aux_values).sum() if aux_values else torch.zeros((), device=device)

            # ---- Guard 2: loss finiteness (non-finite on any rank → all ranks skip, no backward, weights untouched)
            bad_local = torch.tensor(float(not torch.isfinite(loss.detach())), device=device)
            consecutive_bad = consecutive_bad + 1 if bad_local.item() > 0 else 0
            if consecutive_bad >= 5:
                raise RuntimeError(f"rank{rank}: non-finite loss for {consecutive_bad} consecutive steps (different samples), this process's weights/environment are broken, exiting")
            if accelerator.reduce(bad_local, reduction="sum").item() > 0:
                skipped_steps += 1
                if bad_local.item() > 0:
                    print(f"{TAG} rank{rank} step {step}: non-finite loss main={float(loss_main):.4g} aux={float(loss_aux):.4g} "
                          f"batch samples src={[s_.get('src_video_path', '?') for s_ in batch]} tgt={[s_.get('tgt_video_path', '?') for s_ in batch]}", flush=True)
                if is_main:
                    print(f"{TAG} step {step}: non-finite loss on some rank, all ranks skip this step (skipped so far: {skipped_steps})", flush=True)
                del pred, loss
                continue

            # ---- 6. backward + update (accelerate's DeepSpeed wrapper also performs the step inside backward; clipping is in the ds config)
            lr_new, lr_gate = set_learning_rates(opt, step, args)   # this step's learning rates (routed experts / gates transition linearly with step, the rest constant)
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
            timing["dit"] += sync_time() - t_mark
            timing["n"] += 1
            step += 1

            # ---- 7. statistics and logging
            stat[0] += float(loss_main.detach()); stat[1] += float(loss_motion.detach()) if loss_motion is not None else 0.0
            stat[2] += float(loss_aux); stat[3] += float(loss.detach()) + float(loss_aux); stat[4] += 1
            for layer_i, ffn in enumerate(moe_ffns):                # per layer, per expert: token mean of 1 if selected / 0 if not
                start = NUM_LOSS_SLOTS + layer_i * num_experts
                stat[start:start + num_experts] += ffn.last_selected
            # category × expert: per-sample selection counts of each expert (all tokens, all slots), accumulated by sample category
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
                    count = max(totals[4], 1)                                       # = 10 steps × 24 GPUs
                    scalars = {"loss/main": totals[0] / count, "loss/motion": totals[1] / count, "loss/aux": totals[2] / count,
                               "loss/total": totals[3] / count,
                               "lr/base": args.learning_rate, "lr/new_layers": lr_new, "lr/gate": lr_gate,
                               "skipped_steps": skipped_steps, "stage": stage_index}
                    # one folder per MoE layer (moe_L{block}) with one curve per expert: fraction of tokens selecting it (mean of 1 selected / 0 not)
                    selected_by_layer = [totals[NUM_LOSS_SLOTS + layer_i * num_experts: NUM_LOSS_SLOTS + (layer_i + 1) * num_experts]
                                         for layer_i in range(len(moe_ffns))]
                    for block, per_expert in zip(moe_blocks, selected_by_layer):
                        for i, value in enumerate(per_expert):
                            scalars[f"moe_L{block:02d}/expert{i:02d}"] = value / count
                    all_selected = [v / count for per_expert in selected_by_layer for v in per_expert]
                    scalars["moe/selected_min"] = min(all_selected)
                    scalars["moe/selected_max"] = max(all_selected)
                    # P(expert i | category c) = selections of expert i in category c / all selections in c; specialization index = max difference of an expert across categories
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
                    accelerator.log(scalars, step=step)
                    # the console prints only each expert's mean across layers (one number per expert, sum = top_k)
                    mean_by_expert = [sum(per_expert[i] for per_expert in selected_by_layer) / (count * len(selected_by_layer))
                                      for i in range(num_experts)]
                    rate = (step - step0) * world * args.batch_size / max(time.time() - t0, 1e-6)
                    print(f"{TAG} step {step}/{max_steps} stage={stage['name']} loss={totals[0] / count:.4f} aux={totals[2] / count:.4f} "
                          f"lr_new={lr_new:.2e} lr_gate={lr_gate:.2e} sel_min={min(all_selected):.3f} sel_max={max(all_selected):.3f} "
                          f"sel={'/'.join(f'{v:.2f}' for v in mean_by_expert)} "
                          f"{rate:.1f}rec/s eta={(max_steps - step) / max(rate, 1e-6) * world * args.batch_size / 3600:.1f}h "
                          f"t(vae/mllm/dit)={timing['vae'] / max(timing['n'], 1):.1f}/{timing['mllm'] / max(timing['n'], 1):.1f}/{timing['dit'] / max(timing['n'], 1):.1f}s "
                          + " ".join(spread_note), flush=True)
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
    accelerator.end_training()
    if is_main:
        print(f"{TAG} DONE @ step {step}", flush=True)


if __name__ == "__main__":
    main()
