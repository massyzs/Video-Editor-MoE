#!/usr/bin/env python
"""explicit-moe (moe_v2) training script.
The model is defined in moe_v2_transformer.py; this file only does: data → same-group batching → noising → forward → loss → update → save/resume.

Each step:
  1. a batch of B samples from one group (all share the same expert multiset, see GroupedBatchSampler)
  2. VAE-encode source/target videos → per-sample sigma and noise, noisy = (1-σ)·tgt + σ·noise
  3. per sample, blank the base instruction with probability base_text_drop_prob (unconditional branch for text CFG; expert instructions always kept)
  4. MoEV2Model.forward → pred; loss = mse(pred, noise - tgt) (+ motion frame-difference term) + router-head CE
  5. update only fusions.* and router_head.*
Multi-node launch: train_multinode.sh (single-node smoke test: train_smoke.sh).
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from moe_v2_transformer import CLASSES, DEFAULT_LORA_EXPERT_TEMPLATE, build_model, load_checkpoint     # noqa: E402

TAG = "[moe-v2]"


# =========================================================================== #
#  arguments                                                                  #
# =========================================================================== #
def parse_args(argv=None):
    p = argparse.ArgumentParser()
    # model and weights
    p.add_argument("--pretrained_wan_path", required=True)
    p.add_argument("--config_path", required=True)
    p.add_argument("--mllm_model", required=True)
    p.add_argument("--init_weights", required=True)
    p.add_argument("--lora_experts_dir", required=True,
                   help="dir with the frozen LoRA experts, e.g. <explicit-moe release>/lora_experts (files <dir>/<name>/dit_query_router.safetensors)")
    p.add_argument("--lora_expert_template", default=DEFAULT_LORA_EXPERT_TEMPLATE,
                   help="expert file path relative to --lora_experts_dir; {name} = expert name")
    p.add_argument("--experts", default="style,add,remove,replace,background")
    p.add_argument("--router_head", required=True,
                   help="initial router head (state_dict + classes/dmodel/heads), e.g. <explicit-moe release>/router_head_online.pt")
    p.add_argument("--dit_lora_rank", type=int, default=128)
    p.add_argument("--dit_lora_target_modules", default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--use_gradient_checkpointing", type=int, default=1)
    # fusion layers
    p.add_argument("--fusion_start", type=int, default=15)
    p.add_argument("--fusion_end", type=int, default=29, help="fusion layers are attached after blocks [fusion_start, fusion_end); 0 = up to and including the last block")
    p.add_argument("--fusion_heads", type=int, default=24)
    p.add_argument("--fusion_init", choices=["zero", "normal", "scaled"], default="normal")
    p.add_argument("--fusion_init_gain", type=float, default=0.05)
    p.add_argument("--rope_expert_offset0", type=int, default=1, help="width-offset multiple of expert 0: 1 = experts are laid out to the right of base")
    # data
    p.add_argument("--vid_dataset_metadata_path", required=True, help="comma-separated json/jsonl files")
    p.add_argument("--dataset_prefixes", default="", help="path prefixes, one per metadata file (comma-separated; empty = paths are already absolute)")
    p.add_argument("--dataset_base_path", required=True)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--max_pixels", type=int, default=399360)
    p.add_argument("--dataset_num_workers", type=int, default=4)
    p.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    p.add_argument("--compound_joiner", default=" Then ", help="the base's full instruction = sub-instructions joined with this string")
    # training
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=2, help="same-group batch size; effective batch = ranks × batch_size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="used only for throughput stats (accumulation is handled by the tail-batch pairing logic)")
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--epoch_steps", type=int, default=0, help="optimizer steps per epoch; >0 makes a mid-epoch resume continue the same data order exactly")
    p.add_argument("--fusion_learning_rate", type=float, default=1e-4)
    p.add_argument("--fusion_lr_anneal_from", type=float, default=0.0)
    p.add_argument("--fusion_lr_anneal_steps", type=int, default=400)
    p.add_argument("--router_learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--base_text_drop_prob", type=float, default=0.1, help="text CFG: per-sample probability of blanking the base instruction")
    p.add_argument("--motion_sub_loss_ratio", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--resume_from", default="", help=".../step-N/moe_v2_state.pt")
    # logging
    p.add_argument("--use_wandb", type=int, default=0)
    p.add_argument("--wandb_project", default="video-editor-moe")
    p.add_argument("--wandb_run_name", default="moe_v2")
    return p.parse_args(argv)


# =========================================================================== #
#  data                                                                       #
# =========================================================================== #
def load_records(meta_paths, prefixes):
    """Read several json/jsonl files into records with one schema (CoinVE / goku / OpenVE fields differ; normalized here)."""
    def norm(r, pre):
        def j(path):
            return path if (not pre or not path or path.startswith("/")) else os.path.join(pre, path)
        plist = r["prompt"] if isinstance(r["prompt"], list) else [r["prompt"]]
        tts = r.get("turn_types_qwen")
        if tts is None and r.get("cls"):
            tts = [r["cls"]]                       # OpenVE balanced subset: cls → single-turn label
        return {"src_video": j(r["src_video"]), "tgt_video": j(r["tgt_video"]), "prompt": plist,
                "turn_types_qwen": tts, "n_turns": len(plist), "mask_mode": "global"}

    recs, stat = [], []
    for mp, pre in zip(meta_paths, prefixes):
        n = 0
        with open(mp) as f:
            it = json.load(f) if mp.endswith(".json") else (json.loads(l) for l in f if l.strip())
            for r in it:
                recs.append(norm(r, pre))
                n += 1
        stat.append(f"{os.path.basename(mp)}:{n}(prefix={pre or '-'})")
    return recs, " + ".join(stat)


def build_dataset(args):
    """Normalized records → temp jsonl → KiwiMultiEditDataset; also returns each sample's group key (sorted class tuple)."""
    from videox_fun.moe_common.kiwi_dataset_multi_moe import KiwiMultiEditDataset
    meta_paths = [s for s in args.vid_dataset_metadata_path.split(",") if s]
    prefixes = args.dataset_prefixes.split(",") if args.dataset_prefixes else []
    prefixes += [""] * (len(meta_paths) - len(prefixes))
    recs, stat = load_records(meta_paths, prefixes)
    fd, merged = tempfile.mkstemp(suffix=".jsonl", prefix="moe_v2_data_")
    with os.fdopen(fd, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    keys = [tuple(sorted(r["turn_types_qwen"] or ())) for r in recs]
    ds = KiwiMultiEditDataset(
        metadata_path=merged, base_path=args.dataset_base_path, max_turns=0, mask_loss=False,
        subset_single_prob=0.0, num_frames=args.num_frames, height=args.height, width=args.width,
        max_pixels=args.max_pixels, repeat=1, require_ref_image=False, pad_frames_to_num=True,
        return_pixels=True)
    os.unlink(merged)
    assert len(ds) == len(keys), f"dataset filtered out {len(keys) - len(ds)} records, group keys misaligned"
    return ds, keys, stat


class GroupedBatchSampler:
    """Same-group batching + even split across ranks. Group = the sample's expert multiset (sorted class tuple); only
    same-group samples can share a batch (expert stream positions must align). Full batches are shuffled; incomplete
    tail batches go to the end of the epoch, with an equal, even tail count per rank; the main loop pairs two tail
    batches into one accumulated step → every rank takes exactly the same number of optimizer steps (no DDP desync).
    skip: batches (per rank) already consumed in this epoch on resume; applied once, on the next iteration only."""

    def __init__(self, keys, batch_size, num_replicas, rank, seed):
        groups = defaultdict(list)
        for i, k in enumerate(keys):
            groups[k].append(i)
        self.groups = dict(sorted(groups.items()))
        self.b, self.R = batch_size, num_replicas
        self.rank, self.seed, self.epoch = rank, seed, 0
        self.skip = 0

    def set_epoch(self, e):
        self.epoch = e

    def skip_for_steps(self, n_steps):
        """n_steps already done in this epoch → batches to skip (per rank): full batches 1 batch = 1 step, tail batches 2 batches = 1 step."""
        fulls, _ = self._batches()
        nf = len(fulls) // self.R
        return n_steps if n_steps <= nf else nf + 2 * (n_steps - nf)

    def _batches(self):
        import random
        rng = random.Random(self.seed * 100003 + self.epoch)
        fulls, tails = [], []
        for _, g in self.groups.items():
            idxs = g[:]
            rng.shuffle(idxs)
            for i in range(0, len(idxs), self.b):
                bt = idxs[i:i + self.b]
                (fulls if len(bt) == self.b else tails).append(bt)
        rng.shuffle(fulls)
        rng.shuffle(tails)
        fulls += fulls[:(-len(fulls)) % self.R]
        if tails:
            need = (-len(tails)) % (2 * self.R)
            tails += [tails[i % len(tails)] for i in range(need)]
        return fulls, tails

    def __iter__(self):
        fulls, tails = self._batches()
        seq = fulls[self.rank::self.R] + tails[self.rank::self.R]
        sk, self.skip = self.skip, 0
        return iter(seq[sk:])

    def __len__(self):
        fulls, tails = self._batches()
        return len(fulls) // self.R + len(tails) // self.R


# =========================================================================== #
#  loss                                                                        #
# =========================================================================== #
def compute_loss(model, vae, samples, scheduler, device, dtype, args):
    """Flow-matching loss (+ router-head CE) for one same-group batch. Returns (loss, info)."""
    from videox_fun.data.kiwi_video_utils import frames_to_video_tensor, uint8_video_to_pixels

    def pixels(s, key_px, key_frames):
        if s.get(key_px) is not None:
            return uint8_video_to_pixels(s[key_px], device, dtype)
        return frames_to_video_tensor(s[key_frames], dtype=torch.float32).to(device, dtype)

    srcs, tgts = [], []
    with torch.no_grad():                                   # VAE-encode one sample at a time to bound peak memory
        for s in samples:
            srcs.append(vae.encode(pixels(s, "src_pixels", "src_video"))[0].sample())
            tgts.append(vae.encode(pixels(s, "tgt_pixels", "tgt_video"))[0].sample())
    src_latents, tgt_latents = torch.cat(srcs, 0), torch.cat(tgts, 0)
    B = len(samples)

    noise = torch.randn_like(tgt_latents)
    idxs = torch.randint(0, scheduler.num_train_timesteps, (B,))
    sigma = scheduler.sigmas[idxs].to(device=device, dtype=tgt_latents.dtype)
    t = scheduler.timesteps[idxs].to(device=device, dtype=dtype)
    noisy = (1.0 - sigma.view(B, 1, 1, 1, 1)) * tgt_latents + sigma.view(B, 1, 1, 1, 1) * noise
    text_drop = [float(torch.rand(())) < args.base_text_drop_prob for _ in range(B)]

    inner = model.module if hasattr(model, "module") else model
    p = inner.dit.patch_size
    f_lat, h_lat, w_lat = tgt_latents.shape[2:]
    seq_len = (f_lat // p[0]) * (h_lat // p[1]) * (w_lat // p[2])

    out = model(samples, noisy, t, src_latents, sigma, seq_len, joiner=args.compound_joiner, text_drop=text_drop)
    pred, kept = out["pred"], out["kept"]

    losses = []
    for bi, i in enumerate(kept):
        pr = pred[bi:bi + 1].float()
        target = (noise[i:i + 1] - tgt_latents[i:i + 1]).float()
        li = F.mse_loss(pr, target)
        r = args.motion_sub_loss_ratio
        if r > 0 and pr.size(2) > 2:                        # frame-to-frame differences must match too (motion term)
            sub = F.mse_loss(pr[:, :, 1:] - pr[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])
            li = li * (1 - r) + sub * r
        losses.append(li)
    loss = torch.stack(losses).mean()
    info = {"loss_fusion": float(loss.detach()), "loss_router": None,
            "routes": out["routes"], "batch": B, "kept": len(kept)}
    if out["router_logits"] is not None:
        ce = F.cross_entropy(out["router_logits"], out["router_labels"])
        info["loss_router"] = float(ce.detach())
        loss = loss + ce
    return loss, info


# =========================================================================== #
#  save / resume                                                              #
# =========================================================================== #
def set_trainable(model):
    """Freeze everything, unfreeze only fusions.* and router_head.* (as fp32 master weights). Returns (fusion_params, router_params)."""
    for p in model.parameters():
        p.requires_grad_(False)
    fusion_params, router_params = [], []
    for n, p in model.named_parameters():
        if n.startswith("fusions."):
            p.requires_grad_(True); p.data = p.data.float(); fusion_params.append(p)
        elif n.startswith("router_head."):
            p.requires_grad_(True); p.data = p.data.float(); router_params.append(p)
    print(f"{TAG} trainable: fusion {sum(p.numel() for p in fusion_params):,} "
          f"+ router {sum(p.numel() for p in router_params):,} params", flush=True)
    return fusion_params, router_params


def save_checkpoint(model, opt, step, args, final=False):
    """Write ckpt/step-N (or final_components): moe_v2_state.pt (trainable params + optimizer + metadata),
    fusion_moe_v2_absorb.safetensors (fusion layers, bf16), router_head_finetuned.pt (router head)."""
    from safetensors.torch import save_file
    inner = model.module if hasattr(model, "module") else model
    trainable = {n: p.detach().cpu() for n, p in inner.named_parameters() if p.requires_grad}
    d = os.path.join(args.output_dir, "ckpt", "final_components" if final else f"step-{step}")
    os.makedirs(d, exist_ok=True)
    torch.save({"trainable": trainable, "opt": opt.state_dict(), "step": step,
                "variant": "moe_v2", "fusion_start": inner.fusion_start, "fusion_end": inner.fusion_end,
                "n_fusion_layers": len(inner.fusions), "rope_expert_offset0": inner.expert_off0,
                "base_text_drop_prob": args.base_text_drop_prob},
               os.path.join(d, "moe_v2_state.pt"))
    fusion_sd = {n: v.to(torch.bfloat16) for n, v in trainable.items() if n.startswith("fusions.")}
    save_file(fusion_sd, os.path.join(d, "fusion_moe_v2_absorb.safetensors"))
    rh = {n[len("router_head."):]: v for n, v in trainable.items() if n.startswith("router_head.")}
    if rh:
        torch.save({"state_dict": rh, "step": step, "arch": "attn_online_textseg",
                    "classes": list(CLASSES), "cotrained_on": "mask_round_qwen_labels"},
                   os.path.join(d, "router_head_finetuned.pt"))
    print(f"{TAG} saved step-{step} (absorb {len(fusion_sd)})", flush=True)
    if not final:
        rotate_checkpoints(os.path.join(args.output_dir, "ckpt"))


def rotate_checkpoints(ckpt_dir, keep_every=1000, keep_last=2):
    """step-N at every 1000 steps is kept forever; of the rest only the newest keep_last are kept (~16GB each). Only touches step-<number> dirs."""
    steps = [int(m.group(1)) for n in os.listdir(ckpt_dir)
             if (m := re.fullmatch(r"step-(\d+)", n)) and os.path.isdir(os.path.join(ckpt_dir, n))]
    for st in sorted(s for s in steps if s % keep_every != 0)[:-keep_last]:
        shutil.rmtree(os.path.join(ckpt_dir, f"step-{st}"), ignore_errors=True)
        print(f"{TAG} rotated out step-{st}", flush=True)


# =========================================================================== #
#  main                                                                        #
# =========================================================================== #
def main(argv=None):
    args = parse_args(argv)
    from datetime import timedelta
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
    from torch.utils.data import DataLoader
    from videox_fun.data import FlowMatchScheduler

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="bf16",
        log_with=(["tensorboard", "wandb"] if args.use_wandb else ["tensorboard"]), project_dir=args.output_dir,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True),
                         InitProcessGroupKwargs(timeout=timedelta(minutes=60))])   # shared-filesystem reads occasionally stall; relax the NCCL timeout
    set_seed(args.seed + accelerator.process_index)
    device, dtype = accelerator.device, torch.bfloat16
    is_main = accelerator.is_main_process
    guard = torch.empty(1536 * 1024 * 1024, dtype=torch.uint8, device=device)   # hold GPU memory while building the model so another process cannot grab it

    # ---- model
    model, vae = build_model(args, dtype)
    fusion_params, router_params = set_trainable(model)
    model, vae = model.to(device), vae.to(device)
    del guard
    torch.cuda.empty_cache()

    init_flr = args.fusion_lr_anneal_from if args.fusion_lr_anneal_from > 0 else args.fusion_learning_rate
    opt = torch.optim.AdamW([{"params": fusion_params, "lr": init_flr},
                             {"params": router_params, "lr": args.router_learning_rate}],
                            weight_decay=args.weight_decay)

    # ---- data
    ds, keys, data_stat = build_dataset(args)
    if is_main:
        print(f"{TAG} data sources: {data_stat} total={len(ds)}", flush=True)
    sampler = GroupedBatchSampler(keys, args.batch_size, accelerator.num_processes, accelerator.process_index, args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.dataset_num_workers,
                        prefetch_factor=args.dataloader_prefetch_factor or None,
                        collate_fn=lambda b: b, persistent_workers=bool(args.dataset_num_workers))
    scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=1000, training=True)

    model = accelerator.prepare(model)
    opt = accelerator.prepare(opt)
    inner = model.module if hasattr(model, "module") else model

    # ---- resume
    step0 = 0
    if args.resume_from:
        ck = load_checkpoint(inner, args.resume_from)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        elif is_main:   # released checkpoints ship without optimizer state: weights are loaded, AdamW starts fresh
            print(f"{TAG} resume: no optimizer state in {args.resume_from}, starting a fresh optimizer", flush=True)
        for g, lr in zip(opt.param_groups, [args.fusion_learning_rate, args.router_learning_rate]):
            if g["lr"] != lr:
                if is_main:
                    print(f"{TAG} resume lr override: {g['lr']} -> {lr}", flush=True)
                g["lr"] = lr
        step0 = ck["step"]
        if is_main:
            print(f"{TAG} resume @ step {step0}", flush=True)

    if is_main:
        os.makedirs(os.path.join(args.output_dir, "ckpt"), exist_ok=True)
        n_fu = sum(p.numel() for p in inner.fusions.parameters())
        print(f"{TAG} start: data={len(ds)} experts={inner.adapter_names} "
              f"fusion_layers={len(inner.fusions)}({n_fu / 1e6:.1f}M params, blocks[{inner.fusion_start},{inner.fusion_end})) "
              f"base_text_drop={args.base_text_drop_prob} bs={args.batch_size}x{accelerator.num_processes} "
              f"steps={args.max_train_steps} motion={args.motion_sub_loss_ratio}", flush=True)
    accelerator.init_trackers(args.wandb_project, config=vars(args),
                              init_kwargs={"wandb": {"name": args.wandb_run_name, "id": args.wandb_run_name,
                                                     "resume": "allow", "dir": args.output_dir}})
    pbar = None
    if is_main:
        from tqdm import tqdm
        pbar = tqdm(total=args.max_train_steps, initial=step0, desc="moe-v2", dynamic_ncols=True)

    # ---- stats (count vector summed across ranks): [0:5] experts used [5:10] label counts [10:15] router-head picks
    #      [15] router-head correct [16] labeled turns [17] loss_fusion sum [18] count [19] loss_router sum [20] count
    stat = torch.zeros(21, device=device)
    agree = Counter()
    gn_sum, gn_cnt = 0.0, 0
    step, t0, epoch = step0, time.time(), 0
    # exact mid-epoch resume: epoch shuffle seed = global step at epoch start; skip batches already consumed this epoch
    ES = args.epoch_steps
    ep_base = (step0 // ES) if ES else 0
    in_ep = (step0 - ep_base * ES) if ES else 0
    while step < args.max_train_steps:
        if ES:
            sampler.set_epoch((ep_base + epoch) * ES)
            if epoch == 0 and in_ep > 0:
                sampler.skip = sampler.skip_for_steps(in_ep)
                if is_main:
                    print(f"{TAG} exact mid-epoch resume: seed_epoch={ep_base * ES} skipping {in_ep} steps={sampler.skip} batches/rank", flush=True)
        else:
            sampler.set_epoch(epoch + max(0, step0))
        pending = 0                                     # tail batches (< batch_size) are paired and accumulated into one step
        for batch in loader:
            import contextlib
            do_sync = (pending + len(batch)) >= args.batch_size
            scale = len(batch) / max(args.batch_size, 1)
            with (contextlib.nullcontext() if do_sync else accelerator.no_sync(model)):
                loss, info = compute_loss(model, vae, batch, scheduler, device, dtype, args)
                accelerator.backward(loss * scale)
            if do_sync:
                params_all = [p for g in opt.param_groups for p in g["params"]]
                total_norm = accelerator.clip_grad_norm_(params_all, args.max_grad_norm if args.max_grad_norm > 0 else 1e9)
                if not torch.isfinite(total_norm):
                    opt.zero_grad(set_to_none=True)
                    if is_main:
                        print(f"{TAG} step {step}: non-finite grad norm, step voided", flush=True)
                else:
                    gn_sum += float(total_norm); gn_cnt += 1
                opt.step()
                opt.zero_grad()
                pending = 0
            else:
                pending += len(batch)

            for ri in info["routes"]:
                for u, g in zip(ri["used"], ri["gt"]):
                    agree["hit" if u == g else "miss"] += 1
                for c in ri["used"]:
                    stat[CLASSES.index(c)] += 1
                for c in ri["gt"]:
                    if c in CLASSES:
                        stat[5 + CLASSES.index(c)] += 1
                for c in ri["raw"]:
                    stat[10 + CLASSES.index(c)] += 1
                for a, g in zip(ri["raw"], ri["gt"]):
                    if g in CLASSES:
                        stat[16] += 1; stat[15] += float(a == g)
            stat[17] += info["loss_fusion"]; stat[18] += 1
            if info["loss_router"] is not None:
                stat[19] += info["loss_router"]; stat[20] += 1

            if not do_sync:
                continue
            step += 1
            if args.fusion_lr_anneal_from > 0 and step <= args.fusion_lr_anneal_steps:
                frac = min(max(step / max(args.fusion_lr_anneal_steps, 1), 0.0), 1.0)
                opt.param_groups[0]["lr"] = args.fusion_lr_anneal_from + (args.fusion_learning_rate - args.fusion_lr_anneal_from) * frac
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{float(loss.detach()):.4f}", refresh=False)
            if step % 10 == 0:
                g = accelerator.reduce(stat.clone(), reduction="sum").tolist()
                if is_main:
                    sc = {"loss/fusion": g[17] / max(g[18], 1), "lr/fusion": opt.param_groups[0]["lr"]}
                    if g[20] > 0:
                        sc["loss/router_ce"] = g[19] / g[20]
                    if gn_cnt > 0:
                        sc["grad/norm"] = gn_sum / gn_cnt; gn_sum, gn_cnt = 0.0, 0
                    if g[16] > 0:
                        sc["route/agree_vs_qwen_gt"] = g[15] / g[16]
                    h, m = agree["hit"], agree["miss"]
                    sc["route/agree_vs_turntype"] = h / max(h + m, 1)
                    n_used, n_gt, n_raw = max(sum(g[0:5]), 1), max(sum(g[5:10]), 1), max(sum(g[10:15]), 1)
                    for i, c in enumerate(CLASSES):
                        sc[f"expert_used/{c}"] = g[i] / n_used
                        sc[f"expert_gt/{c}"] = g[5 + i] / n_gt
                        sc[f"expert_router_raw/{c}"] = g[10 + i] / n_raw
                    accelerator.log(sc, step=step)
                    r = (step - step0) * args.gradient_accumulation_steps * accelerator.num_processes / max(time.time() - t0, 1e-6)
                    eta_h = (args.max_train_steps - step) / max(r, 1e-6) * args.gradient_accumulation_steps * accelerator.num_processes / 3600
                    print(f"{TAG} step {step}/{args.max_train_steps} loss={loss.item():.4f} {r:.1f}rec/s "
                          f"route_vs_turntype={h}/{h + m} eta={eta_h:.1f}h", flush=True)
                stat.zero_()
            if is_main and step % args.save_steps == 0:
                save_checkpoint(model, opt, step, args)
            if step >= args.max_train_steps:
                break
        assert pending == 0, f"unpaired tail batch at end of epoch (pending={pending})"
        epoch += 1

    if is_main:
        save_checkpoint(model, opt, step, args, final=True)
        if pbar is not None:
            pbar.close()
    accelerator.end_training()
    if is_main:
        print(f"{TAG} DONE", flush=True)


if __name__ == "__main__":
    main()
