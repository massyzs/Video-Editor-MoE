#!/usr/bin/env python
"""mlp-moe one-off script: merge the FFN LoRAs of the 5 category experts into kiwi-edit's FFN weights, then add 1 FFN directly fine-tuned on goku, giving the initial weights of 6 base experts (training duplicates each one with added noise = 12).

Experts were trained as LoRA (r=128, alpha=128, scale alpha/r = 1), so merging is simply  W' = W + B·A
Only ffn.0 (3072→14336) and ffn.2 (14336→3072) of the MoE layers are processed (default layers 2,4,...,28 = indices 1,3,...,27).
The attention LoRAs (q/k/v/o) are not used in this version and are skipped. Biases have no LoRA and stay unchanged.

Output: <out_dir>/<out_name> (default experts_merged_6.safetensors), keys:
    blocks.{b}.{expert_name}.ffn.0.weight   [14336, 3072]  bf16
    blocks.{b}.{expert_name}.ffn.2.weight   [3072, 14336]  bf16
    blocks.{b}.direct_finetune.ffn.{0,2}.bias   (only direct_finetune has its own bias; LoRA experts use kiwi's bias, taken at assembly time)
direct_finetune = ffn.0/ffn.2 of output/direct_finetune (full fine-tune of kiwi-edit on goku, step-3581).
Usage (relative-path defaults are rooted at --base_dir):
    python merge_experts.py --base_dir <BASE_DIR> --out_dir <BASE_DIR>/ckpt/init
    LoRA experts default to <BASE_DIR>/output/moe_expert_<name>/final_components/dit_query_router.safetensors (--experts_root changes the root)
"""
import argparse
import json
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_MOE_LAYERS = "1,3,5,7,9,11,13,15,17,19,21,23,25,27"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True, help="working directory; the relative-path defaults below are rooted here")
    p.add_argument("--dit_weights", default="ckpt/kiwi-edit-instruct-only-videoxfun/diffusion_pytorch_model.safetensors", help="relative paths are rooted at --base_dir")
    p.add_argument("--experts_root", default="output", help="directory containing moe_expert_<name>/; relative paths are rooted at --base_dir")
    p.add_argument("--experts", default="style,add,remove,replace,background")
    p.add_argument("--direct_finetune", default="output/direct_finetune/final_components/transformer", help="diffusers sharded checkpoint dir (relative paths are rooted at --base_dir); empty string = skip this expert")
    p.add_argument("--out_name", default="experts_merged_6.safetensors")
    p.add_argument("--moe_layers", default=DEFAULT_MOE_LAYERS, help="comma-separated block indices (0-based)")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    for key in ("dit_weights", "experts_root", "direct_finetune"):
        value = getattr(args, key)
        if value and not os.path.isabs(value):
            setattr(args, key, os.path.join(args.base_dir, value))
    return args


def main():
    args = parse_args()
    expert_names = [s.strip() for s in args.experts.split(",") if s.strip()]
    moe_layers = [int(s) for s in args.moe_layers.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    base = safe_open(args.dit_weights, "pt")
    merged = {}
    report = {}
    for expert_name in expert_names:
        lora_path = os.path.join(args.experts_root, f"moe_expert_{expert_name}", "final_components", "dit_query_router.safetensors")
        assert os.path.isfile(lora_path), f"LoRA file for expert {expert_name} not found: {lora_path}"
        lora = safe_open(lora_path, "pt")
        for block in moe_layers:
            for proj in ("ffn.0", "ffn.2"):
                base_key = f"blocks.{block}.{proj}.weight"
                W = base.get_tensor(base_key).float()                                        # [out, in]
                A = lora.get_tensor(f"dit.blocks.{block}.{proj}.lora_A.default.weight").float()   # [r, in]
                B = lora.get_tensor(f"dit.blocks.{block}.{proj}.lora_B.default.weight").float()   # [out, r]
                delta = B @ A                                                                 # [out, in], scale alpha/r = 1
                assert delta.shape == W.shape, (delta.shape, W.shape)
                merged[f"blocks.{block}.{expert_name}.{proj}.weight"] = (W + delta).to(torch.bfloat16)
                rel = (delta.norm() / W.norm()).item()
                report[f"blocks.{block}.{expert_name}.{proj}"] = round(rel, 5)
        print(f"[merge] expert '{expert_name}' merged: {len(moe_layers)} layers × 2 projections, "
              f"LoRA relative delta |ΔW|/|W| range "
              f"{min(v for k, v in report.items() if expert_name in k):.4f} ~ "
              f"{max(v for k, v in report.items() if expert_name in k):.4f}", flush=True)

    if args.direct_finetune:
        index = json.load(open(os.path.join(args.direct_finetune, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
        shards = {}
        for block in moe_layers:
            for proj in ("ffn.0", "ffn.2"):
                for part in ("weight", "bias"):
                    key = f"blocks.{block}.{proj}.{part}"
                    fname = index[key]
                    if fname not in shards:
                        shards[fname] = safe_open(os.path.join(args.direct_finetune, fname), "pt")
                    tensor = shards[fname].get_tensor(key)
                    if part == "weight":
                        W = base.get_tensor(key).float()
                        report[f"blocks.{block}.direct_finetune.{proj}"] = round(((tensor.float() - W).norm() / W.norm()).item(), 5)
                    merged[f"blocks.{block}.direct_finetune.{proj}.{part}"] = tensor.to(torch.bfloat16)
        expert_names.append("direct_finetune")
        print(f"[merge] expert 'direct_finetune' taken from {args.direct_finetune}: |ΔW|/|W| vs. base, range "
              f"{min(v for k, v in report.items() if 'direct_finetune' in k):.4f} ~ {max(v for k, v in report.items() if 'direct_finetune' in k):.4f}", flush=True)
    out_path = os.path.join(args.out_dir, args.out_name)
    save_file(merged, out_path, metadata={"experts": ",".join(expert_names), "moe_layers": ",".join(map(str, moe_layers)),
                                          "note": "LoRA experts: W' = W + B·A (alpha/r=1); direct_finetune: weights+bias from output/direct_finetune; bf16; ffn.0 / ffn.2 only"})
    with open(os.path.join(args.out_dir, args.out_name.replace(".safetensors", "_report.json")), "w") as f:
        json.dump({"experts": expert_names, "moe_layers": moe_layers, "relative_delta": report}, f, indent=1)
    size_gb = sum(v.numel() * 2 for v in merged.values()) / 2 ** 30
    print(f"[merge] wrote {out_path}: {len(merged)} tensors, {size_gb:.1f} GB, took {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
