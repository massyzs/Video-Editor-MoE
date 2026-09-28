#!/usr/bin/env python
"""videoeditor-14b one-off script: build the initial weights of the 6 non-original experts out of 7 (expert 0 = original kiwi, copied from the base model at assembly time, not here).

Each MoE layer (default layers 2,4,...,28 = indices 1,3,...,27) takes four projections: self_attn.k / self_attn.v / ffn.0 / ffn.2, each with weight + bias.
  experts 1..5  style / add / remove / replace / background:  W' = W_kiwi + B·A (LoRA r=128, alpha=128, scale 1), bias = kiwi's
  expert 6      direct_finetune:  same-named weights and biases taken from output/direct_finetune (full fine-tune of kiwi-edit on goku, step-3581)
Output: <out_dir>/experts_v3_1.safetensors, keys:
    blocks.{b}.experts.{i}.{self_attn.k|self_attn.v|ffn.0|ffn.2}.{weight|bias}     i = 1..6, bf16
Usage:
    python merge_experts_v3_1.py --base_dir <BASE_DIR> --out_dir <BASE_DIR>/ckpt/init
    (inputs: the base DiT, the 5 category-expert LoRAs <experts_root>/moe_expert_<name>/final_components/dit_query_router.safetensors,
     and the direct_finetune diffusers sharded dir; these training-initialization files are not in the released weights)
"""
import argparse
import json
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_MOE_LAYERS = "1,3,5,7,9,11,13,15,17,19,21,23,25,27"
PROJS = ("self_attn.k", "self_attn.v", "ffn.0", "ffn.2")
LORA_EXPERTS = ["style", "add", "remove", "replace", "background"]      # experts 1..5
EXPERT_NAMES = ["main"] + LORA_EXPERTS + ["direct_finetune"]           # experts 0..6; training and inference both use this order


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True, help="<BASE_DIR>: the three paths below default to locations under it when omitted")
    p.add_argument("--dit_weights", default="", help="empty = <BASE_DIR>/ckpt/kiwi-edit-instruct-only-videoxfun/diffusion_pytorch_model.safetensors")
    p.add_argument("--experts_root", default="", help="directory containing moe_expert_<name>/final_components/dit_query_router.safetensors; empty = <BASE_DIR>/output")
    p.add_argument("--direct_finetune", default="", help="diffusers sharded checkpoint dir; empty = <BASE_DIR>/output/direct_finetune/final_components/transformer")
    p.add_argument("--moe_layers", default=DEFAULT_MOE_LAYERS)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    args.dit_weights = args.dit_weights or os.path.join(args.base_dir, "ckpt/kiwi-edit-instruct-only-videoxfun/diffusion_pytorch_model.safetensors")
    args.experts_root = args.experts_root or os.path.join(args.base_dir, "output")
    args.direct_finetune = args.direct_finetune or os.path.join(args.base_dir, "output/direct_finetune/final_components/transformer")
    return args


class ShardedWeights:
    """Diffusers sharded safetensors dir: look up which file holds each key via index.json."""

    def __init__(self, folder):
        index = json.load(open(os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")))
        self.files = {}
        self.where = index["weight_map"]
        self.folder = folder

    def get_tensor(self, key):
        fname = self.where[key]
        if fname not in self.files:
            self.files[fname] = safe_open(os.path.join(self.folder, fname), "pt")
        return self.files[fname].get_tensor(key)


def main():
    args = parse_args()
    moe_layers = [int(s) for s in args.moe_layers.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    base = safe_open(args.dit_weights, "pt")
    direct = ShardedWeights(args.direct_finetune)
    loras = {}
    for name in LORA_EXPERTS:
        path = os.path.join(args.experts_root, f"moe_expert_{name}", "final_components", "dit_query_router.safetensors")
        assert os.path.isfile(path), f"LoRA file for expert {name} not found: {path}"
        loras[name] = safe_open(path, "pt")

    out, report = {}, {}
    for block in moe_layers:
        for proj in PROJS:
            W = base.get_tensor(f"blocks.{block}.{proj}.weight").float()
            b = base.get_tensor(f"blocks.{block}.{proj}.bias")
            # experts 1..5: merge the LoRA
            for i, name in enumerate(LORA_EXPERTS, start=1):
                A = loras[name].get_tensor(f"dit.blocks.{block}.{proj}.lora_A.default.weight").float()   # [r, in]
                B = loras[name].get_tensor(f"dit.blocks.{block}.{proj}.lora_B.default.weight").float()   # [out, r]
                delta = B @ A
                assert delta.shape == W.shape, (proj, delta.shape, W.shape)
                out[f"blocks.{block}.experts.{i}.{proj}.weight"] = (W + delta).to(torch.bfloat16)
                out[f"blocks.{block}.experts.{i}.{proj}.bias"] = b.to(torch.bfloat16).clone()
                report[f"blocks.{block}.experts.{i}({name}).{proj}"] = round((delta.norm() / W.norm()).item(), 5)
            # expert 6: directly fine-tuned weights
            W6 = direct.get_tensor(f"blocks.{block}.{proj}.weight")
            b6 = direct.get_tensor(f"blocks.{block}.{proj}.bias")
            assert W6.shape == W.shape and b6.shape == b.shape, (proj, W6.shape, W.shape)
            out[f"blocks.{block}.experts.6.{proj}.weight"] = W6.to(torch.bfloat16)
            out[f"blocks.{block}.experts.6.{proj}.bias"] = b6.to(torch.bfloat16)
            report[f"blocks.{block}.experts.6(direct_finetune).{proj}"] = round(((W6.float() - W).norm() / W.norm()).item(), 5)
        print(f"[merge-v3.1] block {block}: 4 projections × 6 experts done, {time.time() - t0:.0f}s", flush=True)

    for i, name in enumerate(EXPERT_NAMES[1:], start=1):
        vals = [v for k, v in report.items() if f"experts.{i}(" in k]
        print(f"[merge-v3.1] expert {i} {name}: weight delta vs. base |ΔW|/|W| range {min(vals):.4f} ~ {max(vals):.4f}", flush=True)
    path = os.path.join(args.out_dir, "experts_v3_1.safetensors")
    save_file(out, path, metadata={"experts": ",".join(EXPERT_NAMES), "moe_layers": ",".join(map(str, moe_layers)),
                                   "projs": ",".join(PROJS), "note": "experts 1-5: W_kiwi + B·A (alpha/r=1); expert 6: direct_finetune step-3581; bf16"})
    json.dump({"experts": EXPERT_NAMES, "moe_layers": moe_layers, "projs": list(PROJS), "relative_delta": report},
              open(os.path.join(args.out_dir, "experts_v3_1_report.json"), "w"), indent=1)
    print(f"[merge-v3.1] wrote {path}: {len(out)} tensors, {sum(v.numel() * 2 for v in out.values()) / 2 ** 30:.1f} GB, took {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
