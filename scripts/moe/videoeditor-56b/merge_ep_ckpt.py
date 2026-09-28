#!/usr/bin/env python
"""Merge an expert-parallel (EP) checkpoint (dense moe_v3_2_dense.safetensors + node 0's experts_ep_lr{0..7}.safetensors) back into the full moe_v3_2_trainable.safetensors used for inference (key names unchanged).
Usage: python merge_ep_ckpt.py <ckpt/step-N dir> [--check_against <old full file; compares key sets only>]  (CPU, ~120 GB RAM; reads/writes key by key)"""
import argparse, glob, json, os
import torch
from safetensors import safe_open
from safetensors.torch import save_file

a = argparse.ArgumentParser(); a.add_argument("step_dir"); a.add_argument("--check_against", default=""); a.add_argument("--out", default="")
a = a.parse_args()
files = [os.path.join(a.step_dir, "moe_v3_2_dense.safetensors")] + sorted(glob.glob(os.path.join(a.step_dir, "experts_ep_lr*.safetensors")))
assert os.path.isfile(files[0]), files[0]
merged, srcs = {}, {}
for f in files:
    with safe_open(f, framework="pt", device="cpu") as h:
        for k in h.keys():
            assert k not in merged, f"duplicate key {k}: {srcs[k]} and {f}"
            merged[k] = h.get_tensor(k); srcs[k] = f
print(f"{len(files)} files -> {len(merged)} keys, {sum(v.numel() for v in merged.values()) / 1e9:.2f}B params")
if a.check_against:
    with safe_open(a.check_against, framework="pt", device="cpu") as h:
        ref = set(h.keys())
    missing, extra = sorted(ref - set(merged)), sorted(set(merged) - ref)
    assert not missing and not extra, f"key set differs from {a.check_against}: missing {missing[:3]} extra {extra[:3]}"
    print("key set matches the reference")
out = a.out or os.path.join(a.step_dir, "moe_v3_2_trainable.safetensors")
save_file({k: v.contiguous() for k, v in merged.items()}, out)
print("wrote", out)
