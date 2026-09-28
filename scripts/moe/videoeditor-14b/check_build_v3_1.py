"""videoeditor-14b build self-check (CPU, real weights, no forward pass): build_dit loads the 7 experts -> mark_trainable groups params -> key/param counts; validates the merged-expert file and assembly code without a GPU.
usage: CUDA_VISIBLE_DEVICES= python check_build_v3_1.py --base_dir <BASE_DIR> [--experts_merged <output of merge_experts_v3_1.py>]
     (the merged-expert file is the training starting point and is not in the released weights; default <BASE_DIR>/ckpt/init/experts_v3_1.safetensors)"""
import argparse, os, sys, time, types, torch
_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from moe_v3_1_model import EXPERT_NAMES, build_dit, mark_trainable, trainable_state_keys
_cli = argparse.ArgumentParser()
_cli.add_argument("--base_dir", required=True, help="<BASE_DIR>: holds the base model under ckpt/kiwi-edit-instruct-only-videoxfun/")
_cli.add_argument("--experts_merged", default="", help="empty = <BASE_DIR>/ckpt/init/experts_v3_1.safetensors")
_cli = _cli.parse_args()
B = _cli.base_dir
args = types.SimpleNamespace(pretrained_wan_path=f"{B}/ckpt/kiwi-edit-instruct-only-videoxfun", config_path=os.path.join(_repo, "config/wan2.2/wan_civitai_5b.yaml"),
                             init_weights=f"{B}/ckpt/kiwi-edit-instruct-only-videoxfun/moe_expert_init.safetensors",
                             experts_merged=_cli.experts_merged or f"{B}/ckpt/init/experts_v3_1.safetensors", moe_layers="1,3,5,7,9,11,13,15,17,19,21,23,25,27",
                             moe_num_experts=7, moe_top_k=3, moe_aux_loss_alpha=0.001, use_gradient_checkpointing=1)
t0 = time.time()
dit = build_dit(args, torch.bfloat16, verbose=True)
print(f"build_dit took {time.time() - t0:.0f}s")
base, experts, gates = mark_trainable(dit)
keys = trainable_state_keys(dit)
print(f"trainable keys {len(keys)}; total params {sum(p.numel() for p in dit.parameters()) / 1e9:.2f}B")
blk = dit.blocks[1]
print("self-check: k_experts[0] exists and shared_kv is off (self.k removed):", blk.self_attn.k_experts[0].weight.shape, not hasattr(blk.self_attn, "k") or blk.self_attn.k is None,
      "| ffn.0 relative diff, expert 6 vs 0:", ((blk.ffn.experts[6][0].weight.float() - blk.ffn.experts[0][0].weight.float()).norm() / blk.ffn.experts[0][0].weight.float().norm()).item(),
      "| gate shapes:", tuple(blk.self_attn.gate.weight.shape), tuple(blk.ffn.gate.weight.shape), blk.self_attn.gate.weight.dtype)
print("CHECK_BUILD_DONE")
