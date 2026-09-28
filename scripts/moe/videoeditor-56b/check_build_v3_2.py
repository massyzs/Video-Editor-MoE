"""videoeditor-56b build self-check (CPU, real weights, no forward pass): build_dit loads 35 experts (7 sources x 5 noised copies) -> mark_trainable grouping -> key/parameter counts; validates the merged expert file and the assembly code without a GPU.
Usage: CUDA_VISIBLE_DEVICES= python scripts/moe/videoeditor-56b/check_build_v3_2.py --base_dir <BASE_DIR> --experts_merged <experts_v3_1.safetensors>
  <BASE_DIR> contains ckpt/kiwi-edit-instruct-only-videoxfun/; --experts_merged is the training-init file written by videoeditor-14b/merge_experts_v3_1.py."""
import argparse, os, sys, time, types, torch
_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from moe_v3_2_model import EXPERT_NAMES, build_dit, mark_trainable, trainable_state_keys
ap = argparse.ArgumentParser()
ap.add_argument("--base_dir", required=True, help="<BASE_DIR>: contains ckpt/kiwi-edit-instruct-only-videoxfun/")
ap.add_argument("--experts_merged", required=True, help="experts_v3_1.safetensors (output of videoeditor-14b/merge_experts_v3_1.py)")
cli = ap.parse_args()
B = cli.base_dir
args = types.SimpleNamespace(pretrained_wan_path=f"{B}/ckpt/kiwi-edit-instruct-only-videoxfun", config_path=os.path.join(_repo, "config/wan2.2/wan_civitai_5b.yaml"),
                             init_weights=f"{B}/ckpt/kiwi-edit-instruct-only-videoxfun/moe_expert_init.safetensors",
                             experts_merged=cli.experts_merged, moe_layers="1,3,5,7,9,11,13,15,17,19,21,23,25,27",
                             moe_num_experts=35, moe_top_k=3, moe_aux_loss_alpha=0.001, use_gradient_checkpointing=1,
                             expert_sources="", expert_copies=5, expert_copy_noise=0.01)
t0 = time.time()
dit = build_dit(args, torch.bfloat16, verbose=True)
print(f"build_dit took {time.time() - t0:.0f}s")
base, experts, gates = mark_trainable(dit)
keys = trainable_state_keys(dit)
print("Experts:", dit.expert_names[:6], "...", dit.expert_names[-1], "total", len(dit.expert_names))
print(f"Trainable keys {len(keys)}; total params {sum(p.numel() for p in dit.parameters()) / 1e9:.2f}B")
blk = dit.blocks[1]
print("Check: k_experts[0] exists and shared_kv is off (self.k removed):", blk.self_attn.k_experts[0].weight.shape, not hasattr(blk.self_attn, "k") or blk.self_attn.k is None,
      "| ffn.0 relative diff, expert 30 (direct_finetune_0) vs 0 (main_0):", ((blk.ffn.experts[30][0].weight.float() - blk.ffn.experts[0][0].weight.float()).norm() / blk.ffn.experts[0][0].weight.float().norm()).item(),
      "| two copies of the same source, main_0 vs main_1:", ((blk.ffn.experts[1][0].weight.float() - blk.ffn.experts[0][0].weight.float()).norm() / blk.ffn.experts[0][0].weight.float().norm()).item(),
      "| gate shapes:", tuple(blk.self_attn.gate.weight.shape), tuple(blk.ffn.gate.weight.shape), blk.self_attn.gate.weight.dtype)
print("CHECK_BUILD_DONE")
