#!/usr/bin/env bash
# videoeditor-14b, low-memory inference: one 40 GB GPU (peak 35 GiB with RESERVE_GIB=10, all MoE blocks resident, default-mode speed).
# The MLLM encoder stays in pinned host memory except while a case is encoded; MoE blocks that do not fit are streamed from host memory.
# Same weights and the same computation as the default script. Host RAM: about 45 GB.
# Edit the configuration block, then run from anywhere:  bash scripts/moe/videoeditor-14b/infer_lowmem.sh
# One video in, one video out: <OUT_DIR>/<NAME>.mp4 plus lossless PNG frames in <OUT_DIR>/frames/<NAME>/.
BASE_DIR=/path/to/base_dir                    # contains ckpt/kiwi-edit-instruct-only-videoxfun/ (shared base components)
CKPT=$BASE_DIR/ckpt/videoeditor-14b           # released weights: moe_v3_1_trainable.safetensors + meta.json (videoeditor-14b-long for long videos)
SRC_VIDEO=/path/to/input.mp4
PROMPT="Remove the dog"                        # the editing instruction
PROMPTS=""                                     # multi-step instruction instead of PROMPT: "Remove the dog ||| Convert to oil painting style"
NAME=edited                                    # output file name (without extension)
OUT_DIR=outputs/videoeditor-14b
NUM_FRAMES=49                                  # 121 or 81 with the videoeditor-14b-long weights
GPU=0                                          # one GPU with >= 20 GiB free
RESERVE_GIB=10                                 # GPU memory kept free for the MLLM encoder, activations and VAE decoding; 10 keeps all 14 MoE blocks resident on 40 GB
RESIDENT_BLOCKS=-1                             # -1 = as many MoE blocks resident as fit; N = exactly N (the rest streamed)
EXTRA_ARGS=""                                  # any further options of the python script, e.g. "--steps 50 --seed 42 --frames_png 0"

set -e
cd "$(dirname "$0")/../../.."                  # repository root
if [ -n "$PROMPTS" ]; then PROMPT_ARGS=(--prompts "$PROMPTS"); else PROMPT_ARGS=(--prompt "$PROMPT"); fi
CUDA_VISIBLE_DEVICES=$GPU python scripts/moe/videoeditor-14b/moe_v3_1_infer.py --base_dir "$BASE_DIR" --ckpt "$CKPT" \
    --src_video "$SRC_VIDEO" "${PROMPT_ARGS[@]}" --name "$NAME" --out_dir "$OUT_DIR" --num_frames $NUM_FRAMES \
    --low_mem 1 --reserve_gib $RESERVE_GIB --resident_blocks $RESIDENT_BLOCKS $EXTRA_ARGS
