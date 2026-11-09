#!/usr/bin/env bash
# videoeditor-56b, low-memory inference on ONE GPU: 80 GB (6 MoE blocks resident + 8 streamed, peak 71 GiB, about 6 min per
# 49-frame video on H100) or 40 GB (all 14 MoE blocks streamed, peak 28 GiB, about 10 min per video). The MLLM encoder stays in pinned host memory
# except while a case is encoded; MoE blocks that do not fit are streamed from pinned host memory block by block. Same weights and the same
# computation as the two-GPU script. Host RAM: 112 GB for the model plus 8 GiB per streamed block (75 GB resident set on 80 GB GPUs,
# 125 GB on 40 GB GPUs); starting takes a few minutes longer (pinning).
# Edit the configuration block, then run from anywhere:  bash scripts/moe/videoeditor-56b/infer_lowmem.sh
# One video in, one video out: <OUT_DIR>/<NAME>.mp4 plus lossless PNG frames in <OUT_DIR>/frames/<NAME>/.
BASE_DIR=/path/to/base_dir                    # contains ckpt/kiwi-edit-instruct-only-videoxfun/ (shared base components)
CKPT=$BASE_DIR/ckpt/videoeditor-56b           # released weights: moe_v3_2_trainable.safetensors + meta.json (videoeditor-56b-long for long videos)
SRC_VIDEO=/path/to/input.mp4
PROMPT="Remove the dog"                        # the editing instruction
PROMPTS=""                                     # multi-step instruction instead of PROMPT: "Remove the dog ||| Convert to oil painting style"
NAME=edited                                    # output file name (without extension)
OUT_DIR=outputs/videoeditor-56b
NUM_FRAMES=49                                  # 121 or 81 with the videoeditor-56b-long weights
GPU=0                                          # one GPU with >= 30 GiB free
RESERVE_GIB=12                                 # GPU memory kept free for the MLLM encoder, activations and VAE decoding when choosing how many MoE blocks stay resident
RESIDENT_BLOCKS=-1                             # -1 = as many MoE blocks resident as fit; N = exactly N (the rest streamed)
EXTRA_ARGS=""                                  # any further options of the python script, e.g. "--steps 50 --seed 42 --frames_png 0"

set -e
cd "$(dirname "$0")/../../.."                  # repository root
if [ -n "$PROMPTS" ]; then PROMPT_ARGS=(--prompts "$PROMPTS"); else PROMPT_ARGS=(--prompt "$PROMPT"); fi
CUDA_VISIBLE_DEVICES=$GPU python scripts/moe/videoeditor-56b/moe_v3_2_infer.py --base_dir "$BASE_DIR" --ckpt "$CKPT" --gpus 1 --low_mem 1 \
    --src_video "$SRC_VIDEO" "${PROMPT_ARGS[@]}" --name "$NAME" --out_dir "$OUT_DIR" --num_frames $NUM_FRAMES \
    --reserve_gib $RESERVE_GIB --resident_blocks $RESIDENT_BLOCKS $EXTRA_ARGS
