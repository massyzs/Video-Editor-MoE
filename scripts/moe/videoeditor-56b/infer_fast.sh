#!/usr/bin/env bash
# videoeditor-56b, fast inference: two 80 GB GPUs, about 37 s per 49-frame video instead of 46 s (H100): fused expert weights,
# grouped expert GEMMs, the two CFG branches pipelined on separate CUDA streams. Same weights; outputs equal to the default script up to bf16 rounding.
# Reserved GPU memory grows to about 78 GiB per GPU; CFG_PIPELINE=0 CFG_BATCH=1 keeps it at 60-62 GiB at nearly the same speed.
# Edit the configuration block, then run from anywhere:  bash scripts/moe/videoeditor-56b/infer_fast.sh
# One video in, one video out: <OUT_DIR>/<NAME>.mp4 plus lossless PNG frames in <OUT_DIR>/frames/<NAME>/.
BASE_DIR=/path/to/base_dir                    # contains ckpt/kiwi-edit-instruct-only-videoxfun/ (shared base components)
CKPT=$BASE_DIR/ckpt/videoeditor-56b           # released weights: moe_v3_2_trainable.safetensors + meta.json (videoeditor-56b-long for long videos)
SRC_VIDEO=/path/to/input.mp4
PROMPT="Remove the dog"                        # the editing instruction
PROMPTS=""                                     # multi-step instruction instead of PROMPT: "Remove the dog ||| Convert to oil painting style"
NAME=edited                                    # output file name (without extension)
OUT_DIR=outputs/videoeditor-56b
NUM_FRAMES=49                                  # 121 or 81 with the videoeditor-56b-long weights
GPUS=0,1                                       # two GPUs with >= 60 GiB free each
CFG_PIPELINE=1                                 # 1 = the two CFG branches on separate CUDA stream pairs (bit-identical to sequential)
CFG_BATCH=0                                    # 1 = the two CFG branches as one batch-2 forward (alternative to CFG_PIPELINE)
EXTRA_ARGS=""                                  # any further options of the python script, e.g. "--steps 50 --seed 42 --frames_png 0"

set -e
cd "$(dirname "$0")/../../.."                  # repository root
if [ -n "$PROMPTS" ]; then PROMPT_ARGS=(--prompts "$PROMPTS"); else PROMPT_ARGS=(--prompt "$PROMPT"); fi
CUDA_VISIBLE_DEVICES=$GPUS python scripts/moe/videoeditor-56b/moe_v3_2_infer_fast.py --base_dir "$BASE_DIR" --ckpt "$CKPT" --gpus 2 \
    --src_video "$SRC_VIDEO" "${PROMPT_ARGS[@]}" --name "$NAME" --out_dir "$OUT_DIR" --num_frames $NUM_FRAMES \
    --moe_impl grouped --cfg_pipeline $CFG_PIPELINE --cfg_batch $CFG_BATCH $EXTRA_ARGS
