#!/usr/bin/env bash
# videoeditor-14b, fast inference: two 80 GB GPUs (unconditional CFG branch on a DiT replica on the second GPU, CUDA graphs,
# grouped expert GEMMs), about 30 s per 49-frame video instead of 35 s (H100). Same weights; outputs equal to the default script up to bf16 rounding.
# With a single GPU set GPUS=0 DP_CFG=0 CUDA_GRAPH=0 (the single-GPU gain is only a few percent).
# Edit the configuration block, then run from anywhere:  bash scripts/moe/videoeditor-14b/infer_fast.sh
# One video in, one video out: <OUT_DIR>/<NAME>.mp4 plus lossless PNG frames in <OUT_DIR>/frames/<NAME>/.
BASE_DIR=/path/to/base_dir                    # contains ckpt/kiwi-edit-instruct-only-videoxfun/ (shared base components)
CKPT=$BASE_DIR/ckpt/videoeditor-14b           # released weights: moe_v3_1_trainable.safetensors + meta.json (videoeditor-14b-long for long videos)
SRC_VIDEO=/path/to/input.mp4
PROMPT="Remove the dog"                        # the editing instruction
PROMPTS=""                                     # multi-step instruction instead of PROMPT: "Remove the dog ||| Convert to oil painting style"
NAME=edited                                    # output file name (without extension)
OUT_DIR=outputs/videoeditor-14b
NUM_FRAMES=49                                  # 121 or 81 with the videoeditor-14b-long weights
GPUS=0,1                                       # two GPUs with >= 60 GiB free each
DP_CFG=1                                       # 1 = unconditional CFG branch on the second GPU (needs two GPUs)
CUDA_GRAPH=1                                   # 1 = capture the DiT forward in CUDA graphs (needs --moe_impl grouped and the fa2 attention backend)
EXTRA_ARGS=""                                  # any further options of the python script, e.g. "--steps 50 --seed 42 --frames_png 0"

set -e
cd "$(dirname "$0")/../../.."                  # repository root
if [ -n "$PROMPTS" ]; then PROMPT_ARGS=(--prompts "$PROMPTS"); else PROMPT_ARGS=(--prompt "$PROMPT"); fi
MOE_ATTN_BACKEND=fa2 CUDA_VISIBLE_DEVICES=$GPUS python scripts/moe/videoeditor-14b/moe_v3_1_infer_fast.py --base_dir "$BASE_DIR" --ckpt "$CKPT" \
    --src_video "$SRC_VIDEO" "${PROMPT_ARGS[@]}" --name "$NAME" --out_dir "$OUT_DIR" --num_frames $NUM_FRAMES \
    --moe_impl grouped --dp_cfg $DP_CFG --cuda_graph $CUDA_GRAPH $EXTRA_ARGS
