#!/usr/bin/env bash
# =============================================================================
# moe_v3_1 (videoeditor-14b) single-node 8-GPU smoke test (6 steps, weights saved at step 4, DeepSpeed state at step 6, no wandb):
#     bash scripts/moe/videoeditor-14b/train_smoke.sh
# Structure: see the header of wan_transformer3d_kiwi_moe_v3_1.py. In the 14 even-numbered layers both self-attention k/v and the FFN are 7-expert top-3 MoE (0 = original / 1..5 = category experts / 6 = GOKU direct fine-tune), no shared FFN;
# in self-attention each token brings 3 expert (k,v) copies concatenated into the sequence (no shared copy), with log ĝ as a prior. Training logic: see the header of moe_v3_1_train.py.
# Requires: <BASE_DIR>/ckpt/kiwi-edit-instruct-only-videoxfun/ (base model) and EXPERTS_MERGED (expert init weights produced by merge_experts_v3_1.py; not part of the released weights).
# Resume: write RESUME_DS_STATE=<ckpt/ds_state_step-N directory> into $OUT/stage.env.
# =============================================================================
set -euo pipefail

# ---------------------------- configuration (edit for your environment) ----------------------------
BASE_DIR=/path/to/workdir          # contains ckpt/kiwi-edit-instruct-only-videoxfun/, datasets/, output/
DATASET_BASE=/path/to/video/root   # prefix for the relative video paths inside the dataset json files
MASTER_ADDR=127.0.0.1              # single node
MASTER_PORT=29840
NUM_NODES=1
GPUS_PER_NODE=8
NODE_RANK=0
USE_EFA=0                          # 1 = export AWS EFA network settings (see below)

EXPERIMENT=videoeditor-14b_smoke
REPO_ROOT=$(cd "$(dirname "$(realpath "$0")")/../../.." && pwd)
cd "$REPO_ROOT"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

NUM_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))

OUT=${BASE_DIR}/output/${EXPERIMENT}
LOG_DIR=${OUT}/logs
mkdir -p "$LOG_DIR" "$OUT/ckpt"
LOG_FILE=${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log
touch "$LOG_FILE"
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/node${NODE_RANK}_latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[log] $(date '+%F %T')  node=${NODE_RANK}/${NUM_NODES}  →  $LOG_FILE"

# Prepend a directory to an environment variable (works whether or not the variable is already set)
prepend_path() {
  local cur
  cur=$(printenv "$1" || true)
  if [ -n "$cur" ]; then export "$1=$2:$cur"; else export "$1=$2"; fi
}

# ---- network / NCCL
export NCCL_DEBUG=INFO
if [ "$USE_EFA" = 1 ]; then          # AWS EFA (p4d/p5-class instances); change the interface name eth0 to match your nodes
  prepend_path LD_LIBRARY_PATH /opt/amazon/efa/lib
  prepend_path LD_LIBRARY_PATH /opt/amazon/ofi-nccl/lib/x86_64-linux-gnu
  export FI_PROVIDER=efa
  export FI_EFA_USE_DEVICE_RDMA=1
  export NCCL_ALGO=NVLS,TREE,RING
  export NCCL_PROTO=SIMPLE,LL,LL128
  export NCCL_BUFFSIZE=8388608
  export NCCL_SOCKET_IFNAME=eth0
  export GLOO_SOCKET_IFNAME=eth0
  export UCX_NET_DEVICES=eth0
  export NCCL_IB_DISABLE=0
  export NCCL_P2P_DISABLE=0
  export NCCL_IB_GID_INDEX=3
  export NCCL_IB_SL=3
  export NCCL_IB_TC=160
  export NCCL_IB_QPS_PER_CONNECTION=4
  export NCCL_IB_CUDA_SUPPORT=1
  export NCCL_CHECK_DISABLE=1
  export NCCL_LL_THRESHOLD=16384
  export NCCL_COLLNET_ENABLE=0
  export SHARP_COLL_ENABLE_SAT=0
  export NCCL_NET_GDR_LEVEL=2
  export NCCL_PXN_DISABLE=0
fi
export VIDEO_READER_TIMEOUT=120
export PYTHONWARNINGS="ignore::Warning"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # disabled: VMM memory + EFA GDR suspected of making ZeRO-3 all-gathers on some ranks receive corrupt data
export MALLOC_ARENA_MAX=4
export MALLOC_TRIM_THRESHOLD_=134217728

USE_WANDB=0                          # smoke test: no wandb
WANDB_PROJECT=video-editor-moe
WANDB_RUN=${EXPERIMENT}

CKPT=${BASE_DIR}/ckpt/kiwi-edit-instruct-only-videoxfun
WAN_PATH=${CKPT}
WAN_CONFIG=config/wan2.2/wan_civitai_5b.yaml
MLLM_MODEL=${CKPT}/qwen
INIT_WEIGHTS=${CKPT}/moe_expert_init.safetensors
EXPERTS_MERGED=${BASE_DIR}/ckpt/init/experts_v3_1.safetensors      # output of merge_experts_v3_1.py (experts 1..6; expert 0 is copied from the base model at assembly)

STAGE_A_DATA=${BASE_DIR}/datasets/filtered/openve_original_nosub_all.json   # OpenVE-3M original, all 1,788,804 pairs (only the subtitles_edit category removed)
STAGE_B_DATA=${BASE_DIR}/datasets/classified/goku_with_masks.jsonl,${BASE_DIR}/datasets/classified/CoinVE.jsonl
STAGE_A_EPOCHS=1
STAGE_B_EPOCHS=2                     # only used by two_stage
DATA_SCHEDULE=mixed                  # OpenVE x1 + GOKU/CoinVE x MULTI_REPEAT each, shuffled together for STAGE_A_EPOCHS epochs; two_stage = A then B
MULTI_REPEAT=2
EXCLUDE_PROMPT_WORDS=overlay,subtitle   # in the original OpenVE the "overlay" category is named subtitles_edit and its instructions say subtitles; also filter by these words as a fallback

MOE_LAYERS=1,3,5,7,9,11,13,15,17,19,21,23,25,27   # 0-based block indices = layers 2,4,...,28
MOE_NUM_EXPERTS=7                    # 0 = original / 1..5 = category experts / 6 = GOKU direct fine-tune
MOE_TOP_K=3                          # top-3 for both self-attention k/v and the FFN
MOE_AUX_LOSS_ALPHA=0.0001

BATCH_SIZE=1                         # per GPU; effective batch = 8 x 1 = 8
LEARNING_RATE=1e-5                   # base learning rate of full fine-tuning; the original parameters always use it
NEW_LAYERS_LR=1e-4                   # starting lr of the new layers (routed experts + gates), decayed linearly to LEARNING_RATE over the first NEW_LAYERS_LR_STEPS steps
NEW_LAYERS_LR_STEPS=2000
GATE_LR=1e-3                         # starting lr of the gates, linear to GATE_LR_END over the first NEW_LAYERS_LR_STEPS steps, then constant
GATE_LR_END=1e-4
WEIGHT_DECAY=0.01
TEXT_DROP_PROB=0.1
MOTION_SUB_LOSS_RATIO=0.1
VAE_ENCODE_GROUP=4                   # videos per VAE encode call; 1 = one sample at a time
SAVE_STEPS=4
DS_SAVE_STEPS=6
NUM_FRAMES=49
HEIGHT=480
WIDTH=832
MAX_PIXELS=399360
MAX_TRAIN_STEPS=6                    # 0 = derived from the number of epochs

RESUME_DS_STATE=""
STAGE_ENV=${OUT}/stage.env
if [ -f "$STAGE_ENV" ]; then
  source "$STAGE_ENV"
  echo "[moe-v3.1] stage.env: RESUME_DS_STATE=${RESUME_DS_STATE}"
fi

# ---- DeepSpeed ZeRO-3 config
DS_CONFIG=${OUT}/ds_zero3.json
cat > "$DS_CONFIG" <<JSON
{
  "bf16": {"enabled": true},
  "train_micro_batch_size_per_gpu": ${BATCH_SIZE},
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "steps_per_print": 1000000,
  "wall_clock_breakdown": false,
  "zero_allow_untested_optimizer": true,
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "sub_group_size": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 2e9,
    "stage3_max_reuse_distance": 2e9,
    "stage3_gather_16bit_weights_on_model_save": true,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"}
  }
}
JSON
echo "[moe-v3.1] batch=${BATCH_SIZE}/GPU  world=${NUM_PROCESSES}  ds=${DS_CONFIG}"

accelerate launch \
  --num_processes=${NUM_PROCESSES} --num_machines=${NUM_NODES} --machine_rank=${NODE_RANK} \
  --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} \
  --mixed_precision=bf16 \
  --use_deepspeed --deepspeed_config_file=${DS_CONFIG} --zero3_init_flag=false \
  --deepspeed_multinode_launcher=standard \
  scripts/moe/videoeditor-14b/moe_v3_1_train.py \
  --pretrained_wan_path ${WAN_PATH} \
  --config_path ${WAN_CONFIG} \
  --mllm_model ${MLLM_MODEL} \
  --init_weights ${INIT_WEIGHTS} \
  --experts_merged ${EXPERTS_MERGED} \
  --stage_a_data ${STAGE_A_DATA} \
  --stage_b_data ${STAGE_B_DATA} \
  --stage_a_epochs ${STAGE_A_EPOCHS} \
  --stage_b_epochs ${STAGE_B_EPOCHS} \
  --data_schedule ${DATA_SCHEDULE} --multi_repeat ${MULTI_REPEAT} \
  --exclude_prompt_words "${EXCLUDE_PROMPT_WORDS}" \
  --dataset_base_path "${DATASET_BASE}" \
  --output_dir ${OUT} \
  --moe_layers ${MOE_LAYERS} --moe_num_experts ${MOE_NUM_EXPERTS} --moe_top_k ${MOE_TOP_K} \
  --moe_aux_loss_alpha ${MOE_AUX_LOSS_ALPHA} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} --new_layers_lr ${NEW_LAYERS_LR} --new_layers_lr_steps ${NEW_LAYERS_LR_STEPS} \
  --gate_lr ${GATE_LR} --gate_lr_end ${GATE_LR_END} --weight_decay ${WEIGHT_DECAY} \
  --text_drop_prob ${TEXT_DROP_PROB} --motion_sub_loss_ratio ${MOTION_SUB_LOSS_RATIO} --vae_encode_group ${VAE_ENCODE_GROUP} \
  --save_steps ${SAVE_STEPS} --ds_save_steps ${DS_SAVE_STEPS} \
  --max_train_steps ${MAX_TRAIN_STEPS} \
  --resume_ds_state "${RESUME_DS_STATE}" \
  --use_wandb ${USE_WANDB} --wandb_project ${WANDB_PROJECT} --wandb_run_name ${WANDB_RUN} \
  --num_frames ${NUM_FRAMES} --height ${HEIGHT} --width ${WIDTH} --max_pixels ${MAX_PIXELS} \
  --use_gradient_checkpointing 1
