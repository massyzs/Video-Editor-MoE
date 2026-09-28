#!/usr/bin/env bash
# =============================================================================
# mlp-moe (moe_v3) multi-node launcher: DeepSpeed ZeRO-3, full training of the MoE DiT, 3 nodes x 8 GPUs = 24 ranks.
# Run the same script on every node, passing the node rank as the first argument:
#     node 0: bash scripts/moe/mlp-moe/train_multinode.sh 0
#     node 1: bash scripts/moe/mlp-moe/train_multinode.sh 1
#     node 2: bash scripts/moe/mlp-moe/train_multinode.sh 2
# Activate your Python environment first. Model structure: wan_transformer3d_kiwi_moe.py; training loop: moe_v3_train.py.
# Two-stage curriculum in one process: A = OpenVE single-instruction pairs (samples containing overlay/subtitle removed), 1 epoch;
# B = GOKU + CoinVE multi-instruction pairs (sub-instructions joined with " Then "), 1 epoch. No masks.
# 12 experts (6 base experts: 5 category LoRA experts merged into the kiwi FFN + 1 GOKU direct fine-tune, each duplicated with
# 1% noise), no shared expert (Mixtral form), top-3, aux-loss alpha 1e-4.
# EXPERTS_MERGED is produced by merge_experts.py (it needs the category LoRA experts and the direct fine-tune, which are not released).
# Resume: write RESUME_DS_STATE=<OUT>/ckpt/ds_state_step-N into ${OUT}/stage.env (same file for all nodes) and restart every node.
# =============================================================================

# ---- configuration (edit these) ---------------------------------------------
BASE_DIR=/path/to/workdir          # contains ckpt/kiwi-edit-instruct-only-videoxfun/, datasets/, output/
DATASET_BASE=/path/to/video/root   # prefix for the relative video paths inside the dataset json files
MASTER_ADDR=127.0.0.1              # IP of node 0
MASTER_PORT=29839
NUM_NODES=3
GPUS_PER_NODE=8
NODE_RANK=$1                       # pass 0..NUM_NODES-1 as the first argument on each node
USE_EFA=0                          # 1 = export the AWS EFA / NCCL settings used on the original cluster (see below)
EXPERIMENT=mlp-moe                 # output goes to ${BASE_DIR}/output/${EXPERIMENT}
WANDB_PROJECT=video-editor-moe     # wandb is used only when WANDB_API_KEY is set in the environment
WANDB_RUN=${EXPERIMENT}

CKPT=${BASE_DIR}/ckpt/kiwi-edit-instruct-only-videoxfun
WAN_PATH=${CKPT}
WAN_CONFIG=config/wan2.2/wan_civitai_5b.yaml
MLLM_MODEL=${CKPT}/qwen
INIT_WEIGHTS=${CKPT}/moe_expert_init.safetensors
EXPERTS_MERGED=${BASE_DIR}/ckpt/init/experts_merged_6.safetensors    # merge_experts.py output: 5 category experts + direct_finetune

STAGE_A_DATA=${BASE_DIR}/datasets/filtered/openve_original_nosub_all.json   # OpenVE-3M original metadata, all 1,788,804 pairs (subtitles_edit category removed)
STAGE_B_DATA=${BASE_DIR}/datasets/classified/goku_with_masks.jsonl,${BASE_DIR}/datasets/classified/CoinVE.jsonl
STAGE_A_EPOCHS=1
STAGE_B_EPOCHS=1
EXCLUDE_PROMPT_WORDS=overlay,subtitle

MOE_LAYERS=1,3,5,7,9,11,13,15,17,19,21,23,25,27   # 0-based block indices
MOE_NUM_EXPERTS=12                   # 6 base experts x 2 copies (the second copy gets noise)
MOE_TOP_K=3
MOE_AUX_LOSS_ALPHA=0.0001
EXPERT_COPY_NOISE=0.01

BATCH_SIZE=8                         # per GPU; effective batch = 24 x 8 = 192
LEARNING_RATE=1e-5                   # base learning rate of all pre-existing parameters
NEW_LAYERS_LR=1e-4                   # routed experts start here and decay linearly to LEARNING_RATE over NEW_LAYERS_LR_STEPS
NEW_LAYERS_LR_STEPS=2000
GATE_LR=1e-3                         # gates start here and move linearly to GATE_LR_END over NEW_LAYERS_LR_STEPS, then stay constant
GATE_LR_END=1e-4
WEIGHT_DECAY=0.01
TEXT_DROP_PROB=0.1
MOTION_SUB_LOSS_RATIO=0.1
VAE_ENCODE_GROUP=4
SAVE_STEPS=250
DS_SAVE_STEPS=1000
NUM_FRAMES=49
HEIGHT=480
WIDTH=832
MAX_PIXELS=399360
MAX_TRAIN_STEPS=0                    # 0 = derived from the two stages' epochs
RESUME_DS_STATE=""
# ------------------------------------------------------------------------------

set -euo pipefail
if [ -z "$NODE_RANK" ]; then
  echo "usage: bash $0 <node_rank 0..$((NUM_NODES - 1))>"
  exit 1
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)    # repository root (contains videox_fun/)
cd "$ROOT"

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
echo "[log] $(date '+%F %T')  node=${NODE_RANK}/${NUM_NODES}  ->  $LOG_FILE"

# ---- optional: AWS EFA / NCCL settings of the original cluster
if [ "$USE_EFA" = "1" ]; then
  EFA_LIBS=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/efa/lib
  if [ -n "$(printenv LD_LIBRARY_PATH)" ]; then
    export LD_LIBRARY_PATH=${EFA_LIBS}:${LD_LIBRARY_PATH}
  else
    export LD_LIBRARY_PATH=${EFA_LIBS}
  fi
  export NCCL_DEBUG=INFO
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
export MALLOC_ARENA_MAX=4
export MALLOC_TRIM_THRESHOLD_=134217728

# ---- wandb: only when WANDB_API_KEY is set in the environment
if [ -n "$(printenv WANDB_API_KEY)" ]; then
  USE_WANDB=1
else
  USE_WANDB=0
fi

STAGE_ENV=${OUT}/stage.env
if [ -f "$STAGE_ENV" ]; then
  source "$STAGE_ENV"
  echo "[moe-v3] stage.env: RESUME_DS_STATE=${RESUME_DS_STATE}"
fi

# ---- DeepSpeed ZeRO-3 config (every node writes the same content; OUT should be on a filesystem shared by all nodes)
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
echo "[moe-v3] batch=${BATCH_SIZE}/gpu  world=${NUM_PROCESSES}  ds=${DS_CONFIG}  wandb=${USE_WANDB}"

accelerate launch \
  --num_processes=${NUM_PROCESSES} --num_machines=${NUM_NODES} --machine_rank=${NODE_RANK} \
  --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} \
  --mixed_precision=bf16 \
  --use_deepspeed --deepspeed_config_file=${DS_CONFIG} --zero3_init_flag=false \
  --deepspeed_multinode_launcher=standard \
  scripts/moe/mlp-moe/moe_v3_train.py \
  --pretrained_wan_path ${WAN_PATH} \
  --config_path ${WAN_CONFIG} \
  --mllm_model ${MLLM_MODEL} \
  --init_weights ${INIT_WEIGHTS} \
  --experts_merged ${EXPERTS_MERGED} \
  --stage_a_data ${STAGE_A_DATA} \
  --stage_b_data ${STAGE_B_DATA} \
  --stage_a_epochs ${STAGE_A_EPOCHS} \
  --stage_b_epochs ${STAGE_B_EPOCHS} \
  --exclude_prompt_words "${EXCLUDE_PROMPT_WORDS}" \
  --dataset_base_path "${DATASET_BASE}" \
  --output_dir ${OUT} \
  --moe_layers ${MOE_LAYERS} --moe_num_experts ${MOE_NUM_EXPERTS} --moe_top_k ${MOE_TOP_K} \
  --moe_aux_loss_alpha ${MOE_AUX_LOSS_ALPHA} --expert_copy_noise ${EXPERT_COPY_NOISE} \
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
