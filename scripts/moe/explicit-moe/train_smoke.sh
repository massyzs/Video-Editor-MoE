#!/usr/bin/env bash
# =============================================================================
# explicit-moe (moe_v2) single-node smoke test: 8 GPUs, 10 steps, checkpoint every 5 steps, no wandb.
# Same model / data / hyperparameters as train_multinode.sh; only the node count, step count, save interval,
# output dir and port differ. Use it to check the environment, data paths and checkpoint writing before a real run.
#
# Usage: bash scripts/moe/explicit-moe/train_smoke.sh
# =============================================================================
set -eo pipefail

# ---------------- config (edit these) ----------------
BASE_DIR=/path/to/workdir          # contains ckpt/kiwi-edit-instruct-only-videoxfun/, ckpt/explicit-moe/ (extracted release), datasets/, output/
DATASET_BASE=/path/to/video/root   # prefix for the relative video paths inside the dataset json files
MASTER_ADDR=127.0.0.1              # single node
MASTER_PORT=29830
NUM_NODES=1
GPUS_PER_NODE=8
NODE_RANK=0
USE_EFA=0                          # 1 = AWS EFA / libfabric NCCL settings (see block below)

EXPERIMENT=explicit-moe_smoke
BASE_CKPT=${BASE_DIR}/ckpt/kiwi-edit-instruct-only-videoxfun      # frozen base: DiT, VAE, qwen/, moe_expert_init.safetensors
EXPERTS_CKPT=${BASE_DIR}/ckpt/explicit-moe                        # extracted explicit-moe release: lora_experts/, router_head_online.pt
WAN_PATH=${BASE_CKPT}
WAN_CONFIG=config/wan2.2/wan_civitai_5b.yaml
MLLM_MODEL=${BASE_CKPT}/qwen
INIT_WEIGHTS=${BASE_CKPT}/moe_expert_init.safetensors
ROUTER_HEAD=${EXPERTS_CKPT}/router_head_online.pt                 # router head before co-training (fine-tuned jointly here)
LORA_EXPERTS_DIR=${EXPERTS_CKPT}/lora_experts                     # <dir>/<name>/dit_query_router.safetensors
EXPERTS=style,add,remove,replace,background

# three metadata files passed as-is, one-to-one with DATA_PREFIX (empty entry = paths are absolute or relative to DATASET_BASE)
DATA=${BASE_DIR}/datasets/classified/CoinVE.jsonl,${BASE_DIR}/datasets/classified/goku_with_masks.jsonl,${BASE_DIR}/datasets/classified/openve_v2_equal_223855.json
DATA_PREFIX=",,"

FUSION_LR=1e-5                       # anneal end point (first 600 steps 1e-4 -> 1e-5)
FUSION_LR_ANNEAL_FROM=1e-4           # anneal start point
FUSION_LR_ANNEAL_STEPS=600
ROPE_EXPERT_OFFSET0=1                # experts laid out to the right of the base (0 = expert 0 overlaps the base)
MOTION_SUB_LOSS_RATIO=0.1
FUSION_HEADS=24
FUSION_START=15                      # fusion layers after blocks [15,29) = 14 layers, none after the last block
FUSION_END=29
BASE_TEXT_DROP_PROB=0.1              # text CFG: 10% of samples get an empty base instruction (experts always keep theirs)
MAX_TRAIN_STEPS=10                   # smoke test: 10 steps only
FUSION_INIT=normal
BATCH_SIZE=2                         # same-group batch (per GPU)
GRAD_ACCUM=1
SAVE_STEPS=5
EPOCH_STEPS=13992                    # optimizer steps per epoch; lets a mid-epoch resume continue the same data order
NUM_FRAMES=49
HEIGHT=480
WIDTH=832
MAX_PIXELS=399360
RESUME_FROM=""                       # e.g. ${BASE_DIR}/output/${EXPERIMENT}/ckpt/step-N/moe_v2_state.pt; empty = train from scratch
WANDB_PROJECT=video-editor-moe
WANDB_RUN=moe_v2
# ---------------- end of config ----------------

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO"
NUM_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VIDEO_READER_TIMEOUT=120
export PYTHONWARNINGS="ignore::Warning"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=4
export MALLOC_TRIM_THRESHOLD_=134217728

# ---- optional: AWS EFA NCCL settings (every node must set LD_LIBRARY_PATH so NCCL finds the aws-ofi-nccl plugin) ----
if [ "$USE_EFA" = 1 ]; then
  export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
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

USE_WANDB=0                            # smoke runs never log to wandb

OUT=${BASE_DIR}/output/${EXPERIMENT}
LOG_DIR=${OUT}/logs
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log
touch "$LOG_FILE"
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/node${NODE_RANK}_latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[log] $(date '+%F %T')  node=${NODE_RANK}/${NUM_NODES}  ->  $LOG_FILE"
echo "[moe-v2] fusion_lr=${FUSION_LR}  steps->${MAX_TRAIN_STEPS}  resume=${RESUME_FROM}  wandb=${USE_WANDB}"

accelerate launch \
  --num_processes=${NUM_PROCESSES} --num_machines=${NUM_NODES} --machine_rank=${NODE_RANK} \
  --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} \
  --mixed_precision=bf16 \
  scripts/moe/explicit-moe/moe_v2_train.py \
  --pretrained_wan_path ${WAN_PATH} \
  --config_path ${WAN_CONFIG} \
  --mllm_model ${MLLM_MODEL} \
  --init_weights ${INIT_WEIGHTS} \
  --router_head ${ROUTER_HEAD} \
  --lora_experts_dir ${LORA_EXPERTS_DIR} \
  --experts ${EXPERTS} \
  --vid_dataset_metadata_path ${DATA} \
  --dataset_prefixes "${DATA_PREFIX}" \
  --dataset_base_path ${DATASET_BASE} \
  --output_dir ${OUT} \
  --fusion_learning_rate ${FUSION_LR} \
  --fusion_heads ${FUSION_HEADS} \
  --fusion_start ${FUSION_START} --fusion_end ${FUSION_END} --base_text_drop_prob ${BASE_TEXT_DROP_PROB} \
  --fusion_init ${FUSION_INIT} \
  --rope_expert_offset0 ${ROPE_EXPERT_OFFSET0} \
  --motion_sub_loss_ratio ${MOTION_SUB_LOSS_RATIO} \
  --fusion_lr_anneal_from ${FUSION_LR_ANNEAL_FROM} \
  --fusion_lr_anneal_steps ${FUSION_LR_ANNEAL_STEPS} \
  --max_train_steps ${MAX_TRAIN_STEPS} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --batch_size ${BATCH_SIZE} \
  --save_steps ${SAVE_STEPS} \
  --epoch_steps ${EPOCH_STEPS} \
  --resume_from "${RESUME_FROM}" \
  --use_wandb ${USE_WANDB} \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run_name ${WANDB_RUN} \
  --num_frames ${NUM_FRAMES} --height ${HEIGHT} --width ${WIDTH} --max_pixels ${MAX_PIXELS} \
  --use_gradient_checkpointing 1
