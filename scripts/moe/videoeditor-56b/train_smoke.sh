#!/usr/bin/env bash
# =============================================================================
# videoeditor-56b (moe_v3_2) 24-GPU SMOKE TEST: same model / data / DeepSpeed setup as train_multinode.sh, builds the 35 experts
# from scratch and runs 30 steps to measure memory (mem/reserved_max_gb) and step time. No checkpoints, no wandb.
# Differences from train_multinode.sh: EXPERIMENT, port, 8 OMP threads, batch 4, MLLM attention flash_attention_3,
# prefetch bucket 5e8, no checkpoint saving, 30 steps, no --opt_probe / weights-only-resume / EP / prefetch flags (trainer defaults).
# Run the same script on every node, passing the node rank as the only argument:
#     node 0 (master):  bash scripts/moe/videoeditor-56b/train_smoke.sh 0
#     node 1:           bash scripts/moe/videoeditor-56b/train_smoke.sh 1
#     node 2:           bash scripts/moe/videoeditor-56b/train_smoke.sh 2
# Activate your python environment first. BASE_DIR / OUT must be on a file system shared by all nodes
# (every node writes the same DeepSpeed config and reads the same checkpoints).
#
# Model: the 30-block Kiwi-Edit DiT with MoE self-attention k/v and MoE FFN in blocks 1,3,...,27; 35 experts
# (7 source experts x 5 noisy copies), top-3 routing, 55.9B parameters, full fine-tuning with three learning-rate groups.
# The 7 source experts are read from EXPERTS_MERGED (written by scripts/moe/videoeditor-14b/merge_experts_v3_1.py) plus the base DiT.
# Memory: optimizer states are offloaded to CPU (DeepSpeedCPUAdam, ~28 GB host RAM per GPU); building the model peaks at
# ~126 GB host RAM per process, so a node needs ~1 TB RAM for 8 processes.
# =============================================================================
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "usage: bash $0 <NODE_RANK>    (0 on the master node, 1..NUM_NODES-1 on the others)"
  exit 1
fi

# ---------------------------------------------------------------- cluster / paths (edit these)
BASE_DIR=/path/to/workdir          # contains ckpt/kiwi-edit-instruct-only-videoxfun/, datasets/, output/
DATASET_BASE=/path/to/video/root   # prefix for the relative video paths inside the dataset json files
MASTER_ADDR=127.0.0.1              # IP of node 0
MASTER_PORT=29852
NUM_NODES=3
GPUS_PER_NODE=8
NODE_RANK=$1                       # pass 0..NUM_NODES-1 as the first argument on each node
SOCKET_IFNAME=eth0                 # network interface for NCCL / Gloo bootstrap
USE_EFA=0                          # 1 = AWS EFA settings used for the original run (aws-ofi-nccl plugin, RDMA)
FA3_SITE=                          # optional directory with FlashAttention-3 (flash_attn_interface) to prepend to PYTHONPATH; empty = not used
NUMA_BIND=0                        # 1 = bind each rank to the CPU cores of its GPU's socket; the core lists in moe_v3_2_train.py
                                   #     assume GPU0-3 -> CPUs 0-47,96-143 and GPU4-7 -> CPUs 48-95,144-191 (check `nvidia-smi topo -m`)

EXPERIMENT=videoeditor-56b_smoke
OUT=${BASE_DIR}/output/${EXPERIMENT}

CKPT=${BASE_DIR}/ckpt/kiwi-edit-instruct-only-videoxfun
WAN_PATH=${CKPT}
WAN_CONFIG=config/wan2.2/wan_civitai_5b.yaml
MLLM_MODEL=${CKPT}/qwen
INIT_WEIGHTS=${CKPT}/moe_expert_init.safetensors
EXPERTS_MERGED=${BASE_DIR}/ckpt/init/experts_v3_1.safetensors   # output of videoeditor-14b/merge_experts_v3_1.py (experts 1..6; expert 0 = base DiT)

# ---------------------------------------------------------------- data
STAGE_A_DATA=${BASE_DIR}/datasets/filtered/openve_original_nosub_all.json,${BASE_DIR}/datasets/filtered/reco_all.json   # single-instruction: OpenVE (without subtitles_edit) + ReCo-Data
STAGE_B_DATA=${BASE_DIR}/datasets/classified/goku_with_masks.jsonl,${BASE_DIR}/datasets/classified/CoinVE.jsonl        # multi-instruction: goku + CoinVE
STAGE_A_EPOCHS=2                     # mixed schedule: the mixed data set is trained for STAGE_A_EPOCHS epochs
STAGE_B_EPOCHS=2                     # only used by DATA_SCHEDULE=two_stage
DATA_SCHEDULE=mixed                  # mixed = single-instruction x1 + multi-instruction x MULTI_REPEAT, shuffled together
MULTI_REPEAT=10                      # multi-instruction data repeated to roughly match the single-instruction data
EXCLUDE_PROMPT_WORDS=overlay,subtitle

# ---------------------------------------------------------------- model / optimisation (values of the released run)
MOE_LAYERS=1,3,5,7,9,11,13,15,17,19,21,23,25,27   # 0-based block indices
MOE_NUM_EXPERTS=35                   # = number of EXPERT_SOURCES x EXPERT_COPIES (checked by the trainer)
EXPERT_SOURCES=main,style,add,remove,replace,background,direct_finetune
EXPERT_COPIES=5
EXPERT_COPY_NOISE=0.01               # each copy = source weight + randn x (weight std x 0.01), fixed seed
MOE_TOP_K=3                          # top-3 for both self-attention k/v and FFN
MOE_AUX_LOSS_ALPHA=0.0001
BATCH_SIZE=4                         # per GPU; effective batch = 24 x 4 = 96, no gradient accumulation
LEARNING_RATE=1e-5                   # base learning rate (all original parameters)
NEW_LAYERS_LR=1e-4                   # experts: start lr, linearly decayed to LEARNING_RATE over NEW_LAYERS_LR_STEPS
NEW_LAYERS_LR_STEPS=2000
GATE_LR=1e-3                         # gates: start lr, linearly to GATE_LR_END over NEW_LAYERS_LR_STEPS, then constant
GATE_LR_END=1e-4
WEIGHT_DECAY=0.01
TEXT_DROP_PROB=0.1
MOTION_SUB_LOSS_RATIO=0.1
VAE_ENCODE_GROUP=8
DATASET_NUM_WORKERS=8
DATALOADER_PREFETCH_FACTOR=2
MULTI_CACHE_KEEP=1                   # 1 = keep the multi-edit feature cache across restarts (not deleted at start / exit)
MULTI_CACHE_DELETE_ON_DONE=0
MULTI_CACHE_DIR=${OUT}/cache_multi   # VAE moments + MLLM context cache of the multi-edit samples
OFFLOAD_OPTIMIZER=cpu                # optimizer states on CPU + DeepSpeedCPUAdam; none = keep on GPU
ZERO_HPZ=8                           # ZeRO++ hpZ secondary partition inside each 8-GPU node
VAE_COMPILE=off
MLLM_ATTN_IMPL=flash_attention_3     # needs FlashAttention-3 importable by transformers; the full run (train_multinode.sh) uses sdpa
Z3_LEAF=1                            # MoEFFN / MoEKVSelfAttention as ZeRO-3 leaf modules (one gather per MoE module)
FINAL_SAVE=0                         # smoke test: no final save
SAVE_STEPS=100000                    # smoke test: never reached
DS_SAVE_STEPS=100000
NUM_FRAMES=49
HEIGHT=480
WIDTH=832
MAX_PIXELS=399360
MAX_TRAIN_STEPS=30                   # smoke test: 30 steps

RESUME_DS_STATE=""

# ---------------------------------------------------------------- environment
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "$REPO_DIR"
[ -d "$CKPT" ] || { echo "[moe-v3.2] base weights not found: $CKPT (set BASE_DIR)"; exit 1; }
[ -f "$EXPERTS_MERGED" ] || { echo "[moe-v3.2] merged experts not found: $EXPERTS_MERGED (run videoeditor-14b/merge_experts_v3_1.py)"; exit 1; }
if pgrep -f 'videoeditor-56b/moe_v3_2_trai[n].py' >/dev/null; then
  echo "[moe-v3.2] a moe_v3_2 training process is already running on this node, refusing to start a second one"
  exit 3
fi
NUM_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))

LOG_DIR=${OUT}/logs
mkdir -p "$LOG_DIR" "$OUT/ckpt" "$OUT/render"
LOG_FILE=${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log
touch "$LOG_FILE"
ln -sf "$(basename "$LOG_FILE")" "${LOG_DIR}/node${NODE_RANK}_latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[log] $(date '+%F %T')  node=${NODE_RANK}/${NUM_NODES}  ->  $LOG_FILE"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MOE_ATTN_BACKEND=fa3          # DiT attention: FlashAttention-3 if importable, otherwise the library falls back to FA2 with a warning
if [ -n "$FA3_SITE" ]; then export PYTHONPATH=${FA3_SITE}:${PYTHONPATH:-}; fi
export NCCL_DEBUG=INFO
export NCCL_ALGO=NVLS,TREE,RING
export NCCL_PROTO=SIMPLE,LL,LL128
export NCCL_BUFFSIZE=8388608
export NCCL_SOCKET_IFNAME=${SOCKET_IFNAME}
export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}
export UCX_NET_DEVICES=${SOCKET_IFNAME}
export NCCL_P2P_DISABLE=0
if [ "$USE_EFA" = 1 ]; then
  # AWS EFA: every node must find the aws-ofi-nccl plugin, otherwise it silently falls back to TCP sockets
  export LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
  export FI_PROVIDER=efa
  export FI_EFA_USE_DEVICE_RDMA=1
  export NCCL_IB_DISABLE=0
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=4
export MALLOC_TRIM_THRESHOLD_=134217728
export MOE_NUMA_BIND=${NUMA_BIND}

USE_WANDB=0                          # smoke test: never logs to wandb
WANDB_PROJECT=video-editor-moe
WANDB_RUN=videoeditor-56b_smoke

STAGE_ENV=${OUT}/stage.env
if [ -f "$STAGE_ENV" ]; then
  source "$STAGE_ENV"
  echo "[moe-v3.2] stage.env: RESUME_DS_STATE=${RESUME_DS_STATE}"
fi

CPU_ADAM=0; OFFLOAD_EXTRA=""
if [ "$OFFLOAD_OPTIMIZER" = cpu ]; then CPU_ADAM=1; OFFLOAD_EXTRA=', "pin_memory": true'; fi

# ---------------------------------------------------------------- DeepSpeed ZeRO-3 config (every node writes the same content)
DS_CONFIG=${OUT}/ds_zero3.json
cat > "$DS_CONFIG" <<EOF
{
  "bf16": {"enabled": true},
  "train_micro_batch_size_per_gpu": ${BATCH_SIZE},
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "steps_per_print": 1000000,
  "wall_clock_breakdown": false,
  "zero_allow_untested_optimizer": true,
  "zero_force_ds_cpu_optimizer": false,
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "sub_group_size": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e5,
    "stage3_max_live_parameters": 4e9,
    "stage3_max_reuse_distance": 2e9,
    "zero_hpz_partition_size": ${ZERO_HPZ},
    "stage3_gather_16bit_weights_on_model_save": true,
    "offload_optimizer": {"device": "${OFFLOAD_OPTIMIZER}"${OFFLOAD_EXTRA}},
    "offload_param": {"device": "none"}
  }
}
EOF
echo "[moe-v3.2] batch=${BATCH_SIZE}/GPU  world=${NUM_PROCESSES}  experts=${MOE_NUM_EXPERTS}  optimizer offload=${OFFLOAD_OPTIMIZER}  ds=${DS_CONFIG}  wandb=${USE_WANDB}"

# ---------------------------------------------------------------- multi-edit feature cache cleanup
# With MULTI_CACHE_KEEP=0 the cache directory is renamed and deleted in the background at start and on exit
# (the trainer also deletes it when training finishes); with MULTI_CACHE_KEEP=1 nothing is deleted.
cleanup_multi_cache() {
  local d
  if [ "$MULTI_CACHE_KEEP" = 1 ]; then return 0; fi
  if [ -d "$MULTI_CACHE_DIR" ]; then mv "$MULTI_CACHE_DIR" "${MULTI_CACHE_DIR}.del.$$.$(date +%s)" 2>/dev/null || true; fi
  for d in "${MULTI_CACHE_DIR}".del.*; do
    [ -d "$d" ] || continue
    nohup bash -c 'find "$1" -mindepth 1 -maxdepth 1 -print0 | xargs -0 -r -P 16 -n 256 rm -rf; rm -rf "$1"' _ "$d" >/dev/null 2>&1 &
  done
}
trap cleanup_multi_cache EXIT
cleanup_multi_cache

accelerate launch \
  --num_processes=${NUM_PROCESSES} --num_machines=${NUM_NODES} --machine_rank=${NODE_RANK} \
  --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} \
  --mixed_precision=bf16 \
  --use_deepspeed --deepspeed_config_file=${DS_CONFIG} --zero3_init_flag=false \
  --deepspeed_multinode_launcher=standard \
  scripts/moe/videoeditor-56b/moe_v3_2_train.py \
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
  --expert_sources ${EXPERT_SOURCES} --expert_copies ${EXPERT_COPIES} --expert_copy_noise ${EXPERT_COPY_NOISE} \
  --moe_aux_loss_alpha ${MOE_AUX_LOSS_ALPHA} --cpu_adam ${CPU_ADAM} --final_save ${FINAL_SAVE} --ds_config ${DS_CONFIG} --z3_leaf_modules ${Z3_LEAF} --mllm_attn_impl ${MLLM_ATTN_IMPL} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} --new_layers_lr ${NEW_LAYERS_LR} --new_layers_lr_steps ${NEW_LAYERS_LR_STEPS} \
  --gate_lr ${GATE_LR} --gate_lr_end ${GATE_LR_END} --weight_decay ${WEIGHT_DECAY} \
  --text_drop_prob ${TEXT_DROP_PROB} --motion_sub_loss_ratio ${MOTION_SUB_LOSS_RATIO} --vae_encode_group ${VAE_ENCODE_GROUP} \
  --dataset_num_workers ${DATASET_NUM_WORKERS} --dataloader_prefetch_factor ${DATALOADER_PREFETCH_FACTOR} \
  --multi_cache_dir "${MULTI_CACHE_DIR}" --cache_keep_existing ${MULTI_CACHE_KEEP} --cache_delete_on_done ${MULTI_CACHE_DELETE_ON_DONE} --vae_compile ${VAE_COMPILE} \
  --save_steps ${SAVE_STEPS} --ds_save_steps ${DS_SAVE_STEPS} \
  --max_train_steps ${MAX_TRAIN_STEPS} \
  --resume_ds_state "${RESUME_DS_STATE}" \
  --use_wandb ${USE_WANDB} --wandb_project ${WANDB_PROJECT} --wandb_run_name ${WANDB_RUN} \
  --num_frames ${NUM_FRAMES} --height ${HEIGHT} --width ${WIDTH} --max_pixels ${MAX_PIXELS} \
  --use_gradient_checkpointing 1
