# videoeditor-56b: 35-expert scale-up of videoeditor-14b

videoeditor-56b has the same architecture as [videoeditor-14b](../videoeditor-14b/README.md), with more experts. Each of the
7 source experts of videoeditor-14b is replicated 5 times with small independent weight noise, which gives 35 experts per MoE
layer. Routing stays top-3, so the DiT has 55.93B parameters but the same ~8B active parameters per token as videoeditor-14b.
File names, log tags and checkpoint files use the internal name `moe_v3_2`.

## Architecture

The backbone is the 30-block Kiwi-Edit DiT (hidden size 3072, FFN size 14336). In blocks 1, 3, ..., 27 (0-based, 14 blocks),
both the self-attention key/value projections and the FFN are mixture-of-experts layers:

- **FFN**: 35 expert FFNs. Each token goes to its top-3 experts and `y = sum_i g_i * expert_i(x)`, where `g` is the gate
  softmax renormalised over the three selected experts (Mixtral form, no shared expert).
- **Self-attention k/v**: the query and output projections and the q/k norms are shared. Each token contributes the
  `(k_i, v_i)` pairs of its three selected experts. All copies are concatenated into one key/value sequence of length 3L, and a
  single attention is computed, so the softmax normalises over all copies. The routing weight enters as a prior `log g_i` on
  the attention logits. This is implemented by padding the head dimension with 8 channels, so FlashAttention works without an
  attention bias. When every expert equals the base weights, the layer reproduces the dense model exactly.
- **Gates**: each MoE block has two independent gates (attention and FFN). Each gate is a bias-free `Linear(3072 -> 35)`,
  evaluated in fp32 and trained from scratch. Each gate has its own DeepSeek-style per-sample expert-balance loss
  (`alpha = 1e-4`).
- Cross-attention, the other 16 blocks, the embeddings and the output head are dense.

The expert index determines the expert order, and `meta.json` lists the names:

| source (`j // 5`) | initial weights of the source | expert names |
|---|---|---|
| `main` | Kiwi-Edit DiT | `main_0` ... `main_4` |
| `style`, `add`, `remove`, `replace`, `background` | Kiwi-Edit DiT with the rank-128 LoRA of that edit category merged into `self_attn.k`, `self_attn.v`, `ffn.0`, `ffn.2` | `style_0` ... `background_4` |
| `direct_finetune` | a Kiwi-Edit DiT fully fine-tuned on multi-step edits | `direct_finetune_0` ... `direct_finetune_4` |

Expert `j` is copy `j % 5` of source `j // 5`. When training starts, each copy is its source weight plus Gaussian noise with
standard deviation `0.01 x std(W)` for each weight tensor. Biases get no noise. The seed is fixed, so every rank builds identical
experts. The noise only initialises training. At inference, every expert is loaded from the checkpoint.

Parameter counts (DiT only; the frozen MLLM encoder, 3.77B, and the Wan2.2 VAE, 0.70B, are not included):

- total 55.93B = ~3.50B shared + 35 experts x 1.498B (k, v and FFN of the 14 MoE blocks) + 3.0M gate parameters;
- active per token ~8.0B = shared + 3 x 1.498B + gates, the same as videoeditor-14b apart from the larger gates.

In the DiT definition, the only change from videoeditor-14b is the constructor argument `moe_expert_dtype`. It builds the expert
linear layers directly in bf16, which cuts the host memory needed to build the model from ~224 GB (fp32) to ~120 GB per process.
Changes relative to `videox_fun/models/wan_transformer3d_kiwi.py` are marked `MOE-V3.2` in the source.

## Files

| file | purpose |
|---|---|
| `wan_transformer3d_kiwi_moe_v3_2.py` | DiT: a full copy of `videox_fun/models/wan_transformer3d_kiwi.py` plus `route_tokens`, `expert_balance_loss`, `Fp32Gate`, `MoEFFN` and `MoEKVSelfAttention` (search `MOE-V3.2`) |
| `moe_v3_2_model.py` | assembly: `build_dit` (for training it builds the 35 experts from the 7 sources, adds noise and runs self-checks; for inference it builds only the skeleton), `mark_trainable` (base / expert / gate parameter groups), `TextEncoder`, `build_mllm`, `build_vae`, `load_trainable_weights` |
| `moe_v3_2_train.py` | training loop: mixed data, flow-matching loss, three learning-rate groups, DeepSpeed ZeRO-3 saving and resuming, optional expert parallelism |
| `moe_v3_2_ep.py` | optional expert parallelism (`--ep 1`): expert placement, expert weight store with sharded CPU Adam, EP versions of the MoE modules |
| `merge_ep_ckpt.py` | merges an expert-parallel checkpoint into the full `moe_v3_2_trainable.safetensors` |
| `moe_v3_2_infer.py` | inference on two GPUs, for a single video or for benchmark case lists (several CFG / sigma-shift variants per model load) |
| `train_multinode.sh` | training launcher for 3 nodes x 8 GPUs |
| `train_smoke.sh` | 24-GPU smoke test: 30 steps, no checkpoints, no wandb |
| `probe_moe_v3_2_cpu.py`, `test_ep_cpu.py`, `test_mllm_pre_cpu.py`, `check_build_v3_2.py` | CPU tests (see [Tests](#tests)) |

## Weights

Download the shared base components and the model package. Both repositories are gated, so request access first. The package
is 5 chunks, about 90 GB compressed.

```bash
huggingface-cli download Massyzs/kiwi-edit-5b-instruct-only-videoxfun \
    --local-dir <BASE_DIR>/ckpt/kiwi-edit-instruct-only-videoxfun
huggingface-cli download Massyzs/videoeditor-56b --local-dir <BASE_DIR>/hf/videoeditor-56b
cd <BASE_DIR>/hf/videoeditor-56b
cat videoeditor-56b.tar.gz.part-* | tar -xzf - -C <BASE_DIR>/ckpt/       # -> <BASE_DIR>/ckpt/videoeditor-56b/
cp SHA256SUMS.contents <BASE_DIR>/ckpt/ && (cd <BASE_DIR>/ckpt && sha256sum -c SHA256SUMS.contents)   # optional check
```

`<BASE_DIR>/ckpt/videoeditor-56b/` then contains:

- `moe_v3_2_trainable.safetensors`: the full DiT (55.93B parameters, bf16, 111.85 GB, 4,663 tensors) at training step
  42,500, without optimizer state;
- `meta.json`: the MoE configuration (MoE blocks, 35 experts, top-3, expert names / sources / copies) and the training
  learning rates.

From the base directory, inference uses `qwen/` (MLLM encoder), `Wan2.2_VAE.pth`, `moe_expert_init.safetensors` and
`config.json`. The script loads `diffusion_pytorch_model.safetensors` while it builds the DiT, then overwrites every DiT weight
from the package.

## Inference

Requirements:

- Two GPUs with at least 60 GiB free each. The script checks this at start (`--min_free_gib`). The bf16 DiT is 112 GB and does
  not fit one 80 GB GPU.
- About 120-210 GB of free host RAM. The bf16 DiT is first built on the CPU, then the checkpoint is streamed into it, and only
  then does the model move to the GPUs.

With the defaults `--gpus 2 --split_block 15`, blocks 0-14 and the input embeddings go on the first visible GPU (~58 GiB).
Blocks 15-29 and the output head go on the second (~63 GiB). The MLLM encoder also runs on the second GPU and the VAE on the
first. `--gpus 1` is accepted only when the whole bf16 DiT fits in 92% of one GPU's memory, which rules out 80 GB GPUs.
The model is built without random initialisation (`--fast_init 1`). The script asserts that the checkpoint covers every
parameter, then loads it. `--experts_merged` stays empty, because inference needs only the released files.

Run the commands from the repository root.

### A single video

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/moe/videoeditor-56b/moe_v3_2_infer.py \
    --base_dir <BASE_DIR> --ckpt <BASE_DIR>/ckpt/videoeditor-56b \
    --src_video <input.mp4> --prompt "Remove the dog" --name dog_removed --out_dir <OUT_DIR>

# multi-step instruction: the sub-instructions are joined with " Then " into one instruction, as in training
CUDA_VISIBLE_DEVICES=0,1 python scripts/moe/videoeditor-56b/moe_v3_2_infer.py \
    --base_dir <BASE_DIR> --ckpt <BASE_DIR>/ckpt/videoeditor-56b \
    --src_video <input.mp4> --prompts "Remove the dog ||| Convert to oil painting style" --name dog_oil --out_dir <OUT_DIR>
```

Sampling defaults:

- Flow-matching Euler with `--steps 50` and `--sigma_shift 5`.
- Text CFG `pred = u + s * (c - u)` with `--cfg_text_scale 3`, applied only in the first `--cfg_text_steps_frac 0.6` of the
  steps. A scale of 1 disables CFG, and a fraction of 1.0 applies it to all steps.
- `--seed 42`.

The source video is prepared as in training. The script reads its first `--num_frames 49` frames, trims them to 4k+1 and pads with
the last frame. It then resizes the short side and centre-crops to `--width 832` x `--height 480`. The output is written at
`--fps 16`. `--variants 'label:cfg_scale:cfg_frac:sigma_shift,...'` renders several settings with one model load, each into
`<OUT_DIR>/<label>/`.

Outputs in `<OUT_DIR>`:

- `<name>.mp4`;
- `frames/<name>/f000.png ... f048.png`: lossless frames (`--frames_png 0` to skip);
- `<name>_experts_attn.png` and `<name>_experts_ffn.png`: the top-1 expert of every DiT token, recorded on the conditional
  pass at the middle denoising step. Rows are MoE blocks 1, 15 and 27, and columns are the first, middle and last latent frame.
  Each source expert has one hue, and its five copies run from dark to light. `--dump_expert_maps 0` skips these maps;
- `manifest_shard<k>.jsonl`: one record per output with the settings, timings and per-block expert usage counts.

Outputs that already exist and are complete are skipped (`--skip_existing 1`).

### Benchmark case lists

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/moe/videoeditor-56b/moe_v3_2_infer.py \
    --base_dir <BASE_DIR> --ckpt <BASE_DIR>/ckpt/videoeditor-56b \
    --bench_root <BENCH_ROOT> --benches <bench_a>,<bench_b> --out_root <OUT_ROOT> \
    --shard_idx 0 --num_shards 1
```

- **Case files**: each `<BENCH_ROOT>/<bench>/cases.json` is a list of
  `{"name": ..., "src": <video path relative to <BENCH_ROOT>/<bench>/>, "prompts": [<sub-instruction>, ...]}`. The prompts
  of a case are joined with " Then ".
- **Benchmark names**: `--benches` defaults to the authors' benchmark folder names (`openve_bench,coinve_bench_sub,covebench_sub`).
  Pass your own folder names.
- **Variants**: without `--variants`, each case is rendered with four variants:
  `seg_shift5:3:0.6:5,full_shift5:3:1.0:5,seg_shift3:3:0.6:3,full_shift3:3:1.0:3`. These are text CFG 3 on the first 60% of
  the steps or on all steps, each with sigma shift 5 and 3. The MLLM encoding, VAE encoding and initial noise are computed once
  per case and shared by all variants.
- **Outputs**: `<OUT_ROOT>/<bench>/<label>/<case>.mp4`, plus frames, expert maps and manifests laid out as above.
- **Sharding**: to use more GPUs, start one process per GPU pair with the same arguments and
  `--shard_idx 0..N-1 --num_shards N`. Case `i` of the combined list goes to shard `i mod N`.

Exit codes for a retrying runner: 17 = non-finite weights after loading, 18 = the MLLM self-test failed three times,
19 = degenerate output (non-finite or constant video).

Install flash-attn (see `requirements.txt`) for the DiT attention. `MOE_ATTN_BACKEND=fa3` switches the DiT to
FlashAttention-3. The MLLM encoder uses PyTorch SDPA.

## Training

### Setup

- **Parallelism**: 24 GPUs (3 nodes x 8, 80 GB) train the whole DiT (full fine-tuning) with DeepSpeed ZeRO-3. The run uses the
  ZeRO++ secondary parameter partition inside each node (`ZERO_HPZ=8`) and registers every MoE module as a ZeRO-3 leaf module
  (`Z3_LEAF=1`). Training runs in bf16 with gradient checkpointing, gradient clipping at 1.0 and no gradient accumulation.
- **Memory**: optimizer states go to the CPU (`OFFLOAD_OPTIMIZER=cpu`, `DeepSpeedCPUAdam` in AdamW mode, pinned memory), about
  28 GB of host RAM per GPU. Building the model peaks at about 126 GB of host RAM per process, so an 8-GPU node needs about
  1 TB during start-up. Each GPU needs at least 60 GiB free when the trainer starts; the trainer waits up to 10 minutes for it.
- **Optimizer check**: the DeepSpeed config written by the launcher sets `"zero_force_ds_cpu_optimizer": false`. Keep this
  setting. Without it, accelerate replaces the optimizer with `DeepSpeedCPUAdam(adamw_mode=False)`, which is Adam with coupled
  L2 weight decay. The trainer asserts that the optimizer in use is `DeepSpeedCPUAdam` with `adam_w_mode=True`. The launcher
  also passes `--opt_probe 50`, which makes rank 0 print Adam-moment / weight statistics of the fp32 shards every 50 steps.
- **Storage**: `BASE_DIR` and the output directory must be on a file system shared by all nodes. A weight checkpoint is 112 GB,
  and a DeepSpeed state is about 670 GB.

### Initial experts

Training starts from the videoeditor-14b initial expert file `experts_v3_1.safetensors`. It holds the `style` ...
`direct_finetune` source experts for the 14 MoE blocks; `main` is read from the base DiT. The file is built by
[`../videoeditor-14b/merge_experts_v3_1.py`](../videoeditor-14b/merge_experts_v3_1.py) from the base DiT, the five category
LoRAs and the direct fine-tune checkpoint:

```bash
python scripts/moe/videoeditor-14b/merge_experts_v3_1.py --base_dir <BASE_DIR> --out_dir <BASE_DIR>/ckpt/init
```

The file and the direct fine-tune checkpoint are not released. The launcher expects the file at
`EXPERTS_MERGED=${BASE_DIR}/ckpt/init/experts_v3_1.safetensors`. It checks for the file even when resuming, because
`build_dit` assembles the source experts before a checkpoint overwrites them. `check_build_v3_2.py` verifies this assembly on
the CPU.

### Data

Dataset files are JSON lists (`.json`) or JSON lines (`.jsonl`). Each record has this shape:

```json
{"src_video": "<path>", "tgt_video": "<path>", "prompt": "<instruction>"}
{"src_video": "<path>", "tgt_video": "<path>", "prompt": ["<step 1>", "<step 2>"]}
```

- **Paths**: relative paths are joined with `DATASET_BASE` (`--dataset_base_path`).
- **Multi-step prompts**: a list of sub-instructions is joined into one instruction with " Then " (`--compound_joiner`).
  Records whose `src_video` follows the GOKU layout (`videos/case_<n>/`) use GOKU wording instead ("A, then b, and finally c").
- **Filtering**: samples whose instruction contains `overlay` or `subtitle` are dropped (`EXCLUDE_PROMPT_WORDS`).
- **Categories**: the optional fields `cat`, `cls` or `turn_types_qwen` give the edit category. They only feed the category x
  expert routing statistics in the logs.

The released run used the mixed schedule (`DATA_SCHEDULE=mixed`). The single-instruction data (`STAGE_A_DATA`) appears once and
the multi-instruction data (`STAGE_B_DATA`) is repeated `MULTI_REPEAT=10` times. Both are shuffled into one pool, which is
trained for `STAGE_A_EPOCHS=2` passes:

| launcher variable | data | samples after filtering | copies per pass |
|---|---|---|---|
| `STAGE_A_DATA` | OpenVE-3M without the `subtitles_edit` category | 1,788,788 | 1 |
| `STAGE_A_DATA` | ReCo-Data | 520,510 | 1 |
| `STAGE_B_DATA` | GOKU composite multi-step edits | 22,856 | 10 |
| `STAGE_B_DATA` | CoinVE multi-instruction edits | 200,669 | 10 |

One pass is 4,544,548 samples.

- **Frames**: each sample uses 49 frames at 480x832 (`NUM_FRAMES`, `HEIGHT`, `WIDTH`).
- **Text CFG**: the instruction is replaced by the empty instruction with probability 0.1.
- **Cache**: for multi-instruction samples, the VAE latent distribution and the MLLM context are cached in `MULTI_CACHE_DIR` on
  first use, because each such sample is seen 10 times per pass. With `MULTI_CACHE_KEEP=1` the cache is kept across restarts.
- **Two-stage schedule**: `DATA_SCHEDULE=two_stage` instead trains `STAGE_A_DATA` for `STAGE_A_EPOCHS`, then `STAGE_B_DATA`
  for `STAGE_B_EPOCHS`.

### Launch

1. Edit the configuration block at the top of `train_multinode.sh`: `BASE_DIR`, `DATASET_BASE`, `MASTER_ADDR`, `MASTER_PORT`,
   `SOCKET_IFNAME`, the dataset files and so on.
2. Activate your Python environment.
3. Run the launcher on every node:

```bash
bash scripts/moe/videoeditor-56b/train_multinode.sh <NODE_RANK>     # 0 on the master node, 1 and 2 on the others
```

The released weights are step 42,500 of this configuration (all values are set in the launcher):

| setting | value |
|---|---|
| experts | 35 = 7 sources x 5 copies (`EXPERT_SOURCES`, `EXPERT_COPIES=5`, `EXPERT_COPY_NOISE=0.01`); top-3 for attention and FFN (`MOE_TOP_K=3`) |
| balance loss | `MOE_AUX_LOSS_ALPHA=1e-4` for each gate |
| batch | 5 per GPU (`BATCH_SIZE`), effective 120 |
| learning rates | base parameters 1e-5; experts 1e-4, decayed linearly to 1e-5 over 2,000 steps; gates 1e-3, linearly to 1e-4 over 2,000 steps, then constant |
| weight decay | 0.01 (AdamW) |
| loss | flow-matching MSE (target = noise - latent), mixed 0.9 / 0.1 with the same loss on frame-to-frame differences (`MOTION_SUB_LOSS_RATIO=0.1`) |
| text dropout | 0.1 (`TEXT_DROP_PROB`) |
| checkpoints | every 500 steps (`SAVE_STEPS`, `DS_SAVE_STEPS`) |

Other launcher options:

- `USE_EFA=1` exports AWS EFA / aws-ofi-nccl settings.
- `FA3_SITE=<dir>` prepends a FlashAttention-3 installation to `PYTHONPATH`. The launcher sets `MOE_ATTN_BACKEND=fa3`, and the
  library falls back to FlashAttention-2 when FA3 cannot be imported.
- `NUMA_BIND=1` binds each rank to the CPU socket of its GPU. The core lists in `moe_v3_2_train.py` assume one specific
  two-socket layout.
- `VAE_PREFETCH=1` encodes the next batch with the VAE while the CPU runs Adam, and `MLLM_WORKER_PRE=1` runs the MLLM
  preprocessing in the dataloader workers. Neither changes the numerics.
- The launcher exports `MOE_DS_STEP_PATCH=1`, which applies two bitwise-neutral patches to the DeepSpeed ZeRO-3 optimizer
  instance. One skips the gradient multiply when the clip coefficient is exactly 1. The other copies bf16 parameters back
  through pinned double buffers. Set it to 0 to disable the patches; expert parallelism always applies them.
- wandb logging is enabled only when `WANDB_API_KEY` is set.

Smoke test: run `bash scripts/moe/videoeditor-56b/train_smoke.sh <NODE_RANK>` on each of the 3 nodes. It uses the same model,
data and DeepSpeed setup and builds the 35 experts from the initial expert file. It then runs 30 steps at batch 4 and logs peak
memory (`mem/alloc_max_gb`, `mem/reserved_max_gb`) and step times. It writes no checkpoints and never uses wandb. It sets
`MLLM_ATTN_IMPL=flash_attention_3`, which needs FlashAttention-3; set it to `sdpa` otherwise.

### Resume and fine-tuning

To resume, write the variables below into `<BASE_DIR>/output/videoeditor-56b/stage.env`, which every node sources, then relaunch all
nodes:

- `RESUME_DS_STATE=<OUT>/ckpt/ds_state_step-N` loads the full DeepSpeed state (weights and optimizer). It needs the same number
  of GPUs.
- `RESUME_WEIGHTS=<dir>/moe_v3_2_trainable.safetensors` loads weights only, and the optimizer restarts. Add
  `RESUME_STEP_FROM=<OUT>/ckpt/ds_state_step-N` to continue the step count, the learning-rate schedule and the data position
  from that state's `step.json`. The batch size or GPU count may differ, because the position is converted by the number of
  samples already consumed. With an unchanged GPU count, the number of samples each GPU has consumed must be divisible by the
  new batch size. `RESUME_CKPT_GLOBAL_BATCH` is only used when a `step.json` does not record its global batch.

To fine-tune from the released weights, set `RESUME_WEIGHTS=<BASE_DIR>/ckpt/videoeditor-56b/moe_v3_2_trainable.safetensors`.
Without `RESUME_STEP_FROM`, the schedule starts at step 0. The launcher still requires `EXPERTS_MERGED`. If you do not have that
file, you need to edit the launcher: `build_dit` skips the file when the trainer gets `--experts_merged ""`, so remove the
file check and pass a quoted empty value.

### Expert parallelism (optional)

With `EP=1`, the launcher passes `--ep 1 --ep_local_world 8` to the trainer, and `moe_v3_2_ep.py` takes over the experts from
ZeRO-3:

- **Groups**: the 8 GPUs of a node form an expert-parallel group that holds one replica of all 35 experts. Tokens are exchanged
  with `all_to_all` inside the node, so only expert gradients and weights cross nodes.
- **Placement**: with 35 experts on 8 GPUs, the GPUs are paired and the four pairs hold 9, 9, 9 and 8 experts. In a pair
  with an odd count, both GPUs hold one of the experts and split its tokens. When the number of experts is divisible by the group size, each GPU
  simply holds the experts `e` with `e % group_size == local_rank`.
- **Weights and optimizer**: expert weights are bf16 GPU tensors outside DeepSpeed. Their gradients are averaged over the
  replicas, and their optimizer (fp32 master weights and Adam moments on the CPU, `DeepSpeedCPUAdam`) is sharded over the
  replicas. ZeRO-3 only handles the dense ~3.5B parameters and the gates. Gradient clipping uses one global norm over both parts.
- **Starting point**: expert weights must come from a checkpoint. Set `RESUME_WEIGHTS` to a full
  `moe_v3_2_trainable.safetensors` (optionally with `RESUME_STEP_FROM`), or set `RESUME_DS_STATE` to the state of an EP run.
- **Launcher settings**: EP was run with `ZERO_HPZ=1`, and `Z3_LEAF` is ignored in EP mode. `--ep_local_world 8` is fixed in
  the launcher.

EP checkpoints have a different layout:

- `ckpt/step-N/` contains `moe_v3_2_dense.safetensors` (dense part and gates), `experts_ep_lr0.safetensors` ...
  `experts_ep_lr7.safetensors` (the experts, written by the 8 GPUs of node 0) and `meta.json`.
- `ckpt/ds_state_step-N/` also holds one `ep_opt_rank<r>.pt` per GPU.

To get the full file for inference, merge the weights on the CPU. This needs about 120 GB of RAM:

```bash
python scripts/moe/videoeditor-56b/merge_ep_ckpt.py <OUT>/ckpt/step-N
# writes <OUT>/ckpt/step-N/moe_v3_2_trainable.safetensors; --out <file> writes elsewhere,
# --check_against <full .safetensors> compares the key set with an existing full checkpoint
```

You can then pass the step directory as `--ckpt` to `moe_v3_2_infer.py`.

### Outputs and logs

Everything is written under `<BASE_DIR>/output/videoeditor-56b/`:

- `ckpt/step-N/moe_v3_2_trainable.safetensors` and `meta.json`, every `SAVE_STEPS`. Step directories that are multiples of
  5,000 are kept; of the others, only the latest two are kept.
- `ckpt/ds_state_step-N/`, every `DS_SAVE_STEPS`, with `step.json` (step, per-GPU batch, world size). Only the latest state is
  kept.
- `logs/node<rank>_<timestamp>.log` (plus `node<rank>_latest.log`), TensorBoard event files, and wandb logs when enabled.

These scalars are logged every 10 steps, averaged over ranks and steps:

- Losses: `loss/main`, `loss/motion`, `loss/aux_attn`, `loss/aux_ffn`, `loss/aux`, `loss/total`.
- Optimisation and timing: `lr/base`, `lr/new_layers`, `lr/gate`, `grad/norm`, `skipped_steps`, `time/*`, `mem/*`.
- Expert usage per MoE block: `moe_attn_L<block>/expert<i>` and `moe_ffn_L<block>/expert<i>` (for example
  `moe_ffn_L01/expert00`) give the fraction of tokens that selected expert `i`; each block sums to 3.
- Routing by category: `route_attn/<category>/expert<i>` and `route_ffn/<category>/expert<i>` give expert `i`'s share of the
  top-3 selections made by samples of that category. `route/attn_spread` and `route/ffn_spread` give the mean over experts of
  the largest difference between categories.

## Tests

All tests run on the CPU.

```bash
# MoE structure on a small random model (35 experts, top-3): experts equal to the dense weights reproduce the dense model,
# the log-prior injection matches an explicit attention bias, both gates receive gradients, selections sum to 3,
# gradient checkpointing on/off agree. Prints "RESULT: PASS".
VIDEOX_ATTENTION_TYPE=SDPA CUDA_VISIBLE_DEVICES= python scripts/moe/videoeditor-56b/probe_moe_v3_2_cpu.py

# Expert parallelism against the reference MoE modules: 4 gloo processes (2 groups x 2), 5 experts, top-3; outputs,
# input and expert gradients, one optimizer step vs. torch AdamW, replica consistency. Prints "EP_CPU_PARITY PASS".
# Other layouts: EP_TEST_N / EP_TEST_LOCAL / EP_TEST_WORLD (e.g. 6 / 3 / 6 for the unpaired layout); port EP_TEST_PORT (default 29911).
CUDA_VISIBLE_DEVICES= python scripts/moe/videoeditor-56b/test_ep_cpu.py

# MLLM preprocessing in dataloader workers gives exactly the same model inputs as preprocessing in the main process
# (uses only the processor in qwen/). Prints "MLLM_PRE_PARITY PASS".
python scripts/moe/videoeditor-56b/test_mllm_pre_cpu.py --base_dir <BASE_DIR>

# Builds the real 55.9B model from the base DiT and the initial expert file and checks the parameter groups
# (needs experts_v3_1.safetensors and about 130 GB of RAM). Prints "CHECK_BUILD_DONE".
CUDA_VISIBLE_DEVICES= python scripts/moe/videoeditor-56b/check_build_v3_2.py --base_dir <BASE_DIR> \
    --experts_merged <BASE_DIR>/ckpt/init/experts_v3_1.safetensors
```

`probe_moe_v3_2_cpu.py` and `test_ep_cpu.py` take no command-line arguments.
