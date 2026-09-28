# mlp-moe

mlp-moe replaces the feed-forward network (FFN) of 14 of the 30 blocks of the Kiwi-Edit 5B video DiT with a 12-expert,
top-3 mixture of experts, and trains the whole DiT end to end for instruction-based video editing. The experts are not
trained from scratch. Each one starts from an existing editing expert: five per-category LoRA experts merged into the
Kiwi-Edit FFN, plus one directly fine-tuned FFN. Each of these six source experts appears twice, and the second copy is
slightly perturbed. The MLLM instruction encoder and the VAE come from Kiwi-Edit and stay frozen.

| | |
|---|---|
| DiT parameters | 18.57B in total, about 7.5B active per token |
| MoE layers | FFN of blocks 1, 3, ..., 27 (0-based; 14 of 30 blocks) |
| Experts per MoE layer | 12 full-width FFNs, top-3 routing per token, no shared expert |
| Trainable | the whole DiT (MLLM encoder and VAE frozen) |
| Input / output | source video (first 49 frames, 832x480) + one text instruction -> 49 edited frames |
| Released checkpoint | step 10,480, [Massyzs/mlp-moe](https://huggingface.co/Massyzs/mlp-moe) |

In the code, file names, log tags and the checkpoint file use the prefix `moe_v3`.

## Architecture

**Base model.** Kiwi-Edit (instruct-only) uses the Wan2.2-TI2V-5B DiT (30 blocks, hidden size 3072, FFN width 14336,
GELU-tanh). The DiT is conditioned through cross-attention on context tokens from a Qwen2.5-VL-based MLLM, which reads the
instruction together with frames of the source video. The source-video latent is also patch-embedded (`vae_condition`)
and passed to the DiT. mlp-moe keeps all of this unchanged except the FFN of the 14 MoE blocks. The other 16 blocks, all
self- and cross-attention layers, the embeddings and the head exist in a single copy.

**MoE FFN** (`MoEFFN` in `wan_transformer3d_kiwi_moe.py`) is applied to each token `u` independently:

```
s   = softmax(W_g u)                          # W_g: 12 x 3072, no bias, evaluated in fp32
T   = indices of the 3 largest entries of s
g_i = s_i / sum_{j in T} s_j                  # renormalised over the top 3, so sum_{i in T} g_i = 1
y   = sum_{i in T} g_i * E_i(u)               # E_i: Linear(3072->14336) -> GELU(tanh) -> Linear(14336->3072)
```

There is no shared expert, so the routing has the same form as Mixtral. Every expert has the shape and size of the
original FFN (88.1M parameters). An expert only computes on the tokens that selected it. The layer still calls every
expert on every step, with an empty input if no token chose it, because DeepSpeed ZeRO-3 needs every parameter to take
part in each forward pass.

**Load-balancing loss.** Each MoE layer uses a DeepSeek-style, expert-level auxiliary loss, computed per sample:

```
f_i = count_i * N / (k * L)             # count_i: tokens of the sample that selected expert i; N = 12, k = 3, L = tokens per sample
P_i = mean of s_i over the sample's tokens
aux = alpha * mean_over_samples( sum_i f_i * P_i )        # equals alpha under perfectly uniform routing
```

A custom autograd function (`AddAuxiliaryLoss`) attaches the term to the layer output. The training loop back-propagates
only the main loss, and the auxiliary gradient still reaches every gate, including under gradient checkpointing. The 14
per-layer terms add up. The launcher uses `alpha = 1e-4`. The schedule used for the released checkpoint is given under
[Training](#training).

**Expert initialisation.** `build_dit` in `moe_v3_model.py` initialises every MoE block in the same way:

| experts | initialised from |
|---|---|
| 0-4 | `style`, `add`, `remove`, `replace`, `background`: the Kiwi-Edit FFN with the FFN part of that category's LoRA expert merged in, `W + B @ A` (rank 128, alpha 128, scale 1); biases from Kiwi-Edit |
| 5 | `direct_finetune`: FFN weights and biases of a direct full-parameter fine-tune of Kiwi-Edit on multi-step editing data (not released) |
| 6-11 | copies of experts 0-5. Gaussian noise with a standard deviation of 1% of each weight matrix's own standard deviation is added to the weights (`--expert_copy_noise 0.01`, fixed seed, identical on all ranks). Biases are copied unchanged. |
| gate | random, normal with std 0.02 |

Everything outside the MoE FFNs is loaded from the Kiwi-Edit DiT, except the source-video patch embedding, which comes from
`moe_expert_init.safetensors`.

**Parameters.** All DiT parameters are trained. That is 3.77B pre-existing parameters (attention, cross-attention, the 16
dense FFNs, embeddings, head), 14.80B expert parameters and 516,096 gate parameters, for 18.57B in total. About 7.47B are
active per token: 3.77B plus 14 layers x 3 experts x 88.1M. The optimizer uses three learning-rate groups:

| group | learning rate |
|---|---|
| pre-existing parameters | 1e-5, constant |
| experts | 1e-4, decays linearly to 1e-5 over the first 2,000 steps, then constant |
| gates | 1e-3, decays linearly to 1e-4 over the first 2,000 steps, then constant |

**Objective.** The objective is flow matching as in VideoX-Fun Wan2.2: `x_t = (1 - sigma) x_0 + sigma * noise` with target
`noise - x_0`. Timesteps are drawn uniformly from a 1000-step schedule with shift 5. The loss is `0.9 * MSE + 0.1 * MSE`
of the frame-to-frame differences of prediction and target (the motion term). Element-wise residuals larger than 50 are
excluded. The balance loss is added on top. With probability 0.1 the instruction is replaced by the empty string, which is
what makes text classifier-free guidance possible. The MLLM still sees the source video in that case. No edit masks are
used.

## Files

| file | contents |
|---|---|
| `wan_transformer3d_kiwi_moe.py` | The DiT. A complete copy of `videox_fun/models/wan_transformer3d_kiwi.py`; every change is marked `MOE-V3`: `AddAuxiliaryLoss`, `Fp32Gate`, `MoEFFN`, the per-block `moe` argument, the model arguments `moe_layers` / `moe_num_experts` / `moe_top_k` / `moe_aux_loss_alpha`, and a dtype cast of the time-embedding input that ZeRO-3 requires |
| `moe_v3_model.py` | Model assembly: `build_dit` (builds the DiT and initialises the experts; skips expert initialisation when `experts_merged` is empty), `mark_trainable` (the three parameter groups), `TextEncoder` (MLLM -> context tokens), `build_mllm`, `build_vae`, `load_trainable_weights` |
| `moe_v3_train.py` | Training loop: two-stage data, flow-matching + motion loss, three learning-rate groups, DeepSpeed checkpointing and resume, routing statistics |
| `moe_v3_infer.py` | Inference: a single video or a list of cases, segmented text CFG, several sampling variants per model load, optional expert maps |
| `merge_experts.py` | Builds the expert initialisation file for training (`experts_merged_6.safetensors`) |
| `train_multinode.sh` | Multi-node training launcher (reference setup: 3 nodes x 8 GPUs) |
| `train_smoke.sh` | Single-node, 8-GPU, 6-step smoke test of the training pipeline |
| `probe_moe_v3_cpu.py` | CPU test of the MoE architecture on a tiny random model |
| `check_infer_v3_cpu.py` | CPU test that loads a real checkpoint and runs one forward pass |
| `README.md` | this file |

The four models share modules in `videox_fun/moe_common/`, among them the MLLM encoder wrapper (`mllm_encoder_moe.py`), the
training dataset (`kiwi_dataset_multi_moe.py`) and the output completeness check (`bench_output_check.py`).

## Weights

Both weight repositories are gated, so request access on the model page first.

```bash
huggingface-cli login
huggingface-cli download Massyzs/kiwi-edit-5b-instruct-only-videoxfun \
    --local-dir <BASE_DIR>/ckpt/kiwi-edit-instruct-only-videoxfun
huggingface-cli download Massyzs/mlp-moe --local-dir <BASE_DIR>/hf/mlp-moe
cd <BASE_DIR>/hf/mlp-moe
cat mlp-moe.tar.gz.part-* | tar -xzf - -C <BASE_DIR>/ckpt/                               # -> <BASE_DIR>/ckpt/mlp-moe/
(cd <BASE_DIR>/ckpt && sha256sum -c <BASE_DIR>/hf/mlp-moe/SHA256SUMS.contents)          # optional
```

The model package is a tar.gz stream split into 20 GB chunks: about 30 GB to download, 37.1 GB once extracted. It
contains no optimizer state.

```
<BASE_DIR>/ckpt/kiwi-edit-instruct-only-videoxfun/
    config.json, diffusion_pytorch_model.safetensors    Kiwi-Edit DiT (used to construct the model)
    moe_expert_init.safetensors                         MLLM connector, query tokens, source-video patch embedding
    Wan2.2_VAE.pth                                      VAE
    qwen/                                               Kiwi-Edit MLLM encoder
<BASE_DIR>/ckpt/mlp-moe/
    moe_v3_trainable.safetensors                        the complete MoE DiT: 1,457 tensors, 18.57B parameters, bf16
    meta.json                                           step (10480) and structure: moe_layers, moe_num_experts (12), moe_top_k (3), ...
```

## Inference

`moe_v3_infer.py` builds the DiT structure from `meta.json` and loads every DiT parameter from
`moe_v3_trainable.safetensors`. It asserts that the checkpoint covers all model parameters. From the base package it
uses the MLLM encoder, `moe_expert_init.safetensors` (connector and query tokens) and the VAE. `--experts_merged` is
**not** needed for inference. If you pass a `merge_experts.py` output, the script only reproduces the training
initialisation, which the checkpoint then overwrites. That is useful for debugging only.

Relative paths are resolved against the current directory, after which the script changes into the repository root, so
it can be started from anywhere. Each process uses the first visible GPU.

```bash
# one video, one instruction
CUDA_VISIBLE_DEVICES=0 python scripts/moe/mlp-moe/moe_v3_infer.py \
    --base_dir <BASE_DIR> --ckpt <BASE_DIR>/ckpt/mlp-moe \
    --src_video input.mp4 --prompt "Remove the dog" --name dog --out_dir <OUT_DIR>

# compound instruction: sub-instructions separated by "|||" are joined into one sentence with " Then "
CUDA_VISIBLE_DEVICES=0 python scripts/moe/mlp-moe/moe_v3_infer.py \
    --base_dir <BASE_DIR> --ckpt <BASE_DIR>/ckpt/mlp-moe \
    --src_video input.mp4 --prompts "Remove the dog ||| Convert to oil painting style" --name dog_oil --out_dir <OUT_DIR>

# a list of cases, sharded over 4 GPUs (process i handles cases[i::4])
# <BENCH_DIR>/cases.json = [{"name": "...", "src": "<video path relative to BENCH_DIR>", "prompts": ["...", ...]}, ...]
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/moe/mlp-moe/moe_v3_infer.py \
      --base_dir <BASE_DIR> --ckpt <BASE_DIR>/ckpt/mlp-moe \
      --bench_dir <BENCH_DIR> --out_dir <OUT_DIR> --shard_idx $i --num_shards 4 &
done; wait

# several sampling settings with a single model load; results go to <OUT_DIR>/<label>/
CUDA_VISIBLE_DEVICES=0 python scripts/moe/mlp-moe/moe_v3_infer.py \
    --base_dir <BASE_DIR> --ckpt <BASE_DIR>/ckpt/mlp-moe --bench_dir <BENCH_DIR> --out_dir <OUT_DIR> \
    --variants 'seg_shift5:3:0.6:5,seg_shift3:3:0.6:3,full_shift5:3:1.0:5'
```

- **Single-video mode** starts whenever `--src_video` is given. `--prompt` is used verbatim, so a compound instruction can
  also be written as one sentence. `--prompts` splits on `|||` and joins the parts with `" Then "`, the same joiner
  training used for CoinVE. The output name is `--name` (default `single`).
- **Case-list mode** reads `<BENCH_DIR>/cases.json`, or the file given by `--cases_json`. Relative `src` paths are
  resolved against `--bench_dir`, or against the JSON file's directory if `--bench_dir` is not given. Each case's
  `prompts` are joined with `" Then "`.
- **`--variants`** takes a comma-separated list of `label:cfg_text_scale:cfg_text_steps_frac:sigma_shift`. For each case,
  the MLLM encoding, the source VAE encoding and the initial noise are computed once and shared by all variants, so the
  variants differ only in the listed settings. If `--variants` is empty, the script runs the single setting given by
  `--cfg_text_scale` / `--cfg_text_steps_frac` / `--sigma_shift` and writes to `<OUT_DIR>`.

Defaults:

| flag | default | meaning |
|---|---|---|
| `--steps` | 50 | flow-matching Euler steps |
| `--sigma_shift` | 5.0 | timestep shift of the sampling schedule |
| `--cfg_text_scale` | 3.0 | text CFG `pred = u + s * (c - u)`. The unconditional branch `u` uses the empty instruction but the same source video. `1` disables CFG |
| `--cfg_text_steps_frac` | 0.6 | apply CFG only on the first 60% of the steps (30 of 50) and use the conditional prediction afterwards; `1.0` = all steps |
| `--num_frames`, `--height`, `--width` | 49, 480, 832 | output size; must match training |
| `--seed` | 42 | initial noise (CPU generator) |
| `--fps` | 16 | frame rate of the output mp4 |
| `--frames_png` | 1 | also write every frame as a lossless PNG |
| `--dump_expert_maps` | 1 | write an expert-routing map per case |
| `--skip_existing` | 1 | skip a case whose mp4 (size, frame count, fps) and PNG frames are already complete |
| `--min_free_gib` | 60 | exit at start-up if the GPU has less free memory than this (GiB) |
| `--fast_init` | 1 | skip random initialisation while the model is constructed (the checkpoint overwrites every parameter) |
| `--experts_merged` | empty | not needed for inference (see above) |
| `--config_path` | `config/wan2.2/wan_civitai_5b.yaml` | relative to the repository root |
| `--shard_idx`, `--num_shards` | 0, 1 | process `cases[shard_idx::num_shards]` |

**Source preprocessing** matches training. The script takes the first `n = min(49, number of frames)` frames and reduces
`n` to the nearest `4k+1`. It pads with the last frame up to 49, resizes so that the short side covers 832x480 and
center-crops. Frames are not resampled in time, so the edit covers the first 49 source frames at the source frame rate.

**Outputs**, in `<OUT_DIR>` or `<OUT_DIR>/<label>/`:
- `<name>.mp4`
- `frames/<name>/f000.png ... f048.png`
- `<name>_experts.png`
- `manifest_shard<k>.jsonl`: one line per output with the settings, seed, timings and per-layer expert counts

The expert map comes from the middle sampling step. It shows MoE blocks 1, 15 and 27 as rows and the first, middle and
last latent frame as columns. Each patch is coloured by its top-1 expert, the argmax of the gate scores on the
conditional branch, with one colour per expert index. If `--out_dir` is omitted, the script writes to
`<run>/render/<ckpt directory name>/`. Here `<run>` is the parent of the `ckpt/` directory that contains `--ckpt`, which
gives `<BASE_DIR>/render/mlp-moe/` for the layout above.

**Memory.** Each process runs the whole model on one GPU. The DiT weights alone take 37 GB in bf16. Together with the MLLM
encoder (about 7 GB), the VAE and the activations, use a GPU with at least 60 GB free; the model was developed on
80 GB H100s. The script creates its CUDA context first and exits if the free memory is below `--min_free_gib`. The DiT is
constructed on the CPU in fp32 and then cast to bf16, so each process also needs roughly 120 GB of host RAM while it
loads.

**Exit codes** are meant for an external retry loop. Rerunning with `--skip_existing 1` continues where a run stopped.

| code | meaning |
|---|---|
| 17 | non-finite DiT weights after loading |
| 18 | the MLLM self-test produced non-finite values three times |
| 19 | a generated video was non-finite or constant; nothing is written for it |

## Training

### Setup used for the released checkpoint

- **Hardware:** 3 nodes x 8 NVIDIA H100 80 GB (24 ranks), DeepSpeed ZeRO-3 without CPU offload, bf16 mixed precision,
  gradient checkpointing, gradient clipping 1.0.
- **Batch and optimizer:** batch 8 per GPU, effective batch 192, no gradient accumulation. AdamW with weight decay 0.01
  and the three learning-rate groups above.
- **Other settings:** text dropout 0.1, motion-loss ratio 0.1, 49 frames at 832x480.
- **Steps:** stage A takes 1 epoch = 9,316 steps and stage B 1 epoch = 1,164 steps, for 10,480 steps in total. One step
  took about 26 s, roughly 3 days for the whole run.
- **Balance coefficient:** `1e-3` for steps 0-6,999. Training then resumed from the full DeepSpeed state at step 7,000
  and used `1e-4` until step 10,480, with every other setting unchanged. The launcher uses `1e-4` from the start.

### Data

The two stages run one after the other inside the same process:

| stage | launcher variable (default file) | content | pairs after filtering | steps |
|---|---|---|---|---|
| A | `STAGE_A_DATA` (`${BASE_DIR}/datasets/filtered/openve_original_nosub_all.json`) | OpenVE-3M, original single-instruction metadata, without the `subtitles_edit` category | 1,788,788 | 9,316 |
| B | `STAGE_B_DATA` (`${BASE_DIR}/datasets/classified/goku_with_masks.jsonl,${BASE_DIR}/datasets/classified/CoinVE.jsonl`) | multi-instruction edits: GOKU (22,856) and CoinVE (200,669) | 223,525 | 1,164 |

- **Filtering.** Samples whose full instruction contains `overlay` or `subtitle` (case-insensitive, `EXCLUDE_PROMPT_WORDS`)
  are dropped: 16 in stage A and 330 in stage B.
- **Samples.** Every sample is one (source video, target video, instruction) triple. The fields read are `src_video`,
  `tgt_video` and `prompt` (a string or a list of sub-instructions). `cat`, `cls` and `turn_types_qwen` are optional and
  only feed the routing statistics. Masks in the files are ignored.
- **Joining sub-instructions.** Multi-instruction records become one sentence. GOKU records, recognised by `videos/case_<n>/`
  in the source path, are joined in GOKU's native wording, "A, then b, and finally c". All others (CoinVE) are joined with
  `" Then "` (`--compound_joiner`).
- **Video paths.** Relative paths are prefixed with `DATASET_BASE` (`--dataset_base_path`).

How to obtain and prepare these datasets is described in [docs/DATA.md](../../../docs/DATA.md).

### Initial weights (`--experts_merged`)

`moe_v3_train.py` requires `--experts_merged`, which is the output of `merge_experts.py`. For every MoE block the file
holds the FFN weights of the six source experts. It is built from three inputs:

1. **The Kiwi-Edit DiT.** Default: `--dit_weights ckpt/kiwi-edit-instruct-only-videoxfun/diffusion_pytorch_model.safetensors`
   under `--base_dir`.
2. **The five per-category LoRA experts** (`style`, `add`, `remove`, `replace`, `background`). They are released inside the
   [explicit-moe](https://huggingface.co/Massyzs/explicit-moe) package under `lora_experts/<name>/dit_query_router.safetensors`.
   Only their `ffn.0` / `ffn.2` LoRA matrices in the MoE blocks are used; the attention LoRAs are ignored.
3. **A direct full-parameter fine-tune of Kiwi-Edit** (`direct_finetune`, the sixth source expert). **It is not released.**
   The exact initialisation of the released model therefore cannot be rebuilt from public files.

`merge_experts.py` looks for the LoRA files at `<experts_root>/moe_expert_<name>/final_components/dit_query_router.safetensors`.
With the explicit-moe package extracted to `<BASE_DIR>/ckpt/explicit-moe/`, link the files into that layout:

```bash
for n in style add remove replace background; do
  mkdir -p <EXPERTS_ROOT>/moe_expert_$n/final_components
  ln -s <BASE_DIR>/ckpt/explicit-moe/lora_experts/$n/dit_query_router.safetensors \
        <EXPERTS_ROOT>/moe_expert_$n/final_components/dit_query_router.safetensors
done
python scripts/moe/mlp-moe/merge_experts.py --base_dir <BASE_DIR> --experts_root <EXPERTS_ROOT> \
    --direct_finetune <DIRECT_FINETUNE_DIR> --out_dir <BASE_DIR>/ckpt/init
```

The script runs on the CPU and writes two files to `--out_dir`:
- `experts_merged_6.safetensors` (bf16, about 15 GB), which is where the launchers expect it (`EXPERTS_MERGED`)
- `experts_merged_6_report.json`, with the relative update size `|dW| / |W|` of each expert in each layer

Since the direct fine-tune is not released, there are two ways to build the file without it:

- **Use your own sixth expert.** `--direct_finetune` accepts any sharded diffusers checkpoint directory of the same DiT
  (with `diffusion_pytorch_model.safetensors.index.json`), for example your own full fine-tune of Kiwi-Edit.
- **Use only the five LoRA experts.** `--direct_finetune ""` writes a file with only the five LoRA experts. `build_dit`
  expects six source experts (`EXPERT_NAMES` in `moe_v3_model.py`, and it asserts `moe_num_experts == 2 x 6`). This
  variant therefore also requires removing `direct_finetune` from `EXPERT_NAMES` and setting `MOE_NUM_EXPERTS=10`. The
  released model was not trained this way.

### Launch

Edit the configuration block at the top of the launcher and activate your Python environment. Then run the multi-node
launcher on every node; the smoke test runs on a single node:

```bash
bash scripts/moe/mlp-moe/train_multinode.sh <node_rank>     # 0 .. NUM_NODES-1
bash scripts/moe/mlp-moe/train_smoke.sh                      # single node, 8 GPUs, 6 steps, batch 4, no wandb
```

| variable | value used | meaning |
|---|---|---|
| `BASE_DIR` | | working directory with `ckpt/kiwi-edit-instruct-only-videoxfun/`, `datasets/`, `output/` |
| `DATASET_BASE` | | prefix for the relative video paths in the data files |
| `MASTER_ADDR`, `MASTER_PORT`, `NUM_NODES`, `GPUS_PER_NODE` | node 0, 29839, 3, 8 | rendezvous and world size |
| `USE_EFA` | 0 | `1` exports the AWS EFA / NCCL environment of the reference cluster |
| `EXPERIMENT` | `mlp-moe` | outputs go to `${BASE_DIR}/output/${EXPERIMENT}` |
| `WANDB_PROJECT` | | used only when `WANDB_API_KEY` is set in the environment |
| `EXPERTS_MERGED` | `${BASE_DIR}/ckpt/init/experts_merged_6.safetensors` | output of `merge_experts.py` |
| `STAGE_A_DATA`, `STAGE_B_DATA` | see [Data](#data) | comma-separated json / jsonl files |
| `STAGE_A_EPOCHS`, `STAGE_B_EPOCHS` | 1, 1 | epochs per stage |
| `EXCLUDE_PROMPT_WORDS` | `overlay,subtitle` | drop samples whose instruction contains these words |
| `MOE_LAYERS` | `1,3,...,27` | 0-based MoE block indices |
| `MOE_NUM_EXPERTS`, `MOE_TOP_K` | 12, 3 | |
| `MOE_AUX_LOSS_ALPHA` | `0.0001` | balance-loss coefficient |
| `EXPERT_COPY_NOISE` | 0.01 | noise of the second expert copy, relative to the weight std |
| `BATCH_SIZE` | 8 | per GPU |
| `LEARNING_RATE` | 1e-5 | pre-existing parameters |
| `NEW_LAYERS_LR`, `NEW_LAYERS_LR_STEPS` | 1e-4, 2000 | experts: start value, then linear decay to `LEARNING_RATE` |
| `GATE_LR`, `GATE_LR_END` | 1e-3, 1e-4 | gates: linear over `NEW_LAYERS_LR_STEPS`, then constant |
| `WEIGHT_DECAY` | 0.01 | |
| `TEXT_DROP_PROB`, `MOTION_SUB_LOSS_RATIO` | 0.1, 0.1 | |
| `VAE_ENCODE_GROUP` | 4 | videos per batched VAE encoder call |
| `SAVE_STEPS`, `DS_SAVE_STEPS` | 250, 1000 | weight checkpoints / full DeepSpeed state |
| `NUM_FRAMES`, `HEIGHT`, `WIDTH`, `MAX_PIXELS` | 49, 480, 832, 399360 | |
| `MAX_TRAIN_STEPS` | 0 | 0 = derived from the epochs of both stages |
| `RESUME_DS_STATE` | empty | see [Resume](#resume) |

**Launcher details.**
- The launcher writes the ZeRO-3 configuration to `${OUT}/ds_zero3.json` and starts
  `accelerate launch --use_deepspeed ... moe_v3_train.py`. `${OUT}` must be on a filesystem that all nodes share.
- Some defaults of `moe_v3_train.py` differ from the launcher values, for example `--moe_num_experts 10`,
  `--moe_aux_loss_alpha 0.001`, `--batch_size 4` and `--stage_b_epochs 2`. If you call the script directly, pass the
  values above explicitly.

**Start-up and runtime checks.**
- Each rank waits up to 10 minutes for 60 GiB of free memory on its GPU and stops with an error if it does not get it.
- Building the model temporarily needs about 120 GB of host RAM per rank. Ranks on the same node start 15 s apart.
- At start-up the script checks that all weights are finite (rebuilding up to 3 times), that a ZeRO-3 gather checksum is
  identical on all ranks, and that an MLLM self-test gives finite output.
- During training, samples with non-finite latents or MLLM context are dropped. A step with a non-finite loss on any
  rank is skipped on all ranks, and a rank with 5 consecutive non-finite steps exits.

**Outputs**, in `${BASE_DIR}/output/${EXPERIMENT}/`:
- `ckpt/step-N/moe_v3_trainable.safetensors` + `meta.json`, every `SAVE_STEPS` steps. This is the full DiT in bf16, the
  same format as the release and directly usable with `--ckpt`. Steps divisible by 1000 are kept; of the others only the
  two most recent.
- `ckpt/ds_state_step-N/`, every `DS_SAVE_STEPS` steps: the full DeepSpeed state including the optimizer. Only the latest
  is kept.
- `ckpt/final_components/`: a copy of the last `step-N`.
- `logs/node<rank>_<timestamp>.log` (plus a `node<rank>_latest.log` link), TensorBoard event files and, if enabled, wandb.

**Logged scalars**, every 10 steps, averaged over ranks and steps:
- losses and optimizer state: `loss/main`, `loss/motion`, `loss/aux`, `loss/total`, `lr/base`, `lr/new_layers`,
  `lr/gate`, `grad/norm`, `skipped_steps`, `stage`
- step timing: `time/{vae,mllm,dit}` (seconds per step on rank 0)
- `moe_L<block>/expert<i>`: the fraction of tokens in that layer that selected expert `i` (sums to 3 per layer), plus
  `moe/selected_min` and `moe/selected_max`
- `route_ffn/<category>/expert<i>`: P(expert `i` selected | sample category), counted over all MoE layers
- `route/ffn_spread`: the per-expert spread (maximum minus minimum over categories), averaged over experts. 0 means no
  category specialisation.

### Resume

Write `RESUME_DS_STATE=<OUT>/ckpt/ds_state_step-N` into `${OUT}/stage.env`, which the launcher sources on every node, and
restart all nodes. The resume needs the same number of ranks as the run that wrote the state. The step counter,
learning-rate schedule, stage and data position all continue from where they were. The script checks the first batch
after a resume against the sampler.

### Fine-tuning from the released checkpoint

This needs no merged-expert file. `--resume_weights` loads weights only: the optimizer starts fresh and the step counter
starts at 0. The launchers do not pass this flag, so edit the `accelerate launch` command in `train_multinode.sh`:

1. Set `EXPERTS_MERGED=""` and quote it in the command, `--experts_merged "${EXPERTS_MERGED}"`. Unquoted, the empty
   value disappears and argparse fails.
2. Add `--resume_weights ${BASE_DIR}/ckpt/mlp-moe/moe_v3_trainable.safetensors`.
3. Keep `MOE_LAYERS`, `MOE_NUM_EXPERTS` and `MOE_TOP_K` as in `meta.json`, and set your data, epochs or `MAX_TRAIN_STEPS`.

The step counter restarts at 0, so the expert and gate learning rates again start at `NEW_LAYERS_LR` and `GATE_LR` and
decay over `NEW_LAYERS_LR_STEPS`. For constant rates, set `NEW_LAYERS_LR=LEARNING_RATE` and `GATE_LR=GATE_LR_END`.

## Tests

Both tests run on the CPU.

```bash
# architecture test on a tiny random model (seconds, no weights needed)
VIDEOX_ATTENTION_TYPE=SDPA CUDA_VISIBLE_DEVICES= python scripts/moe/mlp-moe/probe_moe_v3_cpu.py

# loading test with real weights
CUDA_VISIBLE_DEVICES= python scripts/moe/mlp-moe/check_infer_v3_cpu.py --base_dir <BASE_DIR> --ckpt <BASE_DIR>/ckpt/mlp-moe
```

**`probe_moe_v3_cpu.py`** builds a dense Kiwi DiT and the MoE DiT with a tiny configuration: 8 blocks, width 64, MoE in
blocks 1/3/5, 4 experts, top-3. It checks:
- **(A)** with every expert a copy of the dense FFN, the MoE model reproduces the dense model's output;
- **(B)** a 1% perturbation of the experts changes the output only slightly, and gradients reach the gate and the
  experts, including those of the last MoE layer;
- **(C)** gradient checkpointing on and off gives identical outputs. The total expert usage equals tokens x top-k, the
  per-layer selection fractions sum to top-k, and the balance-loss gradient reaches the gate (the gate gradient differs
  between alpha = 0 and alpha = 0.01);
- **(D)** with all experts identical, the layer output equals that expert's output, which confirms that the renormalised
  top-k weights sum to 1.

It prints `RESULT: PASS` or `RESULT: FAIL`.

**`check_infer_v3_cpu.py`** follows the inference loading path (`--experts_merged` empty by default):
- It builds the DiT from the checkpoint's `meta.json`, loads `moe_v3_trainable.safetensors` and reports key coverage.
  The fraction of model parameters covered should be `1.0000`.
- It runs one forward pass on a tiny random latent (4 tokens) with random context and checks that the output is finite
  and that the per-layer selection fractions sum to 3.
- It prints `CHECK_INFER_DONE`. It needs the base package, the checkpoint and roughly 120 GB of host RAM.

## Notes and limitations

- **Fixed input format.** The model is trained and run at 49 frames and 832x480. Other aspect ratios are resized and
  center-cropped, and only the first 49 frames of a source video are edited. The output is written at 16 fps by default,
  regardless of the source frame rate.
- **Instructions and masks.** Training instructions are English. A compound edit is given as a single sentence (or via
  `--prompts`), and there is no mask input.
- **Routing.** Nothing supervises routing by edit category. The expert names (`style`, `add`, ...) describe only how each
  expert was initialised, not what it does after training. The `route_ffn/*` statistics are for monitoring only.
- **Implementation.** The MoE layer loops over experts in Python. There is no grouped GEMM and no expert parallelism, and
  the full model is replicated on each inference GPU.
- **Reproducing the initialisation.** It cannot be reproduced exactly, because the `direct_finetune` source expert is not
  released (see [Initial weights](#initial-weights---experts_merged)).
- **Data exposure of the released checkpoint.** At the step-7,000 resume, the batch sampler in use at the time lost its
  resume offset when the DataLoader used worker processes. Steps 7,000-9,315 therefore replayed the first 2,316 batches of
  the identically shuffled stage-A epoch, instead of batches 7,000-9,315. About 445k of the 1.79M stage-A pairs were never
  seen, and about as many were seen twice. Stage B was not affected. The code in this repository fixes this: the offset is
  persistent, and the first batch after a resume is checked against the sampler.
