# explicit-moe

Instruction-based video editing with a frozen Kiwi-Edit 5B DiT, five frozen LoRA task experts, and trainable fusion layers
that let the main denoising stream read the experts' features. File names, the checkpoint `variant` field and log prefixes use
the code name `moe_v2`.

## Overview

A request is given as one or more sub-instructions (for example "Remove the dog" and "Convert to oil painting style"). A router
head assigns each sub-instruction to one of five explicit task experts: `style`, `add`, `remove`, `replace` and `background`.
Each expert is a LoRA fine-tune of the Kiwi-Edit DiT on one edit category. The DiT runs one **base stream** without LoRA on the
full instruction, plus one **expert stream** per sub-instruction with that sub-instruction's expert enabled. Fourteen
fusion-attention layers inject the expert streams into the base stream, and the video is decoded from the base stream only.

The Kiwi-Edit DiT, the MLLM encoder, the VAE and the LoRA experts stay frozen. Only the fusion layers and the router head are
trained.

| Component | Parameters | Status |
|---|---|---|
| Kiwi-Edit DiT (Wan2.2-TI2V-5B architecture, 30 blocks, hidden size 3072) | 5.00B | frozen |
| 5 LoRA experts | 5 x 0.32B | frozen |
| 14 fusion-attention layers | 0.53B | trained |
| Router head | 4.2M | initialised from a pre-trained head, fine-tuned |
| MLLM encoder (Qwen2.5-VL-based, from Kiwi-Edit) and Wan2.2 VAE | | frozen |

The released checkpoint is step 28,200 of the reference training run described under [Training](#training).

## Architecture

All of the following is implemented in `moe_v2_transformer.py`.

- **Conditioning (unchanged from Kiwi-Edit).** The MLLM encoder reads the instruction and the source-video frames. A connector
  maps its output to the DiT's cross-attention context. The source-video latent goes through the DiT's `vae_condition`
  convolution, is scaled by the current noise level sigma, and is added to the patch embeddings.
- **Routing.** Each sub-instruction is encoded by the MLLM together with the source frames. The router head
  (`videox_fun/moe_common/attn_head.py`) takes the last-layer hidden states of the text tokens, with the learnable query tokens
  removed. It has a 2048 -> 512 projection, one self-attention layer (8 heads) with an FFN, mean pooling and a 5-way linear
  classifier. The argmax picks the expert. During training, a sub-instruction that has a category label uses the label
  instead, and the router head is trained with cross-entropy against the labels.
- **Streams.** The K routed sub-instructions are sorted by expert name. This order fixes each expert stream's slot. The base
  stream runs the DiT with every LoRA adapter disabled. Its context is the full instruction: the sub-instructions joined with
  `" Then "`. Expert stream k enables only its expert's LoRA and uses its own sub-instruction as context. Expert streams run
  without gradients and never see the base stream or each other.
- **LoRA experts.** Each expert has rank 128 and alpha 128. The LoRA sits on `q`, `k`, `v` and `o` of self- and cross-attention
  and on both FFN linears (`ffn.0`, `ffn.2`), in all 30 blocks. Each expert has 600 tensors and 0.32B parameters. Only tensors
  whose name contains `.lora_` are read from the expert files.
- **Fusion.** A separate `FusionAttention` layer follows each of blocks 15 to 28 (`fusion_start=15`, `fusion_end=29`; there is
  none after the last block). It uses 24 heads. Q is a linear projection of the LayerNorm-ed base stream. K and V are linear
  projections of the concatenation `[base; expert 1; ...; expert K]`, which goes through a second LayerNorm; the K and V weights
  are shared across streams. 3D RoPE is applied to Q and K.
  The width coordinate of expert j (j = 0..K-1) is shifted by `(j + 1) * W`, where W is the latent width in tokens, so the
  experts sit to the right of the base frame (`rope_expert_offset0=1`). The attention output passes through an output
  projection and is added to the base stream as a residual. Expert streams are not modified. With K = 0 the layer reduces to
  self-attention over the base stream.
- **Output.** The DiT head is applied to the base stream only.
- **Classifier-free guidance.** Text CFG is applied on the base stream: `pred = u + s * (c - u)`. The conditional branch uses
  the full instruction. The unconditional branch uses an empty instruction; the MLLM still sees the source video. The expert
  streams are the same in both branches. During training the base instruction is replaced by an empty string with probability
  0.1, and expert instructions are never dropped.

With 49 frames at 480x832, each stream holds 13 x 15 x 26 = 5,070 tokens, and a fusion layer attends over (K + 1) x 5,070
keys. One denoising step runs K + 1 DiT forward passes, or 2(K + 1) while CFG is active, because the unconditional branch
recomputes the expert streams.

## Files

| File | Purpose |
|---|---|
| `README.md` | This file. |
| `moe_v2_transformer.py` | Model code: width-offset 3D RoPE, `FusionAttention`, per-block LoRA switching, `MoEV2Model` (MLLM encoding, routing, `denoise` over the base and expert streams, training forward), `build_model` (loads the base DiT, VAE, MLLM, LoRA experts and router head) and `load_checkpoint`. |
| `moe_v2_train.py` | Training loop: metadata loading, same-group batching (`GroupedBatchSampler`), flow-matching loss with a motion term plus router cross-entropy, AdamW, checkpointing and resume (accelerate, DDP). |
| `moe_v2_infer.py` | Inference on a single video (one or several sub-instructions) or a case list. Applies text CFG, can shard across GPUs, and writes mp4, PNG frames and a manifest. |
| `train_multinode.sh` | Multi-node training launcher (reference run: 2 nodes x 8 GPUs). Settings are in the config block at the top. |
| `train_smoke.sh` | Single-node 8-GPU smoke test: 10 steps, checkpoints at steps 5 and 10, no wandb. |

Shared modules in `videox_fun/moe_common/`:

- `mllm_encoder_moe.py`: MLLM encoder wrapper.
- `attn_head.py`: router head.
- `kiwi_dataset_multi_moe.py` and `kiwi_dataset_moe.py`: training datasets.
- `bench_output_check.py`: skips cases whose outputs are already complete.

## Weights

All weight repositories are gated: request access on the model page, then run `huggingface-cli login`. explicit-moe needs
both the shared frozen base and its own package. Its LoRA experts and fusion layers are applied on top of the base DiT.

```bash
# shared frozen base: DiT, MLLM encoder, connector/query tokens, VAE
huggingface-cli download Massyzs/kiwi-edit-5b-instruct-only-videoxfun \
    --local-dir $BASE_DIR/ckpt/kiwi-edit-instruct-only-videoxfun

# explicit-moe package: a tar.gz stream in 20 GB chunks (a single chunk, part-00, for this model)
huggingface-cli download Massyzs/explicit-moe --local-dir $BASE_DIR/hf/explicit-moe
cd $BASE_DIR/hf/explicit-moe
cat explicit-moe.tar.gz.part-* | tar -xzf - -C $BASE_DIR/ckpt/                               # -> $BASE_DIR/ckpt/explicit-moe/
(cd $BASE_DIR/ckpt && sha256sum -c $BASE_DIR/hf/explicit-moe/SHA256SUMS.contents)            # optional check
```

Expected layout. The training launchers expect `$BASE_DIR/ckpt/explicit-moe/`. For inference, pass the directory with
`--ckpt`.

```
$BASE_DIR/ckpt/kiwi-edit-instruct-only-videoxfun/
    config.json                              DiT config
    diffusion_pytorch_model.safetensors      Kiwi-Edit DiT
    moe_expert_init.safetensors              MLLM connector + query tokens + source-video patch embedding (vae_condition)
    Wan2.2_VAE.pth                           VAE
    qwen/                                    Kiwi-Edit MLLM encoder
$BASE_DIR/ckpt/explicit-moe/
    moe_v2_state.pt                          188 trainable tensors in fp32 (14 fusion layers + router head), step, fusion range and
                                             RoPE offset; no optimizer state. This is the file the code loads.
    fusion_moe_v2_absorb.safetensors         the same 168 fusion-layer tensors in bf16 (convenience copy, not read by the code)
    router_head_finetuned.pt                 the same 20 router-head tensors as a state_dict (convenience copy, not read by the code)
    router_head_online.pt                    router head before fusion training: initial weights for training; at inference only
                                             its architecture fields are used, its weights are overwritten from moe_v2_state.pt
    meta.json                                checkpoint metadata and file descriptions (not read by the code)
    lora_experts/{style,add,remove,replace,background}/dit_query_router.safetensors    frozen LoRA experts (fp32)
```

## Inference

Commands are run from the repository root. One 80 GB GPU per process.

```bash
# single video, one instruction (routed to one expert)
CUDA_VISIBLE_DEVICES=0 python scripts/moe/explicit-moe/moe_v2_infer.py \
    --base_dir $BASE_DIR --ckpt $BASE_DIR/ckpt/explicit-moe \
    --src_video input.mp4 --prompt "Remove the dog" --name dog --out_dir <OUT_DIR>

# compound instruction: sub-instructions separated by |||, each routed to its own expert;
# the base stream gets "Remove the dog Then Convert to oil painting style"
CUDA_VISIBLE_DEVICES=0 python scripts/moe/explicit-moe/moe_v2_infer.py \
    --base_dir $BASE_DIR --ckpt $BASE_DIR/ckpt/explicit-moe \
    --src_video input.mp4 --prompts "Remove the dog ||| Convert to oil painting style" --name dog_oil --out_dir <OUT_DIR>

# case list: <BENCH_DIR>/cases.json, one process per GPU (here 8 shards)
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python scripts/moe/explicit-moe/moe_v2_infer.py \
      --base_dir $BASE_DIR --ckpt $BASE_DIR/ckpt/explicit-moe \
      --bench_dir <BENCH_DIR> --out_dir <OUT_DIR> --shard_idx $i --num_shards 8 &
done; wait
```

`cases.json` is a list of cases. `src` is either absolute or relative to `--bench_dir`:

```json
[{"name": "dog_oil", "src": "videos/dog.mp4", "prompts": ["Remove the dog", "Convert to oil painting style"]}]
```

Use `--cases_json <file>` to read a case list from any file. Relative `src` paths are then resolved against `--bench_dir`, or
against the file's own directory if `--bench_dir` is not given. Shard i processes `cases[i::num_shards]`.

Key options and defaults:

| Option | Default | Meaning |
|---|---|---|
| `--steps` | 50 | Flow-matching Euler steps. |
| `--sigma_shift` | 5.0 | Scheduler shift. |
| `--cfg_text_scale` | 3.0 | Text CFG scale s in `pred = u + s * (c - u)`. 1 disables CFG and skips the unconditional pass. |
| `--cfg_text_steps_frac` | 1.0 | CFG is applied only in the first fraction of the steps. The default applies it to all steps; 0.6 matches the default of the other three models. |
| `--num_frames` | 49 | Uses the first 49 source frames. If the source is shorter, the count is trimmed to 4k+1 and padded by repeating the last frame. |
| `--height`, `--width` | 480, 832 | Short-side resize, then centre crop. |
| `--seed` | 42 | Seed of the initial noise (CPU generator, so the noise is the same on any GPU). |
| `--fps` | 16 | Frame rate of the output mp4. |
| `--frames_png` | 1 | Also write every frame as a lossless PNG. |
| `--skip_existing` | 1 | Skip a case whose mp4 (size, frame count, fps) and PNG frames are already complete. |
| `--lora_experts_dir`, `--router_head` | `<ckpt>/lora_experts`, `<ckpt>/router_head_online.pt` | Needed when `--ckpt` is a training checkpoint directory (see below). |

Keep `--experts`, `--fusion_heads` (24) and `--config_path` at their defaults. The fusion range and the RoPE offset are read
from `moe_v2_state.pt`.

Outputs in `--out_dir`:

- `<name>.mp4`
- `frames/<name>/f000.png ...`
- `manifest_shard<i>.jsonl`, one line per case. Fields: name, src, prompts, output path, checkpoint, steps, sigma shift, CFG
  setting, seed, frame count, elapsed time, `experts_used` (expert per stream, in stream order), `router_raw`, `router_conf`
  and `k` (number of sub-instructions).

To run a checkpoint written by training (`<OUT>/ckpt/step-N` or `<OUT>/ckpt/final_components`), point `--lora_experts_dir` and
`--router_head` at the released package:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/moe/explicit-moe/moe_v2_infer.py --base_dir $BASE_DIR \
    --ckpt $BASE_DIR/output/explicit-moe/ckpt/step-N \
    --lora_experts_dir $BASE_DIR/ckpt/explicit-moe/lora_experts \
    --router_head $BASE_DIR/ckpt/explicit-moe/router_head_online.pt \
    --src_video input.mp4 --prompt "Remove the dog" --name dog --out_dir <OUT_DIR>
```

The script exits with code 17 for a retrying job runner in two cases: any loaded weight or buffer is non-finite, or a router
confidence is NaN.

## Training

Training starts from these weights:

- The shared frozen base, `$BASE_DIR/ckpt/kiwi-edit-instruct-only-videoxfun/` (all five components).
- The released LoRA experts, `$BASE_DIR/ckpt/explicit-moe/lora_experts/`. They stay frozen.
- The released initial router head, `$BASE_DIR/ckpt/explicit-moe/router_head_online.pt`. Its weights are fine-tuned.

The fusion layers are initialised from scratch (`--fusion_init normal`: PyTorch default initialisation with a zero output
bias). `moe_v2_state.pt` is only needed to continue from the released checkpoint (see Resume below). The code that trained the
LoRA experts and the initial router head is not part of this repository.

### Launcher

Edit the config block at the top of `train_multinode.sh`, activate your environment, then run the same script on every node.
Pass that node's rank (the `NODE_RANK` argument, `0 .. NUM_NODES-1`):

```bash
bash scripts/moe/explicit-moe/train_multinode.sh 0     # on node 0 (MASTER_ADDR)
bash scripts/moe/explicit-moe/train_multinode.sh 1     # on node 1
```

The launcher changes to the repository root and starts `moe_v2_train.py` with `accelerate launch` (DDP, bf16 mixed precision,
gradient checkpointing).

| Variable | Default | Meaning |
|---|---|---|
| `BASE_DIR` | `/path/to/workdir` | Holds `ckpt/kiwi-edit-instruct-only-videoxfun/`, `ckpt/explicit-moe/`, `datasets/` and `output/`. |
| `DATASET_BASE` | `/path/to/video/root` | Root for relative video paths in the metadata files. |
| `MASTER_ADDR`, `MASTER_PORT` | `127.0.0.1`, `29829` | Address of node 0. |
| `NUM_NODES`, `GPUS_PER_NODE` | 2, 8 | |
| `USE_EFA` | 0 | 1 exports the AWS EFA / NCCL settings in the script. |
| `EXPERIMENT` | `explicit-moe` | Output directory `$BASE_DIR/output/$EXPERIMENT`. |
| `BASE_CKPT`, `EXPERTS_CKPT` | `$BASE_DIR/ckpt/kiwi-edit-instruct-only-videoxfun`, `$BASE_DIR/ckpt/explicit-moe` | Frozen base and extracted explicit-moe package. |
| `ROUTER_HEAD`, `LORA_EXPERTS_DIR` | `$EXPERTS_CKPT/router_head_online.pt`, `$EXPERTS_CKPT/lora_experts` | Initial router head and frozen experts. |
| `EXPERTS` | `style,add,remove,replace,background` | |
| `DATA`, `DATA_PREFIX` | three files under `$BASE_DIR/datasets/classified/`, `",,"` | Comma-separated metadata files and an optional path prefix for each file. |
| `RESUME_FROM` | empty | `moe_v2_state.pt` to resume from. |
| `WANDB_PROJECT`, `WANDB_RUN` | `video-editor-moe`, `$EXPERIMENT` | Used only when `WANDB_API_KEY` is set in the environment; otherwise only TensorBoard is written. |

Training settings in the launcher (router learning rate, weight decay, gradient clipping and seed are trainer defaults):

| Setting | Value |
|---|---|
| Hardware | 2 nodes x 8 GPUs (80 GB); each GPU holds the full model (plain DDP). |
| Batch | 2 samples per GPU (`BATCH_SIZE=2`), effective batch 32; `GRAD_ACCUM=1`. |
| Steps | `MAX_TRAIN_STEPS=55968` = 4 epochs x `EPOCH_STEPS=13992` (447,710 records at batch 32). The released checkpoint is step 28,200. |
| Fusion learning rate | Linear from 1e-4 to 1e-5 over the first 600 steps, then 1e-5 (`FUSION_LR_ANNEAL_FROM`, `FUSION_LR`, `FUSION_LR_ANNEAL_STEPS`). |
| Router learning rate | 1e-5 (`--router_learning_rate`). |
| Optimizer | AdamW, weight decay 0.01, gradient-norm clip 1.0. |
| Loss | MSE between the prediction and `noise - target_latent` (flow matching, training shift 5), mixed with a frame-difference (motion) MSE at ratio 0.1 (`MOTION_SUB_LOSS_RATIO`), plus router cross-entropy on labelled sub-instructions. |
| Text CFG dropout | 0.1 on the base instruction (`BASE_TEXT_DROP_PROB`). |
| Video | 49 frames at 480x832, `MAX_PIXELS=399360`. |
| Fusion layers | After blocks [15, 29), 24 heads, `ROPE_EXPERT_OFFSET0=1`, `FUSION_INIT=normal`. |
| Checkpoints | Every 100 steps (`SAVE_STEPS`). |

**Batching.** A batch holds only samples with the same multiset of expert labels, because the expert streams of a batch must
line up. Samples without labels form one group and are routed by the router head. Within a step, a sample whose routed
experts differ from the first sample's is left out of that step. Leftover partial batches go to the end of the epoch and are
paired into one optimizer step, so every rank takes the same number of steps. `EPOCH_STEPS` is used only to seed the shuffle
and to resume mid-epoch at the exact batch. Set it to the number of optimizer steps per epoch for your data and GPU count, or
to 0 to disable exact resume.

**Data.** The reference run passes three metadata files as-is: `CoinVE.jsonl` (200,916 records), `goku_with_masks.jsonl`
(22,939) and `openve_v2_equal_223855.json` (223,855, a five-category sample of OpenVE). The trainer reads these fields:

- `src_video` and `tgt_video`.
- `prompt`: one string, or a list of sub-instructions.
- An optional per-sub-instruction category list, `turn_types_qwen`, or a single category, `cls`. Categories are
  style/add/remove/replace/background.

`.json` files hold a JSON array and other files are read as JSON lines. See [`docs/DATA.md`](../../../docs/DATA.md) for the
data formats and how to prepare them.

**Resume.** Set `RESUME_FROM=$BASE_DIR/output/$EXPERIMENT/ckpt/step-N/moe_v2_state.pt` and restart all nodes. This restores
the fusion and router weights, the AdamW state and the step counter, and resets the learning rates to `FUSION_LR` and the
router learning rate. To continue from the released checkpoint, set `RESUME_FROM=$BASE_DIR/ckpt/explicit-moe/moe_v2_state.pt`.
Training then resumes at step 28,200 with a fresh AdamW state, because the release has no optimizer state.

**Outputs** go to `$BASE_DIR/output/$EXPERIMENT/`:

- `ckpt/step-N/` holds:
  - `moe_v2_state.pt`: fp32 trainable tensors, AdamW state and metadata.
  - `fusion_moe_v2_absorb.safetensors`: bf16 fusion layers.
  - `router_head_finetuned.pt`.
- Retention: step directories at multiples of 1000 are kept, and of the others only the latest two.
- `ckpt/final_components/` is written at the end of training.
- `logs/node<rank>_<timestamp>.log`, with `logs/node<rank>_latest.log` linking to the newest log.
- TensorBoard event files.

Scalars are logged every 10 steps: `loss/fusion`, `loss/router_ce`, `lr/fusion`, `grad/norm`, `route/agree_vs_qwen_gt`,
`route/agree_vs_turntype`, and `expert_used/<c>`, `expert_gt/<c>` and `expert_router_raw/<c>` for each category.

## Tests

There are no CPU unit tests for this model. To check the environment, data paths and checkpoint writing before a real run, use
the smoke test:

```bash
bash scripts/moe/explicit-moe/train_smoke.sh      # edit BASE_DIR / DATASET_BASE in its config block first
```

It uses the same model, data and hyperparameters as `train_multinode.sh`, on 1 node with 8 GPUs (port 29830). It runs 10 steps
and saves at steps 5 and 10 plus `final_components`, writing to `$BASE_DIR/output/explicit-moe_smoke/` without wandb. To test resume
as well, set `RESUME_FROM` to its `ckpt/step-5/moe_v2_state.pt` and raise `MAX_TRAIN_STEPS`.

The loaders also check themselves when they run:

- `build_model` asserts 600 LoRA tensors per expert and the router's class order.
- `load_checkpoint` asserts the variant, fusion range, fusion layer count and RoPE offset stored in `moe_v2_state.pt`.
- Inference scans all weights for non-finite values before sampling.

## Notes and limitations

- **Cost grows with the number of sub-instructions.** Every denoising step runs K + 1 DiT passes, or 2(K + 1) while CFG is
  active, and the fusion attention covers (K + 1) x 5,070 keys. `--cfg_text_steps_frac 0.6` removes the unconditional passes
  from the last 40% of the steps.
- **Five categories only.** Every sub-instruction is sent to one of the five experts by argmax; there is no "none" option.
  Compound requests must be split with `--prompts "a ||| b"`. A multi-step sentence passed with `--prompt` goes to a single
  expert.
- **Keep all five experts loaded** (`--experts` default). The router always chooses among five classes, and routing to an
  expert that is not loaded raises an assertion error.
- **Training resolution only.** Training used only 49 frames at 480x832. Other settings must keep `--num_frames` of the form
  4k+1 and height and width multiples of 32. Output videos are written at `--fps` (16) whatever the source frame rate is.
- **No sequence parallelism.** The forward pass asserts a single sequence-parallel rank.
- **Dataset filtering.** The trainer asserts that the dataset class drops no records. Records whose categories include `i2v` or
  `camera` are dropped by `KiwiMultiEditDataset`, so remove them from the metadata beforehand.
- **The release has no optimizer state.** Resuming from it starts a fresh AdamW state.
