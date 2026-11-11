# videoeditor-14b

videoeditor-14b is Kiwi-Edit 5B in which both the FFN and the self-attention key/value projections of 14 of the 30 DiT blocks
are mixture-of-experts layers. Each MoE layer has 7 experts and routes every token to its top 3. Every expert is a complete
copy of the layer it replaces: the original Kiwi-Edit weights, five category-specialised variants and one direct fine-tune.
The whole DiT is trained end to end. It has 13.99B parameters, of which about 8.0B are active per token.

- Weights: [Massyzs/videoeditor-14b](https://huggingface.co/Massyzs/videoeditor-14b) (gated) and the shared base components in
  [Massyzs/kiwi-edit-5b-instruct-only-videoxfun](https://huggingface.co/Massyzs/kiwi-edit-5b-instruct-only-videoxfun).
- Design notes (expert sources, attention formulation, compute budget, verification plan): [DESIGN.md](DESIGN.md).
- File names, log prefixes (`[moe-v3.1]`) use the development name of this model.

## Architecture

The base DiT is Kiwi-Edit, which uses the Wan2.2-TI2V-5B layout: 30 blocks, hidden size 3072, 24 heads of dimension 128 and
FFN width 14336. The DiT is conditioned in two ways. Cross-attention reads the Kiwi-Edit MLLM encoding of the instruction and
the source frames. The source-video latents enter through a patch embedding.

The MoE blocks are those with 0-based indices 1, 3, 5, ..., 27 (`--moe_layers`). Each of them contains two MoE sub-layers:

- **FFN**: 7 experts. Each expert is a full FFN (Linear 3072→14336, GELU, Linear 14336→3072).
- **Self-attention**: the key and value projections become 7 expert pairs `(W_k^i, W_v^i)`. The query and output projections
  and the q/k RMSNorms stay shared.

Cross-attention, modulation, the other 16 blocks, the embeddings and the output head are single copies.

### Experts

| index | name | initial weights of `self_attn.k`, `self_attn.v`, `ffn.0`, `ffn.2` in each MoE block |
|---|---|---|
| 0 | `main` | the original Kiwi-Edit weights |
| 1-5 | `style`, `add`, `remove`, `replace`, `background` | Kiwi-Edit with the category's LoRA expert merged in: `W = W_kiwi + B·A` (rank 128, scale 1), bias unchanged. The LoRA experts are the ones released with [explicit-moe](../explicit-moe) |
| 6 | `direct_finetune` | a separate full fine-tune of the Kiwi-Edit DiT (3,581 steps) on compound-edit video data mixed with image-editing pairs. Not released |

The expert order is `EXPERT_NAMES` in `moe_v3_1_model.py`, and `meta.json` stores it too. These sources are only the
starting point. All experts, routers and the rest of the DiT are then trained jointly, so the released experts no longer
equal their sources. `merge_experts_v3_1.py` builds the initialisation for experts 1-6 (see [Training](#training)).

### Routing

Each MoE block has two independent routers, one for self-attention and one for the FFN. A router is a bias-free linear layer
`3072 → 7` evaluated in fp32 and initialised from N(0, 0.02). Its input is the input of the sub-layer it routes, taken after
the layer norm and the timestep modulation. For every token t:

```
p       = softmax(router(x_t))                    # 7 probabilities
S_t     = indices of the 3 largest p_i
ĝ_{t,i} = p_i / Σ_{j∈S_t} p_j,   i ∈ S_t          # renormalised to sum to 1
```

There is no shared, always-on expert. The original weights take part in routing as expert 0.

### MoE FFN

```
y_t = Σ_{i∈S_t} ĝ_{t,i} · FFN_i(u_t)
```

Each expert runs only on the tokens routed to it, and the outputs are accumulated in fp32.

### MoE self-attention with a log-gate prior

For each token t and each selected expert i ∈ S_t, the layer builds one key/value pair. Every copy uses the token's own 3D
RoPE position:

```
k_t^i = RoPE_t(norm_k(W_k^i x_t)),    v_t^i = W_v^i x_t
```

The three pairs of every token are interleaved into a key/value sequence of length 3L. The L queries come from the shared
projection and attend to all 3L keys in a single attention call. The softmax is normalised jointly over the 3L keys, and the
router weight enters as a multiplicative prior:

```
score(s; t, i) = q_s · k_t^i / √d + log ĝ_{t,i}

y_s = Σ_t Σ_{i∈S_t} ĝ_{t,i} · exp(q_s·k_t^i / √d) · v_t^i   /   Σ_t Σ_{i∈S_t} ĝ_{t,i} · exp(q_s·k_t^i / √d)
```

This formulation has three properties:

- Key and value are bound: a token's key copy and value copy always come from the same expert.
- If all experts of a layer are identical, the three copies of a token coincide. Because Σ_i ĝ_{t,i} = 1, the layer then
  reproduces the original self-attention exactly.
- The router receives gradients from the main loss through `log ĝ`, not only from the balance loss.

FlashAttention accepts no additive bias, so the prior is injected through 8 extra head dimensions (128 → 136):

- Queries are padded with `[√d, √d, 0, ...]`.
- Keys are padded with `[hi, lo, 0, ...]`, where `hi = bf16(log ĝ)` and `lo = log ĝ − hi`.
- Values are padded with zeros.

The kernel is called with `softmax_scale = 1/√128`. The padded dot product then equals `q·k/√d + log ĝ`, with about 16 bits of
precision for the prior, and the padded output dimensions are dropped (`attention_with_scale`, `MoEKVSelfAttention`).
Without FlashAttention (on CPU, or with `VIDEOX_ATTENTION_TYPE=SDPA`), the layer calls PyTorch SDPA with the same scale. For
49 frames at 480×832, the DiT sequence has 13×15×26 = 5,070 tokens, so each MoE self-attention attends over 15,210 keys.

The released model does not use `MoEKVSelfAttention(shared_kv=True)`. That option adds one extra, always-present pair from a
shared projection with prior 1. With the option off, the layer removes the original `k`/`v` projections, and expert 0 holds
their weights.

### Load balancing

Each of the 28 routers adds the DeepSeek-MoE expert-level balance loss, computed per sample (N = 7 experts, k = 3, L tokens
per sample):

```
f_i = N · (number of the sample's tokens that selected i) / (k · L)
P_i = mean of p_i over the sample's tokens
aux = α · mean_over_samples Σ_i f_i · P_i
```

A custom autograd function (`AddAuxiliaryLoss`) attaches the term to the layer output. A backward pass on the main loss
therefore also carries the balance gradient to the routers. The released model uses α = 1e-4. Set α with
`--moe_aux_loss_alpha`; the script default is 1e-3.

### Size

| component | parameters |
|---|---|
| FFN experts (7 × 88.1M × 14 blocks) | 8.63B |
| self-attention k/v experts (7 × 18.9M × 14 blocks) | 1.85B |
| routers (2 × 14 × 7 × 3072) | 0.6M |
| everything else (shared attention, cross-attention, non-MoE blocks, embeddings, head) | 3.50B |
| **total** | **13.99B** |
| active per token (3 of 7 experts) | about 8.0B |

## Files

| file | purpose |
|---|---|
| `wan_transformer3d_kiwi_moe_v3_1.py` | The DiT: a self-contained copy of `videox_fun/models/wan_transformer3d_kiwi.py`. Additions are marked `MOE-V3.1`: `Fp32Gate`, `route_tokens`, `dispatch_by_expert`, `expert_balance_loss`, `AddAuxiliaryLoss`, `MoEFFN`, `MoEKVSelfAttention`, `attention_with_scale`, the block- and model-level `moe` arguments and `moe_attns()` / `moe_ffns()`. The environment variable `MOE_ATTN_BACKEND` (`fa2` default, `fa3`, `cudnn`, `auto`) selects the attention kernel |
| `moe_v3_1_model.py` | Assembly: `EXPERT_NAMES`, `build_dit` (builds the MoE DiT and optionally loads the merged expert initialisation), `mark_trainable` (three learning-rate groups: base / experts / routers), `TextEncoder` (MLLM), `build_mllm`, `build_vae`, `load_trainable_weights` |
| `moe_v3_1_infer.py` | Inference on one video or on a case list, with text CFG and optional per-token expert maps; `--low_mem 1` for 40 GB GPUs (see [Low-memory mode](#low-memory-mode-40-gb-gpus)) |
| `infer.sh`, `infer_lowmem.sh`, `infer_fast.sh` | Launchers for the three inference modes (default on one 80 GB GPU, low memory on one 40 GB GPU, fast on two GPUs): edit the configuration block at the top and run |
| `moe_v3_1_infer_fast.py`, `moe_v3_1_model_fast.py`, `wan_transformer3d_kiwi_moe_v3_1_fast.py` | Fast-inference copies of the three files above (same weights; fused expert path, grouped GEMMs, second-GPU CFG replica, CUDA graphs; see [Fast inference](#fast-inference-two-gpus)) |
| `moe_v3_1_train.py` | Training loop: DeepSpeed ZeRO-3, full DiT, flow-matching loss, checkpoint and resume |
| `merge_experts_v3_1.py` | One-off: builds the initial weights of experts 1-6 (`experts_v3_1.safetensors`) from the LoRA experts and the direct fine-tune |
| `train_multinode.sh` | Multi-node launcher (default 3 nodes × 8 GPUs). Writes the ZeRO-3 config and runs `accelerate launch` |
| `train_smoke.sh` | Single-node, 8-GPU, 6-step smoke test |
| `check_build_v3_1.py` | CPU check with real weights: `build_dit` + `mark_trainable` from the merged expert file |
| `probe_moe_v3_1_cpu.py` | CPU structural probe on a small random model |
| `probe_moe_v3_1_gpu.py` | Single-GPU probe of one MoE self-attention layer at full size |
| `DESIGN.md` | Design notes |

Shared modules (MLLM encoder wrapper, datasets) are in `videox_fun/moe_common/`.

## Weights

Inference needs the shared base components and the videoeditor-14b package:

```bash
huggingface-cli login    # after your access request has been approved
huggingface-cli download Massyzs/kiwi-edit-5b-instruct-only-videoxfun \
    --local-dir $BASE_DIR/ckpt/kiwi-edit-instruct-only-videoxfun
huggingface-cli download Massyzs/videoeditor-14b --local-dir $BASE_DIR/hf/videoeditor-14b
cd $BASE_DIR/hf/videoeditor-14b
cat videoeditor-14b.tar.gz.part-* | tar -xzf - -C $BASE_DIR/ckpt/      # -> $BASE_DIR/ckpt/videoeditor-14b/
cd $BASE_DIR/ckpt && sha256sum -c $BASE_DIR/hf/videoeditor-14b/SHA256SUMS.contents    # optional
```

```
$BASE_DIR/ckpt/
  kiwi-edit-instruct-only-videoxfun/
    config.json, diffusion_pytorch_model.safetensors   # Kiwi-Edit DiT: only used to construct the model skeleton
    moe_expert_init.safetensors                        # MLLM connector + query tokens + source-video patch embedding
    Wan2.2_VAE.pth                                     # VAE
    qwen/                                              # Kiwi-Edit MLLM encoder
  videoeditor-14b/                                     # <CKPT_DIR>
    moe_v3_1_trainable.safetensors                     # the full DiT (13.99B parameters, bf16, 28 GB), step 18630
    meta.json                                          # expert order, MoE layers, top-k, α, learning rates, step
```

The package contains every DiT parameter, including the routers and the source-video patch embedding. It does not contain
optimizer state.

## Inference

Inference runs on one 80 GB GPU (one 40 GB GPU with `--low_mem 1`, see [Low-memory mode](#low-memory-mode-40-gb-gpus)). The simplest way is
one of the launchers `infer.sh` / `infer_lowmem.sh` / `infer_fast.sh`: edit the configuration block at the top (weights, input video, instruction,
output directory, GPU ids) and run `bash scripts/moe/videoeditor-14b/infer.sh`. The equivalent commands, run from the repository root:

```bash
# one video, one instruction
CUDA_VISIBLE_DEVICES=0 python scripts/moe/videoeditor-14b/moe_v3_1_infer.py \
    --base_dir $BASE_DIR --ckpt $BASE_DIR/ckpt/videoeditor-14b \
    --src_video <in.mp4> --prompt "Remove the dog" --name dog --out_dir <OUT_DIR>

# compound instruction: sub-instructions separated by |||, joined with " Then " into one instruction (as in training)
CUDA_VISIBLE_DEVICES=0 python scripts/moe/videoeditor-14b/moe_v3_1_infer.py \
    --base_dir $BASE_DIR --ckpt $BASE_DIR/ckpt/videoeditor-14b \
    --src_video <in.mp4> --prompts "Remove the dog ||| Convert to oil painting style" --name dog_oil --out_dir <OUT_DIR>

# case list <BENCH_DIR>/cases.json = [{"name": ..., "src": <path relative to BENCH_DIR>, "prompts": [...]}, ...]
# one process per GPU: --shard_idx i --num_shards n processes cases i, i+n, i+2n, ...
CUDA_VISIBLE_DEVICES=0 python scripts/moe/videoeditor-14b/moe_v3_1_infer.py \
    --base_dir $BASE_DIR --ckpt $BASE_DIR/ckpt/videoeditor-14b \
    --bench_dir <BENCH_DIR> --out_dir <OUT_DIR> --shard_idx 0 --num_shards 1
```

The script builds the model structure from `meta.json` and skips random initialisation (`--fast_init 1`). It asserts that
`moe_v3_1_trainable.safetensors` covers every parameter, then loads it. Inference does not need `--experts_merged`.
`--cases_json <file>` reads a case list from another file; relative `src` paths then resolve against `--bench_dir`, which
defaults to the directory of the file.

| option | default | meaning |
|---|---|---|
| `--steps` | 50 | flow-matching Euler steps |
| `--sigma_shift` | 5.0 | sigma shift of the sampling schedule |
| `--cfg_text_scale` | 3.0 | text CFG `pred = u + s·(c − u)`. The unconditional branch uses an empty instruction, and the MLLM still sees the source video. `1` disables CFG |
| `--cfg_text_steps_frac` | 0.6 | apply CFG only in this leading fraction of the steps (`1.0` = all steps) |
| `--variants` | empty | `'label:cfg_text_scale:cfg_text_steps_frac:sigma_shift,...'`. Runs several settings with one model load, writing each to `<OUT_DIR>/<label>/`. All variants of a case share the same initial noise |
| `--num_frames`, `--height`, `--width`, `--fps` | 49, 480, 832, 16 | output size. The source uses its first n frames, trimmed to 4k+1 and padded with the last frame, with a short-side resize and centre crop, as in training |
| `--seed` | 42 | initial-noise seed |
| `--dump_expert_maps` | 1 | write per-token expert maps (see below) |
| `--frames_png` | 1 | also write lossless PNG frames |
| `--skip_existing` | 1 | skip cases whose outputs are already complete |
| `--min_free_gib` | 60 (20 with `--low_mem 1`) | refuse to start if the GPU has less free memory than this |
| `--low_mem` | 0 | `1` = low-memory mode: MLLM in pinned host memory except while encoding, MoE blocks that do not fit streamed from pinned host memory (see below) |
| `--reserve_gib` | 12 | low-memory mode: GPU memory kept free for the MLLM encoder (while encoding), activations and VAE decoding when choosing how many MoE blocks stay resident |
| `--resident_blocks` | -1 | low-memory mode: number of resident MoE blocks; `-1` = as many as fit into (free memory - `--reserve_gib`) |
| `--numa_bind` | 1 | low-memory mode: bind the process to the CPUs of the GPU's NUMA node before pinning host memory |
| `--emulate_gpu_gib` | 0 | testing: cap the PyTorch allocator at this many GiB (e.g. 38 to emulate a 40 GB GPU on a larger one) |
| `--base_only` | 0 | `1` runs the unmodified Kiwi-Edit DiT (no MoE layers) through the same pipeline as a baseline. `--ckpt` must still point to a directory that contains `meta.json` |

Outputs in `<OUT_DIR>` (or `<OUT_DIR>/<label>/` with `--variants`):

- `<name>.mp4` and lossless frames `frames/<name>/f000.png ...`.
- `<name>_experts_attn.png` and `<name>_experts_ffn.png`: expert maps of the self-attention and FFN routers. Each map is
  recorded on the conditional branch at the middle denoising step. The rows are MoE blocks 1, 15 and 27. The columns are the
  first, middle and last latent frames. Each 16×16 cell is one token, coloured by its top-1 expert: main red, style green,
  add yellow, remove blue, replace orange, background purple, direct_finetune cyan.
- `manifest_shard<k>.jsonl`: one record per case with the prompts, sampling settings, timings, map paths and per-layer counts
  of top-1 experts.

If `--out_dir` is omitted, the output goes to `<run>/render/<ckpt dir name>/`, where `<run>` is the directory that
contains the checkpoint's `ckpt/` folder (for `$BASE_DIR/ckpt/videoeditor-14b` this is `$BASE_DIR/render/videoeditor-14b/`).
The exit codes 17 (non-finite weights), 18 (the MLLM self-test failed three times) and 19 (non-finite or constant output; nothing is
written) are meant for a runner that retries.

### Low-memory mode (40 GB GPUs)

The default run peaks at 39.1 GiB of GPU memory (26.1 GiB DiT, 7.1 GiB MLLM encoder, VAE and activations) and does not fit a 40 GB GPU.
`--low_mem 1` keeps the MLLM encoder in pinned host memory and copies it to the GPU only while a case is encoded; the dense blocks, embeddings
and head stay resident, and as many of the 14 MoE blocks (1.5 GiB each) as fit into (free GPU memory - `--reserve_gib`) stay resident, the rest
being streamed from pinned host memory block by block through two GPU slots (the next block is prefetched on a side stream while the current one
computes). The results are those of the default mode up to the pipeline's run-to-run variation (same kernels, same data).

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/moe/videoeditor-14b/moe_v3_1_infer.py --low_mem 1 --reserve_gib 10 \
    --base_dir $BASE_DIR --ckpt $BASE_DIR/ckpt/videoeditor-14b --src_video <in.mp4> --prompt "Remove the dog" --name dog --out_dir <OUT_DIR>
```

| setting (one GPU) | resident / streamed MoE blocks | peak GPU memory | end-to-end time per 49-frame video (MLLM/VLM encoding and VAE included) |
|---|---|---|---|
| default mode, 80 GB | 14 / 0 | 39.1 GiB | 35 s |
| `--low_mem 1 --reserve_gib 10`, 40 GB | 14 / 0 | 35.2 GiB | 35-39 s (default-mode speed) |
| `--low_mem 1`, 40 GB (default reserve 12 GiB) | 11 / 3 | 33.7 GiB | 52-60 s |

The 40 GB rows were measured on an 80 GB GPU with `--emulate_gpu_gib 38` (allocator capped at 38 GiB). With `--reserve_gib 10` the whole DiT
stays resident and the only cost is copying the MLLM in and out (about 2 s per case); each streamed block adds 1.5 GiB of host-to-GPU traffic per
DiT forward (80 forwards per video; 14 GB/s from pinned memory on our server).

### Fast inference (two GPUs)

`moe_v3_1_infer_fast.py` has the same command line as `moe_v3_1_infer.py` plus `--fuse_experts`, `--fuse_kv`, `--moe_impl`, `--cfg_batch`,
`--dp_cfg` and `--cuda_graph`; it builds the DiT from `moe_v3_1_model_fast.py` / `wan_transformer3d_kiwi_moe_v3_1_fast.py`. Recommended:

```bash
MOE_ATTN_BACKEND=fa2 CUDA_VISIBLE_DEVICES=0,1 python scripts/moe/videoeditor-14b/moe_v3_1_infer_fast.py \
    --moe_impl grouped --dp_cfg 1 --cuda_graph 1 \
    --base_dir $BASE_DIR --ckpt $BASE_DIR/ckpt/videoeditor-14b --src_video <in.mp4> --prompt "Remove the dog" --name dog --out_dir <OUT_DIR>
```

- `--fuse_experts 1` / `--fuse_kv 1` (defaults): stacked expert weights (views, no extra memory) and one GEMM per expert for k and v; bit-identical.
- `--moe_impl grouped`: one `torch._grouped_mm` per expert linear over the tokens sorted by expert (PyTorch >= 2.8), no host synchronisation; the
  bias is added after the bf16 GEMM (the only numerical difference to the original; frame PSNR 41-48 dB, equal to the original's run-to-run variation).
- `--dp_cfg 1`: a replica of the DiT on the second GPU runs the unconditional CFG branch concurrently with the conditional branch (bit-identical).
- `--cuda_graph 1`: the DiT forward is captured in a CUDA graph per case and CFG branch and replayed (bit-identical; needs `--moe_impl grouped`
  and `MOE_ATTN_BACKEND=fa2`, whose equal-length attention path is capture-safe). Capturing adds about 3 s per case.

The single-GPU 14B forward is compute-bound (327 ms of kernel time per 333 ms forward), so on one GPU these options gain at most ~5%; with the
replica and CUDA graphs a CFG step takes 330 ms instead of 654 ms: 26.1 s -> 16.5 s of DiT time per video, 35.3 s -> 30.2 s end to end, MLLM/VLM encoding and VAE included
(H100 80GB, 49 x 480 x 832, 50 steps, CFG on the first 60% of the steps).

## Training

Training uses DeepSpeed ZeRO-3 over 3 nodes × 8 GPUs (80 GB each) by default. All DiT parameters are trained, while the MLLM
and the VAE stay frozen.

### 1. Expert initialisation

Training starts from `experts_v3_1.safetensors`, the initial weights of experts 1-6 for the four projections of every MoE block.
Expert 0 is copied from the base DiT at build time. `merge_experts_v3_1.py` builds the file from two inputs:

- the five LoRA experts in the [explicit-moe](../explicit-moe) package (`lora_experts/<name>/dit_query_router.safetensors`);
- the direct fine-tune for expert 6. This is a diffusers-style sharded directory with
  `diffusion_pytorch_model.safetensors.index.json` and its shards, using VideoX-Fun key names (`blocks.{b}.self_attn.k.weight`, ...).

The direct fine-tune used for the released model is not released. To reproduce the initialisation, supply your own full
fine-tune of the Kiwi-Edit DiT in that layout.

The script expects the LoRA files at `<experts_root>/moe_expert_<name>/final_components/dit_query_router.safetensors`. Link
the released files into that layout:

```bash
# $BASE_DIR/ckpt/explicit-moe/ = the extracted explicit-moe package
EXPERTS_ROOT=$BASE_DIR/output/lora_experts
for n in style add remove replace background; do
  mkdir -p $EXPERTS_ROOT/moe_expert_$n/final_components
  ln -sf $BASE_DIR/ckpt/explicit-moe/lora_experts/$n/dit_query_router.safetensors \
         $EXPERTS_ROOT/moe_expert_$n/final_components/dit_query_router.safetensors
done
python scripts/moe/videoeditor-14b/merge_experts_v3_1.py --base_dir $BASE_DIR \
    --experts_root $EXPERTS_ROOT --direct_finetune <DIRECT_FINETUNE_DIR> \
    --out_dir $BASE_DIR/ckpt/init
```

The result is `$BASE_DIR/ckpt/init/experts_v3_1.safetensors` (672 bf16 tensors, 16.7 GiB), which is the launchers'
default `EXPERTS_MERGED`. The script also writes `experts_v3_1_report.json` with the relative weight change of each expert.
`--dit_weights` (base DiT file) and `--moe_layers` can also be overridden.

### 2. Data

The dataset files are JSON lists or JSONL files. Each record needs `src_video`, `tgt_video` and `prompt`:

- `src_video` and `tgt_video` are paths that are absolute or relative to `DATASET_BASE` / `--dataset_base_path`.
- `prompt` is a string or a list of sub-instructions.

A list is joined into one instruction. Records whose `src_video` matches `videos/case_<n>/` use the wording
"A, then b, and finally c". All other records are joined with `--compound_joiner` (default `" Then "`). The optional fields
`cat`, `cls` and `turn_types_qwen` only feed the category × expert routing statistics in the logs. Samples whose instruction
contains a word in `--exclude_prompt_words` are dropped.

The launcher defaults, relative to `BASE_DIR`, are:

- `STAGE_A_DATA=datasets/filtered/openve_original_nosub_all.json`: OpenVE-3M single-instruction pairs, without the
  `subtitles_edit` category.
- `STAGE_B_DATA=datasets/classified/goku_with_masks.jsonl,datasets/classified/CoinVE.jsonl`: multi-step edits. Masks are not
  used.

These files are not part of the release. With `DATA_SCHEDULE=mixed`, one copy of the stage-A data and `MULTI_REPEAT` (8)
copies of the stage-B data are shuffled into one pool and trained for `STAGE_A_EPOCHS` (1) epochs. With the original data,
this is 18,630 steps at 192 samples per step. `DATA_SCHEDULE=two_stage` instead trains stage A for `STAGE_A_EPOCHS`, then
stage B for `STAGE_B_EPOCHS`.

### 3. Launch

Both launchers start with a configuration block. Edit it (`BASE_DIR`, `DATASET_BASE`, `MASTER_ADDR`, `MASTER_PORT`,
`NUM_NODES`, `GPUS_PER_NODE`, `EXPERTS_MERGED`, ...), activate your environment, then run the multi-node launcher on every node:

```bash
bash scripts/moe/videoeditor-14b/train_multinode.sh <node_rank>    # 0 .. NUM_NODES-1; rank 0 runs on MASTER_ADDR
bash scripts/moe/videoeditor-14b/train_smoke.sh                    # single node, 8 GPUs, 6 steps, batch 1, no wandb
```

The smoke test writes to `$BASE_DIR/output/videoeditor-14b_smoke/`. It saves weights at step 4 and the DeepSpeed state at step 6.

`train_multinode.sh` defaults:

- Model and optimisation:
  - batch 8 per GPU (192 in total), no gradient accumulation, gradient clipping 1.0, bf16, gradient checkpointing;
  - top-3, α = 1e-4;
  - AdamW with weight decay 0.01 and three learning-rate groups: 1e-5 for the original parameters, experts from 1e-4 down to
    1e-5, routers from 1e-3 down to 1e-4, both decaying linearly over the first 2,000 steps and then held constant;
  - 10% of the samples get an empty instruction (for text CFG);
  - the loss is `0.9 · MSE + 0.1 · MSE of temporal differences` (`--motion_sub_loss_ratio`) on the flow-matching target.
- Launcher settings:
  - `MOE_ATTN_BACKEND=fa3`, which falls back to `fa2` if `flash_attn_interface` cannot be imported. `FA3_SITE` can add a
    separate FlashAttention-3 install to `PYTHONPATH`.
  - `USE_EFA=1` exports the AWS EFA / NCCL settings.
  - wandb logging is enabled only when `WANDB_API_KEY` is set; tensorboard logging is always on.

The output directory `$BASE_DIR/output/videoeditor-14b/` must be on storage shared by all nodes. At start-up, each rank waits up to
10 minutes for at least 60 GiB of free GPU memory. Each process transiently needs about 120 GB of host memory while it builds
the DiT, so the ranks of a node start 15 s apart.

The launcher also enables a feature cache for multi-step samples (`MULTI_CACHE_DIR=$OUT/cache_multi`). After the first visit,
it stores their VAE moments and MLLM contexts, about 15 MB per sample, so that later repeats skip decoding and encoding. The
launcher keeps this cache across restarts and after training (`MULTI_CACHE_KEEP=1`, `MULTI_CACHE_DELETE_ON_DONE=0`). Delete
it yourself, or set `MULTI_CACHE_DIR=` (empty) to disable it.

### 4. Resume and fine-tuning

- **Resume**: write `RESUME_DS_STATE=$BASE_DIR/output/videoeditor-14b/ckpt/ds_state_step-N` into `$BASE_DIR/output/videoeditor-14b/stage.env`,
  then restart all nodes with the same number of processes.
- **Start from the released weights**: pass `--experts_merged ""` and
  `--resume_weights <CKPT_DIR>/moe_v3_1_trainable.safetensors` to `moe_v3_1_train.py`. The launcher does not expose these, so
  edit its `accelerate launch` command. The optimizer, the step counter and the learning-rate ramps start from zero.

### 5. Outputs and logs

`$BASE_DIR/output/videoeditor-14b/` contains:

- `ckpt/step-N/moe_v3_1_trainable.safetensors` + `meta.json`: the full DiT in bf16, every `SAVE_STEPS` (200). Steps that are
  multiples of 1000 are kept; of the others, only the latest two are kept. The final weights are also copied to
  `ckpt/final_components/`.
- `ckpt/ds_state_step-N/`: the full DeepSpeed state for resuming, every `DS_SAVE_STEPS` (200). Only the latest one is kept.
- `logs/node<rank>_<time>.log` (plus `node<rank>_latest.log`), tensorboard event files, and `ds_zero3.json`.

Scalars are logged every 10 steps, averaged over all ranks:

- losses and learning rates: `loss/main`, `loss/motion`, `loss/aux_attn`, `loss/aux_ffn`, `loss/aux`, `loss/total`, `lr/base`,
  `lr/new_layers`, `lr/gate`, `grad/norm`, `skipped_steps`;
- `moe_attn_L{block}/expert{i}` and `moe_ffn_L{block}/expert{i}`: the fraction of tokens that selected expert i, which sums to 3
  per layer; `moe/{attn,ffn}_selected_min/max` give the extremes;
- `route_{attn,ffn}/{category}/expert{i}`: P(expert i selected | sample category); `route/{attn,ffn}_spread` is the mean
  over experts of (max − min) of that probability across categories, where 0 means no specialisation by category;
- timing: `time/{vae,mllm,dit}`;
- cache statistics: `cache/*`.

## Tests

```bash
# CPU structural probe (small random model, 8 blocks, MoE in blocks 1/3/5, 7 experts, top-3); prints RESULT: PASS or FAIL
VIDEOX_ATTENTION_TYPE=SDPA CUDA_VISIBLE_DEVICES= python scripts/moe/videoeditor-14b/probe_moe_v3_1_cpu.py

# single-GPU probe of one full-size MoE self-attention layer (random weights); prints RESULT: PASS or FAIL
CUDA_VISIBLE_DEVICES=0 python scripts/moe/videoeditor-14b/probe_moe_v3_1_gpu.py

# CPU build check with real weights (needs experts_v3_1.safetensors; several minutes, up to about 120 GB host memory)
CUDA_VISIBLE_DEVICES= python scripts/moe/videoeditor-14b/check_build_v3_1.py --base_dir $BASE_DIR \
    [--experts_merged <path to experts_v3_1.safetensors>]
```

- `probe_moe_v3_1_cpu.py` checks five properties:
  - with all 7 experts equal to the dense weights, the MoE model reproduces the dense Kiwi-Edit DiT;
  - the padded-head-dimension injection of `log ĝ` matches an explicit attention-bias reference;
  - both routers receive gradients from the main loss even with α = 0, and the experts and shared projections receive
    gradients;
  - the per-layer selection fractions sum to 3;
  - outputs are identical with gradient checkpointing on and off.
- `probe_moe_v3_1_gpu.py` uses the real sizes (dim 3072, 24 heads, 5,070 tokens, batch 4). It compares the FlashAttention path
  with the padded prior against an fp32 explicit-bias reference, and checks that experts equal to the base reproduce the
  original `WanSelfAttention`. It also reports forward + backward time and peak memory against the dense layer.
- `check_build_v3_1.py` runs `build_dit` and `mark_trainable` on the real base weights and the merged expert file. It prints
  the parameter groups and a few sanity checks, and ends with `CHECK_BUILD_DONE`.
