# videoeditor-14b: design notes

This note describes the architecture of **videoeditor-14b** (source files use the prefix `moe_v3_1`) and the reasoning behind its main design choices. Usage (inference, training launchers, released weights) is covered in `README.md` in this directory.

videoeditor-14b is derived from **mlp-moe** (`scripts/moe/mlp-moe/`) with two main changes:

1. **The experts are seven distinct, complete weight sets** taken from differently trained Kiwi-Edit variants, including the unmodified base as expert 0, instead of noise-perturbed duplicates.
2. **The self-attention key/value projections are mixture-of-experts layers**, in addition to the FFN.

As in mlp-moe there is no shared expert (Mixtral-style routing). Everything else (full-parameter training of the DiT, three learning-rate groups, DeepSpeed ZeRO-3, logging, checkpoint format) follows mlp-moe.

## 1. Architecture at a glance

The DiT is the Kiwi-Edit transformer (Wan2.2 5B layout: 30 blocks, hidden size 3072, 24 heads of dimension 128, FFN width 14336). In the 14 blocks with 0-based indices 1, 3, ..., 27 (every second block), two sub-modules become 7-expert, top-3 MoE layers:

- the **FFN**, and
- the **key and value projections of self-attention**.

The query and output projections, the q/k RMS norms, cross-attention, layer norms and modulation, the other 16 blocks, the embeddings and the head each exist once and are shared by all tokens.

There is no shared expert. Every expert is a full copy of the replaced Kiwi-Edit projections, and each MoE output is the normalized, gate-weighted combination of the three selected experts.

## 2. Expert initialization

| Index | Name (`EXPERT_NAMES`) | Initial weights |
|---|---|---|
| 0 | `main` | Kiwi-Edit base weights, unchanged. |
| 1-5 | `style`, `add`, `remove`, `replace`, `background` | Kiwi-Edit base with the corresponding per-category LoRA expert merged in: `W + B·A` (rank 128, alpha/r = 1); biases from the base. The LoRA experts cover self-attention q/k/v/o, cross-attention q/k/v/o and `ffn.0`/`ffn.2` in all 30 blocks; only `self_attn.k`, `self_attn.v`, `ffn.0` and `ffn.2` of the 14 MoE blocks are merged. |
| 6 | `direct_finetune` | A full fine-tune of Kiwi-Edit (one epoch, 3,581 steps) on a mix of 75% GOKU composite multi-step video edits and 25% Pico-Banana multi-turn image edits; only its `self_attn.k/v` and `ffn.0/2` weights are used. |

`merge_experts_v3_1.py` builds experts 1-6 once and writes `experts_v3_1.safetensors` (keys `blocks.{b}.experts.{i}.{self_attn.k|self_attn.v|ffn.0|ffn.2}.{weight|bias}`, i = 1..6, bf16; 672 tensors, about 16.7 GB). Expert 0 is copied from the base checkpoint by `build_dit` at model-build time. These files only define the training starting point; they are not part of the videoeditor-14b release, and inference with the released checkpoint does not need them, since the checkpoint contains every expert.

**Why no noise-perturbed duplicates.** mlp-moe uses two copies of each of its six base experts, the second with a 1% random perturbation. Here the seven experts already differ (by a few percent in relative weight norm at step 0) and the gates are randomly initialized, so routing can tell them apart without duplication. Top-k weight renormalization, the balance loss and a high initial gate learning rate (1e-3) drive the separation.

## 3. Forward pass

### 3.1 MoE FFN

The same as in mlp-moe:

```
scores = softmax(gate_ffn(u))                   # [T, 7], fp32
w, idx = topk(scores, 3);  w = w / w.sum(-1)    # renormalize within the top-3
y = sum_{i in idx} w_i * expert_i(u)            # each expert runs only on the tokens that selected it
```

Tokens are grouped by expert with one stable sort (`dispatch_by_expert`, a single host-device synchronization per module). Each expert processes its gathered tokens, and the weighted outputs are accumulated in fp32 with `index_add_` and cast back to bf16 once, which gives the same number of roundings as the dense FFN.

**Why there is no shared expert.** A shared expert in residual form, `shared + Σ_i ĝ_i · (e_i - shared)`, cancels algebraically when Σ_i ĝ_i = 1: it neither affects the output nor receives a gradient, and only costs parameters and memory. The plain Mixtral form is used instead, and the unmodified base weights take part in routing as expert 0.

### 3.2 MoE self-attention: expert keys/values, shared query and output

Each token t is routed by the self-attention gate to three experts and produces three (key, value) pairs (k_t^i, v_t^i); the key and value of a pair always come from the same expert. All pairs are placed in one key/value sequence of length 3L (interleaved by token: position 3t + j holds the j-th selected expert of token t). The single shared query attends to all of them in one attention call, and the softmax is normalized jointly over all 3L keys. The gate weight acts as a multiplicative prior:

```
score(q_s, k_t^i) = q_s · k_t^i / sqrt(d) + log ĝ_{t,i}
y_s = Σ_t Σ_{i in top3(t)} ĝ_{t,i} · exp(q_s · k_t^i / sqrt(d)) · v_t^i  /  Z_s
```

where Z_s is the same double sum without v_t^i.

Properties and implementation details:

- **Exact at initialization.** If every expert equals the base weights, then k_t^i = k_t and v_t^i = v_t for all copies and Σ_i ĝ_{t,i} = 1, so numerator and denominator collapse to the base attention and the MoE model reproduces the base model exactly.
- **The gate is trained by the main loss.** Because log ĝ enters the attention logits, the gate receives gradients from the denoising loss directly, not only from the balance loss.
- **Normalization and RoPE per key copy.** Every expert key passes through the shared `norm_k`, and RoPE is applied to each key copy with the 3D position of its own token, as for the dense key; the positional relation between a query and any key copy is unchanged.
- **Injecting log ĝ into FlashAttention.** FlashAttention does not accept an additive bias, so the head dimension is padded by 8 (128 -> 136; FlashAttention requires a multiple of 8). The query is padded with `[sqrt(d), sqrt(d), 0, ...]`, the key with `[hi, lo, 0, ...]`, where `hi = bf16(log ĝ)` and `lo = log ĝ - hi` (together about 2^-16 relative precision in bf16), and the value with zeros. Attention is called with an explicit `softmax_scale = 1/sqrt(128)`, so the padded channels add exactly `hi + lo = log ĝ` to each logit. The padded output channels are dropped.
- **Kernel cost of the padding.** FlashAttention 2 (2.8.3) has no head-dim-160 kernel, so head dim 136 runs on the 192 template, about 1.5x the cost of the 128 kernel. The attention backend is selected by the environment variable `MOE_ATTN_BACKEND` (`fa2` default, `fa3`, `cudnn`, `auto`); the multi-node launcher `train_multinode.sh` sets `fa3` and falls back to `fa2` when FlashAttention 3 (`flash_attn_interface`) cannot be imported.
- **No shared key/value copy (Mixtral form).** The base Kiwi-Edit projection takes part in routing as expert 0. `MoEKVSelfAttention(shared_kv=True)` adds back one shared (k, v) pair per token with prior 1 (log 1 = 0). With the default `shared_kv=False`, the inherited `k`/`v` projections are deleted; otherwise they would be about 264M trainable parameters that never enter the forward pass.
- **Every expert runs every step.** ZeRO-3 requires every rank to call every expert's forward, so experts that receive no tokens are called on an empty tensor (in both the attention and the FFN MoE).
- **Cross-attention is not routed**, because every token needs to read the complete instruction.
- **Variable-length batches.** Valid keys of each sample are contiguous at the front, and `k_lens = 3 · seq_lens` is passed to FlashAttention; when all samples have the same length, `k_lens` is `None`, which avoids two extra copies of K and V.
- On CPU, or with `VIDEOX_ATTENTION_TYPE=SDPA`, the same computation runs through PyTorch SDPA with the explicit scale (equal-length sequences only).

### 3.3 Gates

- Each MoE block has two independent gates. The **self-attention gate** reads the self-attention input (after `norm1` and modulation); the **FFN gate** reads the FFN input (after cross-attention, `norm2` and modulation).
- Each gate is an `Fp32Gate`: a bias-free linear map 3072 -> 7, initialized from N(0, 0.02^2), with logits computed in fp32. It is written as its own module (instead of reading its weight inside another module's forward) because ZeRO-3 only gathers a module's partitioned parameters while that module's forward is running.
- Routing: softmax over the 7 experts, top-3, renormalization of the three weights to sum to 1. Apart from the top-k renormalization, the N(0, 0.02^2) initialization and fp32 scoring, this is the DeepSeek-MoE `MoEGate` formulation.
- Each gate has its own DeepSeek expert-level balance loss, computed per sample: `aux = α · mean_b Σ_i f_i · P_i`, with `f_i = N · count_i / (k · L)` and `P_i` the mean softmax score of expert i over the sample's tokens (aux = α when perfectly balanced). The loss is attached to the layer output by an autograd function (`AddAuxiliaryLoss`), so a single backward pass on the main loss also delivers the balance-loss gradients to the gates, including under gradient checkpointing.

## 4. Parameter count and cost

Sequence length per sample: 49 frames at 480x832 give a 13x30x52 latent; patchify (1, 2, 2) gives 13x15x26 = **5,070 tokens**.

| Component | Parameters |
|---|---|
| FFN experts | 7 x 88.1M x 14 blocks = 8.63B |
| Self-attention k/v experts | 7 x 18.9M x 14 blocks = 1.85B |
| Gates | 2 x 14 x (7 x 3072) = 0.6M |
| All other (shared) parameters | 3.50B |
| **Total (full DiT, all trainable)** | **13.99B** |

Per token, an MoE block evaluates three FFN experts and three k/v projection pairs, and self-attention attends over 3L keys: roughly 3x the attention FLOPs, times about 1.5 for the head-dim-192 template under FlashAttention 2. At 5,070 tokens the attention term is small; the estimated overhead of the MoE self-attention is about 0.3 s per sample per training step and about 1 GB of additional activation memory per sample with gradient checkpointing.

Most of the step time is outside the DiT. With per-sample processing, the frozen VAE (source and target clips) and MLLM encoders take roughly 2 s of the about 3 s spent per sample, while the DiT accounts for about 1 s by FLOP count. The training script therefore encodes VAE inputs in groups (`--vae_encode_group`) and MLLM inputs as a batch (`--mllm_batch_encode 1`), can cache multi-instruction sample features (`--multi_cache_dir`), and logs `time/vae`, `time/mllm` and `time/dit` to monitor the split.

## 5. Training configuration

Script defaults of `moe_v3_1_train.py` and the values set in `train_multinode.sh`. The `meta.json` written next to each checkpoint records the MoE and learning-rate settings it was trained with.

- **Data schedule** `--data_schedule mixed` (default): OpenVE single-instruction pairs once, plus the GOKU and CoinVE multi-instruction data each repeated `--multi_repeat` times, shuffled into one pool and trained for `--stage_a_epochs` epochs. The script default is `--multi_repeat 2`; the launcher sets 8, which makes the multi-instruction data about as large as OpenVE. The sub-instructions of CoinVE samples, which have no native combined instruction, are joined into one instruction with `--compound_joiner " Then "`. `--data_schedule two_stage` (OpenVE for `--stage_a_epochs`, then the multi-instruction data for `--stage_b_epochs`) is kept as an alternative.
- **Batch**: 8 per GPU on 3 nodes x 8 GPUs (effective batch 192) in the launcher; the script default `--batch_size` is 4.
- **Learning rates** (three parameter groups built by `mark_trainable`):
  - base (all pre-existing parameters): `--learning_rate 1e-5`, constant;
  - experts (`k_experts`, `v_experts`, `ffn.experts`): `--new_layers_lr 1e-4`, linearly decayed to `--learning_rate` over `--new_layers_lr_steps 2000`, then constant;
  - gates: `--gate_lr 1e-3`, linearly to `--gate_lr_end 1e-4` over the same number of steps, then constant.
- **Balance-loss weight** `--moe_aux_loss_alpha`: script default 1e-3, launcher 1e-4 (see Section 6).
- **MoE layout**: `--moe_layers 1,3,...,27`, `--moe_top_k 3`, `--moe_num_experts 7`. The expert count cannot be changed without changing `EXPERT_NAMES`, since expert initialization is tied to it.
- **Other**: `--motion_sub_loss_ratio 0.1`, `--text_drop_prob 0.1` (instruction replaced by the empty instruction for text classifier-free guidance), `--weight_decay 0.01`.
- **Checkpoints**: `ckpt/step-N/moe_v3_1_trainable.safetensors` (the full DiT in bf16) plus `meta.json` every `--save_steps`; the full DeepSpeed state in `ckpt/ds_state_step-N/` every `--ds_save_steps`, used with `--resume_ds_state` (requires the same world size). `--resume_weights` loads only the weights and restarts the optimizer.

Routing statistics logged every 10 steps:

- `moe_attn_L{block}/expert{i}` and `moe_ffn_L{block}/expert{i}`: fraction of tokens that select expert i in that MoE block (sums to 3 over the experts of a block);
- `loss/aux_attn` and `loss/aux_ffn`: balance losses of the two gate types;
- `route_{attn|ffn}/{category}/expert{i}`: P(expert i | sample category), i.e. the share of that category's top-3 selections that go to expert i, and `route/{attn|ffn}_spread`: the per-expert maximum difference of this probability across categories, averaged over experts (0 means no specialization). Batch-averaged selection fractions cannot show specialization; the category-conditioned matrix can.

## 6. Load balancing versus category specialization

The balance loss is computed per sample, and one sample is one video that usually belongs to one edit category. The loss is minimal (aux = α) when every sample spreads its tokens evenly over all 7 experts. The routing one would hope for, where each sample uses {base, direct fine-tune, its own category expert}, is exactly its maximum under top-3 routing (aux = 7/3 · α ≈ 2.33 α). The loss therefore works against category specialization; with α = 1e-2 and 1e-3 it pulled routing to the per-sample uniform point on the related mlp-moe model. The launcher therefore uses α = 1e-4, the small sequence-wise balance weight also used by DeepSeek-V3, which keeps a guard against dead experts while weakening the pull toward uniform routing.

Two alternatives were considered and not implemented:

- (a) compute the balance term over all tokens of the batch instead of per sample, and only push experts whose load falls below a floor τ (a one-sided "keep alive" term with zero gradient while every expert is alive);
- (b) DeepSeek-V3 auxiliary-loss-free balancing: a per-expert bias that affects only the top-k selection (not ĝ) and is updated by ±γ according to load. Its symmetric target (load k/N for every expert) conflicts with a routing in which the base and fine-tuned experts are always active while category experts follow the data proportions, so it would also need a one-sided variant.

**Weak routing signal.** The category experts are close to each other and to the base: on samples of their own category, a category expert's loss advantage is estimated at only 0-4%. This places the routing signal near the boundary between learnable and suppressed by the balance loss. Two diagnostics make this visible: a paired evaluation before training (the same samples and noise, with the whole network switched to expert i, giving a 7 x 5 matrix of relative loss differences over experts and categories), and the category-conditioned selection matrix logged during training (`route_*`, Section 5).

## 7. Alternatives considered

- **Output-level mixing for attention**: a separate softmax per expert copy followed by ĝ-weighted mixing of the outputs. The implemented form uses one shared softmax denominator instead: it needs a single attention call, reduces exactly to the base model when the experts coincide, and lets ĝ act as a prior inside the softmax.
- **A shared (k, v) copy per token**: available through `shared_kv=True`, not used (Mixtral form, base weights act as expert 0).
- **One routing decision per token per block**, with the FFN reusing the self-attention gate's selection: half the gate parameters and easier to interpret, but the FFN input has already passed through attention and carries more information, so the two gates are kept independent.
- **top-2 instead of top-3**: not used; both the attention and FFN MoE use top-3, and memory limits are handled by the batch size.
- **Low-value options not implemented**: gate temperature (does not change the top-k selection), noisy top-k, expert-choice routing, grouped GEMM for the experts (at most about 1% of step time), and gating only the values (about 3% saving at 5,070 tokens, while changing the attention formulation).

## 8. Code layout

All files in this directory are self-contained: the DiT is a complete copy of the Kiwi-Edit DiT with the MoE changes marked `MOE-V3.1`.

| File | Contents |
|---|---|
| `wan_transformer3d_kiwi_moe_v3_1.py` | The DiT. `route_tokens`, `expert_balance_loss`, `Fp32Gate`, `MoEFFN` (no shared expert), `MoEKVSelfAttention` (expert k/v, 3L key sequence, log ĝ injection), `attention_with_scale`; the `moe` argument of the block; the model's MoE configuration and `moe_attns()` / `moe_ffns()`. |
| `merge_experts_v3_1.py` | One-off: merges the five category LoRA experts and extracts the direct fine-tune's k/v/FFN weights into `experts_v3_1.safetensors` (experts 1-6). |
| `moe_v3_1_model.py` | Assembly: `build_dit` (loads the 7 experts and self-checks), `mark_trainable` (base / experts / gates), `TextEncoder`, `build_mllm`, `build_vae`, `load_trainable_weights`. |
| `moe_v3_1_train.py` | Training loop (as in mlp-moe), with 7 experts, two gate types and their logging and balance losses. |
| `moe_v3_1_infer.py` | Inference; additionally saves separate attention and FFN expert maps per case. |
| `train_multinode.sh` / `train_smoke.sh` | Launchers (multi-node / single-node smoke test). |
| `probe_moe_v3_1_cpu.py`, `probe_moe_v3_1_gpu.py`, `check_build_v3_1.py` | Verification (Section 9). |

When `build_dit` loads the base checkpoint, `from_pretrained` prints `Size don't match, skip` for the base `blocks.N.self_attn.{k,v}.*` and `blocks.N.ffn.{0,2}.*` keys of each MoE block and lists the expert and gate keys as missing. This is expected: those modules do not exist in the MoE structure, and `build_dit` copies the base weights into expert 0 explicitly.

## 9. Verification

1. **CPU equivalence probe**, `probe_moe_v3_1_cpu.py` (small configuration: 8 blocks, dim 64, 2 heads; MoE in blocks 1, 3 and 5 with 7 experts, top-3):
   - A. with all experts equal to the base weights, the MoE model output matches the dense Kiwi-Edit model (both the FFN and the attention path, including RoPE) to below 1e-5;
   - B. the padded-head-dim log ĝ injection matches a reference that uses an explicit additive attention bias in SDPA;
   - C. with α = 0, both gate types receive gradients from the main loss (the log ĝ path is connected), and the experts and shared q/o receive gradients;
   - D. in every layer, the selected fractions of the attention and of the FFN experts each sum to 3;
   - E. outputs are identical with and without gradient checkpointing.

   ```bash
   VIDEOX_ATTENTION_TYPE=SDPA CUDA_VISIBLE_DEVICES= python scripts/moe/videoeditor-14b/probe_moe_v3_1_cpu.py   # prints RESULT: PASS
   ```
2. **GPU attention probe**, `probe_moe_v3_1_gpu.py` (one GPU, real sizes: dim 3072, 24 heads, 5,070 tokens, random weights): the FlashAttention path with padding and explicit `softmax_scale` against an fp32 SDPA reference with an explicit bias; agreement with the dense `WanSelfAttention` when the experts equal the base (bf16-level differences); forward + backward time and peak memory compared with the dense layer.

   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/moe/videoeditor-14b/probe_moe_v3_1_gpu.py
   ```
3. **Build check**, `check_build_v3_1.py` (CPU, real weights, no forward pass): runs `build_dit` and `mark_trainable` and prints key and parameter counts plus a few self-checks. It needs the merged-expert file, which is not released.

   ```bash
   CUDA_VISIBLE_DEVICES= python scripts/moe/videoeditor-14b/check_build_v3_1.py --base_dir <BASE_DIR> --experts_merged <experts_v3_1.safetensors>
   ```
4. **Smoke test**, `train_smoke.sh`: single node with 8 GPUs, 6 steps, covering weight saving, DeepSpeed state saving and resuming.
