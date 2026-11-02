# Video-Editor-MoE

Instruction-based video editing with mixture-of-experts diffusion transformers.

All models start from **[Kiwi-Edit](https://github.com/showlab/Kiwi-Edit)** (5B, instruct-only; a Wan2.2-TI2V-5B video DiT conditioned
on a Qwen2.5-VL-based MLLM encoder) and add experts that specialise the DiT for different kinds of edits. This repository contains the
training and inference code for four models:

| Model | Code | Idea | DiT parameters | Weights |
|---|---|---|---|---|
| **explicit-moe** | [`scripts/moe/explicit-moe`](scripts/moe/explicit-moe) | frozen Kiwi-Edit DiT + 5 explicit task experts (LoRA, one per edit category); a router head picks the expert of each sub-instruction and 14 trainable fusion layers merge the expert streams | 5.0B frozen + 2.1B | [Massyzs/explicit-moe](https://huggingface.co/Massyzs/explicit-moe) |
| **mlp-moe** | [`scripts/moe/mlp-moe`](scripts/moe/mlp-moe) | FFN of 14 DiT blocks replaced by a 12-expert top-3 MoE, trained end to end | 18.6B | [Massyzs/mlp-moe](https://huggingface.co/Massyzs/mlp-moe) |
| **videoeditor-14b** | [`scripts/moe/videoeditor-14b`](scripts/moe/videoeditor-14b) | FFN **and** self-attention key/value of 14 blocks as 7-expert top-3 MoEs | 14.0B (~8B active) | [Massyzs/videoeditor-14b](https://huggingface.co/Massyzs/videoeditor-14b) |
| **videoeditor-56b** | [`scripts/moe/videoeditor-56b`](scripts/moe/videoeditor-56b) | videoeditor-14b scaled to 35 experts per MoE layer (7 sources x 5 noised copies) | 55.9B (~8B active) | [Massyzs/videoeditor-56b](https://huggingface.co/Massyzs/videoeditor-56b) |
| **videoeditor-14b-long** | [`scripts/moe/videoeditor-14b`](scripts/moe/videoeditor-14b) | videoeditor-14b continued on the same data re-cut to 121 / 81 frames per sample (half an epoch) | 14.0B (~8B active) | [Massyzs/videoeditor-14b-long](https://huggingface.co/Massyzs/videoeditor-14b-long) |
| **videoeditor-56b-long** | [`scripts/moe/videoeditor-56b`](scripts/moe/videoeditor-56b) | videoeditor-56b continued on the same data re-cut to 121 / 81 frames per sample | 55.9B (~8B active) | [Massyzs/videoeditor-56b-long](https://huggingface.co/Massyzs/videoeditor-56b-long) |

The two `-long` variants use the same code and inference scripts as their base models; pass `--num_frames 121` (or 81) for long inputs.

The frozen components shared by all models (DiT base, MLLM encoder, query tokens/connector, VAE) are in
[Massyzs/kiwi-edit-5b-instruct-only-videoxfun](https://huggingface.co/Massyzs/kiwi-edit-5b-instruct-only-videoxfun).
All weight repositories are gated: request access on the model page.

## Installation

```bash
conda create -n videoeditor python=3.10 -y && conda activate videoeditor
pip install -r requirements.txt
# optional: pip install flash-attn --no-build-isolation
```

Tested on NVIDIA H100 80GB with CUDA 12.8.

## Weights

Choose a working directory `BASE_DIR` and download the shared base components:

```bash
huggingface-cli login     # after your access request has been approved
huggingface-cli download Massyzs/kiwi-edit-5b-instruct-only-videoxfun \
    --local-dir $BASE_DIR/ckpt/kiwi-edit-instruct-only-videoxfun
```

Each model repository ships its weights as a tar.gz stream split into 20 GB chunks (optimizer state is not included). After extraction the layout under `BASE_DIR` is:

```
$BASE_DIR/ckpt/kiwi-edit-instruct-only-videoxfun/   shared base components (DiT base, VAE, qwen/ MLLM encoder, moe_expert_init.safetensors)
$BASE_DIR/ckpt/<model>/                              one folder per downloaded model, e.g. videoeditor-14b/, videoeditor-14b-long/, explicit-moe/
```

For example, videoeditor-14b (replace the name for any other model, e.g. `videoeditor-14b-long`):

```bash
huggingface-cli download Massyzs/videoeditor-14b --local-dir $BASE_DIR/hf/videoeditor-14b
cd $BASE_DIR/hf/videoeditor-14b
cat videoeditor-14b.tar.gz.part-* | tar -xzf - -C $BASE_DIR/ckpt/     # -> $BASE_DIR/ckpt/videoeditor-14b/
cp SHA256SUMS.contents $BASE_DIR/ckpt/ && (cd $BASE_DIR/ckpt && sha256sum -c SHA256SUMS.contents)   # optional check
```

## Inference

Every model edits one source video from a text instruction. Inputs: `--src_video` is any mp4 (the first frames are used, resized/cropped to
480x832; 49 frames by default, `--num_frames 121` or `81` for the `-long` models), `--prompt` is the instruction (or `--prompts "step 1 ||| step 2"`
for a multi-step edit). Outputs go to `--out_dir`: `<name>.mp4` (16 fps) and lossless PNG frames in `frames/<name>/`. Run from the repository root:

```bash
python scripts/moe/videoeditor-14b/moe_v3_1_infer.py \
    --base_dir $BASE_DIR --ckpt $BASE_DIR/ckpt/videoeditor-14b \
    --src_video input.mp4 --prompt "Replace the red car with a blue truck" \
    --name edited --out_dir outputs/
# multi-step instruction: --prompts "Remove the dog ||| Convert to oil painting style"
```

| Model | Inference script | GPUs |
|---|---|---|
| explicit-moe | `scripts/moe/explicit-moe/moe_v2_infer.py` | 1 x 80 GB |
| mlp-moe | `scripts/moe/mlp-moe/moe_v3_infer.py` | 1 x 80 GB |
| videoeditor-14b | `scripts/moe/videoeditor-14b/moe_v3_1_infer.py` | 1 x 80 GB |
| videoeditor-56b | `scripts/moe/videoeditor-56b/moe_v3_2_infer.py --gpus 2` | 2 x 80 GB (the DiT is split across both) |
| videoeditor-14b-long | `scripts/moe/videoeditor-14b/moe_v3_1_infer.py --num_frames 121` | 1 x 80 GB |
| videoeditor-56b-long | `scripts/moe/videoeditor-56b/moe_v3_2_infer.py --gpus 2 --num_frames 121` | 2 x 80 GB |

Defaults: 50 sampling steps, sigma shift 5, text classifier-free guidance with scale 3 (applied on the first 60% of the steps for
mlp-moe / videoeditor-14b / videoeditor-56b, on all steps for explicit-moe). The output is `<out_dir>/<name>.mp4` plus lossless
PNG frames; the MoE models also write per-layer expert-routing maps. Each script also has a batch mode over a JSON case list.
See the README in each model directory for all options.

## Training

Training data formats are described in [`docs/DATA.md`](docs/DATA.md). Each model directory has a multi-node launcher
(`train_multinode.sh`) and a single-node smoke test (`train_smoke.sh`). Edit the configuration block at the top of the launcher
(`BASE_DIR`, `DATASET_BASE`, `MASTER_ADDR`, `NUM_NODES`, ...) and start it on every node with its node rank:

```bash
bash scripts/moe/videoeditor-14b/train_multinode.sh 0     # on node 0; use 1, 2, ... on the other nodes
```

| Model | Setup used for the released weights | Starting point |
|---|---|---|
| explicit-moe | 2 nodes x 8 GPUs, batch 32 | frozen Kiwi-Edit + the five LoRA experts and the initial router head (both in the explicit-moe package) |
| mlp-moe | 3 nodes x 8 GPUs, batch 192, DeepSpeed ZeRO-3 | merged expert init built by `merge_experts.py` |
| videoeditor-14b | 3 nodes x 8 GPUs, batch 192, DeepSpeed ZeRO-3 | merged expert init built by `merge_experts_v3_1.py` |
| videoeditor-56b | 3 nodes x 8 GPUs, batch 120, DeepSpeed ZeRO-3 with CPU-offloaded optimizer | the videoeditor-14b expert init, 5 noised copies per expert |

Long-video training: dataset records may carry `num_frames_bucket` (121 or 81); set `NUM_FRAMES=121` in the launcher and the trainer builds
frame-count-homogeneous batches (see `docs/DATA.md`). The released long-video weights continue from the released checkpoints with the same launchers.

The merged expert initialisations combine the category LoRA experts (released with explicit-moe) with a direct full fine-tune
of Kiwi-Edit that is not released; the model READMEs explain how to fine-tune from the released checkpoints instead.

## Repository layout

```
config/wan2.2/wan_civitai_5b.yaml   DiT/VAE configuration
videox_fun/                         model, data and distributed utilities (from VideoX-Fun) + Kiwi-Edit modules
videox_fun/moe_common/              MLLM encoder wrapper, datasets and helpers shared by the four models
scripts/moe/<model>/                training, inference and test scripts of each model (see its README)
```

## Acknowledgements and license

This code builds on [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) (Apache-2.0), [Kiwi-Edit](https://github.com/showlab/Kiwi-Edit) (MIT)
and [Wan2.2](https://github.com/Wan-Video/Wan2.2) (Apache-2.0); see `NOTICE`. The code is released under the Apache License 2.0 (`LICENSE`).
