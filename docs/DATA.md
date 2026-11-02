# Training data

This page describes the data the four trainers read: the metadata file format, how video paths are resolved, how a
record becomes a training sample, the data schedules, and the optional multi-edit feature cache of videoeditor-14b and
videoeditor-56b.

The metadata lists and videos used for the released models are **not** distributed with this repository. To train,
obtain the datasets from their original sources and write metadata files in the format below.

| Model | Trainer | Data flags |
|---|---|---|
| explicit-moe | `scripts/moe/explicit-moe/moe_v2_train.py` | `--vid_dataset_metadata_path`, `--dataset_prefixes`, `--dataset_base_path` |
| mlp-moe | `scripts/moe/mlp-moe/moe_v3_train.py` | `--stage_a_data`, `--stage_b_data`, `--stage_a_epochs`, `--stage_b_epochs`, `--exclude_prompt_words`, `--dataset_base_path` |
| videoeditor-14b | `scripts/moe/videoeditor-14b/moe_v3_1_train.py` | as mlp-moe, plus `--data_schedule`, `--multi_repeat` and the feature-cache flags (Section 5) |
| videoeditor-56b | `scripts/moe/videoeditor-56b/moe_v3_2_train.py` | as videoeditor-14b, plus `--mllm_worker_pre` |

All four trainers also take `--num_frames` (default 49), `--height` (480), `--width` (832), `--max_pixels` (399360),
`--dataset_num_workers` (4), `--dataloader_prefetch_factor` (2) and `--compound_joiner` (`" Then "`). The launchers
(`train_multinode.sh`, `train_smoke.sh` in each model directory) set the data flags from variables in their
configuration block, for example `DATASET_BASE`, `STAGE_A_DATA`, `NUM_FRAMES`.

The loading code is shared by all four models:
[`videox_fun/moe_common/kiwi_dataset_moe.py`](../videox_fun/moe_common/kiwi_dataset_moe.py) (`KiwiEditDataset`: video
decoding, frame selection, resize and crop, retries) and
[`videox_fun/moe_common/kiwi_dataset_multi_moe.py`](../videox_fun/moe_common/kiwi_dataset_multi_moe.py)
(`KiwiMultiEditDataset`: instruction lists, category labels, feature cache). Each trainer first normalises its metadata
files into records with one schema (`load_records` in each `*_train.py`), writes them to a temporary JSON Lines file
and builds a `KiwiMultiEditDataset` from it.

## 1. Metadata files

A metadata file lists (source video, target video, instruction) records.

* A file whose name ends in `.json` must contain a single JSON array of objects.
* Any other file is read as JSON Lines: one JSON object per line, blank lines ignored. Use the `.jsonl` extension.
  CSV files are not accepted by the trainers.
* Several files are passed as one comma-separated argument; their records are concatenated.

### Fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `src_video` | string | yes | Source (unedited) video; relative or absolute path (Section 2). |
| `tgt_video` | string | yes | Edited video: the source after applying every instruction of the record. |
| `prompt` | string or list of strings | yes | The instruction. A list holds sub-instructions applied in order; the target shows the result of all of them. |
| `turn_types_qwen` | list of strings | no | One edit category per sub-instruction: `style`, `add`, `remove`, `replace` or `background`. |
| `cls` | string | no | Edit category of a single-instruction record (same five values). |
| `cat` | string | no | OpenVE-3M category name: `local_add`, `local_remove`, `local_change`, `global_style` or `background_change`. |

All other fields are ignored (for example `n_turns` or mask paths). The category fields are optional for mlp-moe,
videoeditor-14b and videoeditor-56b, which only use them for logging, but they select the experts in explicit-moe
(Section 3.4).

### Examples

The following records are synthetic. Single-instruction records in a `.json` file:

```json
[
  {"src_video": "openve/local_add/000001_src.mp4",
   "tgt_video": "openve/local_add/000001_tgt.mp4",
   "prompt": "Add a red umbrella to the woman's left hand.",
   "cat": "local_add"},
  {"src_video": "reco/replace/src_videos/000002.mp4",
   "tgt_video": "reco/replace/tar_videos/000002.mp4",
   "prompt": "Replace the wooden chair with a blue armchair.",
   "cls": "replace"}
]
```

Multi-instruction records in a `.jsonl` file, one object per line:

```json
{"src_video": "goku/videos/case_0000001/source.mp4", "tgt_video": "goku/videos/case_0000001/step2.mp4", "prompt": ["Remove the kite from the sky", "Add a flock of seagulls to the scene"], "turn_types_qwen": ["remove", "add"]}
{"src_video": "coinve/src/clip_000001.mp4", "tgt_video": "coinve/tgt/clip_000001.mp4", "prompt": ["Replace the white takeout box with a cardboard burger box.", "Add a metal fork on top of the fries.", "Replace the sidewalk with a wooden table surface."], "turn_types_qwen": ["replace", "add", "background"]}
```

The `videos/case_<digits>/` part of the first path is significant: it selects the GOKU instruction phrasing in
Section 3.3.

## 2. Video paths

Relative paths are joined with the dataset base path using Python's `os.path.join`; absolute paths are used unchanged.

* **mlp-moe, videoeditor-14b, videoeditor-56b:** `--dataset_base_path` (default `""`, meaning paths are used as
  written). The launchers set it from `DATASET_BASE`.
* **explicit-moe:** `--dataset_base_path` is required (launcher variable `DATASET_BASE`). In addition,
  `--dataset_prefixes` gives one prefix per metadata file (comma-separated, in the order of
  `--vid_dataset_metadata_path`; an empty entry means no prefix). The prefix is applied to relative paths first, so a
  relative path resolves to `<dataset_base_path>/<prefix>/<path>`. The launcher sets `DATA_PREFIX=",,"`, meaning no
  prefixes.

For example, with `DATASET_BASE=/data/videos` the first record above is read from
`/data/videos/openve/local_add/000001_src.mp4` and `/data/videos/openve/local_add/000001_tgt.mp4`.

The file type is taken from the extension (case-insensitive): videos `mp4`, `avi`, `mov`, `wmv`, `mkv`, `flv`,
`webm` are decoded with imageio (FFmpeg backend); images `jpg`, `jpeg`, `png`, `webp`, `bmp` are read as a single
frame. Any other extension is an error for that record (Section 6).

## 3. From record to training sample

### 3.1 Frames

Source and target are decoded independently:

1. The first N = `--num_frames` frames (default 49) are read in order, starting at frame 0, at the file's own frame
   rate. There is no temporal stride, frame-rate resampling or random start offset. A shorter file yields all its frames.
2. The frame count is trimmed to the largest value <= N of the form 4k+1 (the Wan2.2 VAE compresses time by 4).
3. Source and target must now have the same number of frames; otherwise the record is rejected (Section 6).
4. If fewer than N frames remain, both clips are padded to exactly N frames by repeating their last frame. All four
   trainers enable this, so every sample has N frames.

Use N = 4k+1; N = 49 gives 13 latent frames. For example, a 30-frame source/target pair gives frames 0-28 followed by 20
copies of frame 28. An image file yields one frame and therefore becomes a static clip of N identical frames. The lists
used for the released models contain only MP4 video pairs.

### 3.2 Resolution

Each frame is resized with bilinear interpolation, keeping its aspect ratio, by the smallest factor at which it covers
`--height` x `--width` (default 480 x 832, landscape). It is then center-cropped to exactly that size. Height and width
are rounded down to multiples of 32. Source and target are resized separately, so their native resolutions may differ.
There is no letterboxing: a portrait clip keeps only a central horizontal band. Because the trainers always pass a
fixed height and width, `--max_pixels` does not affect training.

The VAE and the DiT see all N frames at this resolution. The target is used only as the VAE latent target. The MLLM
instruction encoder sees 16 frames of the processed source clip, sampled uniformly and including the first and last
frame, each resized to at most 262,144 (512 x 512) pixels. If the first and last of these frames are identical (for
example an image padded to a static clip), only one frame is passed.

### 3.3 Instructions

**explicit-moe** keeps the sub-instructions separate. The MLLM encodes each sub-instruction on its own, and that
encoding conditions the sub-instruction's LoRA expert. The frozen base stream receives the full instruction: the
sub-instructions joined with `--compound_joiner` (default `" Then "`).

**mlp-moe, videoeditor-14b, videoeditor-56b** turn every record into one full instruction:

| `prompt` | Full instruction |
|---|---|
| string | used as is |
| list, and `src_video` contains `videos/case_<digits>/` | GOKU phrasing: `A, then b` for two steps, `A, then b, then c, and finally d` for more. Trailing periods are removed and every step after the first starts with a lower-case letter. |
| any other list | the sub-instructions joined verbatim with `--compound_joiner` (default `" Then "`) |

For the two multi-instruction examples above, the full instructions are
`Remove the kite from the sky, then add a flock of seagulls to the scene` and
`Replace the white takeout box with a cardboard burger box. Then Add a metal fork on top of the fries. Then Replace the sidewalk with a wooden table surface.`
Sub-instructions are joined verbatim, so end each one with a period if you want sentence breaks.

These three trainers also drop every record whose full instruction contains one of the comma-separated
`--exclude_prompt_words`. Matching is a case-insensitive substring test. The default is `overlay`; the launchers pass
`overlay,subtitle`. explicit-moe has no such filter.

For text classifier-free guidance, each sample's instruction is replaced by the empty string with probability
`--text_drop_prob` (mlp-moe, videoeditor-14b, videoeditor-56b; default 0.1). In explicit-moe the flag is
`--base_text_drop_prob` (default 0.1) and it blanks only the base stream; the experts keep their sub-instructions. The
MLLM still sees the source frames.

### 3.4 Category labels

**mlp-moe, videoeditor-14b, videoeditor-56b** use categories for logging only. The statistics
`route_ffn/<category>/expert<i>` (and `route_attn/...` for the two videoeditor models, whose self-attention key/value
projections are also MoE layers) give how often expert i is selected for samples of each category. Labels never enter
the model, so unlabelled data trains normally and is logged as `unknown`. Each record gets one category; the first rule that
applies wins:

1. `cls`, if it is one of `add`, `remove`, `replace`, `style`, `background` (`multi` and `unknown` are also accepted);
2. `cat`, mapped `local_add` -> `add`, `local_remove` -> `remove`, `local_change` -> `replace`,
   `global_style` -> `style`, `background_change` -> `background`;
3. `turn_types_qwen`: the shared label if all sub-instructions have the same one, otherwise `multi`;
4. `unknown`.

**explicit-moe** uses the labels to choose experts. A sub-instruction's label is `turn_types_qwen[j]`, or `cls` for a
record without `turn_types_qwen` (`cat` is not read). Valid labels are the five expert names `style`, `add`, `remove`,
`replace`, `background`.

* During training, a sub-instruction with a valid label is sent to that label's LoRA expert, and the label is the
  cross-entropy target of the router head. A sub-instruction without a valid label goes to the expert that the router
  head predicts.
* A batch contains only records with the same multiset of labels, because the experts' streams must line up across the
  batch. Records are grouped by their sorted label list. A group that does not fill a whole batch leaves a smaller
  batch; these partial batches run at the end of each epoch, and two of them are accumulated into one optimizer step.
* Records whose labels contain `i2v` or `camera` are removed by `KiwiMultiEditDataset` (its default `drop_types`). That
  stops explicit-moe with the assertion `dataset filtered out N records`, so remove such records from the lists
  beforehand.

### Masks

None of the four trainers uses masks. Every record is loaded with a plain full-frame loss, and mask fields in a list
are ignored. `KiwiMultiEditDataset` has an optional mask-weighted mode (`mask_loss=True`, which reads `mask_mode`,
`turn_mask_videos` and `combined_mask_video`), but no trainer enables it.

## 4. Data schedules

### explicit-moe

All files in `--vid_dataset_metadata_path` form one dataset, and each record is used once per epoch (a few batches
are repeated so that every GPU gets the same number). Training runs for `--max_train_steps`. Set `--epoch_steps` to
the number of optimizer steps per epoch, so that a resume inside an epoch continues the same data order. The
launcher's `EPOCH_STEPS=13992` matches its three lists (447,710 records) on 16 GPUs with batch size 2; recompute it
when the data or the number of GPUs changes.

### mlp-moe: two stages

Stage A reads `--stage_a_data` for `--stage_a_epochs` epochs (default 1.0). Then stage B reads `--stage_b_data` for
`--stage_b_epochs` epochs (default 2.0), in the same process. Epoch counts may be fractional; a stage's step count is
the steps per epoch times the epochs, rounded up. With `--max_train_steps 0` (the default) both stages are trained
completely. Stage A is meant for single-instruction data and stage B for multi-instruction data, but the code accepts
either kind of record in both stages.

### videoeditor-14b and videoeditor-56b: `--data_schedule`

* `mixed` (default): one dataset containing every `--stage_a_data` record once and every `--stage_b_data` record
  `--multi_repeat` times (default 2), shuffled together and trained for `--stage_a_epochs` epochs. `--stage_b_epochs`
  is ignored. The copies are identical records; what differs between visits is only the sampled noise level, the noise
  and the text dropout.
* `two_stage`: stage A then stage B, as in mlp-moe.

In both schedules, the records from `--stage_b_data` files are the ones that can use the feature cache (Section 5).

### Batching and data order (mlp-moe, videoeditor-14b, videoeditor-56b)

Each epoch shuffles all records with a permutation seeded by `--seed` and the epoch index. With W GPUs, rank r takes
every W-th entry starting at r and splits its share into batches of `--batch_size` (per GPU). Leftover records that do
not fill a batch are dropped, so an epoch has floor(floor(num_records / W) / batch_size) steps. The order depends on the
seed and on W. A resume from a DeepSpeed state with the same W continues at the same position.

### Launcher settings

The multi-node launchers (`train_multinode.sh`) of the four models pass the following lists (paths relative to
`${BASE_DIR}/datasets/`), all with N = 49 frames at 480 x 832:

| Model | Single-instruction lists | Multi-instruction lists | Schedule |
|---|---|---|---|
| explicit-moe | `classified/openve_v2_equal_223855.json` | `classified/CoinVE.jsonl`, `classified/goku_with_masks.jsonl` | one dataset (`DATA`), `DATA_PREFIX=",,"` |
| mlp-moe | `filtered/openve_original_nosub_all.json` (`STAGE_A_DATA`) | `classified/goku_with_masks.jsonl`, `classified/CoinVE.jsonl` (`STAGE_B_DATA`) | two stages, 1 + 1 epochs |
| videoeditor-14b | as mlp-moe | as mlp-moe | `mixed`, `MULTI_REPEAT=8`, `STAGE_A_EPOCHS=1` |
| videoeditor-56b | `filtered/openve_original_nosub_all.json`, `filtered/reco_all.json` | as mlp-moe | `mixed`, `MULTI_REPEAT=10`, `STAGE_A_EPOCHS=2` |

mlp-moe, videoeditor-14b and videoeditor-56b use `EXCLUDE_PROMPT_WORDS=overlay,subtitle`. The lists were built from
the following datasets:

| List | Source | Records | Content |
|---|---|---|---|
| `openve_original_nosub_all.json` | OpenVE-3M | 1,788,804 | all original OpenVE-3M pairs except the `subtitles_edit` category; string `prompt`, `cat` |
| `reco_all.json` | ReCo-Data | 520,660 | the add, remove, replace and style tasks; string `prompt`, `cls` |
| `openve_v2_equal_223855.json` | OpenVE-3M | 223,855 | subset labelled with the five categories (`cls`), sized to match the two multi-instruction lists combined |
| `CoinVE.jsonl` | CoinVE | 200,916 | 2 to 5 sub-instructions per record, `turn_types_qwen` |
| `goku_with_masks.jsonl` | GOKU multi-step editing cases | 22,939 | 2 or 3 sub-instructions per record, only records whose sub-instructions all fall in the five categories; the target is the case's video after exactly these steps; `turn_types_qwen` plus mask fields (unused) |

## 5. Multi-edit feature cache (videoeditor-14b, videoeditor-56b)

In the mixed schedule each multi-instruction record is visited `--multi_repeat` times per epoch. The cache stores the
per-sample encodings the first time a record is processed, so later visits skip video decoding, VAE encoding and MLLM
encoding. It is off by default.

| Flag | Default | Launcher |
|---|---|---|
| `--multi_cache_dir` | `""` (off) | `MULTI_CACHE_DIR=${OUT}/cache_multi` |
| `--cache_keep_existing` | 0: rank 0 clears the directory at start | `MULTI_CACHE_KEEP=1` |
| `--cache_delete_on_done` | 1: rank 0 deletes the directory when training finishes | `MULTI_CACHE_DELETE_ON_DONE=0` |
| `--cache_write_threads` | 4 background writer threads per rank | default |
| `--cache_queue` | 64 pending writes per rank; further writes are dropped, not waited for | default |

With `MULTI_CACHE_KEEP=0` the launchers also delete the directory when they start and when they exit. Put the directory
on storage that all nodes share (the launchers use the output directory).

* **Eligible records:** records from `--stage_b_data` files, in both schedules.
* **Key:** SHA-1 of `<src_video>|<tgt_video>|<full instruction>|<N>x<H>x<W>`. `src_video` and `tgt_video` are the
  strings written in the metadata, before joining with `--dataset_base_path`. Moving the video root therefore keeps the
  cache valid, while changing the frame count or resolution produces new keys. The key does **not** cover the VAE or
  MLLM weights, so clear the directory if you change them.
* **File:** `<multi_cache_dir>/<key>.st`, a safetensors file with six bf16 tensors: `src_mean`, `src_logvar`,
  `tgt_mean`, `tgt_logvar` (VAE posterior mean and log-variance, shape `[48, 13, 30, 52]` at 49 x 480 x 832) and
  `ctx_cond`, `ctx_null` (MLLM context of the full instruction and of the empty instruction, shape `[L, 3072]`). One
  file is about 14 MB, so the cache for the GOKU and CoinVE lists above (about 224k records) needs roughly 3 TB.
* **On a hit:** the source and target latents are re-sampled from the stored mean and log-variance with fresh noise,
  and text dropout picks `ctx_null` or `ctx_cond`. A cached visit therefore matches a recomputed one up to the bf16
  storage precision.
* **Robustness:** files are written to a temporary name and renamed atomically. An unreadable or malformed file is
  deleted and the sample is decoded normally.

## 6. Data loading and failure handling

* Loaders use persistent worker processes (`--dataset_num_workers`, `--dataloader_prefetch_factor`). A worker returns the
  source and target as uint8 tensors `[3, N, H, W]`, about 59 MB each at 49 x 480 x 832. videoeditor-14b and
  videoeditor-56b check the memory budget before building the loaders. If workers x prefetch x batch size x 118 MB x
  processes per node exceeds 60% of the smaller of free `/dev/shm` and `MemAvailable`, they stop with suggested values
  for the two flags.
* videoeditor-56b runs the MLLM input preprocessing (frame selection, chat template, resizing, tokenisation) in the
  loader workers when `--mllm_worker_pre 1` (default).
* **Bad records do not stop training.** A missing file, an unsupported extension, a decoding error, a source/target
  frame-count mismatch or a read that exceeds `VIDEO_READER_TIMEOUT` seconds (environment variable, default 120; set by
  the launchers) prints a warning to stderr, and a random other record is loaded instead. Loading stops with an
  error only when 30 attempts in a row fail for one sample. A list with many broken entries therefore trains silently
  on a subset, so check new lists before a run.

## 7. Checking a metadata list

Run this from the repository root on CPU. It loads the first records of a list the same way the trainers do (path
joining, frame selection, resize and crop) and prints the source path as written in the list, both tensor shapes and the
sub-instructions. For explicit-moe with a non-empty `--dataset_prefixes` entry, append that prefix to `video_root`.

```python
import json, sys, tempfile
sys.path.insert(0, ".")                                   # repository root
from videox_fun.moe_common.kiwi_dataset_multi_moe import KiwiMultiEditDataset

src, video_root = "my_list.json", "/data/videos"          # your metadata file and --dataset_base_path
rows = json.load(open(src)) if src.endswith(".json") else [json.loads(l) for l in open(src) if l.strip()]
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    for r in rows[:8]:
        p = r["prompt"] if isinstance(r["prompt"], list) else [r["prompt"]]
        f.write(json.dumps({"src_video": r["src_video"], "tgt_video": r["tgt_video"], "prompt": p}) + "\n")
ds = KiwiMultiEditDataset(metadata_path=f.name, base_path=video_root, num_frames=49, height=480, width=832,
                          pad_frames_to_num=True, return_pixels=True)
for i in range(len(ds)):
    s = ds[i]
    print(s["src_video_path"], tuple(s["src_pixels"].shape), tuple(s["tgt_pixels"].shape), s["prompts_multi"])
```

Every line should show `(3, 49, 480, 832)` twice. If a record fails to load, a `sample load failed` warning appears on
stderr, and the printed `src_video_path` belongs to the random replacement record rather than the requested one.

## Long-video records

A record may carry `num_frames_bucket` (121 or 81). The datasets then load that many frames for the sample and the trainer groups
samples so that every batch has one frame count; records without the field use the launcher's `NUM_FRAMES`.
