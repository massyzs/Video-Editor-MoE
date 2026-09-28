"""Length-bucketing for Kiwi-Edit (kill the variable-length straggler).

Problem
-------
Under data-parallel training (8 ranks), every rank processes its own batch and
then all ranks sync at the gradient barrier. Videos have **variable frame
counts**, so the rank that drew the longest videos this step is the slowest, and
the other 7 ranks idle at the barrier waiting for it (observed: 7 GPUs at 0%
util, 1 at 93%). The straggler determines wall-clock per step.

Fix
---
Order samples so that, at any given step, the 8 ranks process **similar-length**
videos. The classic recipe (HF `LengthGroupedSampler`):

  1. Split the (shuffled) index list into "mega-batches" of size
     `batch_size * world_size * mega_batch_mult`.
  2. Sort each mega-batch by length.
  3. Re-flatten. Now any contiguous `batch_size * world_size` window has
     near-uniform length -> when the dataloader hands consecutive batches to
     consecutive ranks (accelerate's BatchSamplerShard / DistributedSampler),
     every rank's batch at the same step is similar length. Straggler shrinks.

Randomness is preserved *between* mega-batches (their order is shuffled and the
within-mega sort only groups locally), so training still sees varied batches.

Status / caveats
----------------
- Pure index math here is unit-tested on CPU (`get_length_grouped_indices`
  guarantees the contiguous-window uniformity property).
- It is **OFF by default** (`--length_bucketing` flag). The interaction with
  accelerate's distributed batch distribution and the *actual* straggler
  reduction MUST be confirmed on a real multi-GPU run before trusting it.
- It needs per-sample `lengths` (frame counts). `compute_video_lengths`
  builds + caches them (parallel); cheap proxy = file size if you don't want to
  open every video.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import torch


def get_length_grouped_indices(
    lengths: Sequence[int],
    batch_size: int,
    world_size: int,
    mega_batch_mult: int = 50,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Return a permutation of range(len(lengths)) grouped so that every
    contiguous `batch_size * world_size` window has near-uniform length.

    Mirrors HF Transformers' `get_length_grouped_indices`, generalized so the
    grouping window spans ALL ranks (batch_size * world_size), which is what
    makes the per-step cross-rank lengths match (not just within one rank).
    """
    n = len(lengths)
    if n == 0:
        return []
    group = max(1, batch_size * world_size)
    # Shuffle first so mega-batch contents are random run-to-run.
    perm = torch.randperm(n, generator=generator).tolist()
    mega = max(group, group * mega_batch_mult)
    indices = []
    for i in range(0, n, mega):
        chunk = perm[i:i + mega]
        # Sort this mega-batch by length (desc) so neighbours are similar.
        chunk.sort(key=lambda idx: lengths[idx], reverse=True)
        indices.extend(chunk)
    return indices


def get_exact_grouped_indices(
    lengths: Sequence[int],
    group_keys: Sequence,
    batch_size: int,
    world_size: int,
    mega_batch_mult: int = 50,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Like `get_length_grouped_indices` but FIRST hard-partitions by `group_keys`
    (exact equality — e.g. #sub-prompts N for decompose), so every contiguous
    `batch_size*world_size` window is single-key (→ every per-rank batch has a
    SINGLE N, satisfying the decompose same-N-per-batch assert). Within a key,
    samples are length-grouped (straggler reduction). Each key's remainder
    (< one window) is DROPPED (negligible: thousands of samples per key).
    Window order is shuffled so keys interleave during training.
    """
    n = len(lengths)
    if n == 0:
        return []
    group = max(1, batch_size * world_size)
    by_key: dict = {}
    for idx in range(n):
        by_key.setdefault(group_keys[idx], []).append(idx)

    windows: List[List[int]] = []
    for _key, idxs in by_key.items():
        perm = [idxs[i] for i in torch.randperm(len(idxs), generator=generator).tolist()]
        mega = max(group, group * mega_batch_mult)
        ordered: List[int] = []
        for i in range(0, len(perm), mega):
            chunk = perm[i:i + mega]
            chunk.sort(key=lambda j: lengths[j], reverse=True)   # length-group within key
            ordered.extend(chunk)
        usable = (len(ordered) // group) * group                 # drop < one full window
        for i in range(0, usable, group):
            windows.append(ordered[i:i + group])

    out: List[int] = []
    for wi in torch.randperm(len(windows), generator=generator).tolist():
        out.extend(windows[wi])
    return out


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Sampler yielding length-grouped indices (single-process / pre-accelerate).

    Pass to `DataLoader(sampler=...)` with `shuffle=False`. Under accelerate the
    loader is re-wrapped for distribution; that wrapping must be verified to keep
    the contiguous-window grouping intact (see module docstring caveat).
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        world_size: int = 1,
        mega_batch_mult: int = 50,
        seed: int = 0,
        group_keys: Optional[Sequence] = None,
        rank: int = 0,
        num_replicas: int = 1,
    ):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.world_size = world_size
        self.mega_batch_mult = mega_batch_mult
        self.seed = seed
        # group_keys: if given, bucket exactly by key (e.g. per-sample sub-task count N) so every batch has
        # a single key (decompose requires one N per batch). None -> length-only soft grouping (original behaviour).
        self.group_keys = list(group_keys) if group_keys is not None else None
        # ★ Rank sharding (MUST be set for DDP data parallelism!): the full order is identical on all ranks (fixed seed);
        #   the end of __iter__ slices order[rank::num_replicas] -> each rank gets a disjoint 1/num_replicas share.
        #   The strided slice splits every batch_size*world_size same-length/same-N window evenly across ranks:
        #   one batch on rank r = bs samples from one window -> the grouping (same length, same N) is preserved.
        #   Not set (default 1) = old behaviour: every rank iterates the full set -> all GPUs train on identical data (= one GPU)!
        self.rank = int(rank)
        self.num_replicas = max(1, int(num_replicas))
        # ★ __len__ must match __iter__ exactly: in group_keys mode get_exact_grouped_indices drops each
        #   key's tail shorter than one bs*world window -> actual output < total samples. Precompute with the same formula,
        #   otherwise len(dataloader)/steps_per_epoch is too large -> mixed sampling replays early data at epoch end, epoch accounting drifts.
        if self.group_keys is not None:
            group = max(1, self.batch_size * self.world_size)
            counts: dict = {}
            for k in self.group_keys:
                counts[k] = counts.get(k, 0) + 1
            self._order_len = sum((c // group) * group for c in counts.values())
        else:
            self._order_len = len(self.lengths)
        self.epoch = 0
        # Data-position resume: samples consumed so far in this epoch (index into this rank's sharded sequence). __iter__ skips
        # the first _pos_start and increments while yielding; checkpoints save it, resume restores it -> continue where data left off.
        # After sharding all ranks consume at the same rate -> store rank0's value and share it across ranks.
        self._pos_start = 0

    def set_epoch(self, epoch: int) -> None:
        # ★ New epoch = new permutation -> the leftover data position must be reset. Otherwise an iterator abandoned mid-epoch
        #   (with mixed sampling a sub-loader is often dropped at an epoch boundary) carries a non-zero _pos_start into the new epoch and its first samples are silently skipped.
        #   The resume path is safe: the epoch restored by load_state_dict == first_epoch of the following set_epoch (same value, no reset).
        if int(epoch) != self.epoch:
            self._pos_start = 0
        self.epoch = int(epoch)

    def state_dict(self) -> dict:
        # Saved by the trainer into sampler_pos_start.pkl.
        # ★ Known bias (bounded, tolerated): _pos_start increments when an index is dispatched to a DataLoader worker,
        #   so with num_workers=8 x prefetch_factor=2 it always runs 16 batches ahead of what training consumed (stage1 bs=33
        #   -> 528 samples/rank). After resume those samples are skipped for this epoch (~0.3%, no duplicates, no crash).
        #   A full fix must derive the consumed position from the training step count (requires replaying mixed-sampling decisions).
        return {"epoch": int(self.epoch), "pos_start": int(self._pos_start)}

    def load_state_dict(self, state: dict) -> None:
        # On resume, restore the epoch and the data position within it
        self.epoch = int(state.get("epoch", 0))
        self._pos_start = int(state.get("pos_start", 0))

    def __len__(self) -> int:
        # Samples this rank actually yields after sharding (dataloader/steps_per_epoch depend on it).
        # _order_len already excludes each key's dropped tail in group_keys mode -> matches __iter__ output exactly.
        return self._order_len // self.num_replicas

    def __iter__(self):
        # Order depends only on seed+epoch (reproducible; ★ every rank must compute the same order before slicing,
        # so the seed is the explicitly passed value, independent of the global RNG (set_seed device_specific))
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.group_keys is not None:                    # exact bucketing (one N per batch)
            order = get_exact_grouped_indices(
                self.lengths, self.group_keys, self.batch_size, self.world_size,
                self.mega_batch_mult, generator=g,
            )
        else:
            order = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size,
                self.mega_batch_mult, generator=g,
            )                                              # shape=list[int], length = number of samples
        # ★ Rank sharding: truncate to a multiple of num_replicas, then take a strided slice (equal share per rank -> same step count, no desync;
        #   the stride splits every bs*world window evenly across ranks -> per-step batch length/N grouping is preserved)
        if self.num_replicas > 1:
            usable = (len(order) // self.num_replicas) * self.num_replicas
            order = order[self.rank:usable:self.num_replicas]
        # Continue from the consumed position (on resume _pos_start>0 -> skip samples already trained on)
        start = self._pos_start
        for i in range(start, len(order)):
            self._pos_start = i + 1                        # remember how far we have consumed
            yield order[i]
        self._pos_start = 0                                # epoch finished; reset for the next epoch


def compute_video_lengths(
    metadata_path: str,
    base_path: str = "",
    num_frames: int = 81,
    time_div: int = 4,
    time_rem: int = 1,
    proxy: str = "frames",
    workers: int = 32,
    cache: bool = True,
) -> List[int]:
    """Per-sample length signal for bucketing, with a cached sidecar.

    `proxy`:
      - "frames": true post-trim frame count via decord len() (O(1) metadata, no
        decode; same min(num_frames,total)->4n+1 logic the dataset uses). 2.17M
        videos is a one-time parallel pass; cached to `<metadata>.lengths.json`.
      - "filesize": os.path.getsize of src_video (fast stat, rough proxy for
        compute cost = frames x resolution). No decode.

    Returns a list aligned with the jsonl row order.
    """
    cache_path = f"{metadata_path}.lengths.{proxy}.json"
    if cache and os.path.isfile(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    rows = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    def _resolve(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(base_path, p)

    def _one(row) -> int:
        # Accept both video (src_video) and image (source_img) json field names
        src_field = (row.get("src_video") or row.get("source_img")
                     or row.get("source_image") or row.get("src_image"))
        if not src_field:
            return 0 if proxy == "filesize" else num_frames
        src = _resolve(src_field)
        if proxy == "filesize":
            try:
                return os.path.getsize(src)
            except OSError:
                return 0
        # frames proxy
        try:
            from .utils import VideoReader_contextmanager
            with VideoReader_contextmanager(src, num_threads=2) as vr:
                total = len(vr)   # O(1) container metadata; imageio.count_frames() decoded
                                  # the whole file just to count — brutal over 2.17M S3 videos.
            n = min(num_frames, total)
            while n > 1 and n % time_div != time_rem:
                n -= 1
            return n
        except Exception:
            return num_frames  # fall back to max so it's not mis-bucketed as short

    if workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            lengths = list(ex.map(_one, rows))
    else:
        lengths = [_one(r) for r in rows]

    if cache:
        try:
            with open(cache_path, "w") as f:
                json.dump(lengths, f)
        except OSError:
            pass
    return lengths


def compute_subtask_counts(metadata_path: str) -> List[int]:
    """Per-sample sub-prompt count N (group_key for decompose bucketing): `prompt` is a list -> len,
    a str -> 1. Aligned with the jsonl row order."""
    counts: List[int] = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pr = row.get("prompt")
            counts.append(len(pr) if isinstance(pr, (list, tuple)) else 1)
    return counts
