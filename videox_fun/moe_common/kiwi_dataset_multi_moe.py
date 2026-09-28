#!/usr/bin/env python
"""Multi-instruction dataset (for MoE cascaded fine-tuning).

Data: each record is {src_video, tgt_video, prompt: [sub-instructions...], turn_types_qwen: [...], ...}.
Two sources are supported:
  - goku multi (train_goku_multi_moe*.jsonl): relative paths (joined with base_path) + turn_types + turn_types_qwen;
  - CoinVE (datasets/CoinVE.jsonl): absolute s3 paths (passed through by the parent's _resolve) +
    only turn_types_qwen (the sole label source; turn_types was removed), n_turns in {2..5}.
max_turns: when >0, drop whole records with more sub-instructions than this. The fusion module concatenates features
to a fixed K_MAX and multi-turn data only has the final GT, so truncating to the first K instructions would be wrong
supervision against the final video; the whole record must be dropped.

Implementation: reuses KiwiEditDataset from this directory (kiwi_dataset_moe; video decode/crop/retry all validated).
At init each prompt list is joined into a shadow string [with a unique record tag _REC_TAG + row index appended] and
written to temporary metadata fed to the parent; __getitem__ takes the parent's output and looks up the record by the
"returned tagged string" (the parent's retry on bad samples swaps samples, and returns the swapped record's tagged
string -> the lookup is still correct), then strips the tag to restore the clean prompt.
* Why the tag is required: goku instructions are templated and 26% of records share the same prompt chain, but
masks are stored per video (rec_key); keying on the bare string would give earlier records the mask of the last
record with the same string.
Output = parent dict + {"prompts_multi": [...], "turn_types": [...], "loss_mask": ...}.

Feature cache (videoeditor-14b multi-edit sample feature cache, worker side; constructor arg cache_dir="" = off by default,
and when off the code path is byte-identical to the uncached one):
  In the mixed data every goku / CoinVE record is replicated 8 times, so one epoch visits the same sample 8 times, each
  redoing video decode + VAE + MLLM. After the main process computes it once it writes {VAE Gaussian params mean/logvar
  of src/tgt, MLLM ctx for the instructed and empty-instruction branches} to <cache_dir>/<key>.st; afterwards the worker
  checks that file [before decoding the video], and on a hit returns a slim dict (no pixels/PIL, with "cache": {6 bf16
  CPU tensors} + "cache_key"), letting the main process skip VAE/MLLM.
  Only applies to records with a cache_group field (set by moe_v3_1_train.load_records, multi group only; truthy =
  participates) and exactly one full instruction;
  key = sha1("src|tgt|full instruction|FxHxW"), src/tgt are the raw record strings (= src_video_path), instruction =
  prompts_multi[0] as fed to the MLLM by the main process.
  File missing -> decode normally, the returned dict also carries cache_key/cache_path (main process writes the file
  after computing); corrupt file (any load/validation error) -> delete it, decode normally.
  Cache hits bypass the parent: no retry (bad file falls back to decoding), and the periodic malloc_trim is called here.
  The parent kiwi_dataset_moe.py is left unchanged (shared by many scripts).
"""
import hashlib      # feature cache: key
import json
import os
import random
import tempfile

import torch        # feature cache: tensor loading (parent already imports torch, no extra process memory)

from .kiwi_dataset_moe import KiwiEditDataset, _vis_warn, _worker_periodic_trim   # feature cache: trim / logging on the hit path

_SEP = " <|moe_sep|> "     # sub-instruction join separator (never appears in real prompts)
_REC_TAG = " <|moe_rec|> " # record-tag separator: appended to the shadow prompt + row index, so identical strings from different records stay distinct
# Feature cache: the 6 bf16 CPU tensors a cache file (a saved dict, top level or nested under "cache") must contain:
#   src_mean/src_logvar/tgt_mean/tgt_logvar [C,f,h,w] (vae.model.encode output chunked in half along channels, not clamped);
#   ctx_cond [L,D] MLLM ctx of the full instruction; ctx_null [L0,D] ctx of the empty instruction ("") on the same source frames. Optional "meta": {"key": ...} is checked against the file name.
_CACHE_TENSORS = ("src_mean", "src_logvar", "tgt_mean", "tgt_logvar", "ctx_cond", "ctx_null")


def _load_mask_video(path, num_frames, height, width, probe_frames=0):
    """Timeout guard for decord mask reads (same as kiwi_dataset_moe._load_video): a hang -> RuntimeError;
    the caller's try/except falls back to a global loss and counts/prints the failure."""
    import os
    from func_timeout import FunctionTimedOut, func_timeout
    to = int(os.environ.get("VIDEO_READER_TIMEOUT", "120"))
    try:
        return func_timeout(to, _load_mask_video_impl,
                            args=(path, num_frames, height, width, probe_frames))
    except FunctionTimedOut:
        import sys
        print(f"[dataset rank={os.environ.get('RANK', '?')}] mask read timed out (>{to}s): {path}",
              file=sys.stderr, flush=True)
        raise RuntimeError(f"mask read timeout(>{to}s): {path}")


def _load_mask_video_impl(path, num_frames, height, width, probe_frames=0):
    """Mask video -> torch.uint8 [num_frames,H,W] (0/1).
    Frame sampling mirrors _load_video: [first n frames] (n=min(num_frames,total) trimmed to 4k+1) + pad with the last frame;
    geometry mirrors _crop_and_resize: resize (short side fits) + center crop, NEAREST for masks.
    probe_frames>0: only a few evenly spaced low-res frames (for the overlap check; returns [p, h/8, w/8])."""
    import decord
    import numpy as np
    import torch
    vr = decord.VideoReader(path)
    total = len(vr)
    if probe_frames:
        idx = np.linspace(0, min(num_frames, total) - 1, probe_frames).round().astype(int)
        arrs = vr.get_batch(idx.tolist()).asnumpy()
        m = torch.from_numpy(arrs).float().mean(-1) > 127          # [p,H0,W0]
        import torch.nn.functional as F
        m = F.adaptive_max_pool2d(m.float().unsqueeze(1),
                                  (height // 8, width // 8)).squeeze(1)
        return m > 0.5
    n = min(num_frames, total)
    while n > 1 and n % 4 != 1:
        n -= 1
    arrs = vr.get_batch(list(range(n))).asnumpy()                  # [n,H0,W0,3]
    import torch.nn.functional as F
    m = torch.from_numpy(arrs).float().mean(-1) > 127              # bool [n,H0,W0]
    h0, w0 = m.shape[1:]
    scale = max(height / h0, width / w0)                           # short side fits (same semantics as torchvision resize)
    nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
    m = F.interpolate(m.float().unsqueeze(1), size=(nh, nw), mode="nearest")
    top, left = (nh - height) // 2, (nw - width) // 2              # center crop
    m = m[:, 0, top:top + height, left:left + width]
    if n < num_frames:                                             # pad with the last frame
        import torch as _t
        m = _t.cat([m, m[-1:].expand(num_frames - n, -1, -1)], 0)
    return (m > 0.5).to(torch.uint8)


class KiwiMultiEditDataset(KiwiEditDataset):
    def __init__(self, metadata_path, base_path, drop_types=("i2v", "camera"),
                 max_turns=0, mask_loss=False, subset_single_prob=0.5,
                 subset_overlap_thr=0.3, cache_dir="", cache_latent_shape=None,
                 cache_ctx_dim=None, mllm_pre=None, **kw):
        """Feature cache: a non-empty cache_dir enables the multi-sample feature cache (see module header); default "" = old behaviour, other callers unaffected.
          cache_latent_shape: optional (C,f,h,w); if given, the 4 latent tensors in the file must match it exactly (passed by the main process from the VAE config, e.g. (48,13,30,52));
          cache_ctx_dim: optional; if given, the last dim of ctx_cond/ctx_null must equal it (e.g. 3072). With neither, only self-consistency is checked (4 same-shape 4-D latents, 2-D ctx of equal width).
        mask_loss=True reads the record's mask fields and outputs loss_mask in __getitem__:
          mask_mode="per_turn" (CoinVE): with probability subset_single_prob sample a single-turn subset
            (mask = that turn's mask, GT is still the final video; outside the mask the masked loss learns to "keep the source"),
            which must pass an overlap check (overlap ratio of an excluded turn's mask with the chosen mask < subset_overlap_thr,
            so other edits do not contaminate the chosen region); otherwise use all turns + the combined mask. -> combinatorial augmentation.
          mask_mode="single" (OpenVE+SAM3): use mask_video directly.
          mask_mode="global" / no mask / decode failure: loss_mask=None (plain full-frame loss)."""
        # MLLM preprocessing runs in the workers. mllm_pre = dict(processor_path, system_prompt, max_frames, max_pixels_per_frame, max_pixels_image, num_queries)
        # or None (old behaviour). The processor is lazily loaded once per worker process (on first use after fork).
        self.mllm_pre = dict(mllm_pre) if mllm_pre else None
        self._mllm_proc = None
        self._mllm_proc_pid = None
        """drop_types: records containing any of these turn types are dropped entirely (default drops i2v+camera: neither is in
        the 5-category taxonomy, so there is no matching expert; camera is explicitly excluded). Pass () to disable.
        max_turns: when >0, drop whole records with len(prompt)>max_turns (see module header)."""
        records = []
        n_drop = 0
        n_over = 0
        with open(metadata_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                tts = r.get("turn_types") or r.get("turn_types_qwen") or []
                if drop_types and any(t in drop_types for t in tts):
                    n_drop += 1
                    continue
                n_p = len(r["prompt"]) if isinstance(r["prompt"], list) else 1
                if max_turns and n_p > max_turns:
                    n_over += 1
                    continue
                records.append(r)
        if n_drop:
            print(f"[multi-moe] dropped {n_drop} records containing turn types "
                  f"{drop_types}; kept {len(records)}", flush=True)
        if n_over:
            print(f"[multi-moe] dropped {n_over} records with >{max_turns} turns "
                  f"(fixed K_MAX + final GT only, truncation = wrong supervision); kept {len(records)}", flush=True)
        self.mask_loss = mask_loss
        self.subset_single_prob = subset_single_prob
        self.subset_overlap_thr = subset_overlap_thr
        self._mask_fail = 0
        # feature cache: off by default; counters accumulate only within this process (worker), invisible to the main process; for single-process tests/debugging
        self.cache_dir = str(cache_dir or "")
        self.cache_latent_shape = tuple(int(x) for x in cache_latent_shape) if cache_latent_shape else None
        self.cache_ctx_dim = int(cache_ctx_dim) if cache_ctx_dim else None
        self._cache_hit = self._cache_miss = self._cache_read_fail = 0
        _cg_pool = {}                      # feature cache: intern cache_group values (a separate "multi" string per record over millions of records wastes worker COW memory)
        self._joined2multi = {}
        shadow = []
        for ridx, r in enumerate(records):
            plist = r["prompt"] if isinstance(r["prompt"], list) else [r["prompt"]]
            tagged = _SEP.join(plist) + _REC_TAG + str(ridx)
            minfo = {"mode": r.get("mask_mode", "global"),
                     "turn_masks": r.get("turn_mask_videos"),
                     "combined": r.get("combined_mask_video")}
            if self.cache_dir:             # feature cache: the per-record flag enters minfo only when caching is on (otherwise minfo is unchanged)
                cg = r.get("cache_group")
                if cg is not None:
                    try:
                        cg = _cg_pool.setdefault(cg, cg)
                    except TypeError:      # unhashable values (list/dict) are stored as-is
                        pass
                minfo["cache_group"] = cg
            self._joined2multi[tagged] = (plist, r.get("turn_types"),
                                          r.get("turn_types_qwen"), minfo)
            shadow.append({"src_video": r["src_video"], "tgt_video": r["tgt_video"],
                           "prompt": tagged})
        # Write the shadow metadata to a temp file for the parent (the parent only takes a path and reads it once in __init__).
        # Unlink right after reading: "keep until process exit" effectively meant never deleting, and temp files piled up to hundreds of GB;
        # with the videoeditor-56b mixed dataset (4.54M records) each rank's shadow is ~2.6 GB, so 24 ranks leak ~21 GB per node per launch.
        fd, self._shadow_path = tempfile.mkstemp(suffix=".json", prefix="moe_multi_")
        with os.fdopen(fd, "w") as f:
            json.dump(shadow, f)
        try:
            super().__init__(metadata_path=self._shadow_path, base_path=base_path, **kw)
        finally:
            os.unlink(self._shadow_path)

    def __getitem__(self, idx):
        prior = None
        if self.cache_dir:                 # feature cache: check the cache before decoding; on a hit return directly (no parent call, no decoding)
            hit, prior = self._cache_lookup(idx)
            if hit is not None:
                return hit
        s = super().__getitem__(idx)
        tagged = s["prompt"]
        plist, ttypes, qtypes, minfo = self._joined2multi[tagged]
        s["prompt"] = _SEP.join(plist)     # strip the record tag, restoring the clean joined string
        s["prompts_multi"] = plist
        s["turn_types"] = ttypes
        s["turn_types_qwen"] = qtypes      # Qwen3.6-relabelled turn types (may be None)
        s["loss_mask"] = None
        if self.mllm_pre is not None and s.get("src_pixels") is not None:
            self._attach_mllm_pre(s, plist[0])           # MLLM CPU preprocessing in the worker (two branches: full instruction + empty instruction)
        if self.cache_dir:                 # feature cache: add cache_key etc. for missed/non-participating samples (looked up by the [returned] record: parent retry may swap samples)
            self._cache_annotate(s, tagged, plist, minfo, prior)
        if not self.mask_loss or minfo["mode"] == "global":
            return s
        try:
            if s.get("src_pixels") is not None:
                H, W = s["src_pixels"].shape[-2:]
            else:
                W, H = s["src_video"][0].size
            nf = len(s["src_video"])
            if minfo["mode"] == "single" and minfo["combined"]:
                s["loss_mask"] = _load_mask_video(minfo["combined"], nf, H, W)
                return s
            if minfo["mode"] == "per_turn" and minfo["turn_masks"]:
                k = len(plist)
                tms = minfo["turn_masks"]
                # Global-type turns: style is always global; background only when it has no mask of its own
                # (CoinVE background has an inverted foreground mask = stronger explicit "keep foreground" supervision than global, keep it;
                #  goku background has no mask -> VLM judges it global -> full-frame reconstruction)
                glob_turn = [bool(qtypes) and (
                    qtypes[j] == "style"
                    or (qtypes[j] == "background" and not tms[j])) for j in range(k)]
                pick = -1                                  # -1 = full set
                if k >= 2 and random.random() < self.subset_single_prob:
                    i = random.randrange(k)
                    # Only "local edit" turns may be sampled: (1) for a global turn (style) the GT still carries
                    #   other turns' edits while only one prompt remains -> edits without a prompt are learned as
                    #   positives, contaminated supervision, reject; (2) an excluded turn is global -> it changed the
                    #   whole final GT frame, the subset is necessarily contaminated, reject
                    if tms[i] and not glob_turn[i] \
                            and not any(glob_turn[j] for j in range(k) if j != i):
                        probe_i = _load_mask_video(tms[i], nf, H, W, probe_frames=3)
                        area_i = probe_i.float().sum()
                        ok = bool(area_i > 0)              # empty probe cannot verify contamination -> reject
                        if ok:
                            for j in range(k):             # overlap check: excluded turns must not contaminate the chosen region
                                if j == i:
                                    continue
                                if not tms[j]:             # local turn has no mask (SAM miss)
                                    ok = False             # -> cannot verify contamination, reject subset
                                    break
                                probe_j = _load_mask_video(tms[j], nf, H, W,
                                                           probe_frames=3)
                                inter = (probe_i & probe_j).float().sum()
                                # Ratio in both directions: a small excluded edit fully inside a large chosen region
                                # (swap person + add hat) gives a small inter/area_i and slips through -> also check the excluded area
                                if inter / area_i.clamp_min(1.0) \
                                        > self.subset_overlap_thr \
                                        or inter / probe_j.float().sum().clamp_min(1.0) \
                                        > self.subset_overlap_thr:
                                    ok = False
                                    break
                        if ok:
                            pick = i
                if pick >= 0:                              # load the mask before touching prompts (
                    pick_mask = _load_mask_video(tms[pick], nf, H, W)
                    if not bool(pick_mask.any()):          # empty mask cannot delimit the edit -> fall back to full set
                        pick = -1                          # a load failure goes to except: prompts untouched,
                if pick >= 0:                              #  full frame + all instructions fallback, no contamination)
                    s["prompts_multi"] = [plist[pick]]     # single-turn subset (combinatorial augmentation)
                    s["turn_types"] = [ttypes[pick]] if ttypes else None
                    s["turn_types_qwen"] = [qtypes[pick]] if qtypes else None
                    s["loss_mask"] = pick_mask
                elif any(glob_turn):                       # full set has a global turn -> full reconstruction
                    s["loss_mask"] = None
                elif minfo["combined"]:
                    s["loss_mask"] = _load_mask_video(minfo["combined"], nf, H, W)
                elif all(tms[j] for j in range(k)):        # goku has no combined mask -> union on the fly
                    m = _load_mask_video(tms[0], nf, H, W)
                    for j in range(1, k):
                        m = m | _load_mask_video(tms[j], nf, H, W)
                    s["loss_mask"] = m
                # else: a local turn lacks a mask -> loss_mask=None, full-frame fallback
        except Exception:
            self._mask_fail += 1                           # decode failure -> plain loss fallback
            if self._mask_fail in (1, 100, 1000):
                import traceback
                print(f"[multi-moe] mask decode failed x{self._mask_fail}: "
                      f"{traceback.format_exc(limit=1)}", flush=True)
            s["loss_mask"] = None
        return s

    # ------------------------------------------------------------------ #
    #  Feature cache: worker-side reads / key annotation (cache_dir set)   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def cache_key_of(src_video, tgt_video, instruction, num_frames, height, width):
        """Feature cache: key = sha1("src|tgt|full instruction|FxHxW"). src/tgt are the raw record strings (same strings as src_video_path/tgt_video_path in the returned dict,
        not passed through _resolve), instruction = prompts_multi[0] as fed to the MLLM by the main process, F/H/W = dataset args num_frames/height/width. The main process uses the same function when writing files."""
        s = f"{src_video}|{tgt_video}|{instruction}|{num_frames}x{height}x{width}"
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def cache_path_of(self, key):
        """Feature cache: <cache_dir>/<key>.st (flat, one level; safetensors layout, see the trainer's save_cache_file)."""
        return os.path.join(self.cache_dir, key + ".st")

    def cache_stats(self):
        """Feature cache: per-process hit/miss/read-failure counts (each DataLoader worker keeps its own, invisible to the main process; for single-process tests)."""
        return {"hit": self._cache_hit, "miss": self._cache_miss, "read_fail": self._cache_read_fail}

    def _cache_row_of(self, tagged):
        """Feature cache: tagged string -> parent shadow row (the tag ends with the row index; no separate copy of the src/tgt strings). Returns None on mismatch."""
        try:
            row = self.data[int(tagged.rsplit(_REC_TAG, 1)[1])]
        except (IndexError, ValueError):
            return None
        return row if row.get("prompt") == tagged else None

    def _cache_info(self, plist, minfo, row):
        """Feature cache: whether the sample participates in caching; if so return (key, path, cache_group), else None.
        Conditions: the record has a truthy cache_group (multi group only) and exactly one full instruction (the main process asserts len(prompts_multi)==1; the key uses plist[0])
        and it does not take the mask path (the mask path needs pixels, which the cache lacks)."""
        cg = minfo.get("cache_group")
        if not cg or row is None or len(plist) != 1:
            return None
        if self.mask_loss and minfo["mode"] != "global":
            return None
        key = self.cache_key_of(row["src_video"], row["tgt_video"], plist[0],
                                self.num_frames, self.height, self.width)
        return key, self.cache_path_of(key), cg

    def _cache_validate(self, obj, key):
        """Feature cache: validate the loaded cache object and return {6 tensors} (contiguous, CPU bf16); any mismatch raises ValueError -> the caller treats the file as corrupt."""
        if not isinstance(obj, dict):
            raise ValueError(f"top level is not a dict: {type(obj).__name__}")
        d = obj["cache"] if isinstance(obj.get("cache"), dict) else obj
        meta = obj.get("meta")
        if isinstance(meta, dict) and meta.get("key") not in (None, key):
            raise ValueError(f"meta.key={meta.get('key')} does not match file name {key}")
        out = {}
        for name in _CACHE_TENSORS:
            t = d.get(name)
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"missing tensor {name}")
            if t.dtype != torch.bfloat16 or t.device.type != "cpu":
                raise ValueError(f"{name} should be CPU bf16, got {t.device} {t.dtype}")
            out[name] = t.contiguous()
        lat = [out[n] for n in _CACHE_TENSORS[:4]]
        shp = tuple(lat[0].shape)
        if lat[0].dim() != 4 or any(tuple(t.shape) != shp for t in lat[1:]):
            raise ValueError(f"latents should be 4 same-shape 4-D tensors, got {[tuple(t.shape) for t in lat]}")
        if self.cache_latent_shape is not None and shp != self.cache_latent_shape:
            raise ValueError(f"latent shape {shp} != expected {self.cache_latent_shape}")
        ctx = [out[n] for n in _CACHE_TENSORS[4:]]
        if any(t.dim() != 2 or t.shape[0] < 1 for t in ctx) or ctx[0].shape[1] != ctx[1].shape[1]:
            raise ValueError(f"ctx should be 2-D with equal width, got {[tuple(t.shape) for t in ctx]}")
        if self.cache_ctx_dim is not None and ctx[0].shape[1] != self.cache_ctx_dim:
            raise ValueError(f"ctx width {ctx[0].shape[1]} != expected {self.cache_ctx_dim}")
        return out

    def _cache_lookup(self, idx):
        """Feature cache: look up the cache by the [requested] idx (nothing decoded yet). Returns (hit dict | None, (tagged, state));
        state: "hit" / "miss" (file absent) / "read_fail" (corrupt file, deleted) / None (not participating). Never raises: bad -> normal decoding."""
        row = self.data[idx % len(self.data)]
        tagged = row["prompt"]
        ent = self._joined2multi.get(tagged)
        if ent is None:
            return None, (tagged, None)
        plist, ttypes, qtypes, minfo = ent
        info = self._cache_info(plist, minfo, row)
        if info is None:
            return None, (tagged, None)
        key, path, cg = info
        if not os.path.exists(path):       # a deleted directory (launcher trap) also lands here -> miss
            self._cache_miss += 1
            return None, (tagged, "miss")
        try:
            from safetensors import safe_open      # safetensors (mmap, no pickle); legacy .pt files are no longer read
            with safe_open(path, framework="pt", device="cpu") as f:
                # .clone(): get_tensor returns a view of the mmapped file; the shared-memory conversion breaks when it goes back through the DataLoader queue (the main process SIGBUSes on first touch), so copy into owned storage
                obj = {"cache": {k: f.get_tensor(k).clone() for k in f.keys()}, "meta": f.metadata() or {}}
            cache = self._cache_validate(obj, key)
        except Exception as e:             # partial file / deleted / shape mismatch / pickle error ... all treated as a miss
            self._cache_read_fail += 1
            n = self._cache_read_fail
            if n in (1, 10, 100) or n % 1000 == 0:
                _vis_warn(f"[multi-moe] cache read failed x{n}, deleting the bad file and decoding normally: {path}: {type(e).__name__}: {str(e)[:160]}")
            try:
                os.remove(path)
            except OSError:
                pass
            return None, (tagged, "read_fail")
        self._cache_hit += 1
        _worker_periodic_trim()            # hits bypass the parent __getitem__, so run the periodic malloc_trim here
        return {
            "prompt": _SEP.join(plist),
            "prompts_multi": plist,
            "turn_types": ttypes,
            "turn_types_qwen": qtypes,
            "loss_mask": None,
            "src_video_path": row["src_video"],
            "tgt_video_path": row["tgt_video"],
            "src_video": None, "tgt_video": None, "ref_image": None,
            "src_pixels": None, "tgt_pixels": None,       # no pixels: the main process pixels_of / rebuild_src_frames / encode_batch must not touch it
            "cache": cache,                               # {src_mean, src_logvar, tgt_mean, tgt_logvar, ctx_cond, ctx_null} bf16 CPU
            "cache_key": key, "cache_path": path, "cache_group": cg, "cache_state": "hit",
        }, (tagged, "hit")

    def _attach_mllm_pre(self, s, instruction):
        """Rebuild PIL frames from src_pixels (same layout as the trainer's rebuild_src_frames, byte-identical) and call mllm_encoder_moe.preprocess_mllm_sample for
        frame sampling/template/resizing/normalisation/tokenisation; attach s["mllm_pre"] = {"cond": ids, "null": ids, "pixel_values_videos": bf16 [N,1176], "video_grid_thw": [1,3]}.
        Failure is non-fatal: print once and leave it empty; the main process falls back to the old path (preprocessing itself)."""
        try:
            import os as _os
            if self._mllm_proc is None or self._mllm_proc_pid != _os.getpid():
                from transformers import AutoProcessor
                self._mllm_proc = AutoProcessor.from_pretrained(self.mllm_pre["processor_path"])
                self._mllm_proc_pid = _os.getpid()
            from .mllm_encoder_moe import preprocess_mllm_sample
            from PIL import Image
            u8 = s["src_pixels"]
            frames = u8.permute(1, 2, 3, 0).contiguous().numpy()                 # [F, H, W, 3] uint8 (same as rebuild_src_frames)
            pil = [Image.fromarray(frames[f]) for f in range(frames.shape[0])]
            cfg = self.mllm_pre
            pre = preprocess_mllm_sample(self._mllm_proc, cfg["system_prompt"], [instruction, ""], pil, cfg["max_frames"],
                                         cfg["max_pixels_per_frame"], cfg["max_pixels_image"], cfg["num_queries"])
            s["mllm_pre"] = {"cond": pre["input_ids"][0], "null": pre["input_ids"][1],
                             "pixel_values_videos": pre["pixel_values_videos"], "video_grid_thw": pre["video_grid_thw"]}
        except Exception as e:
            n = getattr(self, "_mllm_pre_fail", 0) + 1
            self._mllm_pre_fail = n
            if n in (1, 10, 100) or n % 1000 == 0:
                _vis_warn(f"[multi-moe] worker-side MLLM preprocessing failed x{n} (falling back to main-process preprocessing): {type(e).__name__}: {str(e)[:160]}")
            s["mllm_pre"] = None

    def _cache_annotate(self, s, tagged, plist, minfo, prior):
        """Feature cache: before returning on the normal decode path, annotate by the [returned] record: participating samples get cache_key/cache_path/cache_group
        (the main process writes the file after computing); all samples get tgt_video_path/cache_state ("miss" | "read_fail" | "none"). prior = _cache_lookup's result for the requested idx, not reused if the parent retry swapped the sample."""
        row = self._cache_row_of(tagged)
        s["tgt_video_path"] = row["tgt_video"] if row is not None else None
        info = self._cache_info(plist, minfo, row)
        if info is None:
            s["cache_state"] = "none"
            return
        key, path, cg = info
        s["cache_key"], s["cache_path"], s["cache_group"] = key, path, cg
        s["cache_state"] = prior[1] if (prior is not None and prior[0] == tagged and prior[1]) else "miss"

