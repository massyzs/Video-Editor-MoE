"""mllm_encoder_moe.py - Kiwi-Edit (Qwen2.5-VL) MLLM encoder, copy dedicated to the MoE expert models.
Includes the transformers 5.x compatibility fix (hidden_size lives under text_config).
In MoE expert training the MLLM is always frozen and connetor/query are always warm-started from init_weights.

Self-contained copy of the base MLLM encoder (no imports from it, no monkey-patching), with two deltas:
  1. `_build_messages`/`forward_batch` accept a per-call `system_prompt` override (needed for an ablation
     with a descriptive system prompt); default None = use the configured template.
  2. config adds `connetor_zero_init` (default False = PyTorch default init): experiments showed that zero init +
     lr 1e-5 leaves connetor.2 as near-zero dead weights, which kept the loss high. Training always warm-starts
     from the trained Kiwi-Edit instruct-only connetor (overwritten by init_weights); this flag only affects the cold-start fallback.
Weight compatibility: video_queries.q / connetor.{0,2} (sic) match the converted Kiwi-Edit instruct-only checkpoint key by key.

Kept from UniVideo:
  * overall structure - the `MLLMInContextConfig` / `MLLMInContext(PreTrainedModel)` shell,
  * automatic drop - `compute_user_start_drop_idx` + `_crop_hidden_bs1`
    + `_extract_text_and_queries_bs1` (automatically crops system prompt / vision tokens, keeping only
    the valid text "after the last <|vision_end|>"; left padding is handled correctly).

Queries follow the original Kiwi-Edit design (not UniVideo's metaquery vocabulary tokens):
  * three learnable `QueryVector` banks (image 256 / video 512 / ref 768),
  * the forked Qwen2.5-VL (`modeling_qwen2_5_vl_kiwi`) masked-scatters the query vectors
    into the `<|object_ref_start|>` placeholder positions of the chat template,
  * after the Qwen forward pass, the auto-drop above extracts the [text + query] hidden states,
  * a zero-initialized connector projects them to the DiT dimension (dit_dim).
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
from torch import nn

from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor

# UniVideo's original DEBUG prints are noisy (printed on every forward); enabled only with KIWI_MLLM_VERBOSE=1.
_VERBOSE = os.environ.get("KIWI_MLLM_VERBOSE", "0") == "1"
def _dprint(*args, **kwargs):
    if _VERBOSE:
        print(*args, **kwargs)


def _is_qwen2_5_vl(mllm_id: str) -> bool:
    """Whether this is Qwen2.5-VL: the repo id contains a tag, or a local directory's config.json declares qwen2_5_vl."""
    import json as _json
    if any(tag in mllm_id for tag in ("Qwen2.5-VL", "Qwen2.5", "qwen2.5", "Qwen2_5")):
        return True
    cfg = os.path.join(mllm_id, "config.json")
    if not os.path.isfile(cfg):
        return False
    try:
        c = _json.load(open(cfg))
    except Exception:
        return False
    if c.get("model_type") == "qwen2_5_vl":
        return True
    return any("Qwen2_5_VL" in a for a in (c.get("architectures") or []))


def _find_subseq(seq, sub):
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


def compute_user_start_drop_idx(tokenizer, system_prompt: str) -> int:
    """[Kept from UniVideo] Automatically compute the token index where user content starts (i.e. the system segment length).

    Runs a sentinel string through the chat template and locates its start in the full sequence, which tells how many
    system/template tokens must be dropped when extracting features. Works for Qwen2.5-VL.
    """
    conv = []
    if system_prompt is not None:
        conv.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    SENTINEL = "<<<__SENTINEL_USER_TEXT__>>>"
    conv.append({"role": "user", "content": [{"type": "text", "text": SENTINEL}]})

    rendered = tokenizer.apply_chat_template(conv, add_generation_prompt=True)
    full_ids = tokenizer(text=rendered, return_tensors="pt", padding=False).input_ids[0].tolist()
    sent_ids = tokenizer(text=SENTINEL, return_tensors="pt", padding=False).input_ids[0].tolist()
    start = _find_subseq(full_ids, sent_ids)
    if start == -1:
        prefix = rendered.split(SENTINEL)[0]
        start = len(tokenizer(prefix, return_tensors="pt").input_ids[0])
    return int(start)


class QueryVector(nn.Module):
    """[Original Kiwi-Edit] One bank of `num_query` learnable vectors of dimension dim."""

    def __init__(self, num_query: int, dim: int, dtype=torch.bfloat16, device: str = "cpu"):
        super().__init__()
        # Small Gaussian init (same as diffsynth)
        self.q = nn.Parameter(torch.randn(num_query, dim, dtype=dtype, device=device) * 0.02)  # shape=[num_query, dim]

    def forward(self) -> torch.Tensor:
        return self.q


def select_mllm_frames(frames, max_frames):
    """Extracted verbatim from _build_messages for reuse in worker-side preprocessing: (1) uniformly sample to max_frames (even count, incl. first and last); (2) collapse a static "video" with identical first/last frames into a single frame."""
    if frames is not None and max_frames:
        tgt = max(2, int(max_frames) - int(max_frames) % 2)
        if len(frames) > tgt:
            idx = torch.linspace(0, len(frames) - 1, tgt).round().long().tolist()
            frames = [frames[i] for i in idx]
    if frames is not None and len(frames) > 1:
        f0, f1 = frames[0], frames[-1]
        try:
            is_static = (f0 is f1) or (getattr(f0, "size", None) == getattr(f1, "size", None)
                                       and f0.tobytes() == f1.tobytes())
        except Exception:
            is_static = False
        if is_static:
            frames = frames[:1]
    return frames


def build_mllm_messages(system_prompt, instruction, frames, max_frames, max_pixels_per_frame, max_pixels_image, num_queries):
    """Pure-function version of _build_messages (no instance state); frames have already gone through select_mllm_frames. Returns messages."""
    system_msg = {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    video_data = {"type": "video", "video": frames, "max_frames": max_frames}
    if max_pixels_per_frame:
        video_data["max_pixels"] = max_pixels_per_frame
    if max_pixels_image and frames is not None and len(frames) == 1:
        video_data["max_pixels"] = max_pixels_image
    return [system_msg, {"role": "user", "content": [
        video_data,
        {"type": "text", "text": instruction},
        {"type": "text", "text": "<|object_ref_start|>" * num_queries},
    ]}]


def preprocess_mllm_sample(processor, system_prompt, instructions, frames, max_frames, max_pixels_per_frame, max_pixels_image, num_queries,
                           pixel_dtype=torch.bfloat16):
    """Preprocessing moved into the dataloader worker: runs all CPU-side MLLM preprocessing for one sample's instructions (in training, two branches [full instruction, ""]) in the worker:
    frame sampling/collapsing -> chat template -> qwen_vl_utils.process_vision_info (resizing) -> processor (pixel normalization + tokenization). Pixels are kept once (identical across branches, asserted equal).
    Returns {"input_ids": [Tensor[L_j] ...] (one per instruction, single sample, no padding), "pixel_values_videos": Tensor[N, 1176] (pixel_dtype),
          "video_grid_thw": Tensor[1, 3]}; the main process's forward_batch_pre right-pads per batch and runs the forward pass, bit-identical to forward_batch (preprocessing in the main process)."""
    from videox_fun.models.qwen_vl_utils import process_vision_info
    frames = select_mllm_frames(frames, max_frames)
    ids, pixels, grid = [], None, None
    for instruction in instructions:
        messages = build_mllm_messages(system_prompt, instruction, frames, max_frames, max_pixels_per_frame, max_pixels_image, num_queries)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[text], images=img_in or None, videos=vid_in or None, padding=True, return_tensors="pt")
        assert bool(inputs["attention_mask"].all()), "a single sample should have no padding"
        ids.append(inputs["input_ids"][0].clone())
        pv = inputs["pixel_values_videos"].to(pixel_dtype)
        if pixels is None:
            pixels, grid = pv, inputs["video_grid_thw"].clone()
        else:
            assert torch.equal(pv, pixels) and torch.equal(inputs["video_grid_thw"], grid), "pixel preprocessing must be identical across instructions of the same sample"
    return {"input_ids": ids, "pixel_values_videos": pixels.contiguous(), "video_grid_thw": grid}


class MLLMInContextConfig(PretrainedConfig):
    model_type = "mllm-in-context"

    def __init__(
        self,
        mllm_id: str = "Qwen2.5-VL",
        # -- original Kiwi-Edit: three query bank sizes (chosen by input modality) --
        num_image_queries: int = 256,
        num_video_queries: int = 512,
        num_ref_queries: int = 768,
        max_object_token: int = 768,
        max_frames: int = 16,
        max_pixels_per_frame: int = 512 * 512,
        max_pixels_image: int = 0,   # delta: separate pixel budget for single-frame (image) samples; 0 = use max_pixels_per_frame
        # -- projection to the DiT dimension --
        dit_dim: int = 3072,
        _gradient_checkpointing: bool = True,
        max_input_text_tokens: int = 1024,
        system_prompt: str = (
            "You will be given an image and instruction. Please describe the "
            "content of the image in detail based on instruction in your own words."
        ),
        # -- [kept from UniVideo] auto-drop switches --
        use_chat_template: bool = True,
        crop_system_tokens: bool = True,
        crop_vision_tokens: bool = True,
        system_tokens_drop_idx: int = 0,
        connetor_zero_init: bool = False,   # delta: False = PyTorch default init (zero init was found to cause a dead pathway)
        **kwargs,
    ):
        super().__init__()
        self.connetor_zero_init = bool(connetor_zero_init)
        self.mllm_id = mllm_id
        self.num_image_queries = num_image_queries
        self.num_video_queries = num_video_queries
        self.num_ref_queries = num_ref_queries
        self.max_object_token = max_object_token
        self.max_frames = max_frames
        self.max_pixels_per_frame = max_pixels_per_frame
        self.max_pixels_image = max_pixels_image
        self.dit_dim = dit_dim
        self._gradient_checkpointing = _gradient_checkpointing
        self.max_input_text_tokens = max_input_text_tokens
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template
        self.crop_system_tokens = crop_system_tokens
        self.crop_vision_tokens = crop_vision_tokens
        self.system_tokens_drop_idx = system_tokens_drop_idx


class MLLMInContext(PreTrainedModel):
    config_class = MLLMInContextConfig

    def __init__(self, config: MLLMInContextConfig) -> None:
        super().__init__(config)
        self.config = config
        if not _is_qwen2_5_vl(config.mllm_id):
            raise ValueError(f"Unsupported model: {config.mllm_id}")
        self.mllm_type = "qwenvl"

        # -- [original Kiwi-Edit] load the forked Qwen2.5-VL (supports learnable_query masked-scatter) --
        # The fork (modeling_qwen2_5_vl_kiwi) is required: stock Qwen does not scatter queries into the placeholders.
        from videox_fun.models.modeling_qwen2_5_vl_kiwi import \
            Qwen2_5_VLForConditionalGeneration
        attn_impl = os.environ.get("MLLM_ATTN_IMPL", "").strip()   # explicit backend override (videoeditor-56b passes sdpa: flash_attention_2 gave random NaN in 3~5% of calls); unset = original behaviour
        if not attn_impl:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
        print(f"Using Qwen MLLM {config.mllm_id} (attn={attn_impl})")
        # Default low_cpu_mem_usage=False: the model is built by a normal __init__ on CPU (non-persistent buffers such as RoPE are really computed), then the checkpoint is loaded over it;
        # True goes through the meta device, where non-persistent buffers must be filled by _init_weights (without it vision RoPE is 100% garbage). The extra memory is one bf16 weight copy (~7.5 GB).
        # To save memory set MLLM_LOW_CPU_MEM=1 (this copy has its own _init_weights, so both paths are correct); the recompute + assert below guards both paths.
        low_cpu = os.environ.get("MLLM_LOW_CPU_MEM", "0").strip() == "1"
        self.mllm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.mllm_id, torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl, low_cpu_mem_usage=low_cpu,
        )
        # Fix: with low_cpu_mem_usage=True (meta-device init), transformers' re-initialization does not recognize this copy's RoPE class, so the non-persistent buffer
        # `visual.rotary_pos_emb.inv_freq` is left holding stale memory garbage (100% of 64 probe runs, max 9e23; if it contains inf/NaN the forward pass yields NaN,
        # the root cause of random 3~5% start-up self-check failures). Unconditionally recompute every RoPE inv_freq from its construction formula (idempotent if already correct), then assert.
        fixed = []
        for _name, _mod in self.mllm_backbone.named_modules():
            _inv = getattr(_mod, "inv_freq", None)
            if not isinstance(_inv, torch.Tensor):
                continue
            with torch.no_grad():
                if hasattr(_mod, "rope_init_fn") and hasattr(_mod, "config"):            # language tower: use its own init function
                    _new, _scaling = _mod.rope_init_fn(_mod.config, _inv.device)
                    _mod.attention_scaling = _scaling
                else:                                                                 # vision tower: 1 / theta^(arange(0, dim, 2)/dim), theta 10000
                    _d = _inv.numel() * 2
                    _new = 1.0 / (10000.0 ** (torch.arange(0, _d, 2, dtype=torch.float, device=_inv.device) / _d))
                _inv.copy_(_new.to(_inv.dtype))
                if hasattr(_mod, "original_inv_freq"):
                    _mod.original_inv_freq = _mod.inv_freq
            assert torch.isfinite(_inv).all() and float(_inv.abs().max()) <= 1.0 + 1e-6, f"{_name}.inv_freq still invalid after recompute"
            fixed.append(_name)
        print(f"Recomputed RoPE inv_freq in {len(fixed)} modules: {fixed}")
        # transformers 5.x compatibility: composite configs no longer proxy hidden_size to the top level (it is in text_config)
        _bc = self.mllm_backbone.config
        self.mllm_hidden_size = (getattr(_bc, "hidden_size", None)
                                 or _bc.text_config.hidden_size)  # 3B=2048
        dtype = torch.bfloat16

        # -- [original Kiwi-Edit] three learnable query banks + register their sizes on the fork (used to check placeholder counts) --
        self.num_image_queries = config.num_image_queries
        self.num_video_queries = config.num_video_queries
        self.num_ref_queries = config.num_ref_queries
        self.max_object_token = config.max_object_token
        for target in (self.mllm_backbone, getattr(self.mllm_backbone, "model", None)):
            if target is None:
                continue
            target.num_image_queries = config.num_image_queries
            target.num_video_queries = config.num_video_queries
            target.num_ref_queries = config.num_ref_queries
        # Only video_queries is kept: image / ref queries were removed (inference and training both use the video path, see _build_messages / forward_batch).
        self.video_queries = QueryVector(config.num_video_queries, self.mllm_hidden_size, dtype=dtype)  # shape=[512, hidden]

        self.max_frames = config.max_frames
        self.max_pixels_per_frame = config.max_pixels_per_frame
        self.max_pixels_image = int(getattr(config, "max_pixels_image", 0) or 0)
        self.gradient_checkpointing = config._gradient_checkpointing

        # -- [original Kiwi-Edit] zero-initialized connector: projects (text + query) hidden states to the DiT dimension --
        # Zero-initialized last layer => the condition is a no-op at step 0, starting from the Wan2.2 pretrained prior.
        self.dit_dim = config.dit_dim
        self.connetor = nn.Sequential(  # keeps the original diffsynth spelling "connetor" so old checkpoints line up
            nn.Linear(self.mllm_hidden_size, self.dit_dim, dtype=dtype),  # input shape=[*, hidden], output shape=[*, dit_dim]
            nn.GELU(approximate="tanh"),
            nn.Linear(self.dit_dim, self.dit_dim, dtype=dtype),           # output shape=[*, dit_dim]
        )
        # delta: NOT zero-initialized by default (PyTorch default init) - experiments showed that with zero init + lr 1e-5
        #   connetor.2 stays near zero for a long time and the MLLM condition pathway is a no-op. Training always overwrites it with the trained connetor from init_weights,
        #   so this only affects the cold-start fallback; for the old behaviour (strict no-op start) set connetor_zero_init=True.
        if getattr(config, "connetor_zero_init", False):
            nn.init.zeros_(self.connetor[2].weight)
            nn.init.zeros_(self.connetor[2].bias)

        # -- processor (chat template + vision preprocessing) --
        self.tokenizer = AutoProcessor.from_pretrained(config.mllm_id)  # Qwen2_5_VLProcessor
        self.tokenizer.tokenizer.padding_side = "left"
        self.system_prompt = config.system_prompt

        # -- [kept from UniVideo] auto-drop config + automatic detection of the system drop index --
        self.tokenizer.use_chat_template = config.use_chat_template
        self.tokenizer.crop_system_tokens = config.crop_system_tokens
        self.tokenizer.max_input_text_tokens = config.max_input_text_tokens
        if config.use_chat_template and config.crop_system_tokens:
            if config.system_tokens_drop_idx > 0:
                drop_idx = config.system_tokens_drop_idx
                print(f"[AUTO-CROP] use provided system_tokens_drop_idx={drop_idx}")
            else:
                drop_idx = compute_user_start_drop_idx(self.tokenizer, config.system_prompt)
                print(f"[AUTO-CROP] detected system_tokens_drop_idx={drop_idx}")
            self.tokenizer.system_tokens_drop_idx = drop_idx
        else:
            self.tokenizer.system_tokens_drop_idx = 0

        # Token id of `<|object_ref_start|>`: the forked Qwen scatters the queries into these positions,
        # and auto-drop also uses it to tell query tokens from text tokens when extracting features.
        _tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        self.object_ref_start_id = _tok.convert_tokens_to_ids("<|object_ref_start|>")

        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.gradient_checkpointing_enable({"use_reentrant": False})
                print("Enable Gradient Checkpoint for MLLM backbone")
            except Exception:
                pass

    @property
    def device(self):
        return self.mllm_backbone.device

    # ------------------------------------------------------------------ #
    #  [Kept from UniVideo] auto-drop helpers                            #
    # ------------------------------------------------------------------ #
    def _tok_id(self, s: str):
        tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        try:
            tid = tok.convert_tokens_to_ids(s)
            return tid if isinstance(tid, int) and tid != -1 else None
        except Exception:
            return None

    def _extract_text_and_queries_bs1(
        self,
        input_ids: torch.Tensor,        # shape=[1, T]
        attention_mask: torch.Tensor,   # shape=[1, T]
        last_hidden: torch.Tensor,      # shape=[1, T, D]
    ):
        """[Kept from UniVideo; query detection changed to <|object_ref_start|>] Extract [text + query].

        First auto-drop: keep what follows the last <|vision_end|> (crops vision tokens), or fall back to
        the system drop_idx (crops the system segment); then separate the query placeholders from the text in the kept span,
        and finally concatenate [text hs ; query hs]. bs=1.
        Returns (embeds shape=[1, L, D], attn shape=[1, L]).
        """
        assert input_ids.shape[0] == 1 and attention_mask.shape[0] == 1 and last_hidden.shape[0] == 1
        ids = input_ids[0]            # shape=[T]
        attn = attention_mask[0]      # shape=[T]
        hs = last_hidden[0]           # shape=[T, D]
        T, D = hs.shape

        # valid token range (compatible with left padding)
        valid = (attn == 1).nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            _dprint("[TEXT+QUERY] no valid tokens; return empty.")
            return hs.new_zeros(1, 0, D), attn.new_zeros(1, 0)
        first_valid = int(valid.min().item())
        end_idx = int(valid.max().item()) + 1

        # start: prefer cropping vision (keep what follows the last vision_end), otherwise crop the system segment
        start_idx = None
        if self.config.crop_vision_tokens:
            ve_id = self._tok_id("<|vision_end|>")
            if ve_id is not None:
                ve_pos = (ids == ve_id).nonzero(as_tuple=False).flatten()
                if ve_pos.numel() > 0:
                    start_idx = int(ve_pos.max().item()) + 1
                    _dprint(f"[TEXT+QUERY] vision_end at {ve_pos.tolist()} -> start_idx={start_idx}")
        if start_idx is None:
            drop_idx = int(getattr(self.tokenizer, "system_tokens_drop_idx", 0))
            start_idx = first_valid + drop_idx
            _dprint(f"[TEXT+QUERY] no vision_end -> system drop_idx={drop_idx}, start_idx={start_idx}")
        start_idx = max(0, min(start_idx, end_idx))

        kept_hs = hs[start_idx:end_idx]      # shape=[L, D]
        kept_ids = ids[start_idx:end_idx]    # shape=[L]
        L = kept_hs.shape[0]
        if L == 0:
            _dprint("[TEXT+QUERY] crop produced empty; return empty.")
            return hs.new_zeros(1, 0, D), attn.new_zeros(1, 0)

        # query positions = <|object_ref_start|> placeholders (the forked Qwen has scattered the learnable queries into them)
        device = kept_hs.device
        query_mask = (kept_ids == self.object_ref_start_id)   # shape=[L]
        text_mask = ~query_mask                               # everything else is text

        text_hs = kept_hs[text_mask]      # shape=[L_text, D]
        query_hs = kept_hs[query_mask]    # shape=[L_query, D]
        # delta (adversarial-review fix): length-limit fallback -
        #   the downstream DiT truncates the context to context_token_len (g[:L_ctx], keeping the head), while queries sit at the tail -
        #   an instruction over ~510 tokens would silently cut off the queries (condition pathway severed). Truncate the text tail first to keep all queries.
        _budget = int(getattr(self.config, "max_input_text_tokens", 1024) or 1024)
        _max_text = max(0, _budget - query_hs.shape[0])
        if text_hs.shape[0] > _max_text:
            print(f"[mllm-kiwi] WARNING: instruction too long: text={text_hs.shape[0]} + query={query_hs.shape[0]} "
                  f"> {_budget}, truncating the text tail to keep the queries", flush=True)
            text_hs = text_hs[:_max_text]
        # concatenate [text ; query]
        concat_hs = torch.cat([text_hs, query_hs], dim=0)     # shape=[L_text+L_query, D]
        concat_attn = torch.ones(concat_hs.shape[0], device=device, dtype=attn.dtype)
        _dprint(f"[TEXT+QUERY] concat={tuple(concat_hs.shape)} (text={tuple(text_hs.shape)}, query={tuple(query_hs.shape)})")
        return concat_hs.unsqueeze(0), concat_attn.unsqueeze(0)  # shape=[1, L, D], [1, L]

    # ------------------------------------------------------------------ #
    #  [Original Kiwi-Edit] chat template + forward                      #
    # ------------------------------------------------------------------ #
    def _build_messages(self, instruction, src_image, src_video, ref_image, system_prompt_override=None):
        """[video-only] Everything goes through the video path: any source is fed as video frames and uses video_queries (image/ref queries removed).

        delta: when system_prompt_override is not None, this call uses that system prompt instead -
        used by the descriptive-prompt ablation; default None = the configured template (same distribution as the editing samples, recommended).
        Note: the crop_system_tokens drop_idx is detected for the configured template; if the override template has a different length,
        the start falls back to the vision_end anchor (video/image inputs always contain vision_end, so semantics are unaffected)."""
        _sys_text = self.system_prompt if system_prompt_override is None else system_prompt_override
        frames = src_video if src_video is not None else src_image    # images are also treated as video frames
        # delta (adversarial-review fix; factored into the pure functions select_mllm_frames / build_mllm_messages, behaviour unchanged):
        # (1) explicitly sample uniformly to max_frames frames (linspace incl. first and last, even count); (2) collapse a static "video" with identical first/last frames into one frame; (3) single-frame samples use the max_pixels_image budget.
        frames = select_mllm_frames(frames, self.max_frames)
        num_queries = self.num_video_queries
        input_query = self.video_queries().to(self.mllm_backbone.device)  # shape=[512, hidden]
        messages = build_mllm_messages(_sys_text, instruction, frames, self.max_frames, self.max_pixels_per_frame, self.max_pixels_image, num_queries)
        return messages, input_query, num_queries, "video"

    def forward(
        self,
        instruction: str,
        src_image: Optional[list] = None,   # list[PIL.Image] (image editing)
        src_video: Optional[list] = None,   # list[PIL.Image] frames (video editing)
        ref_image: Optional[list] = None,   # list[PIL.Image] (reference image)
    ) -> torch.Tensor:
        """instruction (+ source image / source video / reference image) -> DiT context, returns shape=[1, L, dit_dim]."""
        if not (src_image or src_video or ref_image):
            raise ValueError("MLLMInContext.forward needs one of src_image/src_video/ref_image.")

        # build the chat + pick the query bank (with <|object_ref_start|> placeholders)
        messages, input_query, num_queries, mode = self._build_messages(
            instruction, src_image, src_video, ref_image,
        )
        # render the template + vision preprocessing
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        from videox_fun.models.qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.tokenizer(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        target_dtype = next(self.mllm_backbone.parameters()).dtype
        inputs = inputs.to(self.mllm_backbone.device, dtype=target_dtype)

        # run the forked Qwen: masked-scatter the learnable queries into the placeholders, take the last-layer hidden states
        if self.gradient_checkpointing:
            import torch.utils.checkpoint as cp

            def _run(*ckpt_inputs):
                (input_query_t,) = ckpt_inputs
                out = self.mllm_backbone(
                    **inputs, learnable_query=input_query_t,
                    output_attentions=False, output_hidden_states=True, return_dict=True,
                )
                return out.hidden_states[-1]

            hidden_states = cp.checkpoint(_run, input_query, use_reentrant=False)  # shape=[1, T, hidden]
        else:
            outputs = self.mllm_backbone(
                **inputs, learnable_query=input_query,
                output_attentions=False, output_hidden_states=True, return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]                              # shape=[1, T, hidden]

        # auto-drop: crop system/vision tokens, take the [text + query] hidden states
        prompt_embeds, _attn = self._extract_text_and_queries_bs1(
            inputs["input_ids"], inputs["attention_mask"], hidden_states,
        )                                                                          # shape=[1, L, hidden]
        # project to the DiT dimension
        context = self.connetor(prompt_embeds.to(self.connetor[0].weight.dtype))   # shape=[1, L, dit_dim]
        return context

    def forward_batch(
        self,
        instructions: List[str],
        src_list: list,
        is_video: bool,
        system_prompt: Optional[str] = None,   # delta: not None -> this batch uses that system prompt (ablation)
    ) -> List[torch.Tensor]:
        """Batched forward: B same-modality (instruction, src) samples go through Qwen in one pass. Gradients are controlled by the caller's context:
        with the MLLM fully frozen the caller wraps it in no_grad (saves memory); with trainable parameters (query/connetor/LoRA) it does not, so they train.

        Returns List[Tensor] (length B), each shape=[L_b, dit_dim]. L_b differs per sample (= text tokens + queries),
        per-sample equivalent to the per-sample `forward()` output (only numerical jitter of the batched Qwen forward; smoke test verified max|diff| ~ bf16 level).

        Key points:
          - Only the expensive Qwen forward is batched (bs=B, temporarily switched to RIGHT padding to prevent RoPE position drift, see comment below);
            the auto-drop feature extraction `_extract_text_and_queries_bs1` still runs per sample (cheap, and keeps semantics identical).
          - learnable_query is expanded from [nq, dim] to [B, nq, dim]: the fork masked_scatters into each sample's nq <|object_ref_start|>
            placeholders and needs B copies of the queries; after expansion the (b,q,d) order matches the row-major traversal of the [B,seq] placeholder mask.
          - MLLM frozen -> no_grad throughout, no gradient checkpointing (no backward).
          - Training only uses the image / video modalities (ref still uses the per-sample forward). The caller guarantees one modality per batch.
        """
        from videox_fun.models.qwen_vl_utils import process_vision_info

        # Right padding (key): in a causal forward pass the real tokens occupy positions 0..L_b-1, exactly as with bs=1,
        # and trailing pads are masked by the causal mask, so they do not affect real tokens. Left padding (the default, meant for generation) shifts the
        # real tokens' RoPE positions by the pad length -> a sample's context depends on the lengths of other samples in the batch (measured 2~13% drift,
        # worse for shorter / more-padded samples). Temporarily switch to right padding and restore afterwards, so bs=1 forward() is unaffected.
        _tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        _saved_side = _tok.padding_side
        _tok.padding_side = "right"
        try:
            texts, images_all, videos_all = [], [], []
            for instr, src in zip(instructions, src_list):
                messages, _, num_queries, _ = self._build_messages(
                    instr, None, src, None, system_prompt_override=system_prompt)   # unified video path
                texts.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False))
                img_in, vid_in = process_vision_info(messages)
                if img_in:
                    images_all.extend(img_in)
                if vid_in:
                    videos_all.extend(vid_in)

            inputs = self.tokenizer(
                text=texts,
                images=images_all or None,
                videos=videos_all or None,
                padding=True, return_tensors="pt",
            )
        finally:
            _tok.padding_side = _saved_side   # restore so bs=1 forward() is unaffected (never left at right, even on exception)
        return self._forward_inputs(inputs)

    def _forward_inputs(self, inputs) -> List[torch.Tensor]:
        """GPU part shared by forward_batch / forward_batch_pre: move to device -> expand learnable_query to the batch -> Qwen forward -> per-sample [text+query] extraction -> connector.
        logits_to_keep=1: we only use the last-layer hidden states (before lm_head); the default 0 computes 150k-vocab logits for every token (~9 GB of memory and bandwidth per chunk) for nothing."""
        target_dtype = next(self.mllm_backbone.parameters()).dtype
        inputs = inputs.to(self.mllm_backbone.device, dtype=target_dtype)

        B = int(inputs["input_ids"].shape[0])
        q = self.video_queries().to(self.mllm_backbone.device)  # [nq, dim] - only video_queries remain
        # [B, nq, dim] contiguous: masked_scatter fills in the source tensor's flatten order, so it must be contiguous to follow (b,q,d) order.
        learnable_query = q.unsqueeze(0).expand(B, -1, -1).contiguous()

        outputs = self.mllm_backbone(
            **inputs, learnable_query=learnable_query,
            output_attentions=False, output_hidden_states=True, return_dict=True, logits_to_keep=1,
        )
        hidden_states = outputs.hidden_states[-1]            # shape=[B, seq, hidden]

        # per-sample auto-drop extracts [text+query] -> connector projection. L_b varies, so return a list.
        contexts: List[torch.Tensor] = []
        for b in range(B):
            prompt_embeds, _attn = self._extract_text_and_queries_bs1(
                inputs["input_ids"][b:b + 1],
                inputs["attention_mask"][b:b + 1],
                hidden_states[b:b + 1],
            )                                                # shape=[1, L_b, hidden]
            ctx = self.connetor(prompt_embeds.to(self.connetor[0].weight.dtype))  # [1, L_b, dit_dim]
            contexts.append(ctx[0])                          # [L_b, dit_dim]
        return contexts

    def collate_pre(self, seqs):
        """Collate worker-preprocessed sequences [{input_ids: Tensor[L], pixel_values_videos: Tensor[N,1176], video_grid_thw: Tensor[1,3]}, ...]
        into the same batch as processor(text=[...], videos=[...], padding=True, padding_side=right): input_ids right-padded (pad_token_id), attention_mask,
        pixel_values_videos concatenated in sequence order (each branch of a sample carries its own copy, matching how the processor treats a videos list), video_grid_thw stacked."""
        from transformers import BatchFeature
        _tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        pad_id = _tok.pad_token_id
        assert pad_id is not None, "tokenizer has no pad_token_id"
        L = max(int(s["input_ids"].numel()) for s in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for b, s in enumerate(seqs):
            n = int(s["input_ids"].numel())
            ids[b, :n] = s["input_ids"]
            mask[b, :n] = 1
        return BatchFeature({"input_ids": ids, "attention_mask": mask,
                             "pixel_values_videos": torch.cat([s["pixel_values_videos"] for s in seqs], dim=0),
                             "video_grid_thw": torch.cat([s["video_grid_thw"] for s in seqs], dim=0)})

    def forward_batch_pre(self, seqs) -> List[torch.Tensor]:
        """Batched forward on worker-preprocessed inputs (produced by preprocess_mllm_sample); bit-identical to forward_batch (main-process preprocessing)."""
        return self._forward_inputs(self.collate_pre(seqs))
