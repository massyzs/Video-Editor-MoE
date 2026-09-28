#!/usr/bin/env python
"""Bit-exact comparison of the model inputs built by worker-side MLLM preprocessing (preprocess_mllm_sample + collate_pre) versus the old
main-process path (processor(text=[...], videos=[...], padding=True, right padding) inside forward_batch): input_ids / attention_mask /
pixel_values_videos (bf16) / video_grid_thw. CPU only, uses only the processor. Includes the static-video folding case.
Usage: python scripts/moe/videoeditor-56b/test_mllm_pre_cpu.py --base_dir <BASE_DIR>   (<BASE_DIR> contains ckpt/kiwi-edit-instruct-only-videoxfun/qwen/)"""
import argparse, os, sys, types
import numpy as np, torch
_here = os.path.dirname(os.path.realpath(__file__))                  # scripts/moe/<model>/
_repo = os.path.abspath(os.path.join(_here, "..", "..", ".."))       # repository root (contains videox_fun/)
for _p in (_here, _repo):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from transformers import AutoProcessor
from PIL import Image
from videox_fun.models.qwen_vl_utils import process_vision_info
from videox_fun.moe_common.mllm_encoder_moe import MLLMInContext, MLLMInContextConfig, preprocess_mllm_sample

_ap = argparse.ArgumentParser()
_ap.add_argument("--base_dir", required=True, help="<BASE_DIR>: contains ckpt/kiwi-edit-instruct-only-videoxfun/qwen/")
QWEN = os.path.join(_ap.parse_args().base_dir, "ckpt/kiwi-edit-instruct-only-videoxfun/qwen")
cfg = MLLMInContextConfig(mllm_id=QWEN, num_image_queries=256, num_video_queries=512, num_ref_queries=768, max_object_token=768, max_frames=16, max_pixels_per_frame=262144, dit_dim=3072, _gradient_checkpointing=False)
proc = AutoProcessor.from_pretrained(QWEN)
fake = types.SimpleNamespace(tokenizer=proc)

def ref_messages(instruction, frames):           # verbatim copy of the old _build_messages
    system_prompt = {"role": "system", "content": [{"type": "text", "text": cfg.system_prompt}]}
    if frames is not None and cfg.max_frames:
        tgt = max(2, int(cfg.max_frames) - int(cfg.max_frames) % 2)
        if len(frames) > tgt:
            idx = torch.linspace(0, len(frames) - 1, tgt).round().long().tolist(); frames = [frames[i] for i in idx]
    if frames is not None and len(frames) > 1:
        f0, f1 = frames[0], frames[-1]
        try: is_static = (f0 is f1) or (getattr(f0, "size", None) == getattr(f1, "size", None) and f0.tobytes() == f1.tobytes())
        except Exception: is_static = False
        if is_static: frames = frames[:1]
    video_data = {"type": "video", "video": frames, "max_frames": cfg.max_frames}
    if cfg.max_pixels_per_frame: video_data["max_pixels"] = cfg.max_pixels_per_frame
    if cfg.max_pixels_image and frames is not None and len(frames) == 1: video_data["max_pixels"] = cfg.max_pixels_image
    return [system_prompt, {"role": "user", "content": [video_data, {"type": "text", "text": instruction}, {"type": "text", "text": "<|object_ref_start|>" * cfg.num_video_queries}]}]

def ref_inputs(pairs):                            # old forward_batch construction: right padding, one processor call
    _tok = getattr(proc, "tokenizer", proc); saved = _tok.padding_side; _tok.padding_side = "right"
    try:
        texts, videos_all = [], []
        for instr, frames in pairs:
            m = ref_messages(instr, frames); texts.append(proc.apply_chat_template(m, tokenize=False, add_generation_prompt=False))
            _, vid = process_vision_info(m); videos_all.extend(vid)
        return proc(text=texts, images=None, videos=videos_all, padding=True, return_tensors="pt")
    finally:
        _tok.padding_side = saved

rng = np.random.default_rng(0)
def frames_of(seed, static=False, n=49):
    r = np.random.default_rng(seed); f0 = r.integers(0, 255, (480, 832, 3), dtype=np.uint8)
    return [Image.fromarray(f0 if static else r.integers(0, 255, (480, 832, 3), dtype=np.uint8)) for _ in range(n)]

cases = {"two_samples": [("Change the jeep's color to bright yellow. Then Add a golden retriever sitting next to the jeep.", frames_of(1)), ("Remove the laptop.", frames_of(2))],
         "static_fold": [("Apply the Impressionist aesthetic to this video.", frames_of(3, static=True))]}
ok_all = True
for name, samples in cases.items():
    pres = [preprocess_mllm_sample(proc, cfg.system_prompt, [instr, ""], frames, cfg.max_frames, cfg.max_pixels_per_frame, cfg.max_pixels_image, cfg.num_video_queries) for instr, frames in samples]
    order = [(k, "cond") for k in range(len(samples))] + [(k, "null") for k in range(len(samples))]     # primary encodings first, the complementary branch after
    seqs = [{"input_ids": pres[k]["input_ids"][0 if br == "cond" else 1], "pixel_values_videos": pres[k]["pixel_values_videos"], "video_grid_thw": pres[k]["video_grid_thw"]} for k, br in order]
    new = MLLMInContext.collate_pre(fake, seqs)
    ref = ref_inputs([(samples[k][0] if br == "cond" else "", samples[k][1]) for k, br in order])
    same_ids = torch.equal(new["input_ids"], ref["input_ids"]); same_mask = torch.equal(new["attention_mask"], ref["attention_mask"])
    same_pix = torch.equal(new["pixel_values_videos"], ref["pixel_values_videos"].to(torch.bfloat16)); same_grid = torch.equal(new["video_grid_thw"], ref["video_grid_thw"])
    print(f"{name}: input_ids {tuple(new['input_ids'].shape)} same={same_ids} mask same={same_mask} pixel {tuple(new['pixel_values_videos'].shape)} same={same_pix} grid {new['video_grid_thw'].tolist()} same={same_grid}")
    ok_all &= same_ids and same_mask and same_pix and same_grid
print("MLLM_PRE_PARITY", "PASS" if ok_all else "FAIL")
