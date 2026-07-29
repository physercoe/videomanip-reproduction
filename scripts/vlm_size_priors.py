#!/usr/bin/env python3
"""VLM-based object size priors (replaces GPT-4.1 text-only query / kimi-agent estimates).

For each object crop, ask a VLM (Qwen2.5-VL-7B) for the object identity and its
real-world physical dimensions, then derive the bbox diagonal in meters — the same
output the paper's GPT-4.1 stage produces (data/size_priors.json).

Run on the GPU server: python /tmp/vlm_size_priors.py /tmp/vlm_imgs
"""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import os
MODEL = os.environ.get("VLM_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
PROMPT = (
    "You are estimating the real-world metric size of the object in this image.\n"
    "1) Identify the object (brand/type).\n"
    "2) Estimate its physical dimensions (height x width x depth) in centimeters, "
    "using your knowledge of this kind of consumer product.\n"
    "3) Give the 3D bounding-box diagonal in meters.\n"
    "Answer in this exact format:\n"
    "object: <name>\n"
    "dims_cm: <h>x<w>x<d>\n"
    "diag_m: <number>"
)


def main() -> None:
    img_dir = Path(sys.argv[1])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    processor = AutoProcessor.from_pretrained(MODEL)
    results = {}
    for fp in sorted(img_dir.glob("*.png")):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": Image.open(fp).convert("RGB")},
            {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[Image.open(fp).convert("RGB")],
                           return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        resp = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0]
        results[fp.stem] = resp.strip()
        print(f"=== {fp.stem}\n{resp.strip()}\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
