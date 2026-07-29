#!/usr/bin/env python3
"""Stage 1b: object masks per frame.

GroundingDINO (zero-shot detection) proposes a box on frame 0 from a text prompt;
SAM2 video predictor propagates the mask through the whole clip.

Reads  outputs/<obj>/frames/*.png
Writes outputs/<obj>/masks/%05d.png   uint8 {0,255} object mask
       outputs/<obj>/masks/meta.json  prompt, box, scores
       outputs/<obj>/masks/qa.png     overlay QA sheet

Run inside .venv-recon. Requires SAM2 checkpoints (see third_party/sam2/checkpoints).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SAM2_ROOT = ROOT / "third_party" / "sam2"
CKPT = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GDINO_ID = "IDEA-Research/grounding-dino-tiny"

# text prompts with fallbacks (first that yields a box wins)
PROMPTS = {
    "spraybottle": ["spray bottle", "bottle"],
    "bottle": ["bottle"],
    "can": ["can", "tin", "box"],
    "bulb": ["light bulb", "bulb"],
    "hat": ["hat", "cap"],
    "jengabox": ["jenga box", "box"],
}


def detect_box(obj: str, image: Image.Image, device: str):
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    processor = AutoProcessor.from_pretrained(GDINO_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(GDINO_ID).to(device).eval()
    for prompt in PROMPTS[obj]:
        inputs = processor(images=image, text=[[prompt]], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.25, text_threshold=0.25,
            target_sizes=[image.size[::-1]])[0]
        if len(res["boxes"]) > 0:
            best = int(res["scores"].argmax())
            return prompt, res["boxes"][best].cpu().numpy(), float(res["scores"][best])
    raise RuntimeError(f"GroundingDINO found no box for {obj} with {PROMPTS[obj]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(CFG, str(CKPT), device=args.device)

    for obj in args.objects:
        frames_dir = OUT / obj / "frames"
        mask_dir = OUT / obj / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted(frames_dir.glob("*.jpg"))
        img0 = Image.open(frames[0]).convert("RGB")
        prompt, box, score = detect_box(obj, img0, args.device)
        print(f"[{obj}] prompt='{prompt}' box={box.round(1).tolist()} score={score:.2f}")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(frames_dir))
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)
            masks = {}
            for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                m = (mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255
                masks[fidx] = m
        for i, m in masks.items():
            cv2.imwrite(str(mask_dir / f"{i:05d}.png"), m)

        meta = {"prompt": prompt, "box": box.tolist(), "score": score,
                "n_frames": len(masks), "coverage": {str(k): float((v > 0).mean()) for k, v in
                                                     list(masks.items())[:1] + list(masks.items())[-1:]}}
        (mask_dir / "meta.json").write_text(json.dumps(meta, indent=1))

        qa = []
        for i in (0, len(masks) // 2, len(masks) - 1):
            im = cv2.imread(str(frames[i]))
            ov = im.copy()
            ov[masks[i] > 0] = (0.4 * ov[masks[i] > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            # draw box on first
            if i == 0:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), 2)
            qa.append(ov)
        cv2.imwrite(str(mask_dir / "qa.png"), np.vstack(qa))
        cov = [float((m > 0).mean()) for m in masks.values()]
        print(f"[ok] {obj}: {len(masks)} masks, coverage min/median/max = "
              f"{min(cov):.3f}/{sorted(cov)[len(cov)//2]:.3f}/{max(cov):.3f}")


if __name__ == "__main__":
    main()
