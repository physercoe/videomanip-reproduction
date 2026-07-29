#!/usr/bin/env python3
"""Stage 1a: per-frame metric depth + camera intrinsics with MoGe-2.

Reads  outputs/<obj>/frames/*.png
Writes outputs/<obj>/depth/%05d.npy        float32 metric depth (m)
       outputs/<obj>/depth/intrinsics.json per-frame 3x3 K + image size
       outputs/<obj>/depth/qa.png          depth QA contact sheet (first/mid/last frame)

Run inside .venv-recon. Model: Ruicheng/moge-2-vitl-normal (HF hub).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model import import_model_class_by_version

MoGeModel = import_model_class_by_version("v2")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MODEL_ID = "Ruicheng/moge-2-vitl-normal"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+", help="object dir names under outputs/")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = MoGeModel.from_pretrained(MODEL_ID).to(args.device).eval()

    for obj in args.objects:
        frames_dir = OUT / obj / "frames"
        depth_dir = OUT / obj / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted(frames_dir.glob("*.jpg"))
        assert frames, f"no frames in {frames_dir}"
        intr = {}
        qa = []
        for i, fp in enumerate(frames):
            img_bgr = cv2.imread(str(fp))
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ten = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                out = model.infer(ten)
            depth = out["depth"].cpu().numpy().astype(np.float32)
            K_norm = out["intrinsics"].cpu().numpy().astype(float)
            H, W = img.shape[:2]
            # MoGe intrinsics are normalized (cx=cy=0.5, focal in image-size units) -> pixel space
            K = K_norm.copy()
            K[0, 0] *= W; K[0, 2] *= W
            K[1, 1] *= H; K[1, 2] *= H
            np.save(depth_dir / f"{i:05d}.npy", depth)
            intr[f"{i:05d}"] = {"K": K.tolist(), "K_normalized": K_norm.tolist(),
                                 "width": int(W), "height": int(H)}
            if i in (0, len(frames) // 2, len(frames) - 1):
                d = np.clip(depth, 0, np.percentile(depth, 99))
                dvis = (d / (d.max() + 1e-9) * 255).astype(np.uint8)
                dvis = cv2.applyColorMap(dvis, cv2.COLORMAP_TURBO)
                qa.append(np.hstack([img_bgr, dvis]))
            if i % 50 == 0:
                print(f"[{obj}] {i}/{len(frames)}")
        (depth_dir / "intrinsics.json").write_text(json.dumps(intr, indent=1))
        cv2.imwrite(str(depth_dir / "qa.png"), np.vstack(qa))
        Ks = np.array([v["K"] for v in intr.values()])
        print(f"[ok] {obj}: {len(frames)} depth maps; "
              f"fx median {np.median(Ks[:,0,0]):.1f}±{Ks[:,0,0].std():.1f}")


if __name__ == "__main__":
    main()
