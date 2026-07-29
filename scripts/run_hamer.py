#!/usr/bin/env python3
"""Stage 5: per-frame hand pose estimation with HaMeR + metric-depth correction.

Deviation from paper: hand boxes come from GroundingDINO ('hand' prompt) instead of
detectron2+ViTPose (model itself unchanged); avoids the detectron2 CUDA build.

Depth correction (paper Sec. III-A): HaMeR uses a weak-perspective camera (depth-ambiguous).
We compute the corrected hand depth tz' as the median of (MoGe metric depth at the
projected 2D joints - MANO joint z), and place the hand metrically using the MoGe
intrinsics.

Reads  outputs/<obj>/frames/, outputs/<obj>/depth/, outputs/<obj>/masks/
Writes outputs/<obj>/hand/%05d.json  (MANO params + corrected metric translation)
       outputs/<obj>/hand/mesh_%05d.ply (hand mesh in camera frame, meters)
       outputs/<obj>/hand/qa.png

Requires third_party/hamer/_DATA (demo data incl. MANO + checkpoints).
Run inside .venv-recon.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
HAMER_ROOT = ROOT / "third_party" / "hamer"
CKPT = HAMER_ROOT / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
GDINO_ID = "IDEA-Research/grounding-dino-tiny"
os.chdir(HAMER_ROOT)  # hamer resolves _DATA paths relative to cwd


def detect_hand_box(image: Image.Image, processor, model, device, prev_box=None):
    inputs = processor(images=image, text=[["hand"]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.25, text_threshold=0.25,
        target_sizes=[image.size[::-1]])[0]
    if len(res["boxes"]) == 0:
        return prev_box
    best = int(res["scores"].argmax())
    return res["boxes"][best].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    from hamer.models import load_hamer
    from hamer.utils import recursive_to
    from hamer.utils.renderer import cam_crop_to_full
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    device = torch.device(args.device)
    model, model_cfg = load_hamer(str(CKPT))
    model = model.to(device).eval()

    processor = AutoProcessor.from_pretrained(GDINO_ID)
    gdino = GroundingDinoForObjectDetection.from_pretrained(GDINO_ID).to(device).eval()

    for obj in args.objects:
        frames_dir = OUT / obj / "frames"
        hand_dir = OUT / obj / "hand"
        hand_dir.mkdir(parents=True, exist_ok=True)
        frames = sorted(frames_dir.glob("*.jpg"))
        intr = json.loads((OUT / obj / "depth" / "intrinsics.json").read_text())
        prev_box = None
        qa = []
        for i, fp in enumerate(frames):
            key = f"{i:05d}"
            img_cv2 = cv2.imread(str(fp))
            img_pil = Image.open(fp).convert("RGB")
            box = detect_hand_box(img_pil, processor, gdino, device, prev_box)
            if box is None:
                print(f"[{obj}] frame {i}: no hand box, skipping")
                continue
            prev_box = box
            dataset = ViTDetDataset(model_cfg, img_cv2, box[None, :], np.array([1]),
                                    rescale_factor=2.0)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                                 num_workers=0)
            batch = recursive_to(next(iter(loader)), device)
            with torch.no_grad():
                out = model(batch)
            pred_cam = out["pred_cam"]
            right = batch["right"]
            pred_cam[:, 1] = (2 * right - 1) * pred_cam[:, 1]
            img_size = batch["img_size"].float()
            scaled_focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            scaled_focal = float(scaled_focal.detach().cpu()) if torch.is_tensor(scaled_focal) else float(scaled_focal)
            cam_t_full = cam_crop_to_full(pred_cam, batch["box_center"].float(),
                                          batch["box_size"].float(), img_size,
                                          scaled_focal)[0].detach().cpu().numpy()
            verts = out["pred_vertices"][0].detach().cpu().numpy()     # MANO space, meters
            joints3d = out["pred_keypoints_3d"][0].detach().cpu().numpy()
            mano = {k: v[0].detach().cpu().numpy().tolist()
                    for k, v in out["pred_mano_params"].items()}

            # metric correction via MoGe depth at projected 2D joints
            K = np.array(intr[key]["K"])
            depth = np.load(OUT / obj / "depth" / f"{key}.npy")
            H, W = depth.shape
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            # weak-persp full cam gives 2D projection with scaled_focal; convert:
            # proj = f * (X + t_xy) / tz  (weak). Use root joint (wrist, idx 0) for anchoring.
            z_spread = joints3d[:, 2] - joints3d[0, 2]
            # 2D joints in full frame: perspective proj with weak-persp full cam
            j3d_cam = joints3d + cam_t_full[None, :]
            j2d = scaled_focal * j3d_cam[:, :2] / j3d_cam[:, 2:3] \
                + np.array([W / 2.0, H / 2.0])
            us = np.clip(j2d[:, 0].astype(int), 0, W - 1)
            vs = np.clip(j2d[:, 1].astype(int), 0, H - 1)
            z_samples = depth[vs, us]
            valid = z_samples > 0.05
            tz = float(np.median((z_samples - z_spread)[valid])) if valid.sum() >= 3 \
                else float(np.mean(z_samples[z_samples > 0.05]))
            # hand root 2D anchor = wrist joint projection; t_metric places MANO
            # such that joints3d[0] (wrist) lands at the anchor (MANO origin != wrist)
            u0, v0 = j2d[0]
            anchor = np.array([(u0 - cx) * tz / fx, (v0 - cy) * tz / fy, tz])
            t_metric = anchor - joints3d[0]
            rec = {
                "mano": mano,
                "joints3d": joints3d.tolist(),
                "cam_t_weak": cam_t_full.tolist(),
                "t_metric": t_metric.tolist(),
                "box": box.tolist(),
                "valid_kpts": int(valid.sum()),
            }
            (hand_dir / f"{key}.json").write_text(json.dumps(rec, indent=1))
            # metric hand mesh in camera frame
            import trimesh
            m = trimesh.Trimesh(vertices=verts + t_metric, faces=model.mano.faces,
                                process=False)
            m.export(str(hand_dir / f"mesh_{key}.ply"))
            if i in (0, len(frames) // 2, len(frames) - 1):
                ov = img_cv2.copy()
                for u, v in zip(us[valid], vs[valid]):
                    cv2.circle(ov, (int(u), int(v)), 4, (0, 0, 255), -1)
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), 2)
                qa.append(ov)
            if i % 25 == 0:
                print(f"[{obj}] {i}/{len(frames)} tz={tz:.3f}m valid={valid.sum()}")
        if qa:
            cv2.imwrite(str(hand_dir / "qa.png"), np.vstack(qa))
        print(f"[ok] {obj}: hand poses -> {hand_dir}")


if __name__ == "__main__":
    main()
