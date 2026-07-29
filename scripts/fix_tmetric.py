#!/usr/bin/env python3
"""One-off fix: correct t_metric in outputs/*/hand/*.json.

Bug (2026-07-22): t_metric was computed as the wrist's back-projected camera position
and then applied to the MANO ORIGIN. But MANO's wrist joint sits ~9.6 cm from the
model origin, so the whole hand was translated by that offset (seen as projections
landing on the forearm instead of the hand).

Correct: t_metric = backproject(j2d_wrist, tz_wrist) - joints3d[0],
so that joints3d[0] + t_metric lands exactly at the wrist's metric position.

Recomputes from stored records + depth maps; preserves tz (unchanged logic).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FOCAL_256 = 5000.0  # hamer model_cfg EXTRA.FOCAL_LENGTH


def main() -> None:
    for obj_dir in sorted(OUT.iterdir()):
        hand_dir = obj_dir / "hand"
        depth_dir = obj_dir / "depth"
        if not hand_dir.is_dir() or not depth_dir.is_dir():
            continue
        intr = json.loads((depth_dir / "intrinsics.json").read_text())
        for fp in sorted(hand_dir.glob("*.json")):
            if fp.name in ("qa.png",):
                continue
            rec = json.loads(fp.read_text())
            if "joints3d" not in rec or "cam_t_weak" not in rec:
                continue
            key = fp.stem
            K = np.array(intr[key]["K"])
            W, H = intr[key]["width"], intr[key]["height"]
            depth = np.load(depth_dir / f"{key}.npy")
            J = np.array(rec["joints3d"])
            cam_t = np.array(rec["cam_t_weak"])
            scaled_focal = FOCAL_256 / 256.0 * max(H, W)
            j3d_cam = J + cam_t[None, :]
            j2d = scaled_focal * j3d_cam[:, :2] / j3d_cam[:, 2:3] + np.array([W / 2, H / 2])
            z_spread = J[:, 2] - J[0, 2]
            us = np.clip(j2d[:, 0].astype(int), 0, W - 1)
            vs = np.clip(j2d[:, 1].astype(int), 0, H - 1)
            z_samples = depth[vs, us]
            valid = z_samples > 0.05
            if valid.sum() >= 3:
                tz = float(np.median((z_samples - z_spread)[valid]))
            else:
                tz = float(np.mean(z_samples[z_samples > 0.05]))
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            u0, v0 = j2d[0]
            anchor = np.array([(u0 - cx) * tz / fx, (v0 - cy) * tz / fy, tz])
            t_new = anchor - J[0]
            rec["t_metric"] = t_new.tolist()
            rec["t_metric_fix"] = "wrist-anchored (2026-07-22)"
            fp.write_text(json.dumps(rec, indent=1))
        print(f"[ok] {obj_dir.name}: t_metric fixed")


if __name__ == "__main__":
    main()
