#!/usr/bin/env python3
"""Fix HandFlow absolute translation with MoGe metric depth (paper's Sec. III-A
correction, same formula as run_hamer.py) + temporal median filter.

HandFlow's FM-predicted metric translation is biased on some of our videos
(hand-object distance ~11 cm on bottle/can/hat during physical grasp), while its
2D alignment and temporal consistency are excellent. HaMeR's pipeline solved the
same problem (weak-perspective ambiguity) by anchoring to MoGe depth; we apply
the identical correction here so both methods share the same depth prior:

  tz'   = median over joints of (depth[proj(joint)] - (z_joint - z_wrist))
  t'    = backproject(proj(wrist), tz') - joints3d[wrist]

then a temporal median filter (k=7) on the corrected translation — HandFlow's
smoothness advantage is kept, per-frame depth jitter is removed.

Rewrites t_metric in outputs/<obj>/handflow/%05d.json (raw value preserved as
t_metric_raw). Prints contact-distance before/after as validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def median_filter(x: np.ndarray, k: int = 7) -> np.ndarray:
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.median(xp[i:i + k], axis=0) for i in range(len(x))])


def contact_stats(obj: str, recs: list[dict], key: str) -> tuple[float, float]:
    mesh = trimesh.load(str(OUT / obj / "mesh" / "object.glb"), force="mesh")
    factor = json.loads((OUT / obj / "scale" / "scale.json").read_text())["chosen"]["factor"]
    mesh.apply_scale(factor)
    obj_pts = np.asarray(mesh.vertices)[::3]
    pose_files = sorted((OUT / obj / "pose").glob("*.json"))
    Ts = [np.array(json.loads(p.read_text())["T"]) for p in pose_files]
    d_min = []
    for i, rec in enumerate(recs):
        if i >= len(Ts):
            break
        J = np.array(rec["joints3d"]) + np.array(rec[key])
        oc = (Ts[i][:3, :3] @ obj_pts.T).T + Ts[i][:3, 3]
        d_min.append(np.sqrt(((J[:, None, :] - oc[None, :, :]) ** 2).sum(-1)).min())
    d = np.array(d_min) * 1000
    return float(np.median(d)), float((d < 25).mean() * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+")
    args = ap.parse_args()

    for obj in args.objects:
        hand_dir = OUT / obj / "handflow"
        depth_dir = OUT / obj / "depth"
        intr = json.loads((depth_dir / "intrinsics.json").read_text())
        keys = sorted(intr.keys())
        files = [fp for fp in sorted(hand_dir.glob("*.json"))]
        recs = [json.loads(fp.read_text()) for fp in files]
        recs = [r for r in recs if "joints3d" in r]
        files = files[:len(recs)]

        med0, close0 = contact_stats(obj, recs, "t_metric")

        t_corr, t_raw = [], []
        for i, rec in enumerate(recs):
            K = np.array(intr[keys[i]]["K"])
            H, W = intr[keys[i]]["height"], intr[keys[i]]["width"]
            depth = np.load(depth_dir / f"{keys[i]}.npy")
            J = np.array(rec["joints3d"])
            t = np.array(rec["t_metric"])
            t_raw.append(t)
            J_cam = J + t[None, :]
            j2d = np.stack([K[0, 0] * J_cam[:, 0] / J_cam[:, 2] + K[0, 2],
                            K[1, 1] * J_cam[:, 1] / J_cam[:, 2] + K[1, 2]], axis=1)
            us = np.clip(j2d[:, 0].astype(int), 0, W - 1)
            vs = np.clip(j2d[:, 1].astype(int), 0, H - 1)
            z_samples = depth[vs, us]
            valid = z_samples > 0.05
            z_spread = J_cam[:, 2] - J_cam[0, 2]
            if valid.sum() >= 3:
                tz = float(np.median((z_samples - z_spread)[valid]))
                u0, v0 = j2d[0]
                anchor = np.array([(u0 - K[0, 2]) * tz / K[0, 0],
                                   (v0 - K[1, 2]) * tz / K[1, 1], tz])
                t_corr.append(anchor - J[0])
            else:
                t_corr.append(t)  # keep raw (rare: no valid depth)
        t_corr = median_filter(np.stack(t_corr), k=7)
        t_raw = np.stack(t_raw)

        for fp, rec, tc, tr in zip(files, recs, t_corr, t_raw):
            rec["t_metric_raw"] = tr.tolist()
            rec["t_metric"] = tc.tolist()
            fp.write_text(json.dumps(rec, indent=1))

        med1, close1 = contact_stats(obj, recs, "t_metric")
        shift = np.linalg.norm(t_corr - t_raw, axis=1).mean() * 1000
        print(f"[{obj}] mean |dt| {shift:6.1f}mm | contact med {med0:6.1f} -> {med1:5.1f}mm | "
              f"frames<25mm {close0:5.1f}% -> {close1:5.1f}%")


if __name__ == "__main__":
    main()
