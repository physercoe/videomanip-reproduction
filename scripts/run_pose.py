#!/usr/bin/env python3
"""Stage 3+4 driver: scale estimation + per-frame 6D pose tracking.

Reads  outputs/<obj>/mesh/object.glb, data/size_priors.json,
       outputs/<obj>/{frames,depth,masks}
Writes outputs/<obj>/scale/scale.json, outputs/<obj>/pose/%05d.json,
       outputs/<obj>/pose/qa.png (projected mesh samples overlay)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

import sys
sys.path.insert(0, str(ROOT / "src"))
from videomanip.reconstruct.pose import estimate_scale_and_pose, track


def load_frame_data(obj: str, i: int):
    depth = np.load(OUT / obj / "depth" / f"{i:05d}.npy")
    mask = cv2.imread(str(OUT / obj / "masks" / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
    intr = json.loads((OUT / obj / "depth" / "intrinsics.json").read_text())
    return {"depth": depth, "mask": mask, "K": np.array(intr[f"{i:05d}"]["K"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+")
    ap.add_argument("--ref", type=int, default=0, help="reference frame for scale+init")
    args = ap.parse_args()

    priors = json.loads((ROOT / "data" / "size_priors.json").read_text())

    for obj in args.objects:
        mesh = trimesh.load(str(OUT / obj / "mesh" / "object.glb"), force="mesh")
        prior = priors[obj]["diag_m"]
        fd0 = load_frame_data(obj, args.ref)
        est = estimate_scale_and_pose(mesh, fd0["depth"], fd0["mask"], fd0["K"], prior)
        best = est["best"]
        print(f"[{obj}] scale_mult={best['scale_mult']:.3f} factor={best['factor']:.4f} "
              f"cost={best['cost'] * 1e3:.2f}mm (cloud n={est['cloud_n']})")

        scale_dir = OUT / obj / "scale"
        scale_dir.mkdir(parents=True, exist_ok=True)
        (scale_dir / "scale.json").write_text(json.dumps(
            {"prior_diag_m": prior, "chosen": {k: v for k, v in best.items() if k != "T"},
             "candidates": [{k: v for k, v in r.items() if k != "T"} for r in est["all"]]},
            indent=1))

        n_frames = len(sorted((OUT / obj / "frames").glob("*.jpg")))
        frames_data = []
        for i in range(n_frames):
            try:
                frames_data.append(load_frame_data(obj, i))
            except FileNotFoundError:
                frames_data.append(None)
        frames_data = [fd if fd is not None else {"depth": np.zeros((2, 2)), "mask": np.zeros((2, 2), np.uint8), "K": np.eye(3)} for fd in frames_data]

        poses = track(mesh, best["factor"], np.array(best["T"]), frames_data)
        pose_dir = OUT / obj / "pose"
        pose_dir.mkdir(parents=True, exist_ok=True)
        for p in poses:
            (pose_dir / f"{p['frame']:05d}.json").write_text(json.dumps(p, indent=1))

        # QA: project mesh samples on ref / mid / last frame
        pts, _ = trimesh.sample.sample_surface(mesh, 3000)
        pts = pts * best["factor"]
        qa = []
        for i in [args.ref, n_frames // 2, n_frames - 1]:
            fd = load_frame_data(obj, i)
            T = np.array(poses[i]["T"])
            P = (T[:3, :3] @ pts.T).T + T[:3, 3]
            K = fd["K"]
            uv = (K @ (P / P[:, 2:3]).T).T[:, :2].astype(int)
            im = cv2.imread(str(sorted((OUT / obj / "frames").glob("*.jpg"))[i]))
            for u, v in uv:
                if 0 <= u < im.shape[1] and 0 <= v < im.shape[0]:
                    im[v, u] = (0, 0, 255)
            qa.append(im)
        cv2.imwrite(str(pose_dir / "qa.png"), np.vstack(qa))
        costs = [p["cost"] for p in poses if p["cost"] is not None]
        re = sum(1 for p in poses if p["reinit"])
        print(f"[ok] {obj}: poses -> {pose_dir} (median cost {np.median(costs)*1e3:.2f}mm, reinits {re})")


if __name__ == "__main__":
    main()
