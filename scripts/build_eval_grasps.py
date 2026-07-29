#!/usr/bin/env python3
"""Build the object-at-origin eval npz from a retarget directory.

Reconstructs the convention of outputs/<obj>/eval/retarget_grasps.npz:
frames = grasp window (contactopt/summary.json [t1,t2]) strided by 2;
q19 wrist pose transformed camera frame -> object mesh frame (same math as
build_dro_dataset.grasp_to_object_frame); object spawned at origin, identity quat.

Usage: python scripts/build_eval_grasps.py <objects...> [--src retarget_handflow]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def rot_to_zyx_euler(R: np.ndarray) -> np.ndarray:
    """drograsp FK convention R = Rx(roll)Ry(pitch)Rz(yaw) -> (roll,pitch,yaw)."""
    pitch = np.arcsin(np.clip(R[0, 2], -1, 1))
    roll = np.arctan2(-R[1, 2], R[2, 2])
    yaw = np.arctan2(-R[0, 1], R[0, 0])
    return np.array([roll, pitch, yaw])


def zyx_euler_to_rot(e: np.ndarray) -> np.ndarray:
    """drograsp FK convention: R = Rx(roll) @ Ry(pitch) @ Rz(yaw)."""
    roll, pitch, yaw = e
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def grasp_to_object_frame(q_cam: np.ndarray, T_cam_obj: np.ndarray) -> np.ndarray:
    q = q_cam.copy()
    T_inv = np.linalg.inv(T_cam_obj)
    p = np.append(q_cam[1:4], 1.0)
    q[1:4] = (T_inv @ p)[:3]
    R_cam = zyx_euler_to_rot(q_cam[4:7])
    q[4:7] = rot_to_zyx_euler(T_inv[:3, :3] @ R_cam)
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+")
    ap.add_argument("--src", default="retarget",
                    help="retarget dir name under outputs/<obj>/ (default: retarget)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--src-jsons", default=None,
                    help="records dir whose json stems give the true frame ids "
                         "(needed when --src is a re-retargeted contactopt dir: its npys "
                         "are enumerated 0..N-1 in sorted-json order, losing frame ids)")
    ap.add_argument("--out", default=None,
                    help="output npz name under outputs/<obj>/eval/ (default: <src>_grasps.npz)")
    args = ap.parse_args()

    for obj in args.objects:
        src_dir = OUT / obj / args.src
        assert src_dir.is_dir(), f"{src_dir} missing"
        if args.src_jsons:
            frames = [int(p.stem) for p in sorted((OUT / obj / args.src_jsons).glob("*.json"))
                      if p.stem.isdigit()]
        else:
            s = json.loads((OUT / obj / "contactopt" / "summary.json").read_text())
            frames = list(range(s["t1"], s["t2"] + 1, args.stride))
        qs = []
        for npy_i, fi in enumerate(frames):
            # dense retarget dirs are frame-indexed; contactopt-retarget dirs are 0..N-1
            q_p = src_dir / f"{fi:05d}.npy" if not args.src_jsons else src_dir / f"{npy_i:05d}.npy"
            pose_p = OUT / obj / "pose" / f"{fi:05d}.json"
            if not q_p.exists() or not pose_p.exists():
                continue
            q = np.load(q_p)
            T = np.array(json.loads(pose_p.read_text())["T"])
            qs.append(grasp_to_object_frame(q, T))
        qs = np.stack(qs).astype(np.float32)
        out_dir = OUT / obj / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = args.out or f"{args.src}_grasps.npz"
        np.savez(out_dir / name,
                 q=qs,
                 obj_pos=np.zeros((len(qs), 3), dtype=np.float32),
                 obj_quat=np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(qs), 1)))
        print(f"[ok] {obj}: {len(qs)} grasps -> {out_dir / name}")


if __name__ == "__main__":
    main()
