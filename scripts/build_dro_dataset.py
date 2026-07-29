#!/usr/bin/env python3
"""Stage 9a: build the DRO-Grasp training dataset from our optimized grasps.

Creates, inside third_party/drograsp:
  data/data_urdf/object/videomanip/<obj>/<obj>.stl     (scaled object mesh)
  data/CMapDataset_videomanip/cmap_dataset.pt          {"metadata": [(q19, "videomanip+<obj>", "inspire")]}
  data/CMapDataset_videomanip/split_train_validate_objects.json

Train with: DRO_DATASET_DIR=data/CMapDataset_videomanip python train.py
            dataset.robot_names=['inspire']

Grasp sources (priority): outputs/<obj>/retarget_contactopt/*.npy (ContactOpt-optimized,
re-retargeted) > outputs/<obj>/retarget/*.npy within the contactopt grasp window.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DRO = ROOT / "third_party" / "drograsp"


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
    """q19 wrist (x,y,z,roll,pitch,yaw) is in the camera frame; the DRO dataset needs it
    in the object's mesh frame: wrist_obj = T_cam_obj^{-1} * wrist_cam."""
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
    ap.add_argument("--validate", nargs="*", default=[],
                    help="objects to hold out for validation (default: all train)")
    ap.add_argument("--augment", type=int, default=30,
                    help="perturbed copies per grasp (finger noise ±0.15 rad, wrist ±1 cm/±10°)")
    ap.add_argument("--src", nargs="+", default=None, metavar="NPY_DIR[:JSONS_DIR]",
                    help="retarget npy dir(s) under outputs/<obj>/, optionally with the "
                         "records dir giving true frame ids (e.g. "
                         "retarget_contactopt_handflow:contactopt_handflow). "
                         "Default: retarget_contactopt[:contactopt] if it exists else retarget")
    ap.add_argument("--name", default="CMapDataset_videomanip",
                    help="output dataset dir name under third_party/drograsp/data/")
    args = ap.parse_args()

    obj_dir = DRO / "data" / "data_urdf" / "object" / "videomanip"
    ds_dir = DRO / "data" / args.name
    obj_dir.mkdir(parents=True, exist_ok=True)
    ds_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for obj in args.objects:
        # scaled object mesh
        mesh = trimesh.load(str(OUT / obj / "mesh" / "object.glb"), force="mesh")
        factor = json.loads((OUT / obj / "scale" / "scale.json").read_text())["chosen"]["factor"]
        mesh.apply_scale(factor)
        dst = obj_dir / obj
        dst.mkdir(exist_ok=True)
        mesh.export(str(dst / f"{obj}.stl"))

        # grasp sources: list of (npy_dir, jsons_dir|None)
        if args.src:
            sources = []
            for spec in args.src:
                npy_name, _, jsons_name = spec.partition(":")
                sources.append((OUT / obj / npy_name,
                                (OUT / obj / jsons_name) if jsons_name else None))
        else:
            co_dir = OUT / obj / "retarget_contactopt"
            sources = [(co_dir, OUT / obj / "contactopt") if co_dir.exists()
                       else (OUT / obj / "retarget", None)]

        # grasp q19s
        n = 0
        for src_dir, jsons_dir in sources:
            npys = sorted(src_dir.glob("*.npy"))
            if jsons_dir is not None:
                frame_ids = [int(p.stem) for p in sorted(jsons_dir.glob("*.json"))
                             if p.stem.isdigit()]
                assert len(frame_ids) == len(npys), \
                    f"{obj}: {len(npys)} npys vs {len(frame_ids)} jsons in {jsons_dir.name}"
            else:
                frame_ids = [int(p.stem) for p in npys]
            for fp, fi in zip(npys, frame_ids):
                q = np.load(fp)
                assert q.size == 19, f"{fp}: expected q19, got {q.shape}"
                # camera frame -> object mesh frame (CRITICAL: dataset is object-relative)
                pose_p = OUT / obj / "pose" / f"{fi:05d}.json"
                if pose_p.exists():
                    T = np.array(json.loads(pose_p.read_text())["T"])
                    q = grasp_to_object_frame(q, T)
                else:
                    print(f"  [warn] no pose for frame {fi}, skipping")
                    continue
                metadata.append((torch.from_numpy(q).float(), f"videomanip+{obj}", "inspire"))
                n += 1
                rng = np.random.default_rng(fi * 7919)
                for _ in range(args.augment - 1):
                    qa = q.copy()
                    qa[7:] += rng.normal(0, 0.15, 12)            # finger joint noise (rad)
                    qa[1:4] += rng.normal(0, 0.01, 3)            # wrist translation noise (m)
                    qa[4:7] += rng.normal(0, np.deg2rad(10), 3)  # wrist rotation noise (rad)
                    metadata.append((torch.from_numpy(qa).float(), f"videomanip+{obj}", "inspire"))
        print(f"[{obj}] {n} grasps (+{args.augment - 1}x augment) from "
              f"{[s[0].name for s in sources]}")

    torch.save({"metadata": metadata}, ds_dir / "cmap_dataset.pt")
    split = {"train": [f"videomanip+{o}" for o in args.objects if o not in args.validate],
             "validate": [f"videomanip+{o}" for o in args.validate]}
    (ds_dir / "split_train_validate_objects.json").write_text(json.dumps(split, indent=1))
    print(f"[ok] dataset: {len(metadata)} grasps, train={len(split['train'])}, "
          f"validate={len(split['validate'])} -> {ds_dir}")


if __name__ == "__main__":
    main()
