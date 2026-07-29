#!/usr/bin/env python3
"""Stage 5b: HandFlow video hand reconstruction (hand-stage upgrade for HaMeR).

Replaces per-frame HaMeR + MoGe metric-depth correction with HandFlow
(arXiv:2607.11221): WiLoR YOLO detection -> online HaMeR features/landmarks ->
flow-matching temporal denoiser (overlapping windows) -> per-frame MANO pose +
metric translation in the camera frame. Fixed-camera mode (c2w = identity);
intrinsics come from the MoGe depth stage.

Writes the SAME schema as run_hamer.py so all downstream stages consume it
unchanged (retarget: joints3d+t_metric; contactopt: mano rotmats):
    outputs/<obj>/handflow/%05d.json   MANO params + joints3d (OpenPose) + t_metric
    outputs/<obj>/handflow/mesh_%05d.ply
    outputs/<obj>/handflow/handflow_raw.npz  (pose/trans/betas raw sequence)
    outputs/<obj>/handflow/qa.png            (2D overlay contact sheet)

Env (set with defaults here): HAMER_CKPT, DETECTOR_CKPT, MANO_ROOT,
HANDFLOW_NORMALIZATION_STATS.
Run inside .venv-handflow (py3.11, torch 2.7.0+cu128 — same stack as .venv-recon).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
HF_ROOT = ROOT / "third_party" / "HandFlow"
sys.path.insert(0, str(HF_ROOT))

# env must be set BEFORE importing HandFlow utils (they read env at import time)
os.environ.setdefault("HAMER_CKPT",
                      str(ROOT / "third_party" / "hamer" / "_DATA" / "hamer_ckpts"
                          / "checkpoints" / "hamer.ckpt"))
os.environ.setdefault("DETECTOR_CKPT", "/app/models/handflow/detector.pt")
os.environ.setdefault("MANO_ROOT", "/app/models/mano/mano_v1_2/models")
os.environ.setdefault("HANDFLOW_NORMALIZATION_STATS",
                      "/app/models/handflow/normalization_stats.npz")

# NOTE on joint order: manopth master (hassony2/manopth@4f1dcad) returns its 21
# joints ALREADY in OpenPose order (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle,
# 13-16 ring, 17-20 pinky) — verified empirically on the zero pose (four-joint
# continuous chains, thumb toward +z) and by bone-length equality with HaMeR's
# OpenPose joints. Do NOT apply hamer's mano_to_openpose permutation here (that
# map is for smplx MANOLayer's raw MANO order) — applying it scrambles the hand.


def aa_to_rotmats(pose48: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """48D axis-angle -> (3,3) global rotmat + (15,3,3) joint rotmats."""
    go, _ = cv2.Rodrigues(pose48[:3].astype(np.float64))
    hp = np.stack([cv2.Rodrigues(pose48[3 + 3 * i: 6 + 3 * i].astype(np.float64))[0]
                   for i in range(15)])
    return go.astype(np.float32), hp.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+")
    ap.add_argument("--side", default="right", choices=["right", "left"],
                    help="target hand side (FM model is right-hand trained)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fm_ckpt", default="/app/models/handflow/handflow_denoiser.pt")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from utils.checkpoint_utils import load_denoiser_from_ckpt
    from utils.inference_utils import (run_fm_inference_with_overlap,
                                       split_sequence_into_windows)
    from utils.mano_utils import MANOForwardKinematics
    from utils.online_hamer import OnlineHaMeRPipeline

    device = torch.device(args.device)
    cfg = OmegaConf.load(str(HF_ROOT / "configs" / "inference.yaml"))
    model_cfg = OmegaConf.load(str(HF_ROOT / cfg.model_yaml))
    merged = OmegaConf.merge(model_cfg, cfg)

    denoiser = load_denoiser_from_ckpt(merged, args.fm_ckpt, device)
    online = OnlineHaMeRPipeline(device=str(device))
    mano_fk = MANOForwardKinematics(os.environ["MANO_ROOT"], device)

    win = int(cfg.inference.window_size)
    overlap = int(cfg.inference.overlap_size)
    ode_steps = int(cfg.inference.ode_steps)

    for obj in args.objects:
        frames_dir = OUT / obj / "frames"
        out_dir = OUT / obj / "handflow"
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        frames_bgr = [cv2.imread(str(p)) for p in frame_paths]
        T = len(frames_bgr)

        intr_file = json.loads((OUT / obj / "depth" / "intrinsics.json").read_text())
        keys = sorted(intr_file.keys())
        intr_list = []
        for i in range(T):
            K = np.array(intr_file[keys[i]]["K"])
            intr_list.append(np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                                      dtype=np.float32))

        print(f"[{obj}] {T} frames, online HaMeR ...")
        res = online.process_sequence(frames_bgr, intr_list, target_side=args.side)
        det_valid = res["detection_valid"].numpy()
        det_sides = res["sides"]
        print(f"  detections: {int(det_valid.sum())}/{T}")
        if not det_valid.any():
            print(f"  [skip] {obj}: no '{args.side}' hand detected in any frame")
            continue
        dominant_side = max(set(det_sides), key=det_sides.count) if det_sides else args.side

        bf = res["backbone_features"].to(device)
        image_tokens = denoiser.frame_compressor(bf.unsqueeze(0)).squeeze(0)
        seq_batch = {
            "mano_params": torch.zeros((1, T, 48), dtype=torch.float32),
            "mano_trans": torch.zeros((1, T, 3), dtype=torch.float32),
            "mano_betas": torch.zeros((1, 10), dtype=torch.float32),
            "padding_mask": torch.zeros((1, T), dtype=torch.bool),
            "images": res["crop_images"].unsqueeze(0),
            "hamer_landmarks": res["hamer_landmarks"].unsqueeze(0),
            "crop_intrinsics": res["crop_intrinsics"].unsqueeze(0),
            "hamer_confidence": res["hamer_confidence"].unsqueeze(0),
            "image_tokens": image_tokens.unsqueeze(0),
            "side": [dominant_side],
            "source": ["custom"],
        }
        win_batch, _ = split_sequence_into_windows(seq_batch, win, win - overlap, device)
        win_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in win_batch.items()}

        print(f"[{obj}] FM denoiser ({win_batch['mano_params'].shape[0]} windows) ...")
        pose_seq, trans_seq, betas_pred = run_fm_inference_with_overlap(
            denoiser, denoiser, win_batch, win, overlap, ode_steps, device,
            overlap_method=cfg.inference.get("overlap_method", "vblend"))
        nf = pose_seq.shape[0]

        # MANO FK -> camera-frame joints/verts (meters); MANO-space = trans removed
        sides = (det_sides[:nf] if len(det_sides) >= nf
                 else det_sides + [det_sides[-1]] * (nf - len(det_sides)))
        if betas_pred.ndim == 1:
            betas_pred = betas_pred.unsqueeze(0).expand(nf, -1)
        zero_t = torch.zeros_like(trans_seq)
        joints_cam = (mano_fk.joints(pose_seq, betas_pred, trans_seq, sides)
                      / 1000.0).cpu().numpy()
        joints_mano = (mano_fk.joints(pose_seq, betas_pred, zero_t, sides)
                       / 1000.0).cpu().numpy()
        trans_np = trans_seq.cpu().numpy()
        betas_np = betas_pred[0].cpu().numpy()
        faces = mano_fk.get_faces(args.side)

        np.savez(out_dir / "handflow_raw.npz",
                 pose=pose_seq.cpu().numpy(), trans=trans_np, betas=betas_np,
                 det_valid=det_valid, sides=np.array(sides), nf=nf, T=T)

        bbox_xyxy = res["bbox_xyxy"].numpy()
        conf = res["hamer_confidence"].numpy()
        qa = []
        for i in range(min(nf, T)):
            pose48 = pose_seq[i].cpu().numpy()
            go, hp = aa_to_rotmats(pose48)
            rec = {
                "mano": {"global_orient": go.tolist(), "hand_pose": hp.tolist(),
                         "betas": betas_np.tolist()},
                "joints3d": joints_mano[i].tolist(),
                "t_metric": trans_np[i].tolist(),
                "box": bbox_xyxy[i].tolist(),
                "detection_valid": bool(det_valid[i]),
                "hamer_confidence": float(conf[i]),
                "side": sides[i],
            }
            (out_dir / f"{i:05d}.json").write_text(json.dumps(rec, indent=1))

            # QA overlay on first/mid/last: project camera-frame joints
            if i in (0, T // 2, T - 1):
                K = np.array(intr_file[keys[i]]["K"])
                jc = joints_cam[i]
                uv = (K @ (jc / jc[:, 2:3]).T).T[:, :2]
                ov = frames_bgr[i].copy()
                for u, v in uv:
                    cv2.circle(ov, (int(u), int(v)), 4, (0, 0, 255), -1)
                x1, y1, x2, y2 = bbox_xyxy[i].astype(int)
                cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), 2)
                qa.append(ov)

            # hand mesh PLY in camera frame (sparse: every 10th + grasp segment)
            if i % 10 == 0:
                import trimesh
                verts = (mano_fk.verts(pose_seq[i:i + 1], betas_pred[i:i + 1],
                                       trans_seq[i:i + 1], [sides[i]])
                         / 1000.0).cpu().numpy()[0]
                trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(
                    str(out_dir / f"mesh_{i:05d}.ply"))

        if qa:
            cv2.imwrite(str(out_dir / "qa.png"), np.vstack(qa))
        wrist = joints_cam[:, 0]
        accel = np.linalg.norm(np.diff(wrist, n=2, axis=0), axis=1) * 1000  # mm/frame^2
        print(f"[ok] {obj}: {min(nf, T)} frames -> {out_dir}  "
              f"wrist |a| mean {accel.mean():.1f} mm/f^2 (smoothness metric)")


if __name__ == "__main__":
    main()
