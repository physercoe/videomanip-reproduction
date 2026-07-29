#!/usr/bin/env python3
"""Stage 8: grasp-window detection + ContactOpt grasp refinement.

For each object: find the grasp window [t1,t2] (hand within 5 cm of the object at t1,
stable grasp before object moves at t2), then refine the reconstructed MANO grasp with
ContactOpt (DeepContact target contact maps + DiffContact pose optimization), and export
the optimized hand joints for re-retargeting.

Run inside .venv-contactopt, from the repo root.
Requires MANO at third_party/hamer/_DATA/data/mano/MANO_RIGHT.pkl (user-provided).

Reads  outputs/<obj>/{hand,pose,mesh,scale}
Writes outputs/<obj>/contactopt/%05d.json (optimized joints3d + mano, camera frame)
       outputs/<obj>/contactopt/summary.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CO_ROOT = ROOT / "third_party" / "ContactOpt"
sys.path.insert(0, str(CO_ROOT))
os.chdir(CO_ROOT)  # contactopt uses relative paths (checkpoints/, mano/models)

MANO_DIR = ROOT / "third_party" / "hamer" / "_DATA" / "data" / "mano"
CONTACT_WINDOW_DIST = 0.05
OBJ_MOVE_THRESH = 0.02


def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """(3,3) or (...,3,3) rotmat -> axis-angle (...,3), scipy."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(*R.shape[:-2], 3)


def load_object_mesh(obj: str):
    mesh = trimesh.load(str(OUT / obj / "mesh" / "object.glb"), force="mesh")
    factor = json.loads((OUT / obj / "scale" / "scale.json").read_text())["chosen"]["factor"]
    mesh.apply_scale(factor)
    return mesh, factor


def grasp_window(obj: str, hand_records: list[dict], pose_records: list[dict],
                 mesh: trimesh.Trimesh) -> tuple[int, int]:
    """t1: first frame hand<5cm from object; t2: first frame after t1 the object moves."""
    n = len(hand_records)
    Ts = [np.array(p["T"]) for p in pose_records]
    base_t = np.median(np.stack([T[:3, 3] for T in Ts[:10]]), axis=0)
    moved = np.array([np.linalg.norm(T[:3, 3] - base_t) > OBJ_MOVE_THRESH for T in Ts])
    obj_pts = np.asarray(mesh.vertices)
    d_min = np.full(n, np.inf)
    for i in range(n):
        joints = np.array(hand_records[i]["joints3d"]) + np.array(hand_records[i]["t_metric"])
        obj_cam = (Ts[i][:3, :3] @ obj_pts[::5].T).T + Ts[i][:3, 3]
        # joint-to-object-cloud min distance (21 x ~3k -> cheap)
        d = np.sqrt(((joints[:, None, :] - obj_cam[None, :, :]) ** 2).sum(-1)).min()
        d_min[i] = d
    t1 = int(np.argmax(d_min < CONTACT_WINDOW_DIST)) if (d_min < CONTACT_WINDOW_DIST).any() else 0
    after = np.where(moved & (np.arange(n) > t1))[0]
    t2 = int(after[0]) if len(after) else n - 1
    return t1, t2


def mano_to_contactopt(rec: dict, pca_basis: np.ndarray, pca_mean: np.ndarray):
    """HaMeR record -> (beta(10), pose18 (3 aa + 15 pca), mTc 4x4)."""
    mano = rec["mano"]
    go = np.array(mano["global_orient"]).reshape(3, 3)
    hp = np.array(mano["hand_pose"]).reshape(15, 3, 3)
    aa_global = rotmat_to_aa(go)
    aa45 = rotmat_to_aa(hp).reshape(-1)
    pca15 = np.linalg.lstsq(pca_basis.T, (aa45 - pca_mean), rcond=None)[0]
    pose18 = np.concatenate([aa_global, pca15])
    mTc = np.eye(4)
    mTc[:3, 3] = np.array(rec["t_metric"])  # translation only; rotation lives in pose[0:3]
    return np.array(mano["betas"]).reshape(-1), pose18, mTc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=250)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--src", default="hand",
                    help="input records dir under outputs/<obj>/ (default: hand)")
    ap.add_argument("--out", default="contactopt",
                    help="output dir under outputs/<obj>/ (default: contactopt)")
    args = ap.parse_args()

    assert (MANO_DIR / "MANO_RIGHT.pkl").exists(), \
        f"MANO_RIGHT.pkl missing at {MANO_DIR} (user must download from mano.is.tue.mpg.de)"
    mano_models = CO_ROOT / "mano" / "models"
    mano_models.mkdir(parents=True, exist_ok=True)
    for f in MANO_DIR.glob("*.pkl"):
        dst = mano_models / f.name
        if not dst.exists():
            dst.symlink_to(f)

    import contactopt.util as util
    from contactopt.hand_object import HandObject
    from contactopt.deepcontact_net import DeepContactNet
    from contactopt.optimize_pose import optimize_pose
    from manopth.manolayer import ManoLayer

    device = torch.device(args.device)
    model = DeepContactNet()
    ckpt = torch.load(CO_ROOT / "checkpoints" / "deepcontact_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    mano_layer = ManoLayer(mano_root=str(CO_ROOT / "mano" / "models"), use_pca=True,
                           ncomps=15, side="right", flat_hand_mean=False).to(device)
    # PCA basis for aa45 -> pca15 conversion (from the manopth layer buffers)
    pca_basis = mano_layer.th_selected_comps.detach().cpu().numpy()                # (15,45)
    pca_mean = mano_layer.th_hands_mean.detach().cpu().numpy().reshape(-1)         # (45,)

    for obj in args.objects:
        mesh, factor = load_object_mesh(obj)
        hand_records = [json.loads(p.read_text())
                        for p in sorted((OUT / obj / args.src).glob("*.json"))]
        pose_records = [json.loads(p.read_text())
                        for p in sorted((OUT / obj / "pose").glob("*.json"))]
        t1, t2 = grasp_window(obj, hand_records, pose_records, mesh)
        print(f"[{obj}] grasp window [{t1},{t2}] of {len(hand_records)} frames")

        out_dir = OUT / obj / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        n_opt = 0
        for fi in range(t1, t2 + 1, args.stride):
            rec = hand_records[fi]
            T = np.array(pose_records[fi]["T"])
            obj_mesh_cam = mesh.copy().apply_transform(T)

            ho = HandObject()
            beta, pose18, mTc = mano_to_contactopt(rec, pca_basis, pca_mean)
            ho.load_from_mano_params(hand_beta=beta, hand_pose=pose18, hand_trans=mTc[:3, 3],
                                     obj_faces=obj_mesh_cam.faces, obj_verts=obj_mesh_cam.vertices)
            ho_gt = HandObject()
            ho_gt.load_from_ho(ho)
            ho_gt.hand_contact = np.array(ho.hand_contact)  # load_from_ho doesn't copy it
            sample = {"ho_aug": ho, "ho_gt": ho_gt}
            sample["obj_sampled_idx"] = np.random.randint(
                0, len(ho_gt.obj_verts), util.SAMPLE_VERTS_NUM)
            sample["hand_feats_aug"], sample["obj_feats_aug"] = \
                ho.generate_pointnet_features(sample["obj_sampled_idx"])

            from contactopt.loader import ContactDBDataset
            ds = ContactDBDataset([sample], min_num_cont=1)
            data = ContactDBDataset.collate_fn([ds[0]])
            data_gpu = util.dict_to_device(data, device)
            with torch.no_grad():
                net_out = model(data_gpu["hand_verts_aug"], data_gpu["hand_feats_aug"],
                                data_gpu["obj_sampled_verts_aug"], data_gpu["obj_feats_aug"])
            hand_ct = util.class_to_val(net_out["contact_hand"]).unsqueeze(2)
            obj_ct = util.class_to_val(net_out["contact_obj"]).unsqueeze(2)
            out_pose, out_tform, obj_rot, opt_state = optimize_pose(
                data_gpu, hand_ct, obj_ct, n_iter=args.n_iter, lr=0.01,
                w_cont_hand=2.5, w_cont_obj=1, ncomps=15, w_cont_asym=2,
                w_opt_trans=0.3, w_opt_pose=1.0, w_opt_rot=1, caps_rad=0.001,
                contact_norm_method=0, caps_top=0.0005, caps_bot=-0.001, w_pen_cost=320,
                pen_it=0, w_obj_rot=0)
            opt_loss = float(opt_state[-1]["loss"].mean())
            # optimized MANO -> joints3d in camera frame (mano_pose_out already
            # includes the initial pose; out_tform is the aggregated cam transform)
            hand_verts, hand_joints = util.forward_mano(
                mano_layer, out_pose, data_gpu["hand_beta_aug"], [out_tform])
            j3d = hand_joints[0].detach().cpu().numpy()
            mano_out = {
                "global_orient_aa": out_pose[0, 0:3].detach().cpu().numpy().tolist(),
                "pose_pca15": out_pose[0, 3:18].detach().cpu().numpy().tolist(),
                "betas": beta.tolist(),
                "tform_cam": out_tform[0].detach().cpu().numpy().tolist(),
            }
            (out_dir / f"{fi:05d}.json").write_text(json.dumps({
                "frame": fi, "joints3d": j3d.tolist(), "t_metric": [0, 0, 0],
                "mano": mano_out, "opt_loss": opt_loss,
                "src": "contactopt"}, indent=1))
            n_opt += 1
            print(f"  [{obj}] frame {fi} optimized, loss={opt_loss:.4f}")
        (out_dir / "summary.json").write_text(json.dumps(
            {"t1": t1, "t2": t2, "n_optimized": n_opt, "stride": args.stride}, indent=1))
        print(f"[ok] {obj}: {n_opt} optimized grasps -> {out_dir}")


if __name__ == "__main__":
    main()
