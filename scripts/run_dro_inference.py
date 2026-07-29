#!/usr/bin/env python3
"""Stage 9b: DRO inference — predict grasps for our objects (no IsaacGym).

Loads the trained DRO network, samples N initial hand configurations per object,
predicts D(R,O), recovers grasp configurations via multilateration + cvxpylayers IK,
and exports an eval npz per object (q19, object at origin — DRO convention).

Usage: python scripts/run_dro_inference.py --epoch 200 [--n_samples 100] [--objects ...]
Run inside .venv-recon, cwd anywhere.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DRO = ROOT / "third_party" / "drograsp"
sys.path.insert(0, str(DRO))
os.environ.setdefault("DRO_DATASET_DIR", "data/CMapDataset_videomanip")

from model.network import create_network          # noqa: E402
from data_utils.CMapDataset import CMapDataset    # noqa: E402
from utils.multilateration import multilateration # noqa: E402
from utils.se3_transform import compute_link_pose # noqa: E402
from utils.optimization import create_problem, optimization, process_transform  # noqa: E402
from utils.hand_model import create_hand_model    # noqa: E402

RUN_NAME = "videomanip_inspire_aug"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="200", help="epoch number or ckpt suffix (e.g. avg_e1to5)")
    ap.add_argument("--run", default=RUN_NAME, help="run dir under drograsp/output/")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--objects", nargs="*", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--filter", action="store_true",
                    help="also write predicted_grasps_filtered.npz (FK-based contact gate)")
    ap.add_argument("--min_contacts", type=int, default=100)
    ap.add_argument("--max_pen_pts", type=int, default=300)
    args = ap.parse_args()

    os.chdir(DRO)
    device = torch.device(f"cuda:{args.gpu}")
    from omegaconf import OmegaConf
    cfg_model = OmegaConf.load("configs/model.yaml")["model"]
    cfg_model.pretrain = None

    network = create_network(cfg_model, mode="validate").to(device)
    ckpt = DRO / "output" / args.run / "state_dict" / f"epoch_{args.epoch}.pth"
    network.load_state_dict(torch.load(ckpt, map_location=device))
    network.eval()
    print(f"[infer] loaded {ckpt}")

    hand = create_hand_model("inspire", device)
    split = json.loads((DRO / os.environ.get("DRO_DATASET_DIR", "data/CMapDataset_filtered") /
                        "split_train_validate_objects.json").read_text())
    objects = args.objects or [o.split("+")[1] for o in split["train"] + split["validate"]]
    print(f"[infer] objects: {objects}")

    for obj in objects:
        # object point cloud from our scaled mesh (512 pts + noise, their 'random' mode)
        import trimesh
        mesh = trimesh.load(str(OUT / obj / "mesh" / "object.glb"), force="mesh")
        factor = json.loads((OUT / obj / "scale" / "scale.json").read_text())["chosen"]["factor"]
        mesh.apply_scale(factor)
        pc_all, _ = trimesh.sample.sample_surface(mesh, 65536)
        idx = torch.randperm(65536)[:512]
        object_pc = torch.tensor(pc_all[idx], dtype=torch.float32, device=device)
        object_pc = object_pc + torch.randn_like(object_pc) * 0.002
        initial_qs, robot_pcs = [], []
        for _ in range(args.n_samples):
            q0 = hand.get_initial_q()
            initial_qs.append(q0)
            robot_pcs.append(hand.get_transformed_links_pc(q0)[:, :3])
        initial_q = torch.stack(initial_qs).to(device)
        robot_pc = torch.stack(robot_pcs).to(device)
        object_pc_b = object_pc.unsqueeze(0).expand(args.n_samples, -1, -1)

        # chunked like validate.py (split_batch_size=25) to bound GPU memory
        chunk = 25
        tf_l, q_l = [], []
        hand_cpu = create_hand_model("inspire", torch.device("cpu"))
        links_pc_cpu = {k: v.cpu() for k, v in hand.links_pc.items()}
        for s0 in range(0, args.n_samples, chunk):
            s1 = min(s0 + chunk, args.n_samples)
            with torch.no_grad():
                dro = network(robot_pc[s0:s1], object_pc_b[s0:s1])["dro"].detach()
            mlat = multilateration(dro, object_pc_b[s0:s1])
            # IK recovery on CPU (cvxpylayers diffcp needs numpy/cpu tensors)
            mlat_c = mlat.cpu()
            tf, _ = compute_link_pose(links_pc_cpu, mlat_c, is_train=False)
            otf = process_transform(hand_cpu.pk_chain, tf)
            layer = create_problem(hand_cpu.pk_chain, otf.keys())
            q_l.append(optimization(hand_cpu.pk_chain, layer, initial_q[s0:s1].cpu(), otf))
            tf_l.append(otf)
        predict_q = torch.cat(q_l, dim=0)

        q = predict_q.detach().cpu().numpy().astype(np.float32)
        out_dir = ROOT / "outputs" / obj / "dro"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "predicted_grasps.npz",
                 q=q,
                 obj_pos=np.zeros((len(q), 3), dtype=np.float32),
                 obj_quat=np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(q), 1)))
        # FK-based contact filter: FK the RECOVERED q -> hand surface pts -> object
        # distance. Keeps grasps that actually touch the object with limited
        # penetration (replaces the old mlat-pc dmin gate, which was ~0 by
        # construction and gated nothing).
        if args.filter:
            sys.path.insert(0, str(ROOT / "scripts"))
            from run_refine import load_object_surface, hand_surface_points, contact_metrics
            obj_pts, obj_nrm, _ = load_object_surface(obj, device)
            n_c, n_p, m_p = [], [], []
            with torch.no_grad():
                for s0 in range(0, len(q), 25):
                    hp = hand_surface_points(hand, torch.tensor(
                        q[s0:s0 + 25], dtype=torch.float32, device=device))
                    _, _, nc, mp, npn = contact_metrics(hp, obj_pts, obj_nrm)
                    n_c += nc.tolist(); n_p += npn.tolist(); m_p += (mp * 1e3).tolist()
            n_c, n_p, m_p = np.array(n_c), np.array(n_p), np.array(m_p)
            keep = (n_c >= args.min_contacts) & (n_p <= args.max_pen_pts)
            qf = q[keep]
            np.savez(out_dir / "predicted_grasps_filtered.npz",
                     q=qf,
                     obj_pos=np.zeros((len(qf), 3), dtype=np.float32),
                     obj_quat=np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(qf), 1)),
                     n_contacts=n_c, n_pen_pts=n_p, max_pen_mm=m_p)
            print(f"[filter] {obj}: {keep.sum()}/{len(q)} grasps with "
                  f">={args.min_contacts} contacts, <={args.max_pen_pts} pen pts "
                  f"(median contacts {np.median(n_c):.0f})")
        print(f"[ok] {obj}: {len(q)} predicted grasps -> {out_dir / 'predicted_grasps.npz'}")


if __name__ == "__main__":
    main()
