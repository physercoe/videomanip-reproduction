#!/usr/bin/env python3
"""Fit an Inspire-hand q19 to one of the paper's reference grasp GLBs
(Adam on differentiable FK: match URDF-FK surface points to GLB hand mesh vertices).

Usage: python scripts/fit_glb_grasp.py <object>
Output: outputs/<obj>/eval/paper_grasp.npz (q19, object at GLB frame -> origin)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent.parent
DRO = ROOT / "third_party" / "drograsp"
sys.path.insert(0, str(DRO))
from utils.hand_model import create_hand_model  # noqa: E402


def main() -> None:
    obj = sys.argv[1]
    device = "cuda"
    hand = create_hand_model("inspire", torch.device(device))
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = torch.tensor(lower, dtype=torch.float32, device=device)
    upper = torch.tensor(upper, dtype=torch.float32, device=device)

    scene = trimesh.load(str(ROOT / f"data/reference/glb/{obj}.glb"), force="scene")
    glb_hand = np.asarray(scene.geometry[[k for k in scene.geometry if k.startswith("geometry_")][0]].vertices)
    glb_obj = np.asarray(scene.geometry[[k for k in scene.geometry if k.startswith("obj_mesh")][0]].vertices)
    H = torch.tensor(glb_hand, dtype=torch.float32, device=device)

    # init: wrist translation = hand centroid - robot hand centroid (zero pose)
    with torch.no_grad():
        st0 = hand.pk_chain.forward_kinematics(torch.zeros(1, 19, device=device))
        rc = torch.cat([st0[l].get_matrix()[0][:3, 3] for l in st0.keys()]).mean(0)
    q = torch.zeros(19, device=device)
    q[1:4] = torch.tensor(glb_hand.mean(0), device=device) - rc
    q = q.detach().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=1e-2)

    # dense surface points from the URDF link meshes (drograsp hand.vertices, in link frames)
    surf = {}
    for lname, verts in hand.vertices.items():
        if lname.startswith("R_"):
            v = torch.tensor(verts, dtype=torch.float32, device=device)
            sel = torch.randperm(len(v))[:150]  # 150 pts/link keeps it fast
            surf[lname] = v[sel]
    link_names = list(surf.keys())
    for it in range(600):
        opt.zero_grad()
        status = hand.pk_chain.forward_kinematics(q.unsqueeze(0))
        P = torch.cat([
            (torch.cat([surf[l], torch.ones(len(surf[l]), 1, device=device)], dim=1)
             @ status[l].get_matrix()[0].T)[:, :3]
            for l in link_names])
        d = torch.cdist(P.unsqueeze(0), H.unsqueeze(0)).squeeze(0)
        d1 = d.min(dim=1).values.mean()
        d2 = d.min(dim=0).values.mean()
        viol = torch.clamp(lower - q, min=0) + torch.clamp(q - upper, min=0)
        loss = d1 + d2 + 100.0 * (viol ** 2).sum()
        loss.backward()
        opt.step()
        q.data = torch.max(torch.min(q.data, upper), lower)
        if it % 100 == 0:
            print(f"it {it}: chamfer {loss.item()*1000:.2f}mm")
    print("final chamfer:", loss.item() * 1000, "mm")
    q_out = q.detach().cpu().numpy()
    # object pose in GLB frame: keep GLB frame; eval places object at origin => shift both
    out_dir = ROOT / "outputs" / obj / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_center = glb_obj.mean(0)
    q_out[1:4] -= obj_center  # object centroid -> origin
    np.savez(out_dir / "paper_grasp.npz", q=q_out[None].astype(np.float32),
             obj_pos=np.zeros((1, 3), dtype=np.float32),
             obj_quat=np.array([[0, 0, 0, 1]], dtype=np.float32))
    print(f"[ok] fitted paper grasp for {obj} -> {out_dir / 'paper_grasp.npz'}")


if __name__ == "__main__":
    main()
