#!/usr/bin/env python3
"""Stage 6b: robot-side contact refinement (ContactOpt analogue for the robot hand).

v2 — surface-based point-to-plane model (replaces the broken keypoint model that
equilibrated at a 2mm standoff and opened grasps):

Per grasp, optimize wrist (6dof) + 12 finger joints so that
  loss = w_att  * mean_{near hand pts} relu(|d_signed| - standoff)   (pull to surface)
       + w_pen  * mean_{near hand pts} relu(-d_signed)               (push out of object)
       + w_fdev * |dq_fingers|^2 + w_rdev * |dq_rot|^2 + w_tdev * |dq_trans|^2
where d_signed = dot(p_hand - o_nearest, n_nearest) against sampled object surface
points with outward normals. Only near points (< near_m) enter att/pen, so distant
fingers are not dragged (the v1 bug).

Inputs:  --grasps <eval npz>  (q19 + object at origin — the same files run_grasp_eval2x
         consumes, e.g. outputs/<obj>/eval/retarget_grasps.npz,
         outputs/<obj>/dro/predicted_grasps_filtered.npz; '<obj>' in the path is
         substituted per object)
Outputs: <name>_refined.npz next to the input + <name>_refined_qa.json with
         before/after contact counts and penetration depths.

Run inside .venv-recon (torch + pytorch_kinematics + trimesh).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DRO = ROOT / "third_party" / "drograsp"

import sys
sys.path.insert(0, str(DRO))
from utils.hand_model import create_hand_model  # noqa: E402

GAP_OK_M = 0.002    # outside points within this gap count as contact (no attraction)
PEN_ALLOW_M = 0.003   # penetration up to this depth is free (generates grip force)
NEAR_M = 0.008        # attraction/penetration active range
W_ATT, W_PEN = 1.0, 5.0
W_FDEV, W_RDEV, W_TDEV = 1.0, 1.0, 1000.0


def load_object_surface(obj: str, device, n_pts: int = 8192):
    """Scaled object mesh -> (pts, outward normals) torch tensors + quality report."""
    mesh = trimesh.load(str(OUT / obj / "mesh" / "object.glb"), force="mesh")
    factor = json.loads((OUT / obj / "scale" / "scale.json").read_text())["chosen"]["factor"]
    mesh.apply_scale(factor)
    trimesh.repair.fix_normals(mesh)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_pts)
    nrm = mesh.face_normals[face_idx]
    # orientation QA: p + eps*n should be outside, p - eps*n inside (parity rays)
    eps = 1e-3
    sub = np.random.default_rng(0).choice(len(pts), size=min(256, len(pts)), replace=False)
    p = pts[sub]
    out_ok = ~mesh.contains(p + eps * nrm[sub])
    in_ok = mesh.contains(p - eps * nrm[sub])
    agree = float((out_ok & in_ok).mean())
    return (torch.tensor(pts, dtype=torch.float32, device=device),
            torch.tensor(nrm, dtype=torch.float32, device=device),
            {"normal_agreement": agree, "watertight": bool(mesh.is_watertight)})


_VERT_CACHE = {}


def hand_surface_points(hand, q):
    """FK all links; return (B, Npts, 3) surface points for the contact links."""
    dev = q.device
    if dev not in _VERT_CACHE:
        _VERT_CACHE[dev] = {l: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev)
                            for l, v in hand.vertices.items()}
    status = hand.pk_chain.forward_kinematics(q)
    pts = []
    for lname, v in _VERT_CACHE[dev].items():
        m = status[lname].get_matrix()  # (B,4,4)
        p = torch.einsum("bij,nj->bni", m[:, :3, :3], v) + m[:, :3, 3].unsqueeze(1)
        pts.append(p)
    return torch.cat(pts, dim=1)


def contact_metrics(hand_pts, obj_pts, obj_nrm, chunk=2048):
    """(att, pen) means over near pairs + QA counts. hand_pts (B,N,3).
    att: one-sided pull of outside points into the GAP_OK band;
    pen: push penetration deeper than PEN_ALLOW back into the allowance.
    Contacts = points in [-PEN_ALLOW, +GAP_OK]. Chunked over hand points to
    bound the cdist memory."""
    B, N, _ = hand_pts.shape
    att_s = torch.zeros(B, device=hand_pts.device)
    pen_s = torch.zeros(B, device=hand_pts.device)
    near_s = torch.zeros(B, device=hand_pts.device)
    n_contact = torch.zeros(B, device=hand_pts.device)
    n_pen = torch.zeros(B, device=hand_pts.device)
    max_pen = torch.zeros(B, device=hand_pts.device)
    for s0 in range(0, N, chunk):
        hp = hand_pts[:, s0:s0 + chunk]
        d = torch.cdist(hp, obj_pts)                           # (B,n,M)
        dmin, j = d.min(dim=-1)                                # (B,n)
        n = obj_nrm[j]                                         # (B,n,3)
        o = torch.gather(obj_pts.unsqueeze(0).expand(B, -1, -1), 1,
                         j.unsqueeze(-1).expand(-1, -1, 3))    # (B,n,3)
        signed = ((hp - o) * n).sum(-1)                        # (B,n)
        near = dmin < NEAR_M
        att_s += (torch.relu(signed - GAP_OK_M) * near).sum(1)
        pen_s += (torch.relu(-signed - PEN_ALLOW_M) * near).sum(1)
        near_s += near.sum(1)
        n_contact += ((signed >= -PEN_ALLOW_M) & (signed < GAP_OK_M) & near).sum(1)
        n_pen += ((signed < -PEN_ALLOW_M) & near).sum(1)
        max_pen = torch.maximum(max_pen, (torch.relu(-signed) * near).amax(1))
    att = att_s / near_s.clamp(min=1)
    pen = pen_s / near_s.clamp(min=1)
    return att, pen, n_contact, max_pen, n_pen


def refine_batch(hand, q0: torch.Tensor, obj_pts, obj_nrm, iters: int, device):
    """q0 (B,19). Returns refined q (B,19) and QA dict of before/after metrics."""
    q0 = q0.to(device)
    B = q0.shape[0]
    fingers = q0[:, 7:].clone().requires_grad_(True)
    wrist_r = q0[:, 4:7].clone().requires_grad_(True)
    wrist_t = q0[:, 1:4].clone().requires_grad_(True)
    dummy = q0[:, :1]
    opt = torch.optim.Adam([fingers, wrist_r, wrist_t], lr=2e-2)
    lower, upper = hand.pk_chain.get_joint_limits()
    lo = torch.tensor(lower, dtype=torch.float32, device=device)
    hi = torch.tensor(upper, dtype=torch.float32, device=device)

    with torch.no_grad():
        hp0 = hand_surface_points(hand, q0)
        _, _, nc0, mp0, np0 = contact_metrics(hp0, obj_pts, obj_nrm)

    for it in range(iters):
        opt.zero_grad()
        q = torch.cat([dummy, wrist_t, wrist_r, fingers], dim=1)
        hp = hand_surface_points(hand, q)
        att, pen, _, _, _ = contact_metrics(hp, obj_pts, obj_nrm)
        loss = (W_ATT * att + W_PEN * pen
                + W_FDEV * ((fingers - q0[:, 7:]) ** 2).mean(1)
                + W_RDEV * ((wrist_r - q0[:, 4:7]) ** 2).mean(1)
                + W_TDEV * ((wrist_t - q0[:, 1:4]) ** 2).mean(1)).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            fingers.data = torch.max(torch.min(fingers.data, hi[7:]), lo[7:])
            # keep euler near the original branch
            wrist_r.data = q0[:, 4:7] + torch.clamp(wrist_r.data - q0[:, 4:7], -0.5, 0.5)
            wrist_t.data = q0[:, 1:4] + torch.clamp(wrist_t.data - q0[:, 1:4], -0.02, 0.02)

    with torch.no_grad():
        q = torch.cat([dummy, wrist_t, wrist_r, fingers], dim=1)
        hp1 = hand_surface_points(hand, q)
        _, _, nc1, mp1, np1 = contact_metrics(hp1, obj_pts, obj_nrm)
    qa = {
        "contacts_before": [int(x) for x in nc0], "contacts_after": [int(x) for x in nc1],
        "pen_pts_before": [int(x) for x in np0], "pen_pts_after": [int(x) for x in np1],
        "max_pen_mm_before": [float(x * 1e3) for x in mp0],
        "max_pen_mm_after": [float(x * 1e3) for x in mp1],
        "dq_fingers_rad": [float(x) for x in (q[:, 7:] - q0[:, 7:]).abs().amax(1)],
        "dwrist_mm": [float(x * 1e3) for x in (q[:, 1:4] - q0[:, 1:4]).norm(dim=1)],
    }
    return q.detach(), qa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+")
    ap.add_argument("--grasps", required=True,
                    help="eval npz (q19 + obj at origin), e.g. outputs/<obj>/eval/retarget_grasps.npz "
                         "or outputs/<obj>/dro/predicted_grasps_filtered.npz. NOTE: per-frame npy dirs "
                         "are in the reconstruction frame and NOT valid here.")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    hand = create_hand_model("inspire", device)

    for obj in args.objects:
        obj_pts, obj_nrm, mesh_qa = load_object_surface(obj, device)
        print(f"[{obj}] mesh QA: {mesh_qa}")

        src_npz = Path(args.grasps.replace("<obj>", obj))
        z = np.load(src_npz)
        q_all = torch.tensor(z["q"], dtype=torch.float32)
        meta = {k: z[k] for k in z.files if k != "q"}
        out_npz = src_npz.with_name(src_npz.stem + "_refined.npz")

        refined, qa_all = [], {"mesh": mesh_qa, "per_grasp": []}
        for s0 in range(0, len(q_all), args.batch):
            q_ref, qa = refine_batch(hand, q_all[s0:s0 + args.batch], obj_pts, obj_nrm,
                                     args.iters, device)
            refined.append(q_ref.cpu())
            qa_all["per_grasp"].extend([
                {k: v[i] for k, v in qa.items()} for i in range(len(qa["dwrist_mm"]))])
        q_ref = torch.cat(refined, dim=0).numpy().astype(np.float32)

        np.savez(out_npz, q=q_ref, **meta)
        qa_path = out_npz.with_name(out_npz.stem + "_qa.json")
        qa_path.write_text(json.dumps(qa_all, indent=1))
        nc0 = np.mean([g["contacts_before"] for g in qa_all["per_grasp"]])
        nc1 = np.mean([g["contacts_after"] for g in qa_all["per_grasp"]])
        mp0 = np.mean([g["max_pen_mm_before"] for g in qa_all["per_grasp"]])
        mp1 = np.mean([g["max_pen_mm_after"] for g in qa_all["per_grasp"]])
        print(f"[ok] {obj}: {len(q_ref)} grasps refined; "
              f"contacts {nc0:.0f}->{nc1:.0f}, max_pen {mp0:.1f}->{mp1:.1f}mm -> {out_npz}")


if __name__ == "__main__":
    main()
