#!/usr/bin/env python3
"""Wrench-aware grasp refinement (research experiment, 2026-07-24).

Motivation: geometric contact refinement (run_refine.py) increases contact counts but
REDUCES sim success on slip-limited objects — geometric proximity is the wrong objective
for disturbance resistance. This script refines grasps against the actual eval criterion:
the 6-direction 0.5*mass disturbance (paper Sec. IV-A).

Model: per hand-surface contact point, normal force F_c = min(k * penetration, F_max)
(linear spring in penetration, capped). A disturbance along d is resisted by the normal
components pressing against d plus friction-cone support from all contacts:
    R_d = sum_c F_c * ( max(0, -n_c.d) + mu * sqrt(max(0, 1-(n_c.d)^2)) )
Loss maximizes the worst-direction margin softmin_d(R_d - F_req), plus penetration cap
(no ejection) and deviation penalties.

Usage: python scripts/run_refine_wrench.py <objects...> --grasps 'outputs/<obj>/dro/predicted_grasps.npz'
Run inside .venv-recon.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
DRO = ROOT / "third_party" / "drograsp"

import sys
sys.path.insert(0, str(DRO))
sys.path.insert(0, str(ROOT / "scripts"))
from utils.hand_model import create_hand_model  # noqa: E402
from run_refine import load_object_surface, hand_surface_points  # noqa: E402

DIRECTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
K_SPRING = 0.5      # N/m per surface point: 1mm penetration -> 0.5mN
F_MAX = 0.02        # N per contact point cap (~20mN x 2000 pts = ~10N total)
MU = 1.5            # conservative friction (harness uses 3.0 combined)
PEN_CAP_M = 0.003   # above this, ejection risk (penalty)
NEAR_M = 0.008
W_MARGIN, W_PEN, W_DEV = 1.0, 5.0, 1.0


def wrench_margin(hand_pts, obj_pts, obj_nrm, dirs_t, f_req, chunk=1024):
    """Differentiable disturbance-resistance margin. hand_pts (B,N,3) -> (B,) softmin margin."""
    B = hand_pts.shape[0]
    margins = []
    for s0 in range(0, hand_pts.shape[1], chunk):
        hp = hand_pts[:, s0:s0 + chunk]
        d = torch.cdist(hp, obj_pts)
        dmin, j = d.min(dim=-1)
        n = obj_nrm[j]
        o = torch.gather(obj_pts.unsqueeze(0).expand(B, -1, -1), 1,
                         j.unsqueeze(-1).expand(-1, -1, 3))
        signed = ((hp - o) * n).sum(-1)
        near = (dmin < NEAR_M).float()
        f_c = torch.clamp(K_SPRING * torch.relu(-signed), max=F_MAX) * near  # (B,n)
        # normal+friction resistance per direction
        cos = torch.einsum("bnc,dc->bnd", n, dirs_t)                  # (B,n,6)
        resist = (torch.relu(-cos) + MU * torch.sqrt(torch.clamp(1 - cos ** 2, min=1e-6)))
        r_d = (f_c.unsqueeze(-1) * resist).sum(1)                     # (B,6)
        margins.append(r_d)
    r = torch.stack(margins).sum(0)                                   # (B,6)
    # softmin over directions of (R_d - F_req)
    return (-torch.logsumexp(-(r - f_req) / 0.02, dim=1) * 0.02), r


def pen_excess(hand_pts, obj_pts, obj_nrm, chunk=2048):
    B, N = hand_pts.shape[0], hand_pts.shape[1]
    num = torch.zeros(B, device=hand_pts.device)
    den = torch.zeros(B, device=hand_pts.device)
    for s0 in range(0, N, chunk):
        hp = hand_pts[:, s0:s0 + chunk]
        d = torch.cdist(hp, obj_pts)
        dmin, j = d.min(dim=-1)
        n = obj_nrm[j]
        o = torch.gather(obj_pts.unsqueeze(0).expand(B, -1, -1), 1,
                         j.unsqueeze(-1).expand(-1, -1, 3))
        signed = ((hp - o) * n).sum(-1)
        near = (dmin < NEAR_M).float()
        num += (torch.relu(-signed - PEN_CAP_M) * near).sum(1)
        den += near.sum(1)
    return num / den.clamp(min=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+")
    ap.add_argument("--grasps", required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--mass", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    hand = create_hand_model("inspire", device)
    dirs_t = torch.tensor(DIRECTIONS, dtype=torch.float32, device=device)
    f_req = 0.5 * args.mass

    for obj in args.objects:
        obj_pts, obj_nrm, _ = load_object_surface(obj, device)
        src_npz = Path(args.grasps.replace("<obj>", obj))
        z = np.load(src_npz)
        q_all = torch.tensor(z["q"], dtype=torch.float32, device=device)
        meta = {k: z[k] for k in z.files if k != "q"}

        refined = []
        for s0 in range(0, len(q_all), args.batch):
            q0 = q_all[s0:s0 + args.batch]
            fingers = q0[:, 7:].clone().requires_grad_(True)
            wrist_r = q0[:, 4:7].clone().requires_grad_(True)
            wrist_t = q0[:, 1:4].clone().requires_grad_(True)
            opt = torch.optim.Adam([fingers, wrist_r, wrist_t], lr=1e-2)
            for it in range(args.iters):
                opt.zero_grad()
                q = torch.cat([q0[:, :1], wrist_t, wrist_r, fingers], dim=1)
                hp = hand_surface_points(hand, q)
                margin, _ = wrench_margin(hp, obj_pts, obj_nrm, dirs_t, f_req)
                pen = pen_excess(hp, obj_pts, obj_nrm)
                dev = ((fingers - q0[:, 7:]) ** 2).mean(1) + ((wrist_r - q0[:, 4:7]) ** 2).mean(1) \
                    + 100.0 * ((wrist_t - q0[:, 1:4]) ** 2).mean(1)
                loss = (-W_MARGIN * margin + W_PEN * pen + W_DEV * dev).sum()
                loss.backward()
                opt.step()
            with torch.no_grad():
                q = torch.cat([q0[:, :1], wrist_t, wrist_r, fingers], dim=1)
            refined.append(q.cpu())
        q_ref = torch.cat(refined, 0).numpy().astype(np.float32)
        out_npz = src_npz.with_name(src_npz.stem + "_wrench.npz")
        np.savez(out_npz, q=q_ref, **meta)
        # QA: final margins
        with torch.no_grad():
            hp = hand_surface_points(hand, torch.tensor(q_ref, device=device))
            m, r = wrench_margin(hp, obj_pts, obj_nrm, dirs_t, f_req)
        print(f"[ok] {obj}: {len(q_ref)} wrench-refined -> {out_npz}; "
              f"mean softmin margin {m.mean():.3f}N, worst-dir R {(r.min(1).values).mean():.3f}N "
              f"(req {f_req:.3f}N)")


if __name__ == "__main__":
    main()
