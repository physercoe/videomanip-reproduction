#!/usr/bin/env python3
"""Stage 6: retarget human hand motion (HaMeR/MANO) to the Inspire hand.

Method (replaces naive DexPilot-only optimization after QA showed drift):
- Wrist: closed form. Build a palm frame from MANO joints (wrist->middle-MCP = x,
  pinky-MCP->index-MCP = y) and match the robot's equivalent knuckle frame; base
  translation = human wrist. Exact, no optimization.
- Fingers: Adam on the 12 finger joints only, DexPilot scale-invariant bone-direction
  matching + temporal smoothness + joint limits.

Human joints: MANO 21 (wrist + 5x3 finger joints + 5 tips) in camera frame (metric,
from stage 5). Robot FK: drograsp HandModel on the extended inspire URDF (dof=19).

Reads  outputs/<obj>/hand/%05d.json
Writes outputs/<obj>/retarget/%05d.npy   (19,) q19: [dummy tx ty tz roll pitch yaw, 12 joints]
       outputs/<obj>/retarget/qa.json    (fit costs)

Run inside .venv-recon (pytorch_kinematics required).
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
from utils.hand_model import create_hand_model  # noqa: E402

# HaMeR joints3d order = OpenPose hand convention (hamer mano_wrapper joint_map):
# 0 wrist; 1-4 thumb (cmc,mcp,pip,tip); 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky
# chains use [mcp, pip, tip] (thumb keeps all 4); robot thumb has 3 segments
FINGERS = {
    "thumb": {"human": [1, 2, 3, 4],
              "robot": ["R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal", "R_thumb_tip"]},
    "index": {"human": [5, 6, 8],
              "robot": ["R_index_proximal", "R_index_intermediate", "R_index_tip"]},
    "middle": {"human": [9, 10, 12],
               "robot": ["R_middle_proximal", "R_middle_intermediate", "R_middle_tip"]},
    "ring": {"human": [13, 14, 16],
             "robot": ["R_ring_proximal", "R_ring_intermediate", "R_ring_tip"]},
    "pinky": {"human": [17, 18, 20],
              "robot": ["R_pinky_proximal", "R_pinky_intermediate", "R_pinky_tip"]},
}
# palm frame anchors: wrist=0, middle mcp=9, index mcp=5, pinky mcp=17
WRIST, MID_MCP, IDX_MCP, PINKY_MCP = 0, 9, 5, 17


def palm_frame(wrist, mid_mcp, idx_mcp, pinky_mcp):
    """Right-handed palm frame columns [ex,ey,ez]: ex wrist->middle MCP,
    ey pinky MCP->index MCP, ez = ex x ey."""
    ex = mid_mcp - wrist
    ex = ex / (ex.norm() + 1e-8)
    ey_raw = idx_mcp - pinky_mcp
    ez = torch.cross(ex, ey_raw, dim=0)
    ez = ez / (ez.norm() + 1e-8)
    ey = torch.cross(ez, ex, dim=0)
    ey = ey / (ey.norm() + 1e-8)
    return torch.stack([ex, ey, ez], dim=1)


def rot_to_zyx_euler(R: torch.Tensor) -> torch.Tensor:
    """drograsp FK convention R = Rx(roll)Ry(pitch)Rz(yaw) -> (roll, pitch, yaw).
    (named historically; returns the virtual-joint angles for the extended URDF chain)"""
    pitch = torch.asin(R[0, 2].clamp(-1, 1))
    roll = torch.atan2(-R[1, 2], R[2, 2])
    yaw = torch.atan2(-R[0, 1], R[0, 0])
    return torch.stack([roll, pitch, yaw])


def retarget_sequence(hand_records: list[dict], device: str = "cuda",
                      iters: int = 200) -> tuple[np.ndarray, list[float]]:
    hand = create_hand_model("inspire", torch.device(device))
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = torch.tensor(lower, dtype=torch.float32, device=device)
    upper = torch.tensor(upper, dtype=torch.float32, device=device)

    # robot zero-pose calibration: knuckle frame in base-link frame
    with torch.no_grad():
        st0 = hand.pk_chain.forward_kinematics(torch.zeros(1, 19, device=device))
        F_r = st0["R_hand_base_link"].get_matrix()[0][:3, :3]

        def pos0(lname):
            return st0[lname].get_matrix()[0][:3, 3]
        K_r = palm_frame(pos0("R_hand_base_link"), pos0("R_middle_proximal"),
                         pos0("R_index_proximal"), pos0("R_pinky_proximal"))
        K_base = F_r.T @ K_r          # constant: knuckle frame in base-link frame
        K_base_T = K_base.T

    qs, costs = [], []
    q_prev = None
    for fi, rec in enumerate(hand_records):
        H = torch.tensor(np.array(rec["joints3d"]), dtype=torch.float32, device=device)
        t_metric = torch.tensor(rec["t_metric"], dtype=torch.float32, device=device)
        H = H + t_metric  # camera frame, metric

        # --- wrist, closed form ------------------------------------------
        K_h = palm_frame(H[WRIST], H[MID_MCP], H[IDX_MCP], H[PINKY_MCP])
        R_base = K_h @ K_base_T       # robot base rotation matching human palm frame
        euler = rot_to_zyx_euler(R_base)

        q = torch.zeros(19, device=device)
        q[1] = H[0, 0]; q[2] = H[0, 1]; q[3] = H[0, 2]
        q[4:7] = euler
        if q_prev is not None:
            q[7:] = q_prev[7:]
        q = q.detach()
        fingers = q[7:].clone().requires_grad_(True)
        opt = torch.optim.Adam([fingers], lr=2e-2)

        for it in range(iters):
            opt.zero_grad()
            q_cur = torch.cat([q[:7], fingers])
            status = hand.pk_chain.forward_kinematics(q_cur.unsqueeze(0))
            kpts = {}
            for lname in {l for f in FINGERS.values() for l in f["robot"]}:
                m = status[lname].get_matrix()[0]
                kpts[lname] = m[:3, 3]
            loss_vec = 0.0
            for f in FINGERS.values():
                hh = [H[i] for i in f["human"]]
                rr = [kpts[l] for l in f["robot"]]
                for a in range(len(hh) - 1):
                    hv = hh[a + 1] - hh[a]
                    rv = rr[a + 1] - rr[a]
                    hv = hv / (hv.norm() + 1e-8)
                    rv = rv / (rv.norm() + 1e-8)
                    loss_vec = loss_vec + ((hv - rv) ** 2).sum()
            loss_smooth = ((fingers - q_prev[7:]) ** 2).sum() * 5.0 if q_prev is not None \
                else torch.zeros((), device=device)
            viol = torch.clamp(lower[7:] - fingers, min=0) + torch.clamp(fingers - upper[7:], min=0)
            loss_lim = (viol ** 2).sum() * 100.0
            loss = loss_vec + loss_smooth + loss_lim
            loss.backward()
            opt.step()
            fingers.data = torch.max(torch.min(fingers.data, upper[7:]), lower[7:])

        q[7:] = fingers.detach()
        qs.append(q.detach().cpu().numpy().copy())
        costs.append(float(loss_vec.detach().cpu()))
        q_prev = q.detach()
        if fi % 25 == 0:
            print(f"  frame {fi}: vec cost {costs[-1]:.4f}")
    return np.stack(qs), costs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--src", default="hand",
                    help="input records dir under outputs/<obj>/ (writes to retarget/ "
                         "for 'hand', else retarget_<src>/)")
    args = ap.parse_args()

    for obj in args.objects:
        hand_dir = OUT / obj / args.src
        records = []
        for fp in sorted(hand_dir.glob("*.json")):
            rec = json.loads(fp.read_text())
            if "joints3d" in rec:
                records.append(rec)
        assert records, f"no hand records for {obj}"
        qs, costs = retarget_sequence(records, args.device)
        out_dir = OUT / obj / ("retarget" if args.src == "hand" else f"retarget_{args.src}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, q in enumerate(qs):
            np.save(out_dir / f"{i:05d}.npy", q)
        (out_dir / "qa.json").write_text(json.dumps(
            {"n": len(qs), "vec_cost_mean": float(np.mean(costs)),
             "vec_cost_median": float(np.median(costs)), "vec_cost_max": float(np.max(costs))}, indent=1))
        print(f"[ok] {obj}: {len(qs)} retargeted frames, median vec cost {np.median(costs):.4f}")


if __name__ == "__main__":
    main()
