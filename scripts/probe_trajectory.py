#!/usr/bin/env python3
"""Trajectory probe: run the eval protocol for ONE grasp and print object pose
every 10 steps (settle + first force phase)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OBJ = sys.argv[1] if len(sys.argv) > 1 else "spraybottle"
GRASPS = sys.argv[2] if len(sys.argv) > 2 else f"outputs/{OBJ}/eval/retarget_grasps.npz"
GIDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app
import torch
import warp as wp
import omni.physx
import omni.physics.tensors
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext, SimulationCfg
import json

HAND_URDF = ROOT / "third_party/xr_teleoperate/assets/inspire_hand/inspire_hand_right.urdf"
PRIORS = json.loads((ROOT / "data/size_priors.json").read_text())
FJ = ["R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_thumb_intermediate_joint",
      "R_thumb_distal_joint", "R_index_proximal_joint", "R_index_intermediate_joint",
      "R_middle_proximal_joint", "R_middle_intermediate_joint", "R_ring_proximal_joint",
      "R_ring_intermediate_joint", "R_pinky_proximal_joint", "R_pinky_intermediate_joint"]


def zyx_euler_to_quat_wxyz(e):
    roll, pitch, yaw = e[..., 0], e[..., 1], e[..., 2]
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    w = cr * cp * cy - sr * sp * sy
    return np.array([w, x, y, z])


z = np.load(ROOT / GRASPS)
q19 = z["q"][GIDX]
mass = PRIORS[OBJ].get("mass_kg", 0.3)

sim_cfg = SimulationCfg(dt=0.01, device="cuda:0")
sim_cfg.physics_material.static_friction = 3.0
sim_cfg.physics_material.dynamic_friction = 3.0
sim = SimulationContext(sim_cfg)
hand_cfg = ArticulationCfg(
    prim_path="/World/envs/env_0/Hand",
    spawn=sim_utils.UrdfFileCfg(asset_path=str(HAND_URDF), fix_base=True, self_collision=False,
        collider_type="convex_decomposition",
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=1000.0, damping=200.0))),
    init_state=ArticulationCfg.InitialStateCfg(),
    actuators={"fingers": ImplicitActuatorCfg(joint_names_expr=["R_.*"], stiffness=1000.0, damping=200.0)},
)
obj_cfg = RigidObjectCfg(
    prim_path="/World/envs/env_0/Object",
    spawn=sim_utils.UsdFileCfg(usd_path=str(ROOT / f"outputs/{OBJ}/usd/object.usd"),
        rigid_props=sim_utils.schemas.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.schemas.CollisionPropertiesCfg(),
        mass_props=sim_utils.schemas.MassPropertiesCfg(mass=mass)),
    init_state=RigidObjectCfg.InitialStateCfg(),
)
sim_utils.create_prim("/World/envs/env_0", "Xform", translation=tuple(q19[1:4]),
                      orientation=tuple(zyx_euler_to_quat_wxyz(q19[4:7])))
hand = Articulation(hand_cfg)
obj = RigidObject(obj_cfg)
sim.reset()
from isaaclab.sim.utils.stage import get_current_stage_id
view = omni.physics.tensors.create_simulation_view("warp", stage_id=get_current_stage_id())
view.set_subspace_roots("/")
art = view.create_articulation_view("/World/envs/env_0/Hand/root_joint")
rb = view.create_rigid_body_view("/World/envs/env_0/Object")
names = list(hand.data.joint_names)
sim_idx = [names.index(n) for n in FJ]
q12 = torch.from_numpy(q19[7:]).float().cuda()[sim_idx]
idx = torch.tensor([0], dtype=torch.int32, device="cuda:0")
art.set_dof_positions(wp.from_torch(q12.unsqueeze(0).contiguous()), wp.from_torch(idx))
rb.set_transforms(wp.from_torch(torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], device="cuda:0")), wp.from_torch(idx))
physx = omni.physx.get_physx_simulation_interface()
qt = wp.from_torch(q12.unsqueeze(0).contiguous())

for step in range(50):
    alpha = 0.3 + 0.7 * (step + 1) / 25 if step < 25 else 1.0
    art.set_dof_positions(wp.from_torch((q12 * alpha).unsqueeze(0).contiguous()), wp.from_torch(idx))
    for _ in range(4):
        physx.simulate(0.0025, 0.0)
    physx.fetch_results()
    if step % 10 == 0:
        op = wp.to_torch(rb.get_transforms())[0].cpu().numpy()
        print(f"TRAJ settle {step}: {op[:3].round(4)}", flush=True)
# first force phase
f = torch.tensor([[1., 0., 0.]], device="cuda:0") * (0.5 * mass)
for step in range(50):
    rb.apply_forces(wp.from_torch(f), wp.from_torch(idx))
    art.set_dof_positions(qt, wp.from_torch(idx))
    for _ in range(4):
        physx.simulate(0.0025, 0.0)
    physx.fetch_results()
    if step % 10 == 0:
        op = wp.to_torch(rb.get_transforms())[0].cpu().numpy()
        print(f"TRAJ force+x {step}: {op[:3].round(4)}", flush=True)
import os
os._exit(0)
