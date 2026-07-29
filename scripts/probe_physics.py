#!/usr/bin/env python3
"""Physics probe: (a) does the converted object fall under gravity? (b) do the
URDF-imported joint drives hold the hand pose? Prints z/jointpos trajectories."""
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext, SimulationCfg

HAND_URDF = ROOT / "third_party/drograsp/data/data_urdf/robot/inspire/inspire_hand_right_extended.urdf"

sim = SimulationContext(SimulationCfg(dt=0.01, device="cuda:0"))

obj_cfg = RigidObjectCfg(
    prim_path="/World/envs/env_0/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "outputs/spraybottle/usd/object.usd"),
        rigid_props=sim_utils.schemas.RigidBodyBaseCfg(),
        collision_props=sim_utils.schemas.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
)
hand_cfg = ArticulationCfg(
    prim_path="/World/envs/env_1/Hand",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(HAND_URDF),
        fix_base=False,
        self_collision=False,
        collision_type="Convex Decomposition",
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness={".*virtual_joint_[xyz]": 1e6,
                           ".*virtual_joint_(roll|pitch|yaw)": 1e5, ".*": 1000.0},
                damping={".*virtual_joint_[xyz]": 1e4,
                         ".*virtual_joint_(roll|pitch|yaw)": 1e3, ".*": 200.0},
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(3.0, 0.0, 0.3)),
    actuators={},
)
sim_utils.create_prim("/World/envs/env_0", "Xform")
sim_utils.create_prim("/World/envs/env_1", "Xform")
obj = RigidObject(obj_cfg)
hand = Articulation(hand_cfg)
sim.reset()

names = list(hand.data.joint_names)
q = torch.zeros(1, hand.num_joints, device="cuda:0")
q[0, 2] = 0.3
q[0, 6:] = 0.6
env_ids = torch.tensor([0], device="cuda:0")
hand.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
hand.reset()

print("probe: stepping 300...")
for i in range(300):
    hand.set_joint_position_target(target=q, env_ids=env_ids)
    sim.step()
    if i % 50 == 0:
        oz = obj.data.root_pose_w.torch[0, 2].item()
        hz = hand.data.root_pose_w.torch[0, 2].item()
        jp = hand.data.joint_pos.torch[0, 6].item()
        print(f"  step {i}: obj_z={oz:.3f} hand_root_z={hz:.3f} joint[6]={jp:.3f}")
simulation_app.close()
