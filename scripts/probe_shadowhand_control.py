#!/usr/bin/env python3
"""Control experiment: DRO-Grasp's own shadowhand + their YCB orange + their GT grasp,
evaluated in our IsaacLab harness (welded-wrist convention)."""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext, SimulationCfg
import omni.physx, omni.physics.tensors
import warp as wp
import torch, numpy as np
from pathlib import Path

ROOT = Path("/app/project/videomanip")
DRO = ROOT / "third_party/drograsp"
z = np.load("/tmp/sh_grasp.npz")
q30 = z["q"]

def zyx_euler_to_quat_xyzw(e):
    roll, pitch, yaw = e[..., 0], e[..., 1], e[..., 2]
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    x = sr*cp*cy + cr*sp*sy; y = cr*sp*cy - sr*cp*sy
    z = cr*cp*sy + sr*sp*cy; w = cr*cp*cy - sr*sp*sy
    return np.array([x, y, z, w])

sim = SimulationContext(SimulationCfg(dt=0.01, device="cuda:0"))
hand_cfg = ArticulationCfg(
    prim_path="/World/envs/env_0/Hand",
    spawn=sim_utils.UrdfFileCfg(asset_path=str(DRO / "data/data_urdf/robot/shadowhand/shadow_hand_right_extended.urdf"),
        fix_base=False, self_collision=False, collision_type="Convex Decomposition",
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=3.0, dynamic_friction=3.0)),
    init_state=ArticulationCfg.InitialStateCfg(),
    actuators={"fingers": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1000.0, damping=200.0)},
)
obj_cfg = RigidObjectCfg(
    prim_path="/World/envs/env_0/Object",
    spawn=sim_utils.UrdfFileCfg(asset_path=str(DRO / "data/data_urdf/object/ycb/orange/orange.urdf"),
        fix_base=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=3.0, dynamic_friction=3.0)),
    init_state=RigidObjectCfg.InitialStateCfg(),
)
sim_utils.create_prim("/World/envs/env_0", "Xform", translation=(0., 0., 0.), orientation=(0., 0., 0., 1.))
hand = Articulation(hand_cfg)
obj = RigidObject(obj_cfg)
sim.reset()

from isaaclab.sim.utils.stage import get_current_stage_id
view = omni.physics.tensors.create_simulation_view("warp", stage_id=get_current_stage_id())
view.set_subspace_roots("/")
art = view.create_articulation_view("/World/envs/env_0/Hand/Geometry/world/virtual_link_x")
rb = view.create_rigid_body_view("/World/envs/env_0/Object/Geometry/object")
names = list(hand.data.joint_names)
print("PROBE sim joints:", len(names))
idx = torch.tensor([0], dtype=torch.int32, device="cuda:0")
q_t = torch.from_numpy(q30).float().cuda()
print("PROBE sim dof:", art.count if hasattr(art, "count") else "?", " q len:", len(q30))
# Isaac Sim 6 consumes the first URDF joint (virtual_joint_x); q30 -> sim 29 (drop x, ~0)
q_sim = torch.from_numpy(q30[1:]).float().cuda()
art.set_dof_positions(wp.from_torch(q_sim.unsqueeze(0).contiguous()), wp.from_torch(idx))
art.set_dof_positions(wp.from_torch(q_sim.unsqueeze(0).contiguous()), wp.from_torch(idx))
rb.set_transforms(wp.from_torch(torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], device="cuda:0")), wp.from_torch(idx))
physx = omni.physx.get_physx_simulation_interface()
for step in range(50):
    art.set_dof_positions(wp.from_torch(q_sim.unsqueeze(0).contiguous()), wp.from_torch(idx))
    physx.simulate(0.01, 0.0)
    physx.fetch_results()
    if step % 10 == 0:
        op = wp.to_torch(rb.get_transforms())[0].cpu().numpy()
        print(f"PROBE settle {step}: obj pos {op[:3].round(4)}")
simulation_app.close()
