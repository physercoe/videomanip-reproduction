#!/usr/bin/env python3
"""Paper-faithful grasp eval variant: FREE hand via the extended URDF's 6 virtual
root joints (all 18 dof driven at 1000/200, matching drograsp's IsaacGym setup),
object at origin, disturbance protocol (50 settle + 6x50 forces at 0.5x mass, <3cm).

Usage: python scripts/run_grasp_eval_free.py --object spraybottle --grasps <npz> [--num_envs N]
Run inside .venv (Isaac Sim 5.1 + IsaacLab 2.3.2).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HAND_URDF = ROOT / "third_party/drograsp/data/data_urdf/robot/inspire/inspire_hand_right_extended.urdf"
PRIORS = json.loads((ROOT / "data/size_priors.json").read_text())
DIRECTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
ROOT_JOINTS = ["virtual_joint_x", "virtual_joint_y", "virtual_joint_z",
               "virtual_joint_roll", "virtual_joint_pitch", "virtual_joint_yaw"]
FINGER_JOINTS = [
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint", "R_thumb_distal_joint",
    "R_index_proximal_joint", "R_index_intermediate_joint",
    "R_middle_proximal_joint", "R_middle_intermediate_joint",
    "R_ring_proximal_joint", "R_ring_intermediate_joint",
    "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--grasps", required=True)
    ap.add_argument("--num_envs", type=int, default=100)
    ap.add_argument("--settle_steps", type=int, default=50)
    ap.add_argument("--force_steps", type=int, default=50)
    ap.add_argument("--thresh_m", type=float, default=0.03)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

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

    device = "cuda:0"
    num_envs = args.num_envs

    z = np.load(args.grasps)
    q, obj_pos, obj_quat = z["q"], z["obj_pos"], z["obj_quat"]
    assert q.shape[1] == 19
    n_trials = len(q)
    trial_of_env = np.arange(num_envs) % n_trials
    q_env, obj_pos_env = q[trial_of_env], obj_pos[trial_of_env]

    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg
    obj_glb = ROOT / "outputs" / args.object / "mesh" / "object.glb"
    scale_json = json.loads((ROOT / "outputs" / args.object / "scale" / "scale.json").read_text())
    factor = scale_json["chosen"]["factor"]
    mass = PRIORS[args.object].get("mass_kg", 0.3)
    usd_dir = ROOT / "outputs" / args.object / "usd"
    usd_dir.mkdir(parents=True, exist_ok=True)
    import trimesh
    scaled_glb = usd_dir / "object_scaled.glb"
    if not scaled_glb.exists():
        _m = trimesh.load(str(obj_glb), force="mesh")
        _m.apply_scale(factor)
        _m.export(str(scaled_glb))
    mesh_cfg = MeshConverterCfg(
        asset_path=str(scaled_glb),
        usd_dir=str(usd_dir),
        usd_file_name="object.usd",
        mass_props=schemas_cfg.MassPropertiesCfg(mass=mass),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        mesh_collision_props=schemas_cfg.MeshCollisionPropertiesCfg(
            mesh_approximation_name="convexDecomposition"),
        scale=(1.0, 1.0, 1.0),
        force_usd_conversion=True,
    )
    usd_path = usd_dir / "object.usd"
    if not usd_path.exists():
        MeshConverter(mesh_cfg)

    sim_cfg = SimulationCfg(dt=0.01, device=device)
    sim_cfg.physics_material.static_friction = 3.0
    sim_cfg.physics_material.dynamic_friction = 3.0
    sim = SimulationContext(sim_cfg)
    spacing = 2.0
    origins = np.array([[i * spacing, 0.0, 0.0] for i in range(num_envs)], dtype=np.float32)

    hand_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Hand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(HAND_URDF),
            fix_base=False,
            self_collision=False,
            collider_type="convex_decomposition",
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                target_type="position",
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=1000.0, damping=200.0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=1000.0, damping=200.0, effort_limit=100.0, velocity_limit=100.0),
        },
    )
    obj_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
            collision_props=schemas_cfg.CollisionPropertiesCfg(),
            mass_props=schemas_cfg.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=tuple(origins[i]))
    hand = Articulation(hand_cfg)
    obj = RigidObject(obj_cfg)
    sim.reset()
    print("[free] scene ready; hand dof =", hand.num_joints)

    # post-reset views (IsaacLab data stale in this env)
    from isaaclab.sim.utils.stage import get_current_stage_id
    physx = omni.physx.get_physx_simulation_interface()
    view = omni.physics.tensors.create_simulation_view("warp", stage_id=get_current_stage_id())
    view.set_subspace_roots("/")
    art_hand = view.create_articulation_view("/World/envs/env_*/Hand/root_joint")
    rb_obj = view.create_rigid_body_view("/World/envs/env_*/Object")
    def wa(t):
        return wp.from_torch(t.contiguous())
    def to_t(a):
        return wp.to_torch(a) if isinstance(a, wp.array) else torch.as_tensor(a)
    idx_w = wp.from_torch(torch.arange(num_envs, dtype=torch.int32, device=device))

    joint_names = list(hand.data.joint_names)
    sim_idx = [joint_names.index(n) for n in ROOT_JOINTS + FINGER_JOINTS]
    # q19 -> sim 18 dof: drop dummy (index 0)
    q_t = torch.tensor(q_env, dtype=torch.float32, device=device)[:, 1:][:, sim_idx]
    obj_pos_t = torch.tensor(obj_pos_env, dtype=torch.float32, device=device) + \
        torch.tensor(origins, dtype=torch.float32, device=device)
    obj_quat_t = torch.tensor(obj_quat_env, dtype=torch.float32, device=device)

    # reset
    art_hand.set_dof_positions(wa(q_t), idx_w)
    art_hand.set_dof_velocities(wa(torch.zeros_like(q_t)), idx_w)
    obj_pose7 = torch.cat([obj_pos_t, obj_quat_t[:, [1, 2, 3, 0]]], dim=-1)
    rb_obj.set_transforms(wa(obj_pose7), idx_w)
    rb_obj.set_velocities(wa(torch.zeros(num_envs, 6, device=device)), idx_w)

    def step():
        physx.simulate(0.01, 0.0)
        physx.fetch_results()

    # settle: ramp-close
    n_ramp = max(10, args.settle_steps // 2)
    for i in range(args.settle_steps):
        if i < n_ramp:
            alpha = 0.3 + 0.7 * (i + 1) / n_ramp
            qa = q_t.clone()
            qa[:, 6:] *= alpha
            art_hand.set_dof_positions(wa(qa), idx_w)
        else:
            art_hand.set_dof_positions(wa(q_t), idx_w)
        step()
    p0 = to_t(rb_obj.get_transforms())[:, :3].clone()

    force_mag = 0.5 * mass
    for d in DIRECTIONS:
        f = torch.tensor(d, dtype=torch.float32, device=device) * force_mag
        forces = f.expand(num_envs, 3).contiguous()
        for _ in range(args.force_steps):
            rb_obj.apply_forces(wa(forces), idx_w)
            art_hand.set_dof_positions(wa(q_t), idx_w)
            step()
    p1 = to_t(rb_obj.get_transforms())[:, :3].clone()

    disp = (p1 - p0).norm(dim=-1).cpu().numpy()
    success = disp < args.thresh_m
    results = {
        "object": args.object, "num_envs": num_envs, "thresh_m": args.thresh_m,
        "force_N": force_mag, "mass_kg": mass, "variant": "free_hand",
        "success_rate": float(success.mean()),
        "per_env": [{"env": int(i), "trial": int(trial_of_env[i]),
                     "disp_m": float(disp[i]), "success": bool(success[i])}
                    for i in range(num_envs)],
    }
    out = Path(args.out) if args.out else ROOT / "outputs" / args.object / "eval" / "grasp_eval_free_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"[free] success {success.sum()}/{num_envs} = {success.mean() * 100:.1f}% "
          f"(median disp {np.median(disp) * 1e3:.1f}mm) -> {out}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
