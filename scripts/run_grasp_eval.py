#!/usr/bin/env python3
"""Grasp disturbance evaluation in IsaacLab (paper Sec. IV-A protocol, IsaacGym->IsaacLab).

IsaacLab 3.0 beta note: Isaac Sim 6.0.1 + IsaacLab 3.0.0-beta2 has a stale data-readback
path (obj.data / USD transforms do not reflect live physics; updateToUsd=false and the
beta scene-data backend freezes after warmup). We therefore:
  - use IsaacLab for scene/asset setup (URDF import, configs) — works fine;
  - step physics via omni.physx IPhysxSimulation.simulate() — verified working;
  - read/write state via our own omni.physics.tensors views — verified working.
See docs/PROGRESS.md (2026-07-22 entry) for the investigation trail.

Per trial: hand + object initialized at the reconstructed grasp; 50 settle steps;
then 300-step disturbance: forces of 0.5x object mass applied to the object from
±x, ±y, ±z sequentially (50 steps each, dt=0.01). Success = object displacement < 3 cm.

Grasp input npz:
  q         (N,18)  [tx ty tz roll pitch yaw, 12 finger joints] (drograsp convention)
  obj_pos   (N,3)   object root position in the env's world frame
  obj_quat  (N,4)   XYZW quaternion

Usage:
  python scripts/run_grasp_eval.py --object spraybottle --grasps <file.npz> [--num_envs 100]
Smoke test:
  python scripts/run_grasp_eval.py --object spraybottle --synthetic --num_envs 4

Run inside .venv-lab3 with LD_LIBRARY_PATH including nvidia/cu13/lib (PhysX nvrtc).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HAND_URDF = ROOT / "third_party/xr_teleoperate/assets/inspire_hand/inspire_hand_right.urdf"
PRIORS = json.loads((ROOT / "data/size_priors.json").read_text())

FINGER_JOINTS = [
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint", "R_thumb_distal_joint",
    "R_index_proximal_joint", "R_index_intermediate_joint",
    "R_middle_proximal_joint", "R_middle_intermediate_joint",
    "R_ring_proximal_joint", "R_ring_intermediate_joint",
    "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
]
ROOT_JOINTS = ["virtual_joint_x", "virtual_joint_y", "virtual_joint_z",
               "virtual_joint_roll", "virtual_joint_pitch", "virtual_joint_yaw"]
DIRECTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def zyx_euler_to_quat_xyzw(e):
    """drograsp FK convention: R = Rx(roll) @ Ry(pitch) @ Rz(yaw) -> quat (x,y,z,w)."""
    import numpy as np
    roll, pitch, yaw = e[..., 0], e[..., 1], e[..., 2]
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    # q = qroll ⊗ qpitch ⊗ qyaw
    x = sr*cp*cy + cr*sp*sy
    y = cr*sp*cy - sr*cp*sy
    z = cr*cp*sy + sr*sp*cy
    w = cr*cp*cy - sr*sp*sy
    return np.stack([x, y, z, w], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--grasps", default=None, help="npz with q, obj_pos, obj_quat")
    ap.add_argument("--synthetic", action="store_true", help="fabricate a test grasp")
    ap.add_argument("--synthetic_open", action="store_true",
                    help="same grasp but fingers open (object should fall -> large disp)")
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
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    import omni.physx
    import omni.physics.tensors

    device = "cuda:0"
    num_envs = args.num_envs

    # --- grasp data ---------------------------------------------------------
    if args.synthetic or args.synthetic_open:
        q = np.zeros((1, 19), dtype=np.float32)          # [dummy, x y z r p y, 12 fingers]
        q[0, 3] = 0.25                                  # wrist 25cm up
        q[0, 7:] = 0.0 if args.synthetic_open else 0.8  # fingers open/closed
        obj_pos = np.array([[-0.05, -0.15, 0.25]], dtype=np.float32)  # at fingertip cluster (FK)
        obj_quat = np.array([[0, 0, 0, 1]], dtype=np.float32)
    else:
        z = np.load(args.grasps)
        q, obj_pos, obj_quat = z["q"], z["obj_pos"], z["obj_quat"]
        assert q.shape[1] == 19, f"expected q19 (dummy+6+12), got {q.shape}"
    n_trials = len(q)
    trial_of_env = np.arange(num_envs) % n_trials
    q_env, obj_pos_env, obj_quat_env = q[trial_of_env], obj_pos[trial_of_env], obj_quat[trial_of_env]

    # --- object USD conversion ----------------------------------------------
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg
    obj_glb = ROOT / "outputs" / args.object / "mesh" / "object.glb"
    scale_json = json.loads((ROOT / "outputs" / args.object / "scale" / "scale.json").read_text())
    factor = scale_json["chosen"]["factor"]
    mass = PRIORS[args.object].get("mass_kg", 0.3)
    usd_dir = ROOT / "outputs" / args.object / "usd"
    usd_dir.mkdir(parents=True, exist_ok=True)
    # bake the metric scale into the mesh vertices (converter scale= is unreliable)
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
        rigid_props=schemas_cfg.RigidBodyBaseCfg(),
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        mesh_collision_props=schemas_cfg.MeshCollisionBaseCfg(
            mesh_approximation_name="convexDecomposition"),
        scale=(1.0, 1.0, 1.0),
        force_usd_conversion=True,
        make_instanceable=False,
    )
    usd_path = usd_dir / "object.usd"
    if not usd_path.exists():
        MeshConverter(mesh_cfg)
        print(f"[eval] converted object mesh -> {usd_path}")

    # --- scene --------------------------------------------------------------
    sim = SimulationContext(SimulationCfg(dt=0.01, device=device))
    # higher solver iterations via carb settings for contact-rich grasp stability
    try:
        import carb
        carb.settings.get_settings().set("/physics/solverType", 1)  # TGS
        carb.settings.get_settings().set("/physics/numPositionIterations", 32)
        carb.settings.get_settings().set("/physics/numVelocityIterations", 4)
    except Exception as e:
        print("[eval] carb settings not applied:", e)
    spacing = 2.0
    origins = np.array([[i * spacing, 0.0, 0.0] for i in range(num_envs)], dtype=np.float32)
    origins_t = torch.tensor(origins, dtype=torch.float32, device=device)

    # wrist pose per env from q19 (translation + ZYX euler -> quat)
    wrist_pos_env = q_env[:, 1:4] + origins
    wrist_quat_env = zyx_euler_to_quat_xyzw(q_env[:, 4:7])
    hand_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Hand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(HAND_URDF),
            fix_base=True,
            self_collision=False,
            collision_type="Convex Decomposition",
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=3.0, dynamic_friction=3.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["R_.*"],
                stiffness=400.0, damping=80.0, effort_limit_sim=100.0, velocity_limit_sim=100.0),
        },
    )
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=3.0, dynamic_friction=3.0)
    obj_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=schemas_cfg.RigidBodyBaseCfg(),
            collision_props=schemas_cfg.CollisionPropertiesCfg(),
            mass_props=schemas_cfg.MassPropertiesCfg(mass=mass),
            physics_material=mat,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform",
                              translation=tuple(wrist_pos_env[i]),
                              orientation=tuple(wrist_quat_env[i]))
    hand = Articulation(hand_cfg)
    obj = RigidObject(obj_cfg)
    sim.reset()
    print("[eval] scene ready; hand dof =", hand.num_joints)

    # --- own simulation views (beta data-layer workaround) -------------------
    from isaaclab.sim.utils.stage import get_current_stage_id
    stage_id = get_current_stage_id()
    physx = omni.physx.get_physx_simulation_interface()
    view = omni.physics.tensors.create_simulation_view("warp", stage_id=stage_id)
    view.set_subspace_roots("/")
    rb_obj = view.create_rigid_body_view("/World/envs/env_*/Object")
    art_hand = view.create_articulation_view("/World/envs/env_*/Hand/Geometry/R_hand_base_link")

    def to_t(a):
        return wp.to_torch(a) if isinstance(a, wp.array) else torch.as_tensor(a)

    joint_names = list(hand.data.joint_names)
    sim_idx = [joint_names.index(n) for n in FINGER_JOINTS]
    # q19 -> 12 finger joints only (wrist is baked into the env xform)
    q_t = torch.tensor(q_env, dtype=torch.float32, device=device)[:, 7:][:, sim_idx]
    obj_pos_t = torch.tensor(obj_pos_env, dtype=torch.float32, device=device) + \
        torch.tensor(origins, dtype=torch.float32, device=device)
    obj_quat_t = torch.tensor(obj_quat_env, dtype=torch.float32, device=device)

    # --- reset --------------------------------------------------------------
    def wa(t, dt=None):
        a = wp.from_torch(t.contiguous()) if torch.is_tensor(t) else wp.array(t)
        return a
    idx = torch.arange(num_envs, dtype=torch.int32, device=device)
    idx_w = wp.from_torch(idx)
    art_hand.set_dof_positions(wa(q_t), idx_w)
    art_hand.set_dof_velocities(wa(torch.zeros_like(q_t)), idx_w)
    art_hand.set_dof_positions(wa(q_t), idx_w)
    # spawn clearance: drop the object 5 mm toward gravity (slides into the grasp
    # instead of spawning inside the finger colliders)
    obj_pos_spawn = obj_pos_t + torch.tensor([0.0, 0.0, -0.005], device=device)
    pose7 = torch.cat([obj_pos_spawn, obj_quat_t], dim=-1)
    rb_obj.set_transforms(wa(pose7), idx_w)
    rb_obj.set_velocities(wa(torch.zeros(num_envs, 6, device=device)), idx_w)

    def step():
        physx.simulate(0.01, 0.0)
        physx.fetch_results()

    # --- settle ---------------------------------------------------------------
    for _ in range(args.settle_steps):
        step()
    p0 = to_t(rb_obj.get_transforms())[:, :3].clone()

    # --- disturbance ----------------------------------------------------------
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
        "force_N": force_mag, "mass_kg": mass,
        "success_rate": float(success.mean()),
        "per_env": [{"env": int(i), "trial": int(trial_of_env[i]),
                     "disp_m": float(disp[i]), "success": bool(success[i])}
                    for i in range(num_envs)],
    }
    out = Path(args.out) if args.out else ROOT / "outputs" / args.object / "eval" / "grasp_eval_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"[eval] success {success.sum()}/{num_envs} = {success.mean() * 100:.1f}% "
          f"(median disp {np.median(disp) * 1e3:.1f}mm) -> {out}")

    simulation_app.close()


if __name__ == "__main__":
    main()
