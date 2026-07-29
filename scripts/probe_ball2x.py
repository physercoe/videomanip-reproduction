#!/usr/bin/env python3
"""Minimal dynamic check in IsaacLab 2.3.2: does a spawned ball fall?"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, device="cuda:0")
simulation_app = app_launcher.app
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext, SimulationCfg

sim = SimulationContext(SimulationCfg(dt=0.01, device="cuda:0"))
ball_cfg = RigidObjectCfg(
    prim_path="/World/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.05,
        rigid_props=sim_utils.schemas.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.schemas.CollisionPropertiesCfg(),
        mass_props=sim_utils.schemas.MassPropertiesCfg(mass=0.2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
)
ball = RigidObject(ball_cfg)
sim.reset()
for i in range(4):
    for _ in range(50):
        sim.step()
    print(f"BALL2X step {(i + 1) * 50}: z={ball.data.root_pos_w[0, 2].item():.4f}", flush=True)
simulation_app.close()
