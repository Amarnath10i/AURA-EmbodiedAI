"""Run this FIRST on the RTX machine, before trusting anything in
lama/env/isaac_warehouse.py.

Purpose: isolate "does Isaac Lab actually work on this machine" from "does
this project's Isaac backend work". This script touches nothing from `lama`
at all -- it is close to the literal tutorial code from the Isaac Lab docs
(isaac-sim.github.io/IsaacLab, the "Interacting with a rigid object"
tutorial), adapted only to launch, spawn one cuboid, drop it under gravity
for a couple of seconds, print its height, and exit cleanly.

If this script fails, the problem is the Isaac Lab install, the GPU driver,
or this specific API surface having moved since this was written -- not
`lama.env.isaac_warehouse`, which has never been exercised beyond a fake
stand-in for these same calls. Fix this script until it passes before even
looking at that file.

Usage (see docs/ISAAC_LAB_SETUP.md for the environment setup this assumes):
    python scripts/isaac_smoke_test.py
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--show", dest="headless", action="store_false",
                       help="open the Isaac Sim window instead of running headless")
    parser.add_argument("--steps", type=int, default=120,
                       help="physics steps to run (default ~2s at 60Hz)")
    args = parser.parse_args()

    print("[1/6] Launching Isaac Sim -- this can take a couple of minutes "
          "the first time (extension registry download).")
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app
    print("      Launched OK.")

    print("[2/6] Importing isaaclab.sim / isaaclab.assets ...")
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext
    print("      Imported OK.")

    print("[3/6] Building a simulation context and a ground plane ...")
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.0])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/light", light_cfg)
    print("      Scene built OK.")

    print("[4/6] Spawning one dynamic cuboid, 1kg, 1 metre off the ground ...")
    cube_cfg = RigidObjectCfg(
        prim_path="/World/smoke_test_cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    cube = RigidObject(cube_cfg)
    print("      Spawned OK.")

    print("[5/6] Resetting sim and stepping physics; the cube should fall "
          "and land near z=0.1 (half its 0.2m height) ...")
    sim.reset()
    for i in range(args.steps):
        cube.write_data_to_sim()
        sim.step()
        cube.update(sim.get_physics_dt())
        if i % 30 == 0 or i == args.steps - 1:
            z = float(cube.data.root_pos_w[0, 2])
            print(f"      step {i:4d}  height z = {z:.3f} m")

    print("[6/6] Applying a horizontal force and checking it moves ...")
    before = cube.data.root_pos_w[0, :2].clone()
    force = torch.zeros((1, 1, 3), dtype=torch.float32, device=sim.device)
    force[0, 0, 0] = 20.0
    torque = torch.zeros((1, 1, 3), dtype=torch.float32, device=sim.device)
    cube.set_external_force_and_torque(force, torque)
    for _ in range(30):
        cube.write_data_to_sim()
        sim.step()
        cube.update(sim.get_physics_dt())
    cube.set_external_force_and_torque(torch.zeros_like(force), torch.zeros_like(torque))
    cube.write_data_to_sim()
    after = cube.data.root_pos_w[0, :2].clone()
    moved = float(torch.linalg.norm(after - before))
    print(f"      moved {moved:.3f} m under a 20N push")

    ok = moved > 0.01
    print()
    print("RESULT:", "PASS -- basic physics works." if ok else
          "FAIL -- the cube did not move; report this before trusting "
          "IsaacWarehouse's push/pull/rotate/tip results.")

    simulation_app.close()


if __name__ == "__main__":
    main()
