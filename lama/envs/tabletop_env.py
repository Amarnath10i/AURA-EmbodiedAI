"""
Isaac Lab Tabletop Environment Wrapper (Real Physics).
This script sets up the Franka Panda in a tabletop scene using Isaac Lab.
It exposes an RL-friendly step/reset API.
"""
import numpy as np

# Isaac Lab / Omniverse Imports
try:
    from omni.isaac.core.world import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.franka import Franka
    from omni.isaac.sensor import Camera
    import omni.isaac.core.utils.prims as prim_utils
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("WARNING: Omniverse / Isaac Lab not detected. Environment initialization will fail.")

from lama.envs.task_config import TabletopTaskConfig

class TabletopEnv:
    def __init__(self, config: TabletopTaskConfig):
        self.config = config
        
        if not ISAAC_AVAILABLE:
            raise RuntimeError("Isaac Lab imports failed. Ensure you are running within Isaac Sim's Python environment.")
            
        self._setup_isaac_world()

    def _setup_isaac_world(self):
        """Initializes the actual Isaac Lab world, robot, and sensors."""
        self.world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
        
        # Spawn Table
        add_reference_to_stage(usd_path=self.config.table_asset_path, prim_path="/World/Table")
        
        # Spawn Robot
        add_reference_to_stage(usd_path=self.config.robot_asset_path, prim_path="/World/Franka")
        self.robot = Franka(prim_path="/World/Franka", name=self.config.robot_name)
        self.world.scene.add(self.robot)
        
        # Spawn Objects from Config
        for obj_cfg in self.config.objects:
            prim_path = f"/World/Objects/{obj_cfg.name}"
            add_reference_to_stage(usd_path=obj_cfg.asset_path, prim_path=prim_path)
            # Create XForm wrapper to set initial position
            if not prim_utils.get_prim_at_path(prim_path):
                print(f"Failed to load {obj_cfg.name} from {obj_cfg.asset_path}")
                continue
            xform = XFormPrim(prim_path=prim_path, name=obj_cfg.name)
            xform.set_world_pose(position=np.array(obj_cfg.initial_position))
        
        # Setup Sensors (RGB, Depth)
        if self.config.sensors.add_rgb or self.config.sensors.add_depth:
            self.camera = Camera(
                prim_path="/World/Camera",
                position=np.array(self.config.sensors.position),
                resolution=self.config.sensors.resolution
            )
            # Basic look-at target setting
            self.camera.set_local_pose(translation=np.array(self.config.sensors.position))
            self.camera.initialize()
            
        # Ensure physics gets started
        self.world.reset()
        print("Isaac Lab Tabletop Environment successfully initialized.")

    def reset(self):
        """Resets the environment and returns the initial observation."""
        self.world.reset()
        # You would typically reset object positions here as well
        for obj_cfg in self.config.objects:
             prim_path = f"/World/Objects/{obj_cfg.name}"
             if prim_utils.get_prim_at_path(prim_path):
                 xform = XFormPrim(prim_path=prim_path)
                 xform.set_world_pose(position=np.array(obj_cfg.initial_position))
        return self._get_observations()

    def step(self, action: np.ndarray):
        """
        Applies action, steps physics, and returns (obs, reward, done, info).
        Action is expected to be end-effector velocity commands + gripper command.
        """
        # Apply action to robot (placeholder for proper End-Effector controller)
        # In a full setup, you'd use omni.isaac.franka.controllers.RMPFlowController
        self.robot.apply_action(self.robot.get_articulation_controller().apply_action(action))
        
        self.world.step(render=True)
        
        obs = self._get_observations()
        reward = 0.0 # Dense reward logic goes here
        done = False
        
        return obs, reward, done, {}

    def _get_observations(self):
        """Gathers data from sensors and robot proprioception."""
        obs = {}
        if self.config.sensors.add_rgb:
            # Drop alpha channel
            obs["rgb"] = self.camera.get_rgba()[:, :, :3]
            
        if self.config.sensors.add_depth:
            obs["depth"] = self.camera.get_depth()
            
        # Get robot joint positions and velocities
        dof_pos = self.robot.get_joint_positions()
        dof_vel = self.robot.get_joint_velocities()
        obs["robot_state"] = np.concatenate([dof_pos, dof_vel])
        
        return obs
