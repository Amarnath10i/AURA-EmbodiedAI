"""
Isaac Lab Tabletop Environment Wrapper.
This script sets up the Franka Panda in a tabletop scene with the objects
defined in task_config.py. It exposes an RL-friendly step/reset API.
"""
import numpy as np

try:
    # These will import successfully when running inside Isaac Sim / Isaac Lab
    import omni.isaac.core.utils.stage as stage_utils
    from omni.isaac.core.world import World
    from omni.isaac.franka import Franka
    from omni.isaac.core.objects import DynamicCuboid # Placeholder for actual assets
    from omni.isaac.sensor import Camera
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("WARNING: Omniverse / Isaac Lab not detected. Running in mock mode.")

from lama.envs.task_config import TabletopTaskConfig

class TabletopEnv:
    def __init__(self, config: TabletopTaskConfig):
        self.config = config
        self.is_mock = not ISAAC_AVAILABLE
        
        if not self.is_mock:
            self._setup_isaac_world()
        else:
            self._setup_mock_world()

    def _setup_isaac_world(self):
        """Initializes the actual Isaac Lab world, robot, and sensors."""
        self.world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
        
        # Spawn Table
        stage_utils.add_reference_to_stage(usd_path="/Isaac/Props/Table/table.usd", prim_path="/World/Table")
        
        # Spawn Robot
        self.robot = Franka(prim_path="/World/Franka", name=self.config.robot_name)
        self.world.scene.add(self.robot)
        
        # Spawn Objects from Config (Skipping exact USD loading logic for brevity)
        # In a full run, we would loop through self.config.objects and add them
        
        # Setup Sensors
        if self.config.sensors.add_rgb:
            self.camera = Camera(
                prim_path="/World/Camera",
                position=np.array([1.5, 0.0, 1.0]),
                resolution=self.config.sensors.resolution
            )
            self.camera.initialize()
            
    def _setup_mock_world(self):
        """Sets up a mock interface for testing the ML pipeline without an RTX GPU."""
        print("Initialized Mock Tabletop Environment.")
        pass

    def reset(self):
        """Resets the environment and returns the initial observation."""
        if not self.is_mock:
            self.world.reset()
            obs = self._get_observations()
            return obs
        else:
            # Return random mock data matching expected tensor shapes
            res = self.config.sensors.resolution
            return {
                "rgb": np.random.randint(0, 255, (res[0], res[1], 3), dtype=np.uint8),
                "depth": np.random.rand(res[0], res[1]).astype(np.float32),
                "robot_state": np.random.randn(self.config.robot_state_dim).astype(np.float32)
            }

    def step(self, action: np.ndarray):
        """Applies action, steps physics, and returns (obs, reward, done, info)."""
        if not self.is_mock:
            self.robot.apply_action(action)
            self.world.step(render=True)
            obs = self._get_observations()
            reward = 0.0 # Dense reward logic goes here
            done = False
            return obs, reward, done, {}
        else:
            # Mock step
            obs = self.reset()
            return obs, 0.0, False, {}

    def _get_observations(self):
        """Gathers data from sensors and robot proprioception."""
        rgb = self.camera.get_rgba()[:, :, :3] # Drop alpha
        depth = self.camera.get_depth()
        # Get robot joint positions and velocities
        dof_pos = self.robot.get_joint_positions()
        dof_vel = self.robot.get_joint_velocities()
        robot_state = np.concatenate([dof_pos, dof_vel])
        
        return {
            "rgb": rgb,
            "depth": depth,
            "robot_state": robot_state
        }
