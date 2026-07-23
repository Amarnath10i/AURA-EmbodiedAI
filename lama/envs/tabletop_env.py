"""
Parallel Isaac Lab Tabletop Environment (DirectRLEnv).
Initializes `num_envs` parallel simulated worlds on the GPU.
"""
import torch
import numpy as np
from typing import Dict, Tuple

try:
    from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.sim import SimulationCfg
    from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
    from omni.isaac.lab.sensors import CameraCfg
    from omni.isaac.lab.sim.spawners.from_files import UsdFileCfg
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("WARNING: Isaac Lab not detected. Parallel Environment initialization will fail.")

from lama.envs.task_config import TabletopTaskConfig

class ParallelTabletopEnv:
    """
    Wrapper around Isaac Lab's DirectRLEnv to manage parallel physics and
    Domain Randomization on the GPU.
    """
    def __init__(self, config: TabletopTaskConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.num_envs = config.num_envs
        
        if not ISAAC_AVAILABLE:
            raise RuntimeError("Isaac Lab imports failed. Ensure you are running within Isaac Sim's Python environment.")
            
        self._setup_vectorized_env()
        
    def _setup_vectorized_env(self):
        """Builds the DirectRLEnv configuration and instantiates the parallel environments."""
        
        # 1. Base Simulation Config
        sim_cfg = SimulationCfg(
            dt=1.0/60.0,
            use_gpu_pipeline=True,
            gravity=(0.0, 0.0, -9.81)
        )
        
        # 2. Scene Config (Spawning the grid of environments)
        scene_cfg = InteractiveSceneCfg(
            num_envs=self.num_envs,
            env_spacing=self.config.env_spacing,
            replicate_physics=True
        )
        
        # We would programmatically inject the robot, table, and objects into scene_cfg here.
        # Example for robot:
        # scene_cfg.robot = ArticulationCfg(
        #     prim_path="/World/envs/env_.*/Robot",
        #     spawn=UsdFileCfg(usd_path=self.config.robot_asset_path)
        # )
        
        # 3. RL Environment Config
        class TabletopEnvCfg(DirectRLEnvCfg):
            sim = sim_cfg
            scene = scene_cfg
            decimation = 2 # Control at 30Hz
            episode_length_s = 5.0
            
            # Action & Observation Spaces
            num_actions = self.config.action_space
            num_observations = self.config.robot_state_dim # Proprioception
            num_states = 0
            
        self.env_cfg = TabletopEnvCfg()
        
        # 4. Create the Environment
        self.env = DirectRLEnv(cfg=self.env_cfg)
        
        # Domain Randomization (applied at reset)
        self.dr_config = self.config.domain_randomization
        
        print(f"Initialized {self.num_envs} Parallel Isaac Lab Environments on {self.device}.")

    def reset(self) -> Dict[str, torch.Tensor]:
        """Resets all environments and applies Domain Randomization."""
        obs, _ = self.env.reset()
        self._apply_domain_randomization()
        return self._format_observations(obs)
        
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """
        Steps all parallel environments.
        actions: (num_envs, action_dim)
        Returns: (obs_dict, rewards, dones, info)
        """
        obs, rewards, dones, info = self.env.step(actions)
        return self._format_observations(obs), rewards, dones, info
        
    def _apply_domain_randomization(self):
        """Applies GPU-accelerated domain randomization across all environments."""
        if not self.dr_config:
            return
            
        if self.dr_config.randomize_object_pose:
            # Example: adding noise to object root states in the physics engine
            # self.env.scene["drawer"].data.root_state_w[:, :3] += torch.randn(...)
            pass
            
        if self.dr_config.randomize_friction:
            # Physics material properties would be updated here
            pass
            
        if self.dr_config.randomize_lighting:
            # Stage lighting intensity updated here
            pass

    def _format_observations(self, base_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Gathers batched RGB, Depth, and Proprioception from the parallel scene.
        """
        # In a full implementation, we extract these from the Camera sensors in the scene
        # For this skeleton, we assume base_obs contains proprioception.
        # rgb_batch = self.env.scene["camera"].data.output["rgb"]
        
        # Mocking the camera data extraction for structural completeness
        B = self.num_envs
        res = self.config.sensors.resolution
        rgb_batch = torch.zeros((B, res[0], res[1], 3), device=self.device)
        depth_batch = torch.zeros((B, res[0], res[1]), device=self.device)
        
        return {
            "rgb": rgb_batch,
            "depth": depth_batch,
            "robot_state": base_obs # (B, 14)
        }
