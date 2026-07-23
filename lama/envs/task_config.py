"""
Configuration for the LAMA Tabletop Environment in Isaac Lab.
Defines the robot (Franka Panda) and the interactive objects (Drawer, Cabinet, Cube, etc.)
with real Omniverse Nucleus paths.
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ObjectConfig:
    name: str
    asset_path: str # Path to USD file in Omniverse (Nucleus server)
    is_dynamic: bool = True
    mass: float = 1.0
    articulation_type: Optional[str] = None # e.g., 'revolute', 'prismatic'
    initial_position: tuple = (0.0, 0.0, 0.0)

@dataclass
class SensorConfig:
    add_rgb: bool = True
    add_depth: bool = True
    add_segmentation: bool = True
    resolution: tuple = (256, 256)
    position: tuple = (1.5, 0.0, 1.2)
    look_at: tuple = (0.5, 0.0, 0.0)

@dataclass
class TabletopTaskConfig:
    robot_name: str = "franka_panda"
    # Isaac Lab standard Franka Panda asset
    robot_asset_path: str = "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Robots/Franka/franka.usd"
    table_asset_path: str = "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Environments/Simple_Room/Props/table_low.usd"
    
    objects: List[ObjectConfig] = field(default_factory=lambda: [
        ObjectConfig("drawer", "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/Blocks/drawer.usd", articulation_type='prismatic', initial_position=(0.6, -0.2, 0.8)),
        ObjectConfig("cabinet", "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/Blocks/cabinet.usd", articulation_type='revolute', initial_position=(0.6, 0.2, 0.8)),
        ObjectConfig("button", "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/Blocks/button.usd", articulation_type='prismatic', initial_position=(0.4, 0.0, 0.8)),
        ObjectConfig("cube", "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/Blocks/cube.usd", initial_position=(0.5, 0.1, 0.85)),
        ObjectConfig("lever", "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Props/Blocks/lever.usd", articulation_type='revolute', initial_position=(0.5, -0.1, 0.8)),
    ])
    sensors: SensorConfig = field(default_factory=SensorConfig)
    
    # RL/Control parameters
    action_space: int = 7 # 6 DOF end-effector pose + 1 gripper open/close
    robot_state_dim: int = 14 # 7 joint positions + 7 joint velocities
