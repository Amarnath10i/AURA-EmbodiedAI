"""
Configuration for the LAMA Tabletop Environment.
Defines the robot (Franka Panda) and the interactive objects (Drawer, Cabinet, Cube, etc.)
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class ObjectConfig:
    name: str
    asset_path: str # Path to USD file in Omniverse
    is_dynamic: bool = True
    mass: float = 1.0

@dataclass
class SensorConfig:
    add_rgb: bool = True
    add_depth: bool = True
    add_segmentation: bool = True
    resolution: tuple = (256, 256)

@dataclass
class TabletopTaskConfig:
    robot_name: str = "franka_panda"
    objects: List[ObjectConfig] = field(default_factory=lambda: [
        ObjectConfig("drawer", "/Isaac/Props/Blocks/drawer.usd"),
        ObjectConfig("cabinet", "/Isaac/Props/Blocks/cabinet.usd"),
        ObjectConfig("button", "/Isaac/Props/Blocks/button.usd"),
        ObjectConfig("cube", "/Isaac/Props/Blocks/cube.usd"),
        ObjectConfig("bottle", "/Isaac/Props/Blocks/bottle.usd"),
        ObjectConfig("cup", "/Isaac/Props/Blocks/cup.usd"),
        ObjectConfig("door", "/Isaac/Props/Blocks/door.usd"),
        ObjectConfig("lever", "/Isaac/Props/Blocks/lever.usd"),
    ])
    sensors: SensorConfig = field(default_factory=SensorConfig)
    
    # RL/Control parameters
    action_space: int = 7 # e.g., 6 DOF + gripper
    robot_state_dim: int = 14 # e.g., 7 joint positions + 7 joint velocities
