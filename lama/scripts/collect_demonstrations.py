"""
Parallel Data Collection for LAMA.
Collects interaction datasets using 32 parallel Isaac Lab environments and saves to HDF5.
Supports Phase A (50), Phase B (200), and Phase C (1000) scaling.
"""
import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lama.envs.task_config import TabletopTaskConfig
from lama.envs.tabletop_env import ParallelTabletopEnv
from lama.data.replay_buffer import ReplayBuffer

def batched_primitive_actions(batch_size: int, action_dim: int, device: str = "cuda") -> torch.Tensor:
    """
    Generates a batch of scripted interaction primitives (push, lift, rotate, press).
    This acts as a placeholder for a complex motion planner.
    """
    actions = torch.zeros((batch_size, action_dim), dtype=torch.float32, device=device)
    
    # Randomly assign a primitive to each environment in the batch
    primitive_ids = torch.randint(0, 6, (batch_size,), device=device)
    
    # 0: push forward
    actions[primitive_ids == 0, 0] = 0.5
    # 1: push left
    actions[primitive_ids == 1, 1] = 0.5
    # 2: push right
    actions[primitive_ids == 2, 1] = -0.5
    # 3: lift
    actions[primitive_ids == 3, 2] = 0.5
    # 4: rotate cw
    actions[primitive_ids == 4, 5] = 0.3
    # 5: press down
    actions[primitive_ids == 5, 2] = -0.5
    
    return actions

def collect_data(phase: str, steps_per_episode: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Phase mapping
    phase_map = {"A": 50, "B": 200, "C": 1000}
    if phase not in phase_map:
        raise ValueError(f"Unknown phase {phase}. Choose A, B, or C.")
    
    total_episodes_target = phase_map[phase]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = TabletopTaskConfig()
    env = ParallelTabletopEnv(config, device=device)
    
    # Initialize Replay Buffer
    storage_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'hdf5')
    buffer = ReplayBuffer(storage_dir=storage_dir, max_episodes=5000)
    
    num_envs = config.num_envs
    episodes_collected = 0
    
    print(f"--- Starting Phase {phase} Collection ({total_episodes_target} episodes) ---")
    print(f"Using {num_envs} parallel environments on {device}.")
    
    pbar = tqdm(total=total_episodes_target, desc=f"Collecting Phase {phase}")
    
    while episodes_collected < total_episodes_target:
        obs = env.reset() # (32, ...)
        
        # Buffers for the current batch of episodes
        ep_rgb = [[] for _ in range(num_envs)]
        ep_depth = [[] for _ in range(num_envs)]
        ep_state = [[] for _ in range(num_envs)]
        ep_action = [[] for _ in range(num_envs)]
        ep_reward = [[] for _ in range(num_envs)]
        ep_done = [[] for _ in range(num_envs)]
        
        for t in range(steps_per_episode):
            actions = batched_primitive_actions(num_envs, config.action_space, device)
            next_obs, rewards, dones, _ = env.step(actions)
            
            # Store transition for each environment
            for i in range(num_envs):
                ep_rgb[i].append(obs["rgb"][i].cpu().numpy())
                ep_depth[i].append(obs["depth"][i].cpu().numpy())
                ep_state[i].append(obs["robot_state"][i].cpu().numpy())
                ep_action[i].append(actions[i].cpu().numpy())
                ep_reward[i].append(rewards[i] if isinstance(rewards, (list, tuple)) else 0.0) # Placeholder dense reward
                ep_done[i].append(dones[i] if isinstance(dones, (list, tuple)) else False)
                
            obs = next_obs
            
        # Save completed episodes to HDF5
        for i in range(num_envs):
            if episodes_collected >= total_episodes_target:
                break
                
            episode_data = {
                "rgb": np.stack(ep_rgb[i]),
                "depth": np.stack(ep_depth[i]),
                "state": np.stack(ep_state[i]),
                "action": np.stack(ep_action[i]),
                "reward": np.stack(ep_reward[i]),
                "done": np.stack(ep_done[i]),
            }
            metadata = {
                "phase": phase,
                "difficulty": "1",
                "length": str(steps_per_episode)
            }
            buffer.add_episode(episode_data, metadata)
            episodes_collected += 1
            pbar.update(1)
            
    pbar.close()
    print(f"Phase {phase} collection complete! Total episodes in buffer: {len(buffer.episodes)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=str, default="A", choices=["A", "B", "C"], help="Collection Phase")
    p.add_argument("--steps", type=int, default=50, help="Steps per episode")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    collect_data(args.phase, args.steps, args.seed)
