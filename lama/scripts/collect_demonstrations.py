"""
Collect demonstration episodes from the LAMA Tabletop Environment.
Supports both Isaac Lab (real physics) and Mock mode (for pipeline testing).
Stores RGB, depth, robot state, actions, rewards, and done flags per episode.
"""
import os
import sys
import numpy as np
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lama.envs.task_config import TabletopTaskConfig
from lama.envs.tabletop_env import TabletopEnv

# Interaction primitives for active exploration
PRIMITIVES = {
    0: "push_forward",
    1: "push_left",
    2: "push_right",
    3: "lift",
    4: "rotate_cw",
    5: "rotate_ccw",
    6: "press_down",
}

def primitive_to_action(primitive_id: int, action_dim: int) -> np.ndarray:
    """
    Converts a high-level interaction primitive into a low-level
    joint-space action vector for the Franka Panda.
    In a full implementation, these would be scripted motion primitives.
    For now, we generate directional impulses.
    """
    action = np.zeros(action_dim, dtype=np.float32)
    if primitive_id == 0:   # push forward
        action[0] = 0.5
    elif primitive_id == 1: # push left
        action[1] = 0.5
    elif primitive_id == 2: # push right
        action[1] = -0.5
    elif primitive_id == 3: # lift
        action[2] = 0.5
    elif primitive_id == 4: # rotate cw
        action[5] = 0.3
    elif primitive_id == 5: # rotate ccw
        action[5] = -0.3
    elif primitive_id == 6: # press down
        action[2] = -0.5
    return action


def collect(episodes: int, out_dir: str, seed: int, steps_per_episode: int):
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    
    config = TabletopTaskConfig()
    env = TabletopEnv(config)
    
    total_transitions = 0
    
    for ep in range(episodes):
        obs = env.reset()
        
        ep_rgb, ep_depth, ep_state = [], [], []
        ep_actions, ep_rewards, ep_dones = [], [], []
        
        for t in range(steps_per_episode):
            # Active exploration: cycle through interaction primitives
            primitive_id = np.random.randint(0, len(PRIMITIVES))
            action = primitive_to_action(primitive_id, config.action_space)
            
            next_obs, reward, done, info = env.step(action)
            
            ep_rgb.append(obs["rgb"])
            ep_depth.append(obs["depth"])
            ep_state.append(obs["robot_state"])
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_dones.append(done)
            
            obs = next_obs
            total_transitions += 1
            
            if done:
                break
        
        # Save episode
        ep_dir = os.path.join(out_dir, f"episode{ep:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        np.save(os.path.join(ep_dir, "rgb.npy"), np.array(ep_rgb, dtype=np.uint8))
        np.save(os.path.join(ep_dir, "depth.npy"), np.array(ep_depth, dtype=np.float32))
        np.save(os.path.join(ep_dir, "state.npy"), np.array(ep_state, dtype=np.float32))
        np.save(os.path.join(ep_dir, "action.npy"), np.array(ep_actions, dtype=np.float32))
        np.save(os.path.join(ep_dir, "reward.npy"), np.array(ep_rewards, dtype=np.float32))
        np.save(os.path.join(ep_dir, "done.npy"), np.array(ep_dones, dtype=bool))
        
    print(f"Collected {episodes} episodes ({total_transitions} transitions) -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), '..', '..', 'dataset'))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    collect(args.episodes, args.out, args.seed, args.steps)
