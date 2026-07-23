import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Add the warehouse_sim directory to path so internal imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'aura_warehouse_sim', 'warehouse_sim'))
from sim import WarehouseSim

from aura_agent.config import AgentConfig
from aura_agent.agent import AURAAgent

def explore(episodes: int, out_dir: str, seed: int, max_steps: int, obs_size: int, difficulty: int):
    os.makedirs(out_dir, exist_ok=True)
    
    config = AgentConfig()
    agent = AURAAgent(config)

    obs_l, act_l, nxt_l, tag_l, ep_l, t_l = [], [], [], [], [], []
    rew_l, term_l = [], []
    stats = {"steps": 0, "collisions": 0, "successes": 0, "interactions": 0}

    print(f"Starting Active Exploration (Difficulty Level {difficulty})...")
    
    for ep in range(episodes):
        sim = WarehouseSim(seed=seed * 10_000 + ep, obs_size=obs_size, max_steps=max_steps, difficulty=difficulty)
        obs, info = sim.reset()
        agent.reset()
        
        while True:
            # Agent uses hypothesis-driven active exploration to pick actions
            action = agent.step_active_exploration(obs)
            
            nxt, r, term, trunc, info = sim.step(action)
            tags = sim.state_summary()

            obs_l.append(obs)
            act_l.append(action)
            nxt_l.append(nxt)
            rew_l.append(r)
            term_l.append(term or trunc)
            
            tag_l.append([tags["near_human"], tags["near_forklift"],
                          tags["near_door"], tags["near_box"]])
            ep_l.append(ep)
            t_l.append(info["t"])

            stats["steps"] += 1
            if info["collision"] is not None:
                stats["collisions"] += 1
            if action == 4:
                stats["interactions"] += 1

            obs = nxt
            if term or trunc:
                break

    path = os.path.join(out_dir, f"exploration_seed{seed}_diff{difficulty}.npz")
    np.savez_compressed(
        path,
        obs=np.asarray(obs_l, dtype=np.uint8),
        action=np.asarray(act_l, dtype=np.int8),
        next_obs=np.asarray(nxt_l, dtype=np.uint8),
        reward=np.asarray(rew_l, dtype=np.float32),
        term=np.asarray(term_l, dtype=bool),
        event_tags=np.asarray(tag_l, dtype=bool),
        episode=np.asarray(ep_l, dtype=np.int32),
        t=np.asarray(t_l, dtype=np.int32),
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved {path}  ({stats['steps']} transitions, {size_mb:.1f} MB)")
    print(f"Stats: {stats}")
    return path, stats

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--obs-size", type=int, default=96)
    p.add_argument("--difficulty", type=int, default=1)
    args = p.parse_args()
    explore(args.episodes, args.out, args.seed, args.max_steps, args.obs_size, args.difficulty)
