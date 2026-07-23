"""
AURA Stage-0 data collection: random walk -> (obs, action, next_obs) transitions.

This produces exactly the dataset format the rest of the AURA pipeline consumes:
  transitions.npz shards containing
      obs        (N, S, S, 3) uint8   frame at time t
      action     (N,)          int8    action taken
      next_obs   (N, S, S, 3) uint8   frame at time t+1
      event_tags (N, 4)        bool    ground-truth context [human, forklift, door, box]
      episode    (N,)          int32   episode id (for sequence models)
      t          (N,)          int32   timestep within episode

Usage:
    python collect.py --episodes 20 --out data/ --seed 0
Scale up with more --episodes (or run many processes with different seeds);
Isaac Lab later replaces WarehouseSim with no change to this file's structure.
"""

import argparse
import os
import numpy as np
from sim import WarehouseSim


def smart_random_policy(sim: WarehouseSim, rng, last_collided: bool) -> int:
    """Random walk biased toward moving; turns away after collisions.
    (Pure uniform-random walks mostly bump into walls; this explores better.)"""
    if last_collided:
        return int(rng.choice([1, 2]))          # turn after hitting something
    return int(rng.choice([0, 0, 0, 1, 2, 3], p=[0.6, 0.1, 0.05, 0.11, 0.11, 0.03]))


def collect(episodes: int, out_dir: str, seed: int, max_steps: int, obs_size: int):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    obs_l, act_l, nxt_l, tag_l, ep_l, t_l = [], [], [], [], [], []
    stats = {"steps": 0, "collisions": 0, "successes": 0, "surprise_events": 0}

    for ep in range(episodes):
        sim = WarehouseSim(seed=seed * 10_000 + ep, obs_size=obs_size,
                           max_steps=max_steps)
        obs, info = sim.reset()
        collided = False
        while True:
            a = smart_random_policy(sim, rng, collided)
            nxt, r, term, trunc, info = sim.step(a)
            tags = sim.state_summary()

            obs_l.append(obs)
            act_l.append(a)
            nxt_l.append(nxt)
            tag_l.append([tags["near_human"], tags["near_forklift"],
                          tags["near_door"], tags["near_box"]])
            ep_l.append(ep)
            t_l.append(info["t"])

            collided = info["collision"] is not None
            stats["steps"] += 1
            stats["collisions"] += int(collided)
            stats["successes"] += int(info["success"])
            stats["surprise_events"] += sum(
                e in ("human_moved_box", "door_opened", "door_closed")
                for e in info["events"])

            obs = nxt
            if term or trunc:
                break

    path = os.path.join(out_dir, f"transitions_seed{seed}.npz")
    np.savez_compressed(
        path,
        obs=np.asarray(obs_l, dtype=np.uint8),
        action=np.asarray(act_l, dtype=np.int8),
        next_obs=np.asarray(nxt_l, dtype=np.uint8),
        event_tags=np.asarray(tag_l, dtype=bool),
        episode=np.asarray(ep_l, dtype=np.int32),
        t=np.asarray(t_l, dtype=np.int32),
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"saved {path}  ({stats['steps']} transitions, {size_mb:.1f} MB)")
    print(f"stats: {stats}")
    return path, stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--out", type=str, default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--obs-size", type=int, default=96)
    args = p.parse_args()
    collect(args.episodes, args.out, args.seed, args.max_steps, args.obs_size)
