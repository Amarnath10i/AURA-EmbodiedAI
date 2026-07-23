import argparse
import os
import sys
import numpy as np

# Add the warehouse_sim directory to path so internal imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'aura_warehouse_sim', 'warehouse_sim'))
from sim import WarehouseSim
from collect import smart_random_policy

def collect(episodes: int, out_dir: str, seed: int, max_steps: int, obs_size: int):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    obs_l, act_l, nxt_l, tag_l, ep_l, t_l = [], [], [], [], [], []
    rew_l, term_l = [], []
    stats = {"steps": 0, "collisions": 0, "successes": 0, "surprise_events": 0}

    for ep in range(episodes):
        sim = WarehouseSim(seed=seed * 10_000 + ep, obs_size=obs_size, max_steps=max_steps)
        obs, info = sim.reset()
        collided = False
        while True:
            a = smart_random_policy(sim, rng, collided)
            nxt, r, term, trunc, info = sim.step(a)
            tags = sim.state_summary()

            obs_l.append(obs)
            act_l.append(a)
            nxt_l.append(nxt)
            rew_l.append(r)
            term_l.append(term or trunc)
            
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
        reward=np.asarray(rew_l, dtype=np.float32),
        term=np.asarray(term_l, dtype=bool),
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
    p.add_argument("--episodes", type=int, default=50) # default to small prototype
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--obs-size", type=int, default=96)
    args = p.parse_args()
    collect(args.episodes, args.out, args.seed, args.max_steps, args.obs_size)
