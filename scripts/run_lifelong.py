"""Run the LAMA loop for many episodes on one persistent memory, and report
whether it measurably gets better at learning over time.

This is the concrete form "self-improvement" takes in this project. There is
no neural network in the working core to gradient-train -- the affordance
bank IS the model (see imagination/hypothesis.py), so getting better means
something specific and checkable: as confirmed and refuted knowledge
accumulates, later episodes should need fewer interactions to reach each new
piece of confirmed knowledge than earlier episodes did, because budget stops
being wasted on verbs the bank has already settled and starts being spent
where it still matters.

**The goal is derived from the agent's own experience, never from the
oracle.** Once ANY concept is confirmed to reach `Effect.OPENED` -- most
often the door's own `OPEN` verb, discovered through ordinary exploration --
`find_a_goal` targets that same (concept, effect) pair as a standing goal.
From then on, `RegressionPlanner.relevant_keys` biases selection toward
OTHER ways of reaching the same effect, which is exactly how an agent that
already knows a door can be opened one way would go looking for a second,
possibly cheaper or more informative way -- discovering the plate's
secondary affordance because it already has a reason to look for
alternatives, not by chance alone.

Usage:
    python scripts/run_lifelong.py --episodes 200 --budget 60
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lama.env import Effect
from lama.env.warehouse import Warehouse
from lama.memory.memory import AffordanceMemory
from lama.planner import Goal, RegressionPlanner
from lama.verification.loop import run_episode


def find_a_goal(memory: AffordanceMemory) -> Goal | None:
    """A goal drawn from what the agent itself has already confirmed, never
    from hidden ground truth: the first concept confirmed to reach OPENED,
    if any exist yet."""
    for belief in memory.bank.confirmed():
        if belief.dominant_effect is Effect.OPENED:
            return Goal(concept=belief.key[0], effect=Effect.OPENED)
    return None


@dataclass
class EpisodeStats:
    episode: int
    interactions: int
    confirmed_total: int
    refuted_total: int
    stuck_total: int
    new_confirmed: int
    had_goal: bool


def run(
    episodes: int, budget: float, n_objects: int, seed: int
) -> tuple[list[EpisodeStats], AffordanceMemory]:
    memory = AffordanceMemory()
    env = Warehouse(seed=seed, budget=budget, n_objects=n_objects)
    stats: list[EpisodeStats] = []
    prev_confirmed = 0

    for ep in range(episodes):
        goal = find_a_goal(memory)
        before = len(memory.log)
        run_episode(env, memory, seed=seed + ep, goal=goal)
        interactions = len(memory.log) - before

        confirmed = len(memory.bank.confirmed())
        refuted = len(memory.bank.refuted())
        stuck = len(memory.bank.stuck())
        stats.append(EpisodeStats(
            episode=ep, interactions=interactions, confirmed_total=confirmed,
            refuted_total=refuted, stuck_total=stuck,
            new_confirmed=confirmed - prev_confirmed, had_goal=goal is not None,
        ))
        prev_confirmed = confirmed

    return stats, memory


def summarize(stats: list[EpisodeStats], block: int = 20) -> None:
    print(f"{'episodes':>12}  {'interactions':>13}  {'new confirmed':>14}  "
          f"{'interactions/new':>17}  {'goal active':>11}")
    for start in range(0, len(stats), block):
        chunk = stats[start:start + block]
        interactions = sum(s.interactions for s in chunk)
        new_confirmed = sum(s.new_confirmed for s in chunk)
        per_new = interactions / new_confirmed if new_confirmed else float("inf")
        goal_frac = sum(s.had_goal for s in chunk) / len(chunk)
        label = f"{chunk[0].episode}-{chunk[-1].episode}"
        per_new_s = f"{per_new:.1f}" if per_new != float("inf") else "--"
        print(f"{label:>12}  {interactions:>13d}  {new_confirmed:>14d}  "
              f"{per_new_s:>17}  {goal_frac:>10.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--n-objects", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=20,
                       help="episodes per reported block")
    args = parser.parse_args()

    print(f"Running {args.episodes} episodes on one persistent memory "
          f"(budget={args.budget}, n_objects={args.n_objects}, seed={args.seed})")
    stats, memory = run(args.episodes, args.budget, args.n_objects, args.seed)
    print()
    summarize(stats, args.block)
    print()

    first_goal_ep = next((s.episode for s in stats if s.had_goal), None)
    print(f"Goal-directed pursuit began at episode: {first_goal_ep}")
    print(f"Final: {len(memory.bank.confirmed())} confirmed, "
          f"{len(memory.bank.refuted())} refuted, "
          f"{len(memory.bank.stuck())} stuck, "
          f"{len(memory.concepts)} concepts formed, "
          f"{len(memory.log)} total interactions logged")

    remote_confirmed = [b for b in memory.bank.confirmed() if b.remote_rate > 0.15]
    if remote_confirmed:
        print()
        print("Confirmed beliefs with a remote effect (candidate secondary "
              "affordances):")
        for b in remote_confirmed:
            print(f"  concept={b.key[0]:>3} {b.key[1].name:9s} "
                  f"tool_concept={str(b.key[2]):>4}  n={b.total_attempts:3d}  "
                  f"mean={b.mean:.2f}  remote_rate={b.remote_rate:.2f}")


if __name__ == "__main__":
    main()
