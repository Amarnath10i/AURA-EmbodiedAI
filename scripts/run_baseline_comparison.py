"""Compare LAMA's full selection policy against three baselines, across
multiple seeds, on the headline research metric: recall broken out by
affordance role (PRIMARY / SECONDARY / INCIDENTAL).

This is the experiment the whole project's design is organised around (D5 in
docs/DECISIONS.md, and see README's "Evaluation plan"). Any exploration
strategy -- even pure random walking -- can stumble into a PRIMARY affordance,
because it is what the object visibly responds to most. SECONDARY
affordances (using one object as a means to an end, e.g. weighing down a
pressure plate with something heavy) are rare and easy to miss. Whether
targeted, uncertainty-and-safety-aware counterfactual verification finds
SECONDARY affordances faster than blind or purely-novelty-driven exploration
is the actual claim; an aggregate precision/recall number cannot show it.

Four policies, all run through the IDENTICAL environment, memory, and
adjudication (`verification.loop.run_episode`'s `select_fn` seam -- see its
module docstring) so only the selection strategy differs:

  random             uniformly random among affordable hypotheses
  novelty            prefer whatever has never been tried (Plan2Explore-style)
  uncertainty_only   select.py's core acquisition function, safety/curiosity/
                     goal bonuses removed (isolates what those add)
  lama               the full system

Usage:
    python scripts/run_baseline_comparison.py --episodes 25 --seeds 5
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lama.env.warehouse import Warehouse
from lama.evaluation import Evaluator
from lama.evaluation.baselines import (
    NoveltyPolicy,
    RandomPolicy,
    uncertainty_only_policy,
)
from lama.memory.memory import AffordanceMemory
from lama.verification.loop import run_episode
from lama.verification.select import select_next

POLICIES = {
    "random": lambda seed: RandomPolicy(seed=seed),
    "novelty": lambda seed: NoveltyPolicy(seed=seed),
    "uncertainty_only": lambda seed: uncertainty_only_policy,
    "lama": lambda seed: select_next,
}

METRICS = (
    "precision", "recall",
    "primary_recall", "secondary_recall", "incidental_recall",
    "interactions_per_confirmed",
)


def run_one(
    policy_name: str, seed: int, n_episodes: int, budget: float, n_objects: int,
) -> dict:
    policy = POLICIES[policy_name](seed)
    run_fn = functools.partial(run_episode, select_fn=policy)
    evaluator = Evaluator(
        env_factory=lambda seed=seed, **kw: Warehouse(
            seed=seed, budget=budget, n_objects=n_objects, **kw
        ),
        memory_factory=AffordanceMemory,
        run_episode_fn=run_fn,
    )
    pr = evaluator.evaluate_precision_recall(n_episodes=n_episodes, seed=seed)
    eff = evaluator.evaluate_efficiency(n_episodes=n_episodes, seed=seed)
    return {
        "precision": pr["precision"],
        "recall": pr["recall"],
        "primary_recall": pr["by_role"]["primary"]["recall"],
        "secondary_recall": pr["by_role"]["secondary"]["recall"],
        "incidental_recall": pr["by_role"]["incidental"]["recall"],
        "interactions_per_confirmed": eff["interactions_per_confirmed"],
        "confirmed_count": pr["tp"] + pr["fp"],
    }


def summarize(runs: list[dict]) -> dict:
    out = {}
    for metric in METRICS:
        values = [r[metric] for r in runs]
        out[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return out


def print_table(summary: dict[str, dict]) -> None:
    header = f"{'policy':18s}" + "".join(f"{m:>22s}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for policy, stats in summary.items():
        row = f"{policy:18s}"
        for metric in METRICS:
            m, s = stats[metric]["mean"], stats[metric]["std"]
            row += f"{m:>15.3f} +/-{s:<5.2f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--n-objects", type=int, default=10)
    parser.add_argument("--output", type=str, default="outputs/baseline_comparison.json")
    args = parser.parse_args()

    print(f"Comparing {list(POLICIES)} over {args.seeds} seeds x "
          f"{args.episodes} episodes each (budget={args.budget}, "
          f"n_objects={args.n_objects})\n")

    all_runs: dict[str, list[dict]] = {}
    for name in POLICIES:
        runs = []
        for seed in range(args.seeds):
            r = run_one(name, seed, args.episodes, args.budget, args.n_objects)
            runs.append(r)
            print(f"  {name:18s} seed={seed}  precision={r['precision']:.2f}  "
                  f"recall={r['recall']:.2f}  secondary_recall="
                  f"{r['secondary_recall']:.2f}  confirmed={r['confirmed_count']}")
        all_runs[name] = runs
    print()

    summary = {name: summarize(runs) for name, runs in all_runs.items()}
    print_table(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        {"config": vars(args), "raw_runs": all_runs, "summary": summary}, indent=2
    ))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
