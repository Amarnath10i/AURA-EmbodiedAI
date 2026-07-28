"""Run the LAMA imagination/verification loop against IsaacWarehouse.

The entire point of D4 in docs/DECISIONS.md: nothing in `lama.memory`,
`lama.imagination` or `lama.verification` changes to use this backend instead
of the numpy one. If this script runs, that claim is proven, not just stated.

Run `scripts/isaac_smoke_test.py` first. If that fails, this will too, for
the same underlying reason -- fix that one first, since it isolates the
Isaac Lab install from this project's code.

Usage:
    python scripts/run_lama_isaac.py --episodes 3 --budget 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running "python scripts/run_lama_isaac.py" puts scripts/ on sys.path, not
# the repo root, so `lama` would not be importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lama.env.isaac_warehouse import IsaacWarehouse, IsaacWarehouseOracle
from lama.memory.memory import AffordanceMemory
from lama.verification import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--budget", type=float, default=40.0)
    parser.add_argument("--n-objects", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--show", dest="headless", action="store_false")
    args = parser.parse_args()

    print(f"Constructing IsaacWarehouse (seed={args.seed}, "
          f"n_objects={args.n_objects}) -- this launches Isaac Sim once.")
    env = IsaacWarehouse(
        seed=args.seed, n_objects=args.n_objects, budget=args.budget,
        headless=args.headless,
    )
    oracle = IsaacWarehouseOracle(env)
    print("Layout:", oracle.summary())

    mem = AffordanceMemory()
    for ep in range(args.episodes):
        steps = run_episode(env, mem, seed=args.seed + ep)
        tested = sum(1 for s in steps if s.hypothesis is not None)
        approached = sum(1 for s in steps if s.approached is not None)
        print(
            f"episode {ep}: {len(steps)} loop steps "
            f"({tested} tested, {approached} approach-only), "
            f"budget left {env.budget_remaining:.1f}"
        )

    print()
    print(f"concepts formed: {len(mem.concepts)}")
    print(f"beliefs: {len(mem.bank)}  confirmed: {len(mem.bank.confirmed())}  "
          f"refuted: {len(mem.bank.refuted())}")
    print(f"ledger entries: {len(mem.log)}  total budget spent: "
          f"{mem.log.total_cost():.1f}")
    print()
    print("Confirmed beliefs with a remote effect (candidate secondary "
          "affordances -- the same emergent signature the numpy backend "
          "produces for the crate/block trap):")
    for b in mem.bank.confirmed():
        if b.remote_rate > 0.15:
            print(f"  concept={b.key[0]:>3} {b.key[1].name:9s} "
                  f"tool_concept={str(b.key[2]):>4}  n={b.total_attempts:3d}  "
                  f"mean={b.mean:.2f}  remote_rate={b.remote_rate:.2f}")


if __name__ == "__main__":
    main()
