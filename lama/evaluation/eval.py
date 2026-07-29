"""Evaluation harness: transfer, retention, efficiency metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class EvalMetrics:
    """Metrics matching the LAMA evaluation plan."""
    interactions_per_confirmed: float
    precision: float
    recall: float
    retention: float
    transfer_precision: float
    transfer_recall: float


class Evaluator:
    """Runs evaluation episodes and computes metrics."""

    def __init__(self, env_factory, memory_factory, run_episode_fn):
        self.env_factory = env_factory
        self.memory_factory = memory_factory
        self.run_episode = run_episode_fn

    def evaluate_efficiency(
        self,
        n_episodes: int = 10,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Interactions per confirmed affordance (headline metric)."""
        env = self.env_factory(seed=seed)
        mem = self.memory_factory()

        total_interactions = 0
        for ep in range(n_episodes):
            steps = self.run_episode(env, mem, seed=seed + ep)
            tested = sum(1 for s in steps if s.hypothesis is not None)
            total_interactions += tested

        confirmed = len(mem.bank.confirmed())
        return {
            "interactions_per_confirmed": total_interactions / max(1, confirmed),
            "total_interactions": total_interactions,
            "confirmed_count": confirmed,
        }

    def evaluate_precision_recall(
        self,
        n_episodes: int = 20,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Precision/recall of affordance bank vs hidden ground truth."""
        # This requires oracle access to true affordances
        env = self.env_factory(seed=seed)
        oracle = env.oracle  # assumes WarehouseOracle/IsaacWarehouseOracle
        mem = self.memory_factory()

        # Run episodes
        for ep in range(n_episodes):
            self.run_episode(env, mem, seed=seed + ep)

        # Compare confirmed beliefs to ground truth
        true_affordances = oracle.ground_truth_affordances()
        confirmed = mem.bank.confirmed()

        tp = fp = fn = 0
        for belief in confirmed:
            key = belief.key
            if key in true_affordances:
                if true_affordances[key] > 0.5:
                    tp += 1
                else:
                    fp += 1
            else:
                fp += 1

        for key, prob in true_affordances.items():
            if prob > 0.5 and key not in [b.key for b in confirmed]:
                fn += 1

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}

    def evaluate_retention(
        self,
        task_a_episodes: int = 10,
        task_b_episodes: int = 10,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Retention across tasks/layouts."""
        env = self.env_factory(seed=seed)
        mem = self.memory_factory()

        # Task A
        for ep in range(task_a_episodes):
            self.run_episode(env, mem, seed=seed + ep)

        confirmed_a = {b.key: b for b in mem.bank.confirmed()}

        # Task B (different layout)
        for ep in range(task_b_episodes):
            self.run_episode(env, mem, seed=seed + task_a_episodes + ep)

        confirmed_b = {b.key: b for b in mem.bank.confirmed()}

        # Retention: fraction of A beliefs still confirmed after B
        retained = sum(1 for k in confirmed_a if k in confirmed_b)
        retention = retained / max(1, len(confirmed_a))

        return {
            "retention": retention,
            "confirmed_task_a": len(confirmed_a),
            "confirmed_task_b": len(confirmed_b),
            "retained_count": retained,
        }

    def evaluate_transfer(
        self,
        train_episodes: int = 50,
        test_episodes: int = 10,
        held_out_kinds: List[str] = None,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Transfer to held-out object kinds."""
        if held_out_kinds is None:
            held_out_kinds = ["pallet", "drum", "switch"]

        # Train with held-out kinds disabled
        env_train = self.env_factory(seed=seed, held_out=held_out_kinds)
        mem = self.memory_factory()
        for ep in range(train_episodes):
            self.run_episode(env_train, mem, seed=seed + ep)

        # Test with held-out kinds enabled
        env_test = self.env_factory(seed=seed + 1000, held_out=[])
        oracle = env_test.oracle
        true_held_out = {k: v for k, v in oracle.ground_truth_affordances().items()
                         if any(h in str(k) for h in held_out_kinds)}

        # Run test episodes
        for ep in range(test_episodes):
            self.run_episode(env_test, mem, seed=seed + 1000 + ep)

        # Evaluate on held-out
        confirmed = {b.key: b for b in mem.bank.confirmed()}
        tp = sum(1 for k in true_held_out if k in confirmed and true_held_out[k] > 0.5)
        fp = sum(1 for k in confirmed if k in true_held_out and true_held_out[k] <= 0.5)
        fn = sum(1 for k, v in true_held_out.items() if v > 0.5 and k not in confirmed)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        return {
            "transfer_precision": precision,
            "transfer_recall": recall,
            "held_out_kinds": held_out_kinds,
        }

    def run_full_eval(self, output_path: str = "results/eval_results.json"):
        """Run all evaluations and save."""
        results = {}

        print("Running efficiency eval...")
        results["efficiency"] = self.evaluate_efficiency()

        print("Running precision/recall...")
        results["precision_recall"] = self.evaluate_precision_recall()

        print("Running retention...")
        results["retention"] = self.evaluate_retention()

        print("Running transfer...")
        results["transfer"] = self.evaluate_transfer()

        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")
        return results