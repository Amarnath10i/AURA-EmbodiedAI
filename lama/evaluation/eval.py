"""Evaluation harness: the metrics from the README's evaluation plan.

Every method here is oracle-side code -- it is allowed to read hidden ground
truth, unlike anything in `imagination/`, `verification/` or `memory/`. That
is the entire point of the split between `Environment` and `AffordanceOracle`
(`env/interface.py`): the agent never sees what this module sees.

**The one genuinely hard part.** A confirmed belief is keyed by
`(concept_id, verb, tool_concept_id)` -- the agent's own appearance-formed
concepts. Ground truth is keyed by `(kind, verb, tool_kind)` -- real object
kinds. These are different key spaces, and a confirmed belief can only be
scored against ground truth once its concept has been resolved back to the
kind(s) it actually corresponds to. `_ConceptKindTracker` does that
resolution while objects are still live in the current episode -- the oracle
cannot identify an object once a later `reset()` has replaced it, so this
has to happen incrementally, episode by episode, not retrospectively at the
end of a run.

That resolution is also where the project's central honesty check lives: a
concept that resolves to MORE THAN ONE kind (crate and block sharing one
concept, because they look identical) is a blended concept, and a "confirmed"
belief about it is answering for the blend, not for either kind alone. This
module reports that rate explicitly rather than only reporting an aggregate
precision number that would hide it.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..env import catalogue
from ..env.affordance import Role
from ..env.interface import AffordanceOracle, Environment
from ..memory.memory import AffordanceMemory


def _make_oracle(env: Environment) -> AffordanceOracle:
    """The right `AffordanceOracle` for `env`, chosen by its `backend_name`.

    Imports are local so evaluating against the numpy backend never requires
    Isaac Lab to be installed.
    """
    if env.backend_name == "numpy-warehouse":
        from ..env.warehouse import WarehouseOracle

        return WarehouseOracle(env)
    if env.backend_name == "isaac-lab":
        from ..env.isaac_warehouse import IsaacWarehouseOracle

        return IsaacWarehouseOracle(env)
    raise ValueError(f"no oracle registered for backend {env.backend_name!r}")


@dataclass(frozen=True)
class ConceptResolution:
    """What a concept id actually turned out to mean, in ground truth.

    Attributes:
        concept_id: The agent's own concept id.
        kind_counts: How many observed instances of this concept were each
            true kind. More than one key present means this concept is
            blended -- see the module docstring.
        dominant_kind: The most common true kind, or `None` if the concept
            was never resolved against a live object (e.g. formed from an
            object whose episode ended before evaluation looked it up).
    """

    concept_id: int
    kind_counts: Counter
    dominant_kind: str | None

    @property
    def is_blended(self) -> bool:
        return len(self.kind_counts) > 1


class ConceptKindTracker:
    """Resolves concept ids back to the true kinds behind them.

    Must be fed `(env, mem)` right after each episode that used `env`, before
    the next `reset()` invalidates the oracle's view of that episode's
    objects -- `evaluate_*` below always calls `observe_episode` inline with
    `run_episode`, never after the fact.
    """

    def __init__(self) -> None:
        self._counts: dict[int, Counter] = defaultdict(Counter)

    def observe_episode(
        self, env: Environment, mem: AffordanceMemory, log_start: int
    ) -> None:
        """Resolve every ledger entry appended since `log_start` (i.e. during
        the episode just run) against `env`'s oracle, while still valid."""
        oracle = _make_oracle(env)
        for entry in mem.log.entries()[log_start:]:
            self._note(oracle, entry.target_concept, entry.object_id)
            if entry.tool_concept is not None and entry.tool_id is not None:
                self._note(oracle, entry.tool_concept, entry.tool_id)

    def _note(self, oracle: AffordanceOracle, concept_id: int, object_id: str) -> None:
        try:
            kind = oracle.object_kind(object_id)
        except KeyError:
            return
        self._counts[concept_id][kind] += 1

    def resolve(self, concept_id: int) -> ConceptResolution:
        counts = self._counts.get(concept_id, Counter())
        dominant = counts.most_common(1)[0][0] if counts else None
        return ConceptResolution(concept_id, counts, dominant)

    def blended_concepts(self) -> tuple[ConceptResolution, ...]:
        return tuple(
            self.resolve(c) for c in self._counts if self.resolve(c).is_blended
        )


def _confirmed_match(
    tracker: ConceptKindTracker, belief_key: tuple,
):
    """`(matched_affordance, dominant_kind, dominant_tool_kind)` for one
    confirmed belief key. `matched_affordance` is `None` if the dominant kind
    is unresolved, or does not genuinely afford this verb in ground truth --
    otherwise it is the real `Affordance`, whose `.role` is what lets
    precision/recall be broken out by PRIMARY/SECONDARY/INCIDENTAL, the
    headline distinction this project's whole design is organised around
    (D5 in docs/DECISIONS.md): any policy can stumble into a primary
    affordance, the interesting question is whether targeted verification
    finds secondary ones faster than blind exploration does.
    """
    target_concept, action, tool_concept = belief_key
    target_kind = tracker.resolve(target_concept).dominant_kind
    tool_kind = tracker.resolve(tool_concept).dominant_kind if tool_concept is not None else None
    if target_kind is None:
        return None, None, tool_kind
    aff = catalogue.lookup(target_kind, action, tool_kind)
    return (aff if aff.is_real else None), target_kind, tool_kind


class Evaluator:
    """Runs evaluation episodes and computes the README's metrics.

    Args:
        env_factory: `(seed, **kwargs) -> Environment`. `include_held_out` is
            the one kwarg `evaluate_transfer` passes through; it must match
            `Warehouse`/`IsaacWarehouse`'s actual parameter name.
        memory_factory: `() -> AffordanceMemory`.
        run_episode_fn: `verification.loop.run_episode` (or a drop-in),
            injected rather than imported so this module never has to import
            `verification/` and risk a cycle back through `planner/`.
    """

    def __init__(
        self,
        env_factory: Callable[..., Environment],
        memory_factory: Callable[[], AffordanceMemory],
        run_episode_fn: Callable,
    ) -> None:
        self.env_factory = env_factory
        self.memory_factory = memory_factory
        self.run_episode = run_episode_fn

    def _run_tracked(
        self, env: Environment, mem: AffordanceMemory, n_episodes: int, seed: int,
        tracker: ConceptKindTracker,
    ) -> None:
        for ep in range(n_episodes):
            log_start = len(mem.log)
            self.run_episode(env, mem, seed=seed + ep)
            tracker.observe_episode(env, mem, log_start)

    def evaluate_efficiency(self, n_episodes: int = 10, seed: int = 0) -> dict[str, Any]:
        """Interactions per confirmed affordance -- the headline number."""
        env = self.env_factory(seed=seed)
        mem = self.memory_factory()
        tracker = ConceptKindTracker()
        self._run_tracked(env, mem, n_episodes, seed, tracker)

        confirmed = len(mem.bank.confirmed())
        total_interactions = len(mem.log)
        return {
            "interactions_per_confirmed": total_interactions / max(1, confirmed),
            "total_interactions": total_interactions,
            "confirmed_count": confirmed,
        }

    def evaluate_precision_recall(
        self, n_episodes: int = 20, seed: int = 0
    ) -> dict[str, Any]:
        """Precision/recall of confirmed beliefs against hidden ground truth,
        resolved through concept-to-kind mapping (see the module docstring).
        """
        env = self.env_factory(seed=seed)
        mem = self.memory_factory()
        tracker = ConceptKindTracker()
        self._run_tracked(env, mem, n_episodes, seed, tracker)

        confirmed = mem.bank.confirmed()
        matches = {b.key: _confirmed_match(tracker, b.key) for b in confirmed}
        tp = sum(1 for aff, _, _ in matches.values() if aff is not None)
        fp = len(confirmed) - tp

        oracle = _make_oracle(self.env_factory(seed=seed))
        reachable_real = [a for a in oracle.catalogue_affordances() if a.is_real]
        found_kinds = {
            (r.dominant_kind, b.key[1], tracker.resolve(b.key[2]).dominant_kind
             if b.key[2] is not None else None)
            for b in confirmed
            for r in [tracker.resolve(b.key[0])]
            if r.dominant_kind is not None
        }
        fn = sum(1 for a in reachable_real if a.key not in found_kinds)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        # The headline breakdown (D5 in docs/DECISIONS.md): any exploration
        # strategy can stumble into a PRIMARY affordance -- it is what the
        # object is FOR, and easy to trigger by accident. SECONDARY
        # affordances (using an object as a means to an end) are rare and
        # easy to miss; whether targeted verification finds them faster than
        # blind exploration is the actual research question, and an
        # aggregate precision/recall number hides the answer.
        by_role: dict[str, dict[str, Any]] = {}
        for role in (Role.PRIMARY, Role.SECONDARY, Role.INCIDENTAL):
            role_reachable = [a for a in reachable_real if a.role is role]
            role_tp = sum(
                1 for aff, _, _ in matches.values() if aff is not None and aff.role is role
            )
            role_found = {
                (aff.kind, aff.action, aff.tool_kind)
                for aff, _, _ in matches.values() if aff is not None and aff.role is role
            }
            role_fn = sum(1 for a in role_reachable if a.key not in role_found)
            by_role[role.name.lower()] = {
                "confirmed": role_tp,
                "reachable": len(role_reachable),
                "recall": role_tp / max(1, role_tp + role_fn),
            }

        blended = tracker.blended_concepts()
        return {
            "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn,
            "by_role": by_role,
            "blended_concept_count": len(blended),
            "blended_concepts": [
                {"concept_id": r.concept_id, "kinds": dict(r.kind_counts)}
                for r in blended
            ],
        }

    def evaluate_retention(
        self, task_a_episodes: int = 10, task_b_episodes: int = 10, seed: int = 0
    ) -> dict[str, Any]:
        """Fraction of beliefs confirmed after task A that are still
        confirmed after task B runs on the same, persistent memory."""
        env = self.env_factory(seed=seed)
        mem = self.memory_factory()
        tracker = ConceptKindTracker()

        self._run_tracked(env, mem, task_a_episodes, seed, tracker)
        confirmed_a = {b.key for b in mem.bank.confirmed()}

        self._run_tracked(env, mem, task_b_episodes, seed + task_a_episodes, tracker)
        confirmed_b = {b.key for b in mem.bank.confirmed()}

        retained = sum(1 for k in confirmed_a if k in confirmed_b)
        return {
            "retention": retained / max(1, len(confirmed_a)),
            "confirmed_task_a": len(confirmed_a),
            "confirmed_task_b": len(confirmed_b),
            "retained_count": retained,
        }

    def evaluate_transfer(
        self, train_episodes: int = 50, test_episodes: int = 10, seed: int = 0
    ) -> dict[str, Any]:
        """Precision/recall restricted to the held-out kinds, after training
        with them absent and testing with them present."""
        env_train = self.env_factory(seed=seed, include_held_out=False)
        mem = self.memory_factory()
        tracker = ConceptKindTracker()
        self._run_tracked(env_train, mem, train_episodes, seed, tracker)

        env_test = self.env_factory(seed=seed + 1000, include_held_out=True)
        self._run_tracked(env_test, mem, test_episodes, seed + 1000, tracker)

        held_out_kinds = {
            name for name, spec in catalogue.KINDS.items() if spec.held_out
        }
        confirmed = mem.bank.confirmed()
        on_held_out = [
            b for b in confirmed
            if tracker.resolve(b.key[0]).dominant_kind in held_out_kinds
        ]
        tp = sum(
            1 for b in on_held_out if _confirmed_match(tracker, b.key)[0] is not None
        )
        fp = len(on_held_out) - tp

        oracle = _make_oracle(env_test)
        true_held_out_real = [
            a for a in oracle.catalogue_affordances()
            if a.is_real and a.kind in held_out_kinds
        ]
        found = {
            (tracker.resolve(b.key[0]).dominant_kind, b.key[1],
             tracker.resolve(b.key[2]).dominant_kind if b.key[2] is not None else None)
            for b in on_held_out
        }
        fn = sum(1 for a in true_held_out_real if a.key not in found)

        return {
            "transfer_precision": tp / max(1, tp + fp),
            "transfer_recall": tp / max(1, tp + fn),
            "held_out_kinds": sorted(held_out_kinds),
            "confirmed_on_held_out": len(on_held_out),
        }

    def run_full_eval(self, output_path: str = "outputs/eval_results.json") -> dict:
        """Run every evaluation and save the combined result."""
        results = {
            "efficiency": self.evaluate_efficiency(),
            "precision_recall": self.evaluate_precision_recall(),
            "retention": self.evaluate_retention(),
            "transfer": self.evaluate_transfer(),
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        return results
