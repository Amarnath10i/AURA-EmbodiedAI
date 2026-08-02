"""Backward-chaining classical planning over confirmed affordance knowledge.

The problem this exists to solve: with N verbs and M reachable objects, the
raw space of things worth trying grows combinatorially, and it keeps growing
as the verb set grows -- exactly the concern that motivated this module.
Uncertainty-driven selection (`verification/select.py`) already prunes
settled beliefs, but it has no notion of a GOAL: it is equally happy testing
something irrelevant to what the agent is actually trying to achieve as
something on the critical path to it.

This module adds the other half: goal-directed relevance, via the classical
AI technique the problem calls for -- STRIPS-style regression search,
backward from the goal. Given a `Goal` ("some object of concept X should
reach effect E"), `RegressionPlanner.relevant_keys` searches BACKWARD through
CONFIRMED operators, asking "what would have to be true right before this
effect, and what would have to be true right before THAT", and returns the
set of `(concept, verb, tool_concept)` keys that participate in some chain
reaching the goal. `imagination.hypothesis.imagine` looks up this set and
boosts `Hypothesis.relevance` for anything in it, so `verification/select.py`
naturally prefers goal-relevant tests over irrelevant ones -- without ever
hard-restricting what CAN be tried. The agent can still learn about anything;
it just tries goal-relevant things first, which is what turns "try every
combination" into "try the combinations that could plausibly matter".

This deliberately only chains through CONFIRMED knowledge: an operator the
bank is not yet sure about cannot be trusted as a stepping stone. When the
chain runs out of confirmed operators, it returns whatever it found so far --
a partial chain is still useful (it identifies exactly which concept's
behaviour is the missing piece), and ordinary uncertainty-driven exploration
in `select.py` fills the rest of the gap on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..env import Action, Effect
from ..env.actions import spec
from ..memory.bank import AffordanceBank, Belief


@dataclass(frozen=True)
class Operator:
    """A confirmed capability, in a form backward chaining can use.

    Preconditions here are real, derived from the verb's own public metadata
    (`ActionSpec`) rather than a placeholder -- that is what lets chaining
    discover genuine multi-step structure, e.g. "must be holding something of
    tool_concept, which means first grasping one, which needs a free
    gripper".
    """

    target_concept: int
    action: Action
    tool_concept: int | None
    effect: Effect
    reliability: float
    needs_free_gripper: bool
    needs_held_object: bool
    remote_effect: Effect | None
    remote_target_concept: int | None
    remote_reliability: float

    @property
    def key(self) -> tuple[int, Action, int | None]:
        return (self.target_concept, self.action, self.tool_concept)

    def satisfies(self, goal: "Goal") -> bool:
        """Whether firing this operator can make `goal` true, directly or
        through its remote effect -- the two ways a verb can change the
        world (see `env/outcomes.py`)."""
        direct = self.target_concept == goal.concept and self.effect is goal.effect
        remote = (
            self.remote_target_concept == goal.concept
            and self.remote_effect is goal.effect
        )
        return direct or remote


def operator_from_belief(belief: Belief) -> Operator | None:
    """The `Operator` a `CONFIRMED` belief implies, or `None` otherwise.

    Only `CONFIRMED` beliefs produce an operator: a `PROVISIONAL` or `STUCK`
    one is not knowledge a plan can be trusted to stand on.
    """
    if not belief.is_confirmed or belief.dominant_effect is None:
        return None
    target_concept, action, tool_concept = belief.key
    s = spec(action)
    remote = belief.dominant_remote
    return Operator(
        target_concept=target_concept,
        action=action,
        tool_concept=tool_concept,
        effect=belief.dominant_effect,
        reliability=belief.mean,
        needs_free_gripper=s.needs_free_gripper,
        needs_held_object=s.needs_held_object,
        remote_effect=remote[1] if remote else None,
        remote_target_concept=remote[0] if remote else None,
        remote_reliability=belief.remote_rate,
    )


@dataclass(frozen=True)
class Goal:
    """"Some object of `concept` should reach `effect`" -- satisfied either
    by an operator whose own effect is this, or by an operator whose REMOTE
    effect is this (the interesting case: opening a door by pressing a
    plate elsewhere)."""

    concept: int
    effect: Effect


@dataclass(frozen=True)
class PlanStep:
    """One link in a concrete backward-chained plan: an operator, and which
    goal it was chosen to satisfy."""

    operator: Operator
    satisfies: Goal


#: How much a relevance weight decays per hop away from the goal. A verb that
#: directly achieves the goal is weight 1.0; the verb that gets you the tool
#: for THAT verb is weight DECAY, and so on -- distant prerequisites still
#: matter, just less than the immediate next step.
_RELEVANCE_DECAY: float = 0.7


class RegressionPlanner:
    """STRIPS-style backward chaining over the bank's confirmed operators."""

    def __init__(self, bank: AffordanceBank):
        self.bank = bank

    def operators(self) -> tuple[Operator, ...]:
        """Every operator the bank's current confirmed knowledge implies."""
        return tuple(
            op for op in (operator_from_belief(b) for b in self.bank.confirmed())
            if op is not None
        )

    def plan(self, goal: Goal, max_depth: int = 6) -> list[PlanStep] | None:
        """One concrete chain of confirmed operators reaching `goal`, or
        `None` if confirmed knowledge cannot currently reach it.

        Most callers want `relevant_keys` instead -- this exists for
        inspection/logging/tests, where seeing one concrete chain is more
        readable than a weighted key set.
        """
        return self._search(goal, self.operators(), max_depth, frozenset())

    def _search(
        self, goal: Goal, operators: tuple[Operator, ...], depth: int,
        seen: frozenset,
    ) -> list[PlanStep] | None:
        goal_id = (goal.concept, goal.effect)
        if goal_id in seen or depth <= 0:
            return None
        seen = seen | {goal_id}

        for op in operators:
            if not op.satisfies(goal):
                continue

            prefix: list[PlanStep] = []
            if op.needs_held_object and op.tool_concept is not None:
                sub_plan = self._search(
                    Goal(op.tool_concept, Effect.CARRIED), operators, depth - 1, seen
                )
                if sub_plan is None:
                    continue  # this operator's prerequisite is unreachable; try another
                prefix = sub_plan

            return prefix + [PlanStep(op, goal)]

        return None

    def relevant_keys(
        self, goal: Goal, max_depth: int = 6
    ) -> dict[tuple[int, Action, int | None], float]:
        """Every `(concept, verb, tool_concept)` key that participates in
        SOME chain reaching `goal`, weighted by proximity to the goal.

        Unlike `plan`, this collects the union across every valid chain
        rather than committing to one -- ranking wants to boost every
        plausible route, not just the first one found; `verification/
        select.py`'s greedy per-step choice handles actually committing to a
        path as the episode unfolds.
        """
        weights: dict[tuple[int, Action, int | None], float] = {}
        self._collect(goal, self.operators(), max_depth, frozenset(), 1.0, weights)
        return weights

    def _collect(
        self, goal: Goal, operators: tuple[Operator, ...], depth: int,
        seen: frozenset, weight: float,
        out: dict[tuple[int, Action, int | None], float],
    ) -> None:
        goal_id = (goal.concept, goal.effect)
        if goal_id in seen or depth <= 0:
            return
        seen = seen | {goal_id}

        for op in operators:
            if not op.satisfies(goal):
                continue
            out[op.key] = max(out.get(op.key, 0.0), weight)
            if op.needs_held_object and op.tool_concept is not None:
                self._collect(
                    Goal(op.tool_concept, Effect.CARRIED), operators, depth - 1,
                    seen, weight * _RELEVANCE_DECAY, out,
                )
