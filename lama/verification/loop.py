"""Closing the loop: select a hypothesis, test it for real, remember what
happened.

Deliberately thin. Selection decides what is worth testing (`select.py`) and
`AffordanceMemory` decides what a result means -- the Bayesian update and any
revision on contradiction; this module's only job is to turn a choice into an
actual environment step and thread the resulting observation forward.

**Goal-directed pursuit is optional, not required.** Pass `goal` and this
builds a `RegressionPlanner` over the current bank, computes which
`(concept, verb, tool)` keys are relevant to reaching it, and hands that to
`imagine` so `select.py` prioritises them (see `planner/planner.py` and
`imagination/hypothesis.py`). Leave `goal` unset and behaviour is exactly the
uncertainty-driven loop this project had before goals existed -- nothing
about ordinary exploration depends on one being supplied.

**Choosing where to walk is no longer a dummy heuristic.** `imagine`/`select`
only ever propose verbs on objects already within reach; something still has
to decide where to walk when nothing reachable is worth testing.
`exploration.select_exploration_target` picks the not-yet-reachable object
with the best expected-information-gain-per-distance, the same acquisition
principle `select.py` uses for testing, rather than simply the closest thing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..env import Action, Environment, Interaction, Observation, StepResult
from ..env.actions import cost as action_cost
from ..exploration import select_exploration_target
from ..imagination.hypothesis import Hypothesis, imagine
from ..memory.bank import Revision
from ..memory.memory import AffordanceMemory
from ..planner import Goal, RegressionPlanner
from .select import select_next


@dataclass(frozen=True)
class VerificationStep:
    """What happened when the loop tried to make one unit of progress.

    Exactly one of `hypothesis` / `approached` is set when `step` is not
    `None`. Every field `None` means the loop found nothing left to do --
    everything reachable is settled and nothing unreached is affordable --
    and the caller should end the episode.
    """

    hypothesis: Hypothesis | None
    approached: str | None
    step: StepResult | None
    revision: Revision | None


def verify_once(
    env: Environment,
    memory: AffordanceMemory,
    observation: Observation,
    goal: Goal | None = None,
) -> VerificationStep:
    """Imagine, select, and either test a hypothesis or move toward one.

    `observation` must be whatever `env` most recently returned -- from
    `reset` or the previous `step` -- since `Environment` has no way to ask
    for the current observation without stepping.
    """
    relevant_keys = None
    if goal is not None:
        relevant_keys = RegressionPlanner(memory.bank).relevant_keys(goal)

    hypotheses = imagine(observation, memory, relevant_keys)
    chosen = select_next(hypotheses, env.budget_remaining)
    if chosen is not None:
        step = env.step(Interaction(chosen.action, chosen.target_id))
        revision = memory.observe(observation, step.record)
        return VerificationStep(chosen, None, step, revision)

    target = select_exploration_target(observation, memory)
    if target is not None and env.budget_remaining >= action_cost(Action.APPROACH):
        step = env.step(Interaction(Action.APPROACH, target.object_id))
        return VerificationStep(None, target.object_id, step, None)

    return VerificationStep(None, None, None, None)


def run_episode(
    env: Environment,
    memory: AffordanceMemory,
    seed: int | None = None,
    goal: Goal | None = None,
) -> tuple[VerificationStep, ...]:
    """Run one episode of the loop to completion.

    Ends when the budget runs out, the environment reports the episode done,
    or `verify_once` finds nothing left to do. Every real action costs a
    positive amount of budget (`env/actions.py`), so this always terminates.
    """
    observation = env.reset(seed=seed)
    steps: list[VerificationStep] = []
    while env.budget_remaining > 0:
        result = verify_once(env, memory, observation, goal)
        if result.step is None:
            break
        steps.append(result)
        observation = result.step.observation
        if result.step.done:
            break
    return tuple(steps)
