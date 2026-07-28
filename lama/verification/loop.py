"""Closing the loop: select a hypothesis, test it for real, remember what
happened.

Deliberately thin. Selection decides what is worth testing (`select.py`) and
`AffordanceMemory` decides what a result means -- the Bayesian update and any
revision on contradiction; this module's only job is to turn a choice into an
actual environment step and thread the resulting observation forward.

**A real, documented limitation.** `imagine`/`select` only ever propose verbs
on objects already within reach: deciding where to go next is a planning
problem, and `planner/` does not exist yet. Without something filling that
gap, an episode with nothing already close to the agent at reset would end
having tested nothing. `_nearest_unreached` is a minimal, deliberately dumb
stand-in: when nothing reachable is worth testing, move toward whatever is
closest. It has no opinion about which unreached object looks most likely to
teach the bank something -- that is exactly the kind of decision the eventual
planner should make instead of this function.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..env import (
    Action,
    Environment,
    Interaction,
    Observation,
    ObjectView,
    StepResult,
)
from ..env.actions import cost as action_cost
from ..imagination.hypothesis import Hypothesis, imagine
from ..memory.bank import Revision
from ..memory.memory import AffordanceMemory
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


def _nearest_unreached(observation: Observation) -> ObjectView | None:
    """Closest object not currently within reach, or `None` if everything
    visible already is. See the module docstring: a stand-in for planning."""
    candidates = [
        o for o in observation.objects if not o.within_reach and not o.held
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda o: o.distance)


def verify_once(
    env: Environment, memory: AffordanceMemory, observation: Observation
) -> VerificationStep:
    """Imagine, select, and either test a hypothesis or move toward one.

    `observation` must be whatever `env` most recently returned -- from
    `reset` or the previous `step` -- since `Environment` has no way to ask
    for the current observation without stepping.
    """
    hypotheses = imagine(observation, memory)
    chosen = select_next(hypotheses, env.budget_remaining)
    if chosen is not None:
        step = env.step(Interaction(chosen.action, chosen.target_id))
        revision = memory.observe(observation, step.record)
        return VerificationStep(chosen, None, step, revision)

    target = _nearest_unreached(observation)
    if target is not None and env.budget_remaining >= action_cost(Action.APPROACH):
        step = env.step(Interaction(Action.APPROACH, target.object_id))
        return VerificationStep(None, target.object_id, step, None)

    return VerificationStep(None, None, None, None)


def run_episode(
    env: Environment, memory: AffordanceMemory, seed: int | None = None
) -> tuple[VerificationStep, ...]:
    """Run one episode of the loop to completion.

    Ends when the budget runs out, the environment reports the episode done,
    or `verify_once` finds nothing left to do. Every real action costs a
    positive amount of budget (`env/actions.py`), so this always terminates.
    """
    observation = env.reset(seed=seed)
    steps: list[VerificationStep] = []
    while env.budget_remaining > 0:
        result = verify_once(env, memory, observation)
        if result.step is None:
            break
        steps.append(result)
        observation = result.step.observation
        if result.step.done:
            break
    return tuple(steps)
