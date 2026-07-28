"""Turning "what can I do right now" into ranked, evidence-backed guesses.

Imagining an interaction here means looking up what the affordance bank
already believes about the `(concept, verb, tool concept)` it would produce,
via `ConceptCodebook.peek` -- which is read-only, so idle looking never
mutates memory. An object that has never been merged into any concept yields
no belief at all, which is the correct representation of total ignorance:
there is nothing to look up.

This is deliberately not a neural world model. The affordance bank already
*is* a predictive model here: nearest-concept lookup of an accumulated Beta
posterior, which is enough to produce real, testable hypotheses today. A
learned model belongs in `world_model/` once one exists -- for generalising
past what the codebook has already clustered -- and nothing here should need
to change when it arrives, since both answer exactly the same question: what
do I predict, and how sure am I.

**What is and is not consulted.** Only *observable* preconditions gate which
hypotheses are proposed: whether a verb needs the agent's gripper free or
occupied (`ActionSpec.needs_free_gripper` / `needs_held_object`), which is
public knowledge about the agent's own action repertoire, not a fact about any
particular object. Hidden, object-specific preconditions
(`env.affordance.Precondition`, e.g. "the door must already be unlocked") are
never read here, or anywhere in this module -- they are oracle-only, and the
whole point of imagining and then verifying is to find them out the hard way.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..env import Action, Effect, Observation
from ..env.actions import INTERACTION_ACTIONS, spec
from ..memory.bank import Status
from ..memory.memory import AffordanceMemory

#: Verbs imagination ever proposes. `APPROACH` cannot reveal an affordance
#: (`Interaction.is_affordance_test`); `RELEASE` has no target and the bank
#: never keys on it (see `memory.py`), so neither produces a hypothesis.
HYPOTHESIS_ACTIONS: tuple[Action, ...] = tuple(
    a for a in INTERACTION_ACTIONS if a is not Action.RELEASE
)


@dataclass(frozen=True)
class Hypothesis:
    """One candidate interaction, with whatever the bank currently believes.

    Attributes:
        target_id, tool_id: The objects an `Interaction` built from this
            hypothesis would name. `tool_id` is `None` for a unary verb.
        target_concept, tool_concept: Concept ids `peek` resolved, or `None`
            if the object has never been seen before -- nothing to look up.
        status: The bank's status for this key, or `UNTESTED` if there is no
            belief at all -- which covers both "never tried" and "the target
            has never even been matched to a concept".
        predicted_mean: The belief's success-rate estimate, or `None` when
            there is no belief to estimate from.
        credible_width: Width of the belief's 95% credible interval -- the
            bank's own measure of how much a test here would still teach it.
            `None` when there is no belief; treat that as "maximally worth
            testing", never as "zero information", which is the opposite.
        dominant_effect: What the bank thinks would happen, if it has an
            opinion.
        cost: Budget units attempting this would cost.
    """

    target_id: str
    action: Action
    tool_id: str | None
    target_concept: int | None
    tool_concept: int | None
    status: Status
    predicted_mean: float | None
    credible_width: float | None
    dominant_effect: Effect | None
    cost: float

    @property
    def is_relational(self) -> bool:
        return self.tool_id is not None


def imagine(
    observation: Observation, memory: AffordanceMemory
) -> tuple[Hypothesis, ...]:
    """Every affordance hypothesis attemptable right now, each annotated with
    the bank's current belief about it.

    Only reachable objects are considered, and only for verbs whose
    observable preconditions currently hold -- an agent that is not holding
    anything cannot attempt `PLACE_ON`, and knows that without needing to test
    it. The object currently held is never proposed as a target: interacting
    with what is already in your own gripper as though it were a separate
    freestanding object is not a meaningful hypothesis.
    """
    holding = observation.holding
    tool_view = observation.view(holding) if holding is not None else None
    tool_concept = (
        memory.concepts.peek(tool_view.appearance) if tool_view is not None else None
    )

    hypotheses: list[Hypothesis] = []
    for target in observation.reachable():
        if target.held:
            continue
        target_concept = memory.concepts.peek(target.appearance)
        for action in HYPOTHESIS_ACTIONS:
            s = spec(action)
            if s.needs_free_gripper and holding is not None:
                continue
            if s.needs_held_object and holding is None:
                continue

            this_tool_id = holding if s.relational else None
            this_tool_concept = tool_concept if s.relational else None
            belief = (
                memory.bank.belief((target_concept, action, this_tool_concept))
                if target_concept is not None
                else None
            )
            lo_hi = belief.credible_interval if belief is not None else None
            hypotheses.append(
                Hypothesis(
                    target_id=target.object_id,
                    action=action,
                    tool_id=this_tool_id,
                    target_concept=target_concept,
                    tool_concept=this_tool_concept,
                    status=belief.status if belief is not None else Status.UNTESTED,
                    predicted_mean=belief.mean if belief is not None else None,
                    credible_width=(lo_hi[1] - lo_hi[0]) if lo_hi is not None else None,
                    dominant_effect=(
                        belief.dominant_effect if belief is not None else None
                    ),
                    cost=s.cost,
                )
            )
    return tuple(hypotheses)
