"""Tests for counterfactual hypothesis generation.

The properties that matter: only observable preconditions gate what gets
proposed, looking never mutates memory, and a belief already in the bank
actually reaches the hypothesis that asked about it.
"""

from __future__ import annotations

import numpy as np

from lama.env import (
    APPEARANCE_DIM,
    Action,
    Effect,
    Interaction,
    Observation,
    ObjectView,
    Outcome,
)
from lama.env.actions import spec
from lama.env.warehouse import Warehouse
from lama.imagination import HYPOTHESIS_ACTIONS, imagine
from lama.memory.bank import Status
from lama.memory.memory import AffordanceMemory


def obj_view(object_id, appearance_value=0.5, within_reach=True, held=False):
    return ObjectView(
        object_id=object_id,
        position=np.zeros(2),
        extent=np.array([0.3, 0.3]),
        distance=0.5 if within_reach else 5.0,
        bearing=0.0,
        appearance=np.full(APPEARANCE_DIM, appearance_value, dtype=np.float32),
        within_reach=within_reach,
        held=held,
    )


def obs(*views, holding=None):
    return Observation(
        t=0, agent_position=np.zeros(2), agent_heading=0.0, objects=views,
        holding=holding, budget_remaining=20.0,
    )


# --------------------------------------------------------------------------- #
# what gets proposed
# --------------------------------------------------------------------------- #
def test_hypothesis_actions_exclude_approach_and_release():
    assert Action.APPROACH not in HYPOTHESIS_ACTIONS
    assert Action.RELEASE not in HYPOTHESIS_ACTIONS


def test_out_of_reach_objects_get_no_hypotheses():
    mem = AffordanceMemory()
    hs = imagine(obs(obj_view("x", within_reach=False)), mem)
    assert hs == ()


def test_reachable_object_yields_one_hypothesis_per_applicable_verb():
    """With an empty gripper: every verb except PLACE_ON (needs something
    held) should be proposed once."""
    mem = AffordanceMemory()
    hs = imagine(obs(obj_view("x")), mem)
    actions = {h.action for h in hs}
    assert Action.PLACE_ON not in actions
    assert actions == set(HYPOTHESIS_ACTIONS) - {Action.PLACE_ON}


def test_holding_something_blocks_gripper_dependent_verbs():
    """GRASP, LIFT etc. need a free gripper; PLACE_ON needs the opposite."""
    mem = AffordanceMemory()
    target = obj_view("plate_0")
    tool = obj_view("block_0", held=True)
    hs = imagine(obs(target, tool, holding="block_0"), mem)
    actions = {h.action for h in hs}
    for a in HYPOTHESIS_ACTIONS:
        if spec(a).needs_free_gripper:
            assert a not in actions, f"{a.name} should be blocked while holding"
    assert Action.PLACE_ON in actions


def test_place_on_uses_the_held_object_as_tool():
    mem = AffordanceMemory()
    target = obj_view("plate_0")
    tool = obj_view("block_0", held=True)
    hs = imagine(obs(target, tool, holding="block_0"), mem)
    place = next(h for h in hs if h.action is Action.PLACE_ON)
    assert place.target_id == "plate_0"
    assert place.tool_id == "block_0"
    assert place.is_relational


def test_the_held_object_is_never_proposed_as_a_target():
    mem = AffordanceMemory()
    tool = obj_view("block_0", held=True)
    hs = imagine(obs(tool, holding="block_0"), mem)
    assert all(h.target_id != "block_0" for h in hs)


# --------------------------------------------------------------------------- #
# looking does not mutate memory
# --------------------------------------------------------------------------- #
def test_imagining_never_creates_a_concept():
    mem = AffordanceMemory()
    imagine(obs(obj_view("x")), mem)
    assert len(mem.concepts) == 0
    assert len(mem.bank) == 0
    assert len(mem.log) == 0


def test_unfamiliar_appearance_yields_untested_status():
    mem = AffordanceMemory()
    hs = imagine(obs(obj_view("x", appearance_value=0.9)), mem)
    assert all(h.status is Status.UNTESTED for h in hs)
    assert all(h.predicted_mean is None for h in hs)
    assert all(h.credible_width is None for h in hs)
    assert all(h.target_concept is None for h in hs)


# --------------------------------------------------------------------------- #
# a real belief reaches the hypothesis
# --------------------------------------------------------------------------- #
def test_an_existing_belief_is_attached_to_the_matching_hypothesis():
    mem = AffordanceMemory()
    appearance_value = 0.4
    target = obj_view("crate_0", appearance_value)
    for _ in range(8):
        mem.observe(
            obs(target),
            _push_record(target, Outcome(Effect.TRANSLATED)),
        )
    hs = imagine(obs(target), mem)
    pushed = next(h for h in hs if h.action is Action.PUSH)
    assert pushed.status is Status.CONFIRMED
    assert pushed.predicted_mean is not None and pushed.predicted_mean > 0.8
    assert pushed.credible_width is not None


def _push_record(target_view, outcome, t=0):
    from lama.env import InteractionRecord

    return InteractionRecord(
        episode=0, t=t, interaction=Interaction(Action.PUSH, target_view.object_id),
        outcome=outcome, cost=1.0, view_before=target_view,
    )


# --------------------------------------------------------------------------- #
# end to end against the real environment
# --------------------------------------------------------------------------- #
def test_end_to_end_against_the_real_environment():
    mem = AffordanceMemory()
    w = Warehouse(seed=0, layout_seed=1, budget=60.0)
    observation = w.reset()

    # walk up to the first visible object so something is reachable
    target_id = observation.objects[0].object_id
    observation = w.step(Interaction(Action.APPROACH, target_id)).observation
    hs = imagine(observation, mem)
    assert hs
    assert all(h.status is Status.UNTESTED for h in hs)

    pushed = next(h for h in hs if h.action is Action.PUSH)
    step = w.step(Interaction(pushed.action, pushed.target_id))
    mem.observe(observation, step.record)

    hs2 = imagine(step.observation, mem)
    updated = next(
        h for h in hs2 if h.action is Action.PUSH and h.target_id == target_id
    )
    assert updated.status is not Status.UNTESTED
    assert updated.predicted_mean is not None
