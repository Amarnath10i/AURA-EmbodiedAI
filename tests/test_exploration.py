"""Tests for choosing where to walk when nothing reachable is worth testing."""

from __future__ import annotations

import numpy as np

from lama.env import APPEARANCE_DIM, Action, Effect, Interaction, Observation, ObjectView, Outcome
from lama.exploration import select_exploration_target
from lama.memory.memory import AffordanceMemory


def obj_view(object_id, distance=1.0, within_reach=False, held=False, appearance_value=0.5):
    return ObjectView(
        object_id=object_id, position=np.zeros(2), extent=np.array([0.3, 0.3]),
        distance=distance, bearing=0.0,
        appearance=np.full(APPEARANCE_DIM, appearance_value, dtype=np.float32),
        within_reach=within_reach, held=held,
    )


def obs(*views):
    return Observation(
        t=0, agent_position=np.zeros(2), agent_heading=0.0, objects=views,
        budget_remaining=10.0,
    )


def test_ignores_reachable_and_held_objects():
    reachable = obj_view("r", within_reach=True)
    held = obj_view("h", distance=0.0, held=True)
    assert select_exploration_target(obs(reachable, held), AffordanceMemory()) is None


def test_none_with_no_objects():
    assert select_exploration_target(obs(), AffordanceMemory()) is None


def test_prefers_closer_object_when_uncertainty_is_equal():
    """Two never-seen objects (equal, maximal info_gain) -- the closer one
    should win, same as the old nearest-unreached fallback did."""
    near = obj_view("near", distance=2.0, appearance_value=0.1)
    far = obj_view("far", distance=5.0, appearance_value=0.9)
    target = select_exploration_target(obs(far, near), AffordanceMemory())
    assert target.object_id == "near"


def test_prefers_a_more_uncertain_object_even_if_farther():
    """A closer but already-well-understood object should lose to a farther,
    still-uncertain one -- the whole point of using info_gain instead of
    raw distance."""
    mem = AffordanceMemory()
    known_appearance = np.full(APPEARANCE_DIM, 0.3, dtype=np.float32)
    known_view = obj_view("known", distance=1.5, within_reach=True,
                          appearance_value=0.3)
    # settle this concept: many consistent successes -> CONFIRMED -> 0 uncertainty
    for i in range(10):
        before = obs(known_view)
        record_view = known_view
        target_concept = mem.concepts.assign(record_view.appearance, 0, i)
        mem.bank.observe(target_concept, Action.PUSH, None,
                         Outcome(Effect.TRANSLATED, displacement=1.0, force_required=1.0),
                         episode=0, t=i, object_id="known")

    known_far = obj_view("known", distance=2.0, appearance_value=0.3)
    unknown_far = obj_view("unknown", distance=2.2, appearance_value=0.9)
    target = select_exploration_target(obs(known_far, unknown_far), mem)
    assert target.object_id == "unknown"


def test_distance_weight_shifts_the_tradeoff():
    mem = AffordanceMemory()
    close_known = obj_view("close", distance=1.0, appearance_value=0.9)
    far_unknown = obj_view("far", distance=20.0, appearance_value=0.1)
    # with a huge distance weight, the far object's uncertainty edge cannot
    # outweigh the walk -- the close one should win regardless of info_gain
    target = select_exploration_target(
        obs(close_known, far_unknown), mem, distance_weight=1000.0
    )
    assert target.object_id == "close"


def test_expected_info_gain_is_reported():
    mem = AffordanceMemory()
    target = select_exploration_target(obs(obj_view("x")), mem)
    assert target.expected_info_gain == 1.0, "never-seen object is maximal uncertainty"
