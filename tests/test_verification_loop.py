"""Tests for the verification loop that closes observe -> imagine -> select
-> verify -> adjudicate -> remember.
"""

from __future__ import annotations

import numpy as np

from lama.env import APPEARANCE_DIM, Action, Interaction, Observation, ObjectView
from lama.env.warehouse import Warehouse
from lama.memory.memory import AffordanceMemory
from lama.verification.loop import _nearest_unreached, run_episode, verify_once


def obj_view(object_id, distance=1.0, within_reach=False, held=False):
    return ObjectView(
        object_id=object_id, position=np.zeros(2), extent=np.array([0.3, 0.3]),
        distance=distance, bearing=0.0,
        appearance=np.zeros(APPEARANCE_DIM, dtype=np.float32),
        within_reach=within_reach, held=held,
    )


def obs(*views, budget=10.0):
    return Observation(
        t=0, agent_position=np.zeros(2), agent_heading=0.0, objects=views,
        budget_remaining=budget,
    )


# --------------------------------------------------------------------------- #
# _nearest_unreached
# --------------------------------------------------------------------------- #
def test_nearest_unreached_picks_the_closest_out_of_reach_object():
    near = obj_view("near", distance=2.0)
    far = obj_view("far", distance=5.0)
    assert _nearest_unreached(obs(far, near)) is near


def test_nearest_unreached_ignores_reachable_and_held_objects():
    reachable = obj_view("r", within_reach=True)
    held = obj_view("h", distance=0.0, held=True)
    assert _nearest_unreached(obs(reachable, held)) is None


def test_nearest_unreached_is_none_with_no_objects():
    assert _nearest_unreached(obs()) is None


# --------------------------------------------------------------------------- #
# verify_once against the real environment
# --------------------------------------------------------------------------- #
def test_verify_once_tests_a_hypothesis_when_something_is_reachable():
    w = Warehouse(seed=0, layout_seed=1, budget=60.0)
    mem = AffordanceMemory()
    observation = w.reset()
    target_id = observation.objects[0].object_id
    observation = w.step(Interaction(Action.APPROACH, target_id)).observation

    result = verify_once(w, mem, observation)
    assert result.hypothesis is not None
    assert result.approached is None
    assert result.step is not None
    assert len(mem.log) == 1


def test_verify_once_approaches_when_nothing_reachable_is_worth_testing():
    """At reset, objects are scattered across a 12x9 floor with REACH=1.0, so
    almost nothing starts in reach -- the loop must move instead of stalling."""
    w = Warehouse(seed=0, layout_seed=1, budget=60.0)
    mem = AffordanceMemory()
    observation = w.reset()
    assert not observation.reachable(), "test assumes nothing starts in reach"

    result = verify_once(w, mem, observation)
    assert result.hypothesis is None
    assert result.approached is not None
    assert result.step is not None
    assert len(mem.log) == 0, "approaching must not be logged as an affordance test"


def test_verify_once_finds_nothing_to_do_with_no_objects_at_all():
    class _StubEnv:
        budget_remaining = 10.0

    result = verify_once(_StubEnv(), AffordanceMemory(), obs())
    assert result.hypothesis is None
    assert result.approached is None
    assert result.step is None
    assert result.revision is None


# --------------------------------------------------------------------------- #
# run_episode
# --------------------------------------------------------------------------- #
def test_run_episode_spends_budget_and_terminates():
    w = Warehouse(seed=0, layout_seed=1, budget=30.0)
    mem = AffordanceMemory()
    steps = run_episode(w, mem)
    assert steps
    assert w.budget_remaining < 30.0
    assert w.budget_remaining >= 0.0


def test_run_episode_records_only_real_interactions_to_memory():
    w = Warehouse(seed=0, layout_seed=1, budget=30.0)
    mem = AffordanceMemory()
    steps = run_episode(w, mem)
    n_tested = sum(1 for s in steps if s.hypothesis is not None)
    assert len(mem.log) == n_tested


def test_run_episode_populates_the_bank():
    w = Warehouse(seed=0, layout_seed=1, budget=60.0)
    mem = AffordanceMemory()
    run_episode(w, mem)
    assert len(mem.bank) > 0


def test_a_cheap_persistently_failing_verb_can_dominate_a_whole_episode():
    """Documented, observed limitation: because score is per-budget-unit and
    refuting a belief takes ~27 clean failures (see bank.py), the greedy
    per-key acquisition function can spend nearly an entire episode grinding
    down one cheap, unproductive verb on a single object before the fallback
    ever reaches a second one. This is not asserted as *desirable* -- it is
    pinned down so a future change to selection has a concrete regression to
    compare against."""
    w = Warehouse(seed=0, layout_seed=1, budget=60.0)
    mem = AffordanceMemory()
    steps = run_episode(w, mem)
    tested_targets = {s.hypothesis.target_id for s in steps if s.hypothesis is not None}
    assert len(tested_targets) <= 2, (
        "if this now explores more broadly, update the note in loop.py -- the "
        "acquisition function's cross-object behaviour has changed"
    )
