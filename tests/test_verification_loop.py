"""Tests for the verification loop that closes observe -> imagine -> select
-> verify -> adjudicate -> remember.
"""

from __future__ import annotations

import numpy as np

from lama.env import APPEARANCE_DIM, Action, Interaction, Observation, ObjectView
from lama.env.warehouse import Warehouse
from lama.imagination.hypothesis import imagine
from lama.memory.memory import AffordanceMemory
from lama.planner import RegressionPlanner
from lama.verification.loop import run_episode, verify_once


def obj_view(object_id, distance=1.0, within_reach=False, held=False, appearance=None):
    return ObjectView(
        object_id=object_id, position=np.zeros(2), extent=np.array([0.3, 0.3]),
        distance=distance, bearing=0.0,
        appearance=(
            appearance if appearance is not None
            else np.zeros(APPEARANCE_DIM, dtype=np.float32)
        ),
        within_reach=within_reach, held=held,
    )


def obs(*views, budget=10.0):
    return Observation(
        t=0, agent_position=np.zeros(2), agent_heading=0.0, objects=views,
        budget_remaining=budget,
    )


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


# --------------------------------------------------------------------------- #
# goal-directed pursuit: does supplying a goal actually change what gets
# tested, not just get accepted as an unused parameter?
# --------------------------------------------------------------------------- #
def test_verify_once_without_a_goal_behaves_exactly_as_before():
    """The default (goal=None) path must be unaffected by the goal machinery
    existing at all."""
    w = Warehouse(seed=0, layout_seed=1, budget=60.0)
    mem = AffordanceMemory()
    observation = w.reset()
    target_id = observation.objects[0].object_id
    observation = w.step(Interaction(Action.APPROACH, target_id)).observation

    result = verify_once(w, mem, observation)
    assert result.hypothesis is not None
    assert result.hypothesis.relevance == 0.0


def test_a_goal_makes_relevant_hypotheses_score_above_irrelevant_ones():
    """Seed the bank with the flagship two-step chain (grasp a heavy object,
    place it on the plate to open the door) as CONFIRMED knowledge, put both
    the tool and an unrelated never-seen object in reach, set a goal for the
    door, and check the tool -- not the unrelated object -- is what
    imagination marks relevant."""
    from lama.env import Effect, Outcome, RemoteEffect
    from lama.planner import Goal

    mem = AffordanceMemory()
    tool_appearance = np.full(APPEARANCE_DIM, 0.2, dtype=np.float32)
    plate_appearance = np.full(APPEARANCE_DIM, 0.7, dtype=np.float32)
    door_appearance = np.full(APPEARANCE_DIM, 0.95, dtype=np.float32)

    tool_concept = mem.concepts.assign(tool_appearance, 0, 0)
    plate_concept = mem.concepts.assign(plate_appearance, 0, 0)
    door_concept = mem.concepts.assign(door_appearance, 0, 0)

    for i in range(8):
        mem.bank.observe(tool_concept, Action.GRASP, None, Outcome(Effect.CARRIED),
                         episode=0, t=i, object_id="tool_0")
    for i in range(8):
        outcome = Outcome(Effect.SUPPORTED, remote=(RemoteEffect("door_0", Effect.OPENED),))
        mem.bank.observe(plate_concept, Action.PLACE_ON, tool_concept, outcome,
                         episode=0, t=i, object_id="plate_0", tool_id="tool_0",
                         remote_concepts=(door_concept,))

    tool_view = obj_view("tool_0", within_reach=True, appearance=tool_appearance)
    unrelated_view = obj_view(
        "unrelated_0", within_reach=True, distance=1.5,
        appearance=np.full(APPEARANCE_DIM, 0.5, dtype=np.float32),
    )
    observation = obs(tool_view, unrelated_view)

    goal = Goal(concept=door_concept, effect=Effect.OPENED)
    hypotheses = imagine(observation, mem, RegressionPlanner(mem.bank).relevant_keys(goal))
    tool_grasp = next(
        h for h in hypotheses if h.target_id == "tool_0" and h.action is Action.GRASP
    )
    unrelated_grasp = next(
        h for h in hypotheses if h.target_id == "unrelated_0" and h.action is Action.GRASP
    )
    assert tool_grasp.relevance > 0.0
    assert unrelated_grasp.relevance == 0.0
