"""Tests for `AffordanceMemory`, the facade tying concepts, bank and log
together.

Most of these build `Observation`/`InteractionRecord` by hand so the id
translation logic can be tested in isolation from the environment. One
integration test runs the real `Warehouse` end to end, because the facade's
actual point -- that appearance-based generalisation blends the crate/block
trap into one blended belief -- is only convincing if it happens against the
real environment, not a hand-built stand-in for it.
"""

from __future__ import annotations

import numpy as np
import pytest

from lama.env import (
    APPEARANCE_DIM,
    Action,
    Effect,
    Interaction,
    InteractionRecord,
    Observation,
    ObjectView,
    Outcome,
)
from lama.env.warehouse import Warehouse, WarehouseOracle
from lama.memory.bank import Status
from lama.memory.memory import AffordanceMemory


def view(object_id: str, appearance_value: float = 0.5, held: bool = False) -> ObjectView:
    return ObjectView(
        object_id=object_id,
        position=np.zeros(2),
        extent=np.array([0.3, 0.3]),
        distance=0.5,
        bearing=0.0,
        appearance=np.full(APPEARANCE_DIM, appearance_value, dtype=np.float32),
        within_reach=True,
        held=held,
    )


def record(
    action: Action,
    target: str | None,
    outcome: Outcome,
    *,
    tool_id: str | None = None,
    view_before: ObjectView | None = None,
    episode: int = 0,
    t: int = 0,
) -> InteractionRecord:
    return InteractionRecord(
        episode=episode, t=t, interaction=Interaction(action, target),
        outcome=outcome, cost=1.0, tool_id=tool_id, view_before=view_before,
    )


def observation(*views: ObjectView) -> Observation:
    return Observation(
        t=0, agent_position=np.zeros(2), agent_heading=0.0, objects=views,
        budget_remaining=10.0,
    )


# --------------------------------------------------------------------------- #
# what gets ignored
# --------------------------------------------------------------------------- #
def test_approach_is_ignored():
    mem = AffordanceMemory()
    target = view("crate_0")
    rev = mem.observe(
        observation(target),
        record(Action.APPROACH, "crate_0", Outcome(Effect.NOTHING), view_before=target),
    )
    assert rev is None
    assert len(mem.log) == 0
    assert len(mem.bank) == 0
    assert len(mem.concepts) == 0


def test_release_is_ignored_because_it_has_no_target():
    mem = AffordanceMemory()
    rev = mem.observe(
        observation(), record(Action.RELEASE, None, Outcome(Effect.RELEASED)),
    )
    assert rev is None
    assert len(mem.log) == 0


def test_target_not_visible_anywhere_is_safely_ignored():
    """view_before absent and the target missing from `before` too -- must
    not raise, and must not fabricate a concept from nothing."""
    mem = AffordanceMemory()
    rev = mem.observe(
        observation(), record(Action.PUSH, "ghost_0", Outcome(Effect.TRANSLATED)),
    )
    assert rev is None
    assert len(mem.concepts) == 0
    assert len(mem.log) == 0


# --------------------------------------------------------------------------- #
# basic wiring
# --------------------------------------------------------------------------- #
def test_a_single_unary_interaction_forms_one_concept_and_one_log_entry():
    mem = AffordanceMemory()
    target = view("crate_0", 0.2)
    rev = mem.observe(
        observation(target),
        record(Action.PUSH, "crate_0", Outcome(Effect.TRANSLATED), view_before=target),
    )
    assert rev is None  # UNTESTED -> PROVISIONAL is not a settled-status crossing
    assert len(mem.concepts) == 1
    assert len(mem.log) == 1
    assert len(mem.bank) == 1
    key = next(iter(mem.bank.beliefs())).key
    assert key == (0, Action.PUSH, None)


def test_view_before_missing_falls_back_to_the_observation():
    """InteractionRecord.view_before is the usual source, but the facade must
    still work from `before` when a backend leaves it unset."""
    mem = AffordanceMemory()
    target = view("crate_0", 0.7)
    mem.observe(
        observation(target),
        record(Action.PUSH, "crate_0", Outcome(Effect.TRANSLATED), view_before=None),
    )
    assert len(mem.concepts) == 1


def test_relational_interaction_resolves_the_tool_from_before():
    """The tool has no view on the record at all -- it can only come from the
    observation immediately before the step, which is the whole reason
    `observe` takes `before` as an argument."""
    mem = AffordanceMemory()
    target = view("plate_0", 0.2)
    tool = view("block_0", 0.9, held=True)
    rev = mem.observe(
        observation(target, tool),
        record(Action.PLACE_ON, "plate_0", Outcome(Effect.SUPPORTED),
               tool_id="block_0", view_before=target),
    )
    assert rev is None
    assert len(mem.concepts) == 2
    key = next(iter(mem.bank.beliefs())).key
    assert key[1] is Action.PLACE_ON
    assert key[2] is not None, "tool concept must be resolved, not left as None"


def test_missing_tool_view_leaves_the_key_without_a_tool_concept():
    """Defensive path: a tool id that is not in `before` for some reason must
    not raise, and must not be silently confused with 'no tool'."""
    mem = AffordanceMemory()
    target = view("plate_0", 0.2)
    mem.observe(
        observation(target),
        record(Action.PLACE_ON, "plate_0", Outcome(Effect.SUPPORTED),
               tool_id="phantom_tool", view_before=target),
    )
    key = next(iter(mem.bank.beliefs())).key
    assert key[2] is None


def test_repeated_observations_of_the_same_appearance_share_a_concept():
    mem = AffordanceMemory()
    for t in range(5):
        target = view("crate_0", 0.3)
        mem.observe(
            observation(target),
            record(Action.PUSH, "crate_0", Outcome(Effect.TRANSLATED),
                   view_before=target, t=t),
        )
    assert len(mem.concepts) == 1
    assert len(mem.log) == 5
    assert mem.bank.belief((0, Action.PUSH, None)).total_attempts == 5


# --------------------------------------------------------------------------- #
# the point of the whole thing: the trap survives into the bank
# --------------------------------------------------------------------------- #
def test_crate_and_block_blend_into_one_belief_end_to_end():
    """Run the real environment: fetch a crate half the time and a block the
    other half, carry it to the plate, and check what memory ends up
    believing. It should be confident that placing something on the plate
    works (both do), and estimate the door-opening rate at roughly 50% --
    correctly reflecting that it cannot tell which half of its own concept it
    is holding."""
    mem = AffordanceMemory()
    for ep in range(24):
        w = Warehouse(seed=ep, layout_seed=1, budget=60.0)
        oracle = WarehouseOracle(w)
        plate = oracle.ids_of_kind("plate")[0]
        tool = oracle.ids_of_kind("crate" if ep % 2 == 0 else "block")[0]
        for action, target in (
            (Action.APPROACH, tool), (Action.GRASP, tool),
            (Action.APPROACH, plate), (Action.PLACE_ON, plate),
        ):
            before = w._observe()
            step = w.step(Interaction(action, target))
            mem.observe(before, step.record)

    # crate and block are visually identical: exactly one concept for both.
    tool_concepts = {
        b.key[0] for b in mem.bank.beliefs() if b.key[1] is Action.GRASP
    }
    assert len(tool_concepts) == 1

    place_beliefs = [b for b in mem.bank.beliefs() if b.key[1] is Action.PLACE_ON]
    assert len(place_beliefs) == 1
    belief = place_beliefs[0]
    assert belief.status is Status.CONFIRMED, "placing on the plate always works"
    assert 0.3 < belief.remote_rate < 0.7, (
        "but only the block half opens the door, and the concept cannot "
        "tell which half it is holding"
    )


def test_repeated_crate_block_encounters_do_not_cascade_split_forever():
    """Regression test for a real bug found by running the system for 150
    episodes (docs/DECISIONS.md): because crate and block are appearance-
    identical, splitting can never actually separate them, so every fresh
    STUCK verdict on a new episode's crate/block instances re-split an
    already-split descendant -- 27 STUCK beliefs and 73 concepts from what
    should have been about a dozen, with the same seed that this test now
    pins down to a bounded number.

    Pushes crate and block, alternating, across many episodes on one
    persistent memory -- enough to trigger STUCK/split more than once if
    nothing were capping it -- and checks the cap actually holds."""
    from lama.env.warehouse import WarehouseOracle
    from lama.memory.memory import MAX_SPLIT_GENERATIONS

    mem = AffordanceMemory()
    for ep in range(40):
        w = Warehouse(seed=ep, layout_seed=1, budget=60.0)
        oracle = WarehouseOracle(w)
        target = oracle.ids_of_kind("crate" if ep % 2 == 0 else "block")[0]
        before = w.reset()
        before = w.step(Interaction(Action.APPROACH, target)).observation
        step = w.step(Interaction(Action.PUSH, target))
        mem.observe(before, step.record)

    retired = sum(
        1 for c in mem.concepts.concepts() if float(c.mean[0]) == float("inf")
    )
    assert retired <= 1, (
        f"expected at most one crate/block split (generation cap = "
        f"{MAX_SPLIT_GENERATIONS}), got {retired} retired concepts -- the "
        f"cascade regression is back"
    )
    max_generation = max(c.generation for c in mem.concepts.concepts())
    assert max_generation <= MAX_SPLIT_GENERATIONS
