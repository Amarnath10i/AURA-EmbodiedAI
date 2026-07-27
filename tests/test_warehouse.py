"""Tests for the numpy warehouse backend.

Beyond checking that the backend works, these pin the properties the research
depends on: that the flagship secondary affordance is reachable, that the
look-alike trap actually bites, and that nothing leaks ground truth to the
agent.
"""

from __future__ import annotations

import numpy as np
import pytest

from lama.env import (
    ALL_ACTIONS,
    INTERACTION_ACTIONS,
    Action,
    Effect,
    Environment,
    Interaction,
    Role,
)
from lama.env.appearance import prototype
from lama.env.warehouse import REACH, Warehouse, WarehouseOracle


def ids(world: Warehouse, kind: str) -> list[str]:
    """Ids of a kind. Ground truth, so it goes through the oracle -- the ids
    themselves are opaque precisely so the agent cannot do this."""
    return list(WarehouseOracle(world).ids_of_kind(kind))


def first(world: Warehouse, kind: str) -> str:
    found = ids(world, kind)
    assert found, f"no {kind} in this layout"
    return found[0]


def carry_to(world: Warehouse, tool: str, target: str) -> Effect:
    """Fetch `tool`, bring it to `target`, put it down. Returns the effect."""
    world.step(Interaction(Action.APPROACH, tool))
    world.step(Interaction(Action.GRASP, tool))
    world.step(Interaction(Action.APPROACH, target))
    return world.step(Interaction(Action.PLACE_ON, target)).record.outcome.effect


# --------------------------------------------------------------------------- #
# contract compliance
# --------------------------------------------------------------------------- #
def test_warehouse_implements_the_contract():
    assert isinstance(Warehouse(seed=0), Environment)


def test_backend_is_named_for_traceability():
    assert Warehouse(seed=0).backend_name == "numpy-warehouse"


def test_same_seed_and_actions_reproduce_the_trajectory():
    def run(seed: int) -> list[tuple]:
        world = Warehouse(seed=seed, layout_seed=1)
        rng = np.random.default_rng(0)
        out = []
        for _ in range(60):
            a = Action(rng.integers(1, len(ALL_ACTIONS)))
            t = None if a is Action.RELEASE else str(rng.choice(list(world._objects)))
            r = world.step(Interaction(a, t))
            out.append(
                (int(r.record.outcome.effect),
                 float(r.observation.objects[0].appearance.sum()))
            )
        return out

    assert run(3) == run(3)
    assert run(3) != run(4), "different seeds must diverge"


def test_every_verb_is_accepted_for_every_target():
    """Doomed attempts return an outcome; refusing them would hand the agent
    knowledge it should have had to pay for."""
    world = Warehouse(seed=0, layout_seed=2)
    for target in list(world._objects):
        for action in INTERACTION_ACTIONS:
            i = Interaction(action, None if action is Action.RELEASE else target)
            world.step(i)  # must not raise


def test_unknown_target_raises():
    """Malformed input is a bug, not a doomed attempt."""
    with pytest.raises(KeyError):
        Warehouse(seed=0).step(Interaction(Action.PUSH, "no_such_object"))


def test_every_attempt_costs_budget_including_failures():
    world = Warehouse(seed=0, layout_seed=2, budget=20.0)
    pillar = ids(world, "pillar") or ids(world, "door")
    before = world.budget_remaining
    r = world.step(Interaction(Action.LIFT, pillar[0]))
    assert r.record.outcome.effect in (Effect.NOTHING, Effect.BLOCKED)
    assert world.budget_remaining < before
    assert r.record.cost > 0


def test_episode_ends_when_the_budget_runs_out():
    world = Warehouse(seed=0, layout_seed=2, budget=3.0)
    target = first(world, "door")
    done = False
    for _ in range(20):
        done = world.step(Interaction(Action.PUSH, target)).done
        if done:
            break
    assert done
    assert world.budget_remaining == 0.0


# --------------------------------------------------------------------------- #
# no semantic leakage
# --------------------------------------------------------------------------- #
def test_observations_never_reveal_object_kinds():
    world = Warehouse(seed=0, layout_seed=2)
    obs = world.reset()
    kinds = {o.kind for o in world._objects.values()}
    blob = repr(obs) + repr(world.step(Interaction(Action.PUSH, first(world, "door"))))
    leaked = {k for k in kinds if k in blob}
    assert not leaked, f"observation leaked kinds: {leaked}"


def test_object_ids_are_stable_within_an_episode():
    world = Warehouse(seed=0, layout_seed=2)
    before = {o.object_id for o in world.reset().objects}
    world.step(Interaction(Action.PUSH, first(world, "crate")))
    assert {o.object_id for o in world._observe().objects} == before


# --------------------------------------------------------------------------- #
# mechanisms: the flagship secondary affordance
# --------------------------------------------------------------------------- #
def test_doors_start_locked_so_a_mechanism_is_required():
    world = Warehouse(seed=0, layout_seed=1)
    door = first(world, "door")
    world.step(Interaction(Action.APPROACH, door))
    r = world.step(Interaction(Action.OPEN, door))
    assert r.record.outcome.effect is Effect.BLOCKED


def test_standing_on_the_plate_opens_the_door_only_while_standing():
    world = Warehouse(seed=0, layout_seed=1)
    door, plate = first(world, "door"), first(world, "plate")
    world.step(Interaction(Action.APPROACH, plate))
    r = world.step(Interaction(Action.PRESS, plate))
    assert r.record.outcome.effect is Effect.ACTUATED
    assert world._objects[door].is_open
    world.step(Interaction(Action.APPROACH, door))
    assert not world._objects[door].is_open, "the plate must not latch"


def test_a_heavy_object_holds_the_plate_down_permanently():
    """The affordance the whole project is built around."""
    world = Warehouse(seed=0, layout_seed=1)
    door, plate = first(world, "door"), first(world, "plate")
    assert carry_to(world, first(world, "block"), plate) is Effect.SUPPORTED
    world.step(Interaction(Action.APPROACH, door))
    assert world._objects[door].is_open


def test_the_look_alike_crate_is_not_heavy_enough():
    """Same appearance, different outcome: this is why verification exists."""
    world = Warehouse(seed=0, layout_seed=1)
    door, plate = first(world, "door"), first(world, "plate")
    assert carry_to(world, first(world, "crate"), plate) is Effect.SUPPORTED
    world.step(Interaction(Action.APPROACH, door))
    assert not world._objects[door].is_open


def test_the_trap_pair_is_visually_identical():
    assert np.array_equal(prototype("crate"), prototype("block"))


def test_latching_mechanisms_stay_open():
    for kind in ("button", "lever", "valve"):
        for layout in range(12):
            world = Warehouse(seed=0, layout_seed=layout)
            if not ids(world, kind):
                continue
            door, mech = first(world, "door"), first(world, kind)
            verb = {"button": Action.PRESS, "lever": Action.PULL,
                    "valve": Action.ROTATE}[kind]
            world.step(Interaction(Action.APPROACH, mech))
            r = world.step(Interaction(verb, mech))
            assert r.record.outcome.effect is Effect.ACTUATED
            assert r.record.outcome.had_remote_effect
            world.step(Interaction(Action.APPROACH, door))
            assert world._objects[door].is_open, f"{kind} must latch"
            break


# --------------------------------------------------------------------------- #
# preconditions and reach
# --------------------------------------------------------------------------- #
def test_out_of_reach_attempts_are_blocked_not_absent():
    world = Warehouse(seed=0, layout_seed=1)
    crate = first(world, "crate")
    world._agent = world._objects[crate].position + np.array([REACH + 5.0, 0.0])
    r = world.step(Interaction(Action.PUSH, crate))
    assert r.record.outcome.effect is Effect.BLOCKED


def test_grasping_is_blocked_while_already_holding_something():
    world = Warehouse(seed=0, layout_seed=1)
    crate, block = first(world, "crate"), first(world, "block")
    world.step(Interaction(Action.APPROACH, crate))
    world.step(Interaction(Action.GRASP, crate))
    world.step(Interaction(Action.APPROACH, block))
    assert world.step(Interaction(Action.GRASP, block)).record.outcome.effect \
        is Effect.BLOCKED


def test_placing_with_an_empty_gripper_is_blocked():
    world = Warehouse(seed=0, layout_seed=1)
    plate = first(world, "plate")
    world.step(Interaction(Action.APPROACH, plate))
    assert world.step(Interaction(Action.PLACE_ON, plate)).record.outcome.effect \
        is Effect.BLOCKED


def test_release_puts_the_held_object_down():
    world = Warehouse(seed=0, layout_seed=1)
    crate = first(world, "crate")
    world.step(Interaction(Action.APPROACH, crate))
    world.step(Interaction(Action.GRASP, crate))
    assert world._observe().holding == crate
    assert world.step(Interaction(Action.RELEASE)).record.outcome.effect \
        is Effect.RELEASED
    assert world._observe().holding is None


# --------------------------------------------------------------------------- #
# hidden mass leaks through displacement
# --------------------------------------------------------------------------- #
def test_heavier_objects_move_less_when_shoved():
    """Mass is never observable, but its consequence is -- and that is the only
    channel through which the trap pair can ever be told apart."""
    world = Warehouse(seed=0, layout_seed=1)
    moved = {}
    for kind in ("crate", "block"):
        target = first(world, kind)
        world.step(Interaction(Action.APPROACH, target))
        moved[kind] = world.step(Interaction(Action.PUSH, target)) \
            .record.outcome.displacement
    assert moved["crate"] > moved["block"] > 0


# --------------------------------------------------------------------------- #
# stochastic affordances
# --------------------------------------------------------------------------- #
def test_an_unreliable_affordance_sometimes_fails():
    """Shoving a button works about half the time. An agent that cannot tell
    this from missing knowledge will retest it forever."""
    hits = 0
    trials = 0
    for seed in range(120):
        world = Warehouse(seed=seed, layout_seed=1)
        if not ids(world, "button"):
            continue
        button = first(world, "button")
        world.step(Interaction(Action.APPROACH, button))
        hits += world.step(Interaction(Action.PUSH, button)) \
            .record.outcome.effect is Effect.ACTUATED
        trials += 1
    assert trials > 50
    assert 0.30 < hits / trials < 0.70


# --------------------------------------------------------------------------- #
# layouts
# --------------------------------------------------------------------------- #
def test_required_kinds_are_always_present():
    for seed in range(8):
        present = WarehouseOracle(Warehouse(seed=seed)).present_kinds()
        assert {"door", "plate", "block", "crate"} <= present


def test_layouts_vary_across_episodes_when_unpinned():
    world = Warehouse(seed=0)
    seen = set()
    for _ in range(6):
        seen.add(tuple(sorted(WarehouseOracle(world).present_kinds())))
        world.reset()
    assert len(seen) > 1


def test_layouts_are_fixed_when_pinned():
    world = Warehouse(seed=0, layout_seed=42)
    seen = set()
    for _ in range(4):
        seen.add(tuple(sorted(WarehouseOracle(world).present_kinds())))
        world.reset()
    assert len(seen) == 1


def test_held_out_kinds_are_excluded_from_training_layouts():
    for seed in range(10):
        present = WarehouseOracle(Warehouse(seed=seed)).present_kinds()
        assert not present & {"switch", "drum", "bench"}


def test_held_out_kinds_can_be_enabled_for_transfer():
    seen: set[str] = set()
    for seed in range(30):
        seen |= WarehouseOracle(
            Warehouse(seed=seed, include_held_out=True)
        ).present_kinds()
    assert seen & {"switch", "drum", "bench"}


def test_layouts_need_room_for_the_required_kinds():
    with pytest.raises(ValueError):
        Warehouse(seed=0, n_objects=2)


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #
def test_reachable_affordances_exclude_absent_objects_and_tools():
    """The recall denominator must not punish an agent for its layout."""
    oracle = WarehouseOracle(Warehouse(seed=0, layout_seed=1))
    present = oracle.present_kinds()
    reachable = oracle.reachable_affordances()
    assert reachable
    assert len(reachable) < len(oracle.catalogue_affordances())
    for a in reachable:
        assert a.kind in present
        assert a.tool_kind is None or a.tool_kind in present


def test_every_layout_offers_something_secondary_to_discover():
    for seed in range(8):
        oracle = WarehouseOracle(Warehouse(seed=seed))
        assert oracle.reachable_by_role(Role.SECONDARY)


def test_oracle_reports_the_backend_that_produced_a_result():
    assert WarehouseOracle(Warehouse(seed=0)).summary()["backend"] \
        == "numpy-warehouse"
