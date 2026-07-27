"""Tests for the environment contract.

These pin down the invariants every backend must honour, so that adding the
Isaac Lab backend later cannot quietly change the meaning of a stored record or
a reported metric.
"""

from __future__ import annotations

import numpy as np
import pytest

import lama.env as env
from lama.env import (
    APPEARANCE_DIM,
    Action,
    Affordance,
    Effect,
    Interaction,
    Observation,
    ObjectView,
    Outcome,
    Precondition,
    RemoteEffect,
    Role,
)


# --------------------------------------------------------------------------- #
# verbs
# --------------------------------------------------------------------------- #
def test_every_verb_has_a_spec():
    assert set(env.SPECS) == set(Action)


def test_verb_values_are_contiguous_from_zero():
    """Verb values index model outputs, so gaps would silently waste logits."""
    assert [int(a) for a in Action] == list(range(len(Action)))


def test_approach_is_not_an_affordance_test():
    """Repositioning must never be counted as a verification."""
    assert Action.APPROACH not in env.INTERACTION_ACTIONS
    assert not Interaction(Action.APPROACH, "crate_0").is_affordance_test
    assert Interaction(Action.PUSH, "crate_0").is_affordance_test


def test_every_verb_costs_something():
    """A free verb would let an agent verify without spending its budget."""
    assert all(env.cost(a) > 0 for a in Action)


def test_place_on_is_the_relational_verb():
    relational = [a for a in Action if env.is_relational(a)]
    assert relational == [Action.PLACE_ON]


def test_relational_verbs_require_a_held_object():
    for a in Action:
        if env.is_relational(a):
            assert env.spec(a).needs_held_object


def test_no_verb_both_requires_and_forbids_a_held_object():
    for a in Action:
        s = env.spec(a)
        assert not (s.needs_free_gripper and s.needs_held_object)


# --------------------------------------------------------------------------- #
# outcomes
# --------------------------------------------------------------------------- #
def test_null_effects_do_not_change_the_world():
    for effect in env.NULL_EFFECTS:
        assert not Outcome(effect).changed_world


def test_nothing_and_blocked_stay_distinct():
    """Collapsing them would teach the agent a locked door cannot be opened."""
    assert env.nothing().effect is not env.blocked().effect


def test_remote_effect_counts_as_changing_the_world():
    """A verb whose only result is elsewhere is still a real affordance."""
    out = Outcome(Effect.NOTHING, remote=(RemoteEffect("door_0", Effect.OPENED),))
    assert out.changed_world
    assert out.had_remote_effect


def test_outcome_without_remote_effect_is_not_a_means_to_an_end():
    assert not Outcome(Effect.TRANSLATED, displacement=0.5).had_remote_effect


# --------------------------------------------------------------------------- #
# affordances
# --------------------------------------------------------------------------- #
def test_roles_order_by_informativeness():
    assert Role.PRIMARY > Role.SECONDARY > Role.INCIDENTAL > Role.NONE


def test_relational_affordance_must_name_its_tool():
    with pytest.raises(ValueError):
        Affordance("plate", Action.PLACE_ON, Effect.SUPPORTED, Role.SECONDARY)


def test_unary_affordance_must_not_name_a_tool():
    with pytest.raises(ValueError):
        Affordance(
            "button", Action.PRESS, Effect.ACTUATED, Role.PRIMARY,
            tool_kind="crate",
        )


def test_reliability_is_a_probability():
    with pytest.raises(ValueError):
        Affordance("cart", Action.PUSH, Effect.TRANSLATED, Role.PRIMARY,
                   reliability=1.5)


def test_role_none_implies_no_effect():
    with pytest.raises(ValueError):
        Affordance("wall", Action.LIFT, Effect.LIFTED, Role.NONE)


def test_key_ignores_effect_and_role():
    """Testing the right verb but predicting the wrong effect is a different
    error from never testing the verb, and must stay distinguishable."""
    a = Affordance("cart", Action.PUSH, Effect.TRANSLATED, Role.PRIMARY)
    b = Affordance("cart", Action.PUSH, Effect.TOPPLED, Role.INCIDENTAL)
    assert a.key == b.key


def test_key_separates_relational_affordances_by_tool():
    """The same verb on the same target is a different capability depending on
    what is held: a crate holds the plate down, a cup merely sits on it."""
    heavy = Affordance("plate", Action.PLACE_ON, Effect.SUPPORTED,
                       Role.SECONDARY, tool_kind="crate",
                       remote_effect=Effect.OPENED)
    light = Affordance("plate", Action.PLACE_ON, Effect.SUPPORTED,
                       Role.INCIDENTAL, tool_kind="cup")
    assert heavy.key != light.key
    assert heavy.uses_object_as_means
    assert not light.uses_object_as_means


def test_unreliable_affordance_is_real_but_not_deterministic():
    a = Affordance("trolley", Action.PUSH, Effect.TRANSLATED, Role.PRIMARY,
                   reliability=0.6)
    assert a.is_real
    assert not a.is_deterministic


def test_zero_reliability_affordance_is_not_real():
    a = Affordance("wall", Action.PUSH, Effect.NOTHING, Role.NONE,
                   reliability=0.0)
    assert not a.is_real


def test_remote_effect_marks_use_as_a_means():
    a = Affordance("plate", Action.PLACE_ON, Effect.SUPPORTED, Role.SECONDARY,
                   tool_kind="crate", preconditions=(Precondition.HOLDING_HEAVY,),
                   remote_effect=Effect.OPENED)
    assert a.uses_object_as_means
    assert a.is_conditional
    assert "crate -> plate" in a.describe()


# --------------------------------------------------------------------------- #
# agent-visible data
# --------------------------------------------------------------------------- #
def _view(object_id: str = "crate_0", **kw) -> ObjectView:
    return ObjectView(
        object_id=object_id,
        position=np.zeros(2),
        extent=np.array([0.4, 0.4]),
        distance=kw.pop("distance", 1.0),
        bearing=kw.pop("bearing", 0.0),
        appearance=np.zeros(APPEARANCE_DIM, dtype=np.float32),
        **kw,
    )


def test_object_view_carries_no_semantic_label():
    """The agent must infer what a thing is; it is never told."""
    fields = set(ObjectView.__dataclass_fields__)
    assert not fields & {"kind", "label", "name", "category", "affordances"}


def test_appearance_dimension_is_enforced():
    with pytest.raises(ValueError):
        ObjectView("x", np.zeros(2), np.zeros(2), 0.0, 0.0,
                   np.zeros(APPEARANCE_DIM + 1, dtype=np.float32))


def test_observation_lookup_and_reachability():
    near, far = _view("crate_0", within_reach=True), _view("shelf_0")
    obs = Observation(t=0, agent_position=np.zeros(2), agent_heading=0.0,
                      objects=(near, far), budget_remaining=10.0)
    assert obs.view("crate_0") is near
    assert obs.view("absent") is None
    assert obs.reachable() == (near,)


def test_interaction_requires_a_target_except_release():
    with pytest.raises(ValueError):
        Interaction(Action.PUSH)
    with pytest.raises(ValueError):
        Interaction(Action.RELEASE, "crate_0")
    assert Interaction(Action.RELEASE).target is None


def test_interaction_reports_the_cost_of_its_verb():
    assert Interaction(Action.LIFT, "crate_0").cost == env.cost(Action.LIFT)


# --------------------------------------------------------------------------- #
# the ground-truth boundary
# --------------------------------------------------------------------------- #
def test_oracle_is_not_reachable_from_the_package_surface():
    """Reaching hidden ground truth must require an explicit, reviewable
    import of lama.env.interface, never a convenient package-level one."""
    assert not hasattr(env, "AffordanceOracle")
    assert "AffordanceOracle" not in env.__all__


def test_step_result_exposes_no_ground_truth_fields():
    fields = set(env.StepResult.__dataclass_fields__)
    assert not fields & {"affordances", "kinds", "ground_truth"}


def test_environment_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        env.Environment()
