"""Tests for backward-chaining regression planning.

The property under test throughout: given confirmed knowledge, the planner
should discover genuine multi-step structure (you need to be holding the
right tool before PLACE_ON works) rather than treating every operator as
immediately achievable -- that structure is what actually prunes the
combinatorial search the planner exists to tame.
"""

from __future__ import annotations

from lama.env import Action, Effect, Outcome, RemoteEffect
from lama.memory.bank import AffordanceBank
from lama.planner import Goal, RegressionPlanner, operator_from_belief


def confirm(bank, target_concept, action, tool_concept, outcome, n=8):
    """Push enough consistent evidence through to reach CONFIRMED."""
    for i in range(n):
        bank.observe(target_concept, action, tool_concept, outcome,
                    episode=0, t=i, object_id="x", tool_id=None)


# --------------------------------------------------------------------------- #
# operator_from_belief
# --------------------------------------------------------------------------- #
def test_unconfirmed_belief_has_no_operator():
    bank = AffordanceBank()
    bank.observe(0, Action.PUSH, None, Outcome(Effect.TRANSLATED, displacement=1.0,
                                                force_required=1.0),
                episode=0, t=0, object_id="x")
    belief = bank.belief((0, Action.PUSH, None))
    assert operator_from_belief(belief) is None


def test_confirmed_belief_carries_real_preconditions():
    bank = AffordanceBank()
    confirm(bank, 0, Action.GRASP, None, Outcome(Effect.CARRIED))
    belief = bank.belief((0, Action.GRASP, None))
    op = operator_from_belief(belief)
    assert op is not None
    assert op.needs_free_gripper is True, "GRASP needs a free gripper -- real metadata"
    assert op.effect is Effect.CARRIED


def test_unresolved_remote_concept_still_reports_the_effect_type():
    """observe() without remote_concepts still learns WHAT tends to happen
    remotely (the effect type), just not WHICH concept it happens to."""
    bank = AffordanceBank()
    outcome = Outcome(Effect.ACTUATED, remote=(RemoteEffect("door_0", Effect.OPENED),))
    confirm(bank, 5, Action.PRESS, None, outcome)
    belief = bank.belief((5, Action.PRESS, None))
    op = operator_from_belief(belief)
    assert op.remote_effect is Effect.OPENED
    assert op.remote_target_concept is None


def test_operator_from_belief_with_resolved_remote_concept():
    bank = AffordanceBank()
    outcome = Outcome(Effect.ACTUATED, remote=(RemoteEffect("door_0", Effect.OPENED),))
    for i in range(8):
        bank.observe(5, Action.PRESS, None, outcome, episode=0, t=i,
                    object_id="plate_0", remote_concepts=(3,))
    op = operator_from_belief(bank.belief((5, Action.PRESS, None)))
    assert op.remote_target_concept == 3
    assert op.remote_effect is Effect.OPENED


# --------------------------------------------------------------------------- #
# direct (single-step) planning
# --------------------------------------------------------------------------- #
def test_plans_a_direct_operator():
    bank = AffordanceBank()
    outcome = Outcome(Effect.ACTUATED, remote=(RemoteEffect("door_0", Effect.OPENED),))
    for i in range(8):
        bank.observe(1, Action.PRESS, None, outcome, episode=0, t=i,
                    object_id="plate_0", remote_concepts=(0,))
    planner = RegressionPlanner(bank)
    plan = planner.plan(Goal(concept=0, effect=Effect.OPENED))
    assert plan is not None
    assert len(plan) == 1
    assert plan[0].operator.action is Action.PRESS
    assert plan[0].operator.target_concept == 1


def test_no_plan_when_nothing_confirmed_reaches_the_goal():
    bank = AffordanceBank()
    planner = RegressionPlanner(bank)
    assert planner.plan(Goal(concept=0, effect=Effect.OPENED)) is None


# --------------------------------------------------------------------------- #
# multi-step planning: the actual point of backward chaining
# --------------------------------------------------------------------------- #
def _flagship_bank():
    """Confirms GRASP(block_concept) and PLACE_ON(plate_concept, block_concept)
    -> remote OPENED on door_concept -- the project's flagship secondary
    affordance, expressed as confirmed operators."""
    bank = AffordanceBank()
    for i in range(8):
        bank.observe(2, Action.GRASP, None, Outcome(Effect.CARRIED),
                    episode=0, t=i, object_id="block_0")
    for i in range(8):
        outcome = Outcome(Effect.SUPPORTED, remote=(RemoteEffect("door_0", Effect.OPENED),))
        bank.observe(1, Action.PLACE_ON, 2, outcome, episode=0, t=i,
                    object_id="plate_0", tool_id="block_0", remote_concepts=(0,))
    return bank


def test_plans_a_two_step_chain_through_a_prerequisite():
    planner = RegressionPlanner(_flagship_bank())
    plan = planner.plan(Goal(concept=0, effect=Effect.OPENED))
    assert plan is not None
    actions = [step.operator.action for step in plan]
    assert actions == [Action.GRASP, Action.PLACE_ON], (
        "must grasp the tool before using it -- that ordering is the whole "
        "point of backward chaining through needs_held_object"
    )
    assert plan[-1].operator.remote_target_concept == 0


def test_plan_gives_up_below_max_depth():
    """A goal genuinely unreachable within the depth budget returns None
    rather than an infinite or truncated-but-wrong plan."""
    planner = RegressionPlanner(_flagship_bank())
    assert planner.plan(Goal(concept=0, effect=Effect.OPENED), max_depth=0) is None


def test_plan_does_not_loop_forever_on_a_cycle():
    """Two operators whose goals reference each other must not hang the
    search -- the `seen` set breaks cycles."""
    bank = AffordanceBank()
    for i in range(8):
        bank.observe(1, Action.PLACE_ON, 2, Outcome(Effect.SUPPORTED,
                     remote=(RemoteEffect("x", Effect.CARRIED),)),
                    episode=0, t=i, object_id="a", tool_id="b", remote_concepts=(2,))
        bank.observe(2, Action.GRASP, None, Outcome(Effect.CARRIED,
                     remote=()), episode=0, t=i, object_id="b")
    planner = RegressionPlanner(bank)
    # concept 2's GRASP effect is CARRIED directly; goal for concept 2 CARRIED
    # is satisfied without recursion, so this must terminate promptly.
    plan = planner.plan(Goal(concept=2, effect=Effect.CARRIED), max_depth=10)
    assert plan is not None


# --------------------------------------------------------------------------- #
# relevant_keys: what imagination/select actually consume
# --------------------------------------------------------------------------- #
def test_relevant_keys_includes_the_direct_operator_at_full_weight():
    planner = RegressionPlanner(_flagship_bank())
    weights = planner.relevant_keys(Goal(concept=0, effect=Effect.OPENED))
    assert weights[(1, Action.PLACE_ON, 2)] == 1.0


def test_relevant_keys_includes_the_prerequisite_at_decayed_weight():
    planner = RegressionPlanner(_flagship_bank())
    weights = planner.relevant_keys(Goal(concept=0, effect=Effect.OPENED))
    assert 0.0 < weights[(2, Action.GRASP, None)] < 1.0


def test_relevant_keys_is_empty_for_an_unreachable_goal():
    planner = RegressionPlanner(AffordanceBank())
    assert planner.relevant_keys(Goal(concept=99, effect=Effect.OPENED)) == {}


def test_relevant_keys_ignores_unconfirmed_beliefs():
    bank = AffordanceBank()
    # only 2 trials: PROVISIONAL, not CONFIRMED
    bank.observe(1, Action.PRESS, None,
                Outcome(Effect.ACTUATED, remote=(RemoteEffect("d", Effect.OPENED),)),
                episode=0, t=0, object_id="p", remote_concepts=(0,))
    planner = RegressionPlanner(bank)
    assert planner.relevant_keys(Goal(concept=0, effect=Effect.OPENED)) == {}
