"""Tests for the baseline selection policies used in the research comparison.

The property that matters: each baseline actually implements what it claims
to, not just "runs without crashing" -- since the whole point is a controlled
comparison, a baseline that secretly behaves like LAMA (or crashes silently
into always picking nothing) would invalidate the comparison without anyone
noticing.
"""

from __future__ import annotations

from lama.env import Action
from lama.evaluation.baselines import (
    NoveltyPolicy,
    RandomPolicy,
    lama_no_safety_policy,
    uncertainty_only,
    uncertainty_only_policy,
)
from lama.imagination.hypothesis import Hypothesis
from lama.memory.bank import Status


def hyp(target_id="x", status=Status.UNTESTED, credible_width=None, cost=1.0,
       irreversible_risk=0.0, info_gain=0.0, relevance=0.0):
    return Hypothesis(
        target_id=target_id, action=Action.PUSH, tool_id=None,
        target_concept=0, tool_concept=None, status=status,
        predicted_mean=None, credible_width=credible_width,
        dominant_effect=None, predicted_effect=None,
        irreversible_risk=irreversible_risk, info_gain=info_gain,
        cost=cost, relevance=relevance,
    )


# --------------------------------------------------------------------------- #
# RandomPolicy
# --------------------------------------------------------------------------- #
def test_random_policy_ignores_settledness():
    """Unlike every other policy, random must be willing to pick an already-
    CONFIRMED hypothesis -- that is the whole point of it being the floor."""
    settled = hyp(status=Status.CONFIRMED)
    policy = RandomPolicy(seed=0)
    # with only one option, it must return it even though it is settled
    assert policy((settled,), budget_remaining=10.0) is settled


def test_random_policy_respects_budget():
    affordable = hyp(target_id="cheap", cost=1.0)
    unaffordable = hyp(target_id="expensive", cost=100.0)
    policy = RandomPolicy(seed=0)
    for _ in range(20):
        chosen = policy((affordable, unaffordable), budget_remaining=5.0)
        assert chosen.target_id == "cheap"


def test_random_policy_is_deterministic_given_a_seed():
    options = tuple(hyp(target_id=str(i)) for i in range(10))
    a = RandomPolicy(seed=42)
    b = RandomPolicy(seed=42)
    picks_a = [a(options, 10.0).target_id for _ in range(10)]
    picks_b = [b(options, 10.0).target_id for _ in range(10)]
    assert picks_a == picks_b


def test_random_policy_none_when_nothing_affordable():
    policy = RandomPolicy(seed=0)
    assert policy((hyp(cost=100.0),), budget_remaining=1.0) is None


# --------------------------------------------------------------------------- #
# NoveltyPolicy
# --------------------------------------------------------------------------- #
def test_novelty_policy_prefers_untested_over_provisional():
    tested = hyp(target_id="tested", status=Status.PROVISIONAL, credible_width=0.9)
    fresh = hyp(target_id="fresh", status=Status.UNTESTED)
    policy = NoveltyPolicy(seed=0)
    for _ in range(20):
        assert policy((tested, fresh), 10.0).target_id == "fresh"


def test_novelty_policy_falls_back_to_unsettled_when_nothing_untested():
    provisional = hyp(target_id="p", status=Status.PROVISIONAL, credible_width=0.5)
    confirmed = hyp(target_id="c", status=Status.CONFIRMED)
    policy = NoveltyPolicy(seed=0)
    for _ in range(20):
        assert policy((provisional, confirmed), 10.0).target_id == "p"


def test_novelty_policy_none_when_everything_settled():
    settled = hyp(status=Status.REFUTED)
    policy = NoveltyPolicy(seed=0)
    assert policy((settled,), 10.0) is None


def test_novelty_policy_ignores_cost_entirely():
    """Unlike select_next, novelty has no cost-normalisation -- an expensive
    untested hypothesis is exactly as attractive as a cheap one."""
    cheap = hyp(target_id="cheap", status=Status.UNTESTED, cost=0.2)
    expensive = hyp(target_id="expensive", status=Status.UNTESTED, cost=2.0)
    policy = NoveltyPolicy(seed=1)
    picks = {policy((cheap, expensive), 10.0).target_id for _ in range(30)}
    assert picks == {"cheap", "expensive"}, "both should get picked over enough trials"


# --------------------------------------------------------------------------- #
# uncertainty_only_policy: the ablation
# --------------------------------------------------------------------------- #
def test_uncertainty_only_matches_select_score_with_neutral_bonuses():
    """With irreversible_risk=0, info_gain=0, relevance=0 (all bonuses at
    their neutral value), uncertainty_only must equal select.score exactly --
    confirming it really is the same base function, not a reimplementation
    that happens to look similar."""
    from lama.verification.select import score

    h = hyp(status=Status.PROVISIONAL, credible_width=0.6, cost=1.5)
    assert uncertainty_only(h) == score(h)


def test_uncertainty_only_ignores_safety_curiosity_and_relevance():
    """The entire point of the ablation: these three factors must NOT change
    its ranking, unlike the real select.score."""
    plain = hyp(target_id="plain", status=Status.PROVISIONAL, credible_width=0.5)
    risky = hyp(target_id="risky", status=Status.PROVISIONAL, credible_width=0.5,
               irreversible_risk=0.9)
    curious = hyp(target_id="curious", status=Status.PROVISIONAL, credible_width=0.5,
                  info_gain=1.0)
    relevant = hyp(target_id="relevant", status=Status.PROVISIONAL, credible_width=0.5,
                   relevance=1.0)
    values = {uncertainty_only(h) for h in (plain, risky, curious, relevant)}
    assert len(values) == 1, "all four should score identically without the bonuses"


def test_uncertainty_only_policy_never_picks_a_settled_hypothesis():
    settled = hyp(status=Status.CONFIRMED)
    assert uncertainty_only_policy((settled,), 10.0) is None


def test_uncertainty_only_policy_still_uses_cost_normalisation():
    """Unlike NoveltyPolicy, this ablation keeps select.py's cost-per-unit
    normalisation -- only the three multiplicative bonuses are removed."""
    cheap = hyp(target_id="cheap", status=Status.UNTESTED, cost=0.5)
    expensive = hyp(target_id="expensive", status=Status.UNTESTED, cost=5.0)
    chosen = uncertainty_only_policy((cheap, expensive), 10.0)
    assert chosen.target_id == "cheap"


# --------------------------------------------------------------------------- #
# lama_no_safety_policy: the targeted ablation
# --------------------------------------------------------------------------- #
def test_no_safety_policy_ignores_irreversible_risk():
    """The one thing this ablation must NOT respond to."""
    safe = hyp(target_id="safe", status=Status.PROVISIONAL, credible_width=0.5,
              irreversible_risk=0.0)
    risky = hyp(target_id="risky", status=Status.PROVISIONAL, credible_width=0.5,
               irreversible_risk=0.95)
    from lama.evaluation.baselines import _no_safety_score

    assert _no_safety_score(safe) == _no_safety_score(risky)


def test_no_safety_policy_still_responds_to_curiosity_and_relevance():
    """Unlike uncertainty_only, this ablation keeps the OTHER two bonuses --
    only safety is removed."""
    from lama.evaluation.baselines import _no_safety_score

    plain = hyp(status=Status.PROVISIONAL, credible_width=0.5)
    curious = hyp(status=Status.PROVISIONAL, credible_width=0.5, info_gain=1.0)
    relevant = hyp(status=Status.PROVISIONAL, credible_width=0.5, relevance=1.0)
    assert _no_safety_score(curious) > _no_safety_score(plain)
    assert _no_safety_score(relevant) > _no_safety_score(plain)


def test_no_safety_policy_never_picks_a_settled_hypothesis():
    settled = hyp(status=Status.REFUTED)
    assert lama_no_safety_policy((settled,), 10.0) is None


def test_no_safety_policy_prefers_a_risky_hypothesis_full_select_would_discount():
    """The concrete behavioural difference from the real select_next: with
    risk removed from scoring, a risky-but-otherwise-equal hypothesis is no
    longer penalised relative to a safe one."""
    from lama.verification.select import score as full_score

    safe = hyp(target_id="safe", status=Status.PROVISIONAL, credible_width=0.5,
              irreversible_risk=0.0)
    risky = hyp(target_id="risky", status=Status.PROVISIONAL, credible_width=0.5,
               irreversible_risk=0.9)
    assert full_score(risky) < full_score(safe), "full select_next discounts it"
    from lama.evaluation.baselines import _no_safety_score

    assert _no_safety_score(risky) == _no_safety_score(safe), (
        "the ablation must not"
    )
