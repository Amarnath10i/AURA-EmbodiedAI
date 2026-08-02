"""Tests for the hypothesis-selection acquisition function."""

from __future__ import annotations

import pytest

from lama.env import Action, Effect
from lama.imagination.hypothesis import Hypothesis
from lama.memory.bank import Status
from lama.verification.select import rank, score, select_next, uncertainty


def hyp(
    target_id="x", action=Action.PUSH, tool_id=None, status=Status.UNTESTED,
    predicted_mean=None, credible_width=None, cost=1.0,
    irreversible_risk=0.0, info_gain=0.0, relevance=0.0,
):
    return Hypothesis(
        target_id=target_id, action=action, tool_id=tool_id,
        target_concept=0, tool_concept=None, status=status,
        predicted_mean=predicted_mean, credible_width=credible_width,
        dominant_effect=None, predicted_effect=None,
        irreversible_risk=irreversible_risk, info_gain=info_gain,
        cost=cost, relevance=relevance,
    )


# --------------------------------------------------------------------------- #
# uncertainty
# --------------------------------------------------------------------------- #
def test_untested_is_maximally_uncertain():
    assert uncertainty(hyp(status=Status.UNTESTED, credible_width=None)) == 1.0


def test_provisional_uncertainty_is_the_credible_width():
    assert uncertainty(hyp(status=Status.PROVISIONAL, credible_width=0.42)) == 0.42


def test_confirmed_and_refuted_are_zero_uncertainty():
    assert uncertainty(hyp(status=Status.CONFIRMED, credible_width=0.2)) == 0.0
    assert uncertainty(hyp(status=Status.REFUTED, credible_width=0.09)) == 0.0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def test_score_is_uncertainty_over_cost():
    h = hyp(status=Status.PROVISIONAL, credible_width=0.6, cost=2.0)
    assert score(h) == 0.3


def test_settled_hypotheses_score_zero_regardless_of_cost():
    h = hyp(status=Status.CONFIRMED, cost=0.1)
    assert score(h) == 0.0


def test_cheaper_verb_scores_higher_at_equal_uncertainty():
    """The whole reason to divide by cost: a cheap unknown should outrank an
    expensive one of the same uncertainty."""
    cheap = hyp(target_id="a", cost=0.5)
    expensive = hyp(target_id="b", cost=2.0)
    assert score(cheap) > score(expensive)


# --------------------------------------------------------------------------- #
# rank
# --------------------------------------------------------------------------- #
def test_rank_orders_by_score_descending():
    low = hyp(target_id="low", status=Status.PROVISIONAL, credible_width=0.2, cost=1.0)
    high = hyp(target_id="high", status=Status.PROVISIONAL, credible_width=0.9, cost=1.0)
    ordered = rank([low, high])
    assert [h.target_id for h in ordered] == ["high", "low"]


def test_rank_breaks_ties_deterministically():
    a = hyp(target_id="b", action=Action.PUSH)
    b = hyp(target_id="a", action=Action.PUSH)
    assert [h.target_id for h in rank([a, b])] == ["a", "b"]
    assert [h.target_id for h in rank([b, a])] == ["a", "b"]


# --------------------------------------------------------------------------- #
# select_next
# --------------------------------------------------------------------------- #
def test_select_next_picks_the_top_ranked_affordable_hypothesis():
    low = hyp(target_id="low", status=Status.PROVISIONAL, credible_width=0.2)
    high = hyp(target_id="high", status=Status.PROVISIONAL, credible_width=0.9)
    chosen = select_next([low, high], budget_remaining=10.0)
    assert chosen.target_id == "high"


def test_select_next_skips_an_unaffordable_top_choice():
    expensive = hyp(target_id="expensive", credible_width=0.9, cost=5.0)
    cheap = hyp(target_id="cheap", credible_width=0.5, cost=1.0)
    chosen = select_next([expensive, cheap], budget_remaining=2.0)
    assert chosen.target_id == "cheap"


def test_select_next_returns_none_when_everything_is_settled():
    settled = [
        hyp(target_id="a", status=Status.CONFIRMED),
        hyp(target_id="b", status=Status.REFUTED),
    ]
    assert select_next(settled, budget_remaining=100.0) is None


def test_select_next_returns_none_when_nothing_is_affordable():
    h = hyp(status=Status.PROVISIONAL, credible_width=0.5, cost=10.0)
    assert select_next([h], budget_remaining=1.0) is None


def test_select_next_returns_none_for_no_hypotheses():
    assert select_next([], budget_remaining=100.0) is None


# --------------------------------------------------------------------------- #
# STUCK is settled -- the exact bug this pins: an earlier, separately
# maintained copy of "settled" in this module omitted STUCK, so a belief the
# bank had flagged as probably-blended kept scoring as if fully unknown.
# --------------------------------------------------------------------------- #
def test_stuck_is_settled_and_scores_zero():
    h = hyp(status=Status.STUCK, credible_width=0.6)
    assert uncertainty(h) == 0.0
    assert score(h) == 0.0


def test_select_next_skips_stuck_hypotheses():
    stuck = hyp(target_id="stuck", status=Status.STUCK, credible_width=0.9)
    provisional = hyp(target_id="fine", status=Status.PROVISIONAL, credible_width=0.2)
    assert select_next([stuck, provisional], budget_remaining=10.0).target_id == "fine"


# --------------------------------------------------------------------------- #
# safety: irreversible_risk discounts, but never blocks outright
# --------------------------------------------------------------------------- #
def test_irreversible_risk_discounts_score():
    safe = hyp(target_id="safe", status=Status.PROVISIONAL, credible_width=0.5,
              irreversible_risk=0.0)
    risky = hyp(target_id="risky", status=Status.PROVISIONAL, credible_width=0.5,
               irreversible_risk=0.8)
    assert score(risky) < score(safe)
    assert score(risky) == pytest.approx(score(safe) * 0.2)


def test_a_first_attempt_at_a_possibly_irreversible_verb_is_not_blocked():
    """The bank's uninformed prior for an untested verb is 0.5 risk, not 1.0
    -- caution, not paralysis. It must still be selectable when nothing safer
    is available."""
    only_option = hyp(status=Status.UNTESTED, irreversible_risk=0.5, cost=1.6)
    chosen = select_next([only_option], budget_remaining=10.0)
    assert chosen is not None


def test_certain_irreversibility_does_not_fully_zero_the_score():
    """Even a verb the bank is confident is destructive still scores
    positive -- a discount, never a hard block; see select.py's docstring on
    why this is multiplicative rather than a gate."""
    h = hyp(status=Status.PROVISIONAL, credible_width=0.5, irreversible_risk=0.99)
    assert score(h) > 0.0


# --------------------------------------------------------------------------- #
# curiosity and goal-relevance bonuses
# --------------------------------------------------------------------------- #
def test_info_gain_boosts_score_but_does_not_revive_a_settled_hypothesis():
    plain = hyp(target_id="plain", status=Status.PROVISIONAL, credible_width=0.4)
    curious = hyp(target_id="curious", status=Status.PROVISIONAL, credible_width=0.4,
                  info_gain=1.0)
    assert score(curious) > score(plain)

    settled_but_curious = hyp(status=Status.CONFIRMED, info_gain=1.0)
    assert score(settled_but_curious) == 0.0


def test_relevance_boosts_score_but_does_not_revive_a_settled_hypothesis():
    irrelevant = hyp(target_id="irrelevant", status=Status.PROVISIONAL, credible_width=0.4)
    relevant = hyp(target_id="relevant", status=Status.PROVISIONAL, credible_width=0.4,
                   relevance=1.0)
    assert score(relevant) > score(irrelevant)

    settled_but_relevant = hyp(status=Status.REFUTED, relevance=1.0)
    assert score(settled_but_relevant) == 0.0


def test_relevance_can_reorder_selection_toward_a_goal():
    """The entire point of wiring a planner in: a goal-relevant hypothesis
    with somewhat lower raw uncertainty can still outrank an irrelevant one
    with higher raw uncertainty."""
    off_goal = hyp(target_id="off_goal", status=Status.PROVISIONAL, credible_width=0.7)
    on_goal = hyp(target_id="on_goal", status=Status.PROVISIONAL, credible_width=0.5,
                  relevance=1.0)
    assert select_next([off_goal, on_goal], budget_remaining=10.0).target_id == "on_goal"
