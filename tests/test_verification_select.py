"""Tests for the hypothesis-selection acquisition function."""

from __future__ import annotations

from lama.env import Action, Effect
from lama.imagination.hypothesis import Hypothesis
from lama.memory.bank import Status
from lama.verification.select import rank, score, select_next, uncertainty


def hyp(
    target_id="x", action=Action.PUSH, tool_id=None, status=Status.UNTESTED,
    predicted_mean=None, credible_width=None, cost=1.0,
):
    return Hypothesis(
        target_id=target_id, action=action, tool_id=tool_id,
        target_concept=0, tool_concept=None, status=status,
        predicted_mean=predicted_mean, credible_width=credible_width,
        dominant_effect=None, cost=cost,
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
