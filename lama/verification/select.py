"""Deciding which hypothesis is worth spending real budget to test.

The core acquisition function is unchanged in spirit: how much would testing
this still teach the bank, per unit of budget it costs. "How much would it
teach" is read straight off the belief the bank already holds:

* No belief at all -- never matched to a concept, or matched but never tried
  with this verb -- is maximal uncertainty, worth 1.0. There is nothing to
  lose by finding out.
* `PROVISIONAL` or `STUCK` is worth exactly the width of its credible
  interval: literally how unsettled the bank's own estimate still is.
* `CONFIRMED` or `REFUTED` is worth 0. The bank already knows the answer to
  the precision it needs; spending budget to narrow a settled belief further
  has no research value here, even though its interval is not literally zero
  width.

Three further, MULTIPLICATIVE factors adjust that base score -- multiplicative
so that none of them can turn an already-settled (score 0) hypothesis
positive, which an earlier additive version of this function got wrong:

* **Safety.** `(1 - irreversible_risk)` discounts anything the bank
  estimates is likely to cause unrecoverable damage. A never-tested verb
  starts at the uninformed prior (0.5 risk, a real but moderate discount, not
  a block) and the discount relaxes quickly as evidence shows it usually is
  not destructive -- see `Belief.irreversible_alpha/beta` in `memory/bank.py`.
* **Concept-level curiosity.** A mild bonus for objects that are broadly
  uncertain (`Hypothesis.info_gain`), not just uncertain about this one verb.
* **Goal relevance.** A stronger bonus for hypotheses a backward-chaining
  plan identified as being on the way to whatever goal the caller is
  currently pursuing (`Hypothesis.relevance`; see `planner/planner.py` and
  `imagination/hypothesis.py`'s `relevant_keys` argument). Zero unless a goal
  was actually supplied, so this is a pure extension: nothing changes for a
  caller that never sets a goal.

Dividing the base score by cost is what makes this a genuine per-budget-unit
acquisition score, so a cheap, moderately uncertain test can outrank an
expensive, slightly more uncertain one -- which is the whole reason
`env/actions.py` gives verbs different costs in the first place.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..imagination.hypothesis import Hypothesis
from ..memory.bank import SETTLED_STATUSES

#: Weight on the concept-level curiosity bonus. Deliberately modest: this
#: should nudge between similarly-uncertain options, not override the
#: primary per-verb uncertainty signal.
INFO_GAIN_WEIGHT: float = 0.5

#: Weight on the goal-relevance bonus. Larger than the curiosity weight: an
#: active goal should meaningfully reorder what gets tried first, which is
#: the entire point of wiring a planner in at all.
RELEVANCE_WEIGHT: float = 2.0


def uncertainty(hypothesis: Hypothesis) -> float:
    """How much a real test of `hypothesis` would still teach the bank, in
    `[0, 1]`: 0 means settled, 1 means totally unknown."""
    if hypothesis.status in SETTLED_STATUSES:
        return 0.0
    if hypothesis.credible_width is None:
        return 1.0
    return hypothesis.credible_width


def score(hypothesis: Hypothesis) -> float:
    """Expected information gained per unit of budget spent testing this,
    discounted for irreversibility risk and boosted for curiosity/relevance.
    """
    u = uncertainty(hypothesis)
    if u <= 0.0:
        return 0.0
    base = u / hypothesis.cost
    safety = 1.0 - hypothesis.irreversible_risk
    curiosity = 1.0 + INFO_GAIN_WEIGHT * hypothesis.info_gain
    goal = 1.0 + RELEVANCE_WEIGHT * hypothesis.relevance
    return base * safety * curiosity * goal


def rank(hypotheses: Iterable[Hypothesis]) -> tuple[Hypothesis, ...]:
    """`hypotheses` ordered by `score`, highest first.

    Ties break on `(target_id, verb, tool_id)` so ranking is reproducible for
    the same input, rather than depending on incidental iteration order.
    """
    return tuple(
        sorted(
            hypotheses,
            key=lambda h: (-score(h), h.target_id, h.action.value, h.tool_id or ""),
        )
    )


def select_next(
    hypotheses: Iterable[Hypothesis], budget_remaining: float
) -> Hypothesis | None:
    """The single most worthwhile hypothesis affordable right now.

    `None` if nothing reachable is both worth testing and within budget --
    everything is settled, or everything unsettled costs more than is left.
    """
    for h in rank(hypotheses):
        if score(h) <= 0.0:
            break  # rank() is sorted descending; nothing further scores higher
        if h.cost <= budget_remaining:
            return h
    return None
