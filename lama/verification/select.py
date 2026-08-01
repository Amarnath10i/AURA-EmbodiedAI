"""Deciding which hypothesis is worth spending real budget to test.

The acquisition function is deliberately simple: how much would testing this
still teach the bank, per unit of budget it costs. "How much would it teach"
is read straight off the belief the bank already holds:

* No belief at all -- never matched to a concept, or matched but never tried
  with this verb -- is maximal uncertainty, worth 1.0. There is nothing to
  lose by finding out.
* `PROVISIONAL` is worth exactly the width of its credible interval: literally
  how unsettled the bank's own estimate still is.
* `CONFIRMED` or `REFUTED` is worth 0. The bank already knows the answer to
  the precision it needs; spending budget to narrow a settled belief further
  has no research value here, even though its interval is not literally zero
  width.

Dividing by cost turns this into a genuine per-budget-unit acquisition score,
so a cheap, moderately uncertain test can outrank an expensive, slightly more
uncertain one -- which is the whole reason `env/actions.py` gives verbs
different costs in the first place.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..imagination.hypothesis import Hypothesis
from ..memory.bank import Status

#: Statuses the bank considers settled; testing them further scores zero.
_SETTLED = frozenset({Status.CONFIRMED, Status.REFUTED})


def uncertainty(hypothesis: Hypothesis) -> float:
    """How much a real test of `hypothesis` would still teach the bank, in
    `[0, 1]`: 0 means settled, 1 means totally unknown."""
    if hypothesis.status in _SETTLED:
        return 0.0
    if hypothesis.credible_width is None:
        return 1.0
    return hypothesis.credible_width


from ..env.outcomes import IRREVERSIBLE_EFFECTS

def score(hypothesis: Hypothesis) -> float:
    """Expected information gained per unit of budget spent testing this."""
    u = uncertainty(hypothesis)
    
    # Hard-gate: prevent selection of irreversible actions if uncertainty is too high
    if hypothesis.dominant_effect in IRREVERSIBLE_EFFECTS and u > 0.5:
        return 0.0
        
    base_score = u / hypothesis.cost if u > 0.0 else 0.0
    return base_score + hypothesis.relevance + hypothesis.info_gain


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
