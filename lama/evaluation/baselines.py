"""Baseline selection policies, for a controlled comparison against the real
system.

Each policy here matches `verification.loop.SelectFn`'s signature exactly --
`(hypotheses, budget_remaining) -> Hypothesis | None` -- so it can run
through the IDENTICAL `verify_once`/`run_episode` loop that
`verification.select.select_next` does: same environment, same Bayesian
adjudication in `AffordanceMemory`, same exploration-target fallback. Only
the choice of what to test next differs. That is what makes the comparison
in `scripts/run_baseline_comparison.py` a real ablation rather than a
different-environment confound.

Three baselines, chosen to match what the literature and this project's own
prior planning documents (AURA_Project_Plan.pdf, section 8.1) actually
compare against:

* `RandomPolicy` -- the floor. Ignores uncertainty, cost, and settledness
  entirely.
* `NoveltyPolicy` -- "seek novel states", the Plan2Explore-style strategy:
  prefer whatever has never been tried, with no Bayesian calibration, no
  safety discount, no goal-direction.
* `uncertainty_only_policy` -- `select.py`'s core acquisition function
  (uncertainty per unit cost) with its three multiplicative refinements
  (safety discount, curiosity bonus, goal relevance) removed. This is the
  ablation that isolates what those three refinements actually contribute,
  as opposed to what plain uncertainty-driven testing already gets you.

All three are deterministic given a seed (`RandomPolicy`/`NoveltyPolicy`) or
unconditionally (`uncertainty_only_policy`), matching this project's general
determinism guarantee.
"""

from __future__ import annotations

import numpy as np

from ..imagination.hypothesis import Hypothesis
from ..memory.bank import SETTLED_STATUSES, Status
from ..verification.select import uncertainty

__all__ = [
    "RandomPolicy",
    "NoveltyPolicy",
    "uncertainty_only_policy",
    "uncertainty_only",
    "lama_no_safety_policy",
]


def uncertainty_only(hypothesis: Hypothesis) -> float:
    """`select.py`'s base score -- uncertainty per unit cost -- with the
    safety discount, curiosity bonus, and goal-relevance multiplier all
    removed. What `select.score` reduces to without its three refinements;
    used by `uncertainty_only_policy` to isolate what those refinements
    actually contribute in the baseline comparison.
    """
    u = uncertainty(hypothesis)
    return u / hypothesis.cost if u > 0.0 else 0.0


def _affordable(hypotheses, budget_remaining: float) -> list[Hypothesis]:
    return [h for h in hypotheses if h.cost <= budget_remaining]


class RandomPolicy:
    """Uniformly random among affordable hypotheses. The floor every other
    policy is expected to beat."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(
        self, hypotheses: tuple[Hypothesis, ...], budget_remaining: float
    ) -> Hypothesis | None:
        pool = _affordable(hypotheses, budget_remaining)
        if not pool:
            return None
        return pool[int(self._rng.integers(len(pool)))]


class NoveltyPolicy:
    """Prefers whatever has never been tried; falls back to the pool of
    still-unsettled hypotheses when everything reachable has been tried at
    least once. No cost-normalisation, no Bayesian confidence, no safety or
    goal awareness -- pure "have I seen this before".
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(
        self, hypotheses: tuple[Hypothesis, ...], budget_remaining: float
    ) -> Hypothesis | None:
        pool = _affordable(hypotheses, budget_remaining)
        untested = [h for h in pool if h.status is Status.UNTESTED]
        candidates = untested or [
            h for h in pool if h.status not in SETTLED_STATUSES
        ]
        if not candidates:
            return None
        return candidates[int(self._rng.integers(len(candidates)))]


def uncertainty_only_policy(
    hypotheses: tuple[Hypothesis, ...], budget_remaining: float
) -> Hypothesis | None:
    """`select.py`'s acquisition function with the safety/curiosity/goal
    multipliers stripped out -- see `uncertainty_only` above."""
    affordable = _affordable(hypotheses, budget_remaining)
    ranked = sorted(
        affordable,
        key=lambda h: (
            -uncertainty_only(h), h.target_id, h.action.value, h.tool_id or ""
        ),
    )
    for h in ranked:
        if uncertainty_only(h) <= 0.0:
            break
        return h
    return None


def _no_safety_score(hypothesis: Hypothesis) -> float:
    from ..verification.select import INFO_GAIN_WEIGHT, RELEVANCE_WEIGHT

    u = uncertainty(hypothesis)
    if u <= 0.0:
        return 0.0
    base = u / hypothesis.cost
    curiosity = 1.0 + INFO_GAIN_WEIGHT * hypothesis.info_gain
    goal = 1.0 + RELEVANCE_WEIGHT * hypothesis.relevance
    return base * curiosity * goal


def lama_no_safety_policy(
    hypotheses: tuple[Hypothesis, ...], budget_remaining: float
) -> Hypothesis | None:
    """The full `select.score` formula with ONLY the safety discount
    (`1 - irreversible_risk`) removed -- curiosity and goal-relevance stay.

    A targeted ablation, not one of the three standard baselines: it exists
    specifically to test a hypothesis raised by the baseline comparison
    (docs/RESEARCH_FINDINGS.md) -- that discounting irreversible verbs by
    design trades off against discovering exactly the secondary affordances
    that are only reachable through one (e.g. `TIP` on barrel/drum). If
    secondary recall improves here relative to full `select_next` while
    staying below `uncertainty_only_policy`, that supports the hypothesis
    that safety specifically (not curiosity or cost-normalisation) is the
    cost; if it does not, the tension has a different cause.
    """
    affordable = _affordable(hypotheses, budget_remaining)
    ranked = sorted(
        affordable,
        key=lambda h: (
            -_no_safety_score(h), h.target_id, h.action.value, h.tool_id or ""
        ),
    )
    for h in ranked:
        if _no_safety_score(h) <= 0.0:
            break
        return h
    return None
