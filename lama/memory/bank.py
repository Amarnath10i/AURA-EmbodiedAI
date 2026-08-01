"""The lifelong affordance bank: confirmed knowledge, held with calibrated
confidence, and revised when contradicted.

Keys are `(concept_id, action, tool_concept_id | None)` rather than
`(kind, action, tool_kind)` -- the agent never has kind, only the concepts it
has formed from appearance (see `concepts.py`). This has a direct consequence
worth stating plainly: **a belief keyed by a concept that conflates two true
kinds (crate and block look identical, so they land in one concept) will
settle on an intermediate reliability that is correct for the concept and
wrong for either kind alone.** The bank cannot fix this; it only has what the
concept codebook gives it, and that is by design (D2/D5 in
docs/DECISIONS.md). Measuring how much this costs is part of the evaluation,
not a defect to engineer away here.

Belief is Beta-Bernoulli over "did this verb change the world". A Beta
posterior does the right thing for both kinds of uncertainty this project
cares about: few trials leave it wide regardless of the true rate --
epistemic, more testing would narrow it -- while many trials on a genuinely
50%-reliable affordance still concentrate tightly *around 0.5* -- aleatoric,
the rate itself is what stays uncertain, not our knowledge of it. The bank
reports both the estimate and its precision; deciding whether a given
precision is worth spending more budget to improve is the verification
module's job, not this one.

Thresholds below (`REAL_EPS`, `MAX_SETTLED_WIDTH`) were picked by simulating
the resulting Beta posterior, not guessed: under them, roughly 8 consecutive
successes settle a deterministic affordance as `CONFIRMED`, roughly 27
consecutive failures settle one as `REFUTED`, and a genuinely 50%-reliable
affordance needs about 40 trials before the bank calls it `CONFIRMED` rather
than `PROVISIONAL`. Being harder to refute than to confirm is intentional:
declaring something impossible is a stronger claim than declaring it works,
and should cost more evidence.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from ..env import Action, Effect, Outcome


class Status(Enum):
    """What the bank currently believes about one `(concept, verb, tool)`."""

    UNTESTED = auto()     # never attempted
    PROVISIONAL = auto()  # attempted, but not enough evidence to settle
    CONFIRMED = auto()    # confident this is a real, working affordance
    REFUTED = auto()      # confident this does not work
    STUCK = auto()        # irreducible ambiguity (e.g. missing sensing channel)


#: Posterior mass below which the bank calls an affordance not real.
REAL_EPS: float = 0.10

#: Maximum 95%-credible-interval width to call a belief settled rather than
#: merely provisional. See the module docstring for what this implies in
#: trials.
MAX_SETTLED_WIDTH: float = 0.30

#: z-score for the credible interval (~95%, normal approximation to Beta).
_Z: float = 1.96

#: Evidence pointers kept per belief, for provenance without unbounded memory
#: growth at lifelong interaction volumes (10^4-10^5 interactions).
_EVIDENCE_CAP: int = 20

#: Statuses that count as "settled" for deciding whether a status change is a
#: `Revision` worth reporting, as opposed to routine evidence accumulation.
_SETTLED = frozenset({Status.CONFIRMED, Status.REFUTED, Status.STUCK})


@dataclass(frozen=True)
class Evidence:
    """One interaction that contributed to a belief."""

    episode: int
    t: int
    object_id: str
    tool_id: str | None
    effect: Effect
    force_required: float = 0.0


@dataclass
class Belief:
    """Everything the bank has learned about one `(concept, verb, tool)`.

    Attributes:
        key: `(target_concept, action, tool_concept)`.
        alpha, beta: Beta posterior over "this verb changes the world",
            starting from a uniform Beta(1, 1) prior.
        effect_counts: How often each effect was observed; see
            `dominant_effect`.
        remote_alpha, remote_beta: Beta posterior, conditioned on success,
            over "this also changed something else" -- the bank's own proxy
            for `Affordance.uses_object_as_means`, learned without ever
            touching ground truth.
        evidence: A bounded recent sample, for provenance and debugging.
        total_attempts: True count, even once `evidence` has been truncated.
    """

    key: tuple[int, Action, int | None]
    alpha: float = 1.0
    beta: float = 1.0
    effect_counts: dict[Effect, int] = field(default_factory=dict)
    remote_alpha: float = 1.0
    remote_beta: float = 1.0
    evidence: deque = field(default_factory=lambda: deque(maxlen=_EVIDENCE_CAP))
    interval_history: deque = field(default_factory=lambda: deque(maxlen=10))
    total_attempts: int = 0

    @property
    def mean(self) -> float:
        """Point estimate of the true success rate."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def credible_interval(self) -> tuple[float, float]:
        """Approximate 95% credible interval on the true success rate.

        Normal approximation to the Beta posterior. Adequate here: the bank
        only needs this to decide settled-vs-not, not to report a precise
        interval, and the approximation is conservative -- slightly wider
        than exact -- near the edges where that decision actually happens.
        """
        n = self.alpha + self.beta
        var = (self.alpha * self.beta) / (n * n * (n + 1.0))
        std = math.sqrt(var)
        m = self.mean
        return max(0.0, m - _Z * std), min(1.0, m + _Z * std)

    @property
    def remote_rate(self) -> float:
        """Estimated probability that a success also changes something else."""
        return self.remote_alpha / (self.remote_alpha + self.remote_beta)

    @property
    def dominant_effect(self) -> Effect | None:
        """The most frequently observed effect, or `None` with no evidence."""
        if not self.effect_counts:
            return None
        return max(self.effect_counts, key=self.effect_counts.get)

    @property
    def operator(self) -> dict | None:
        """STRIPS-style operator if confirmed."""
        if not self.is_confirmed:
            return None
        return {
            "precondition": ["reachable(target)"],
            "action": (self.key[1].name, "target", "tool" if self.key[2] is not None else None),
            "effect": {"state": self.dominant_effect.name if self.dominant_effect else "UNKNOWN"},
            "confidence": {"alpha": self.alpha, "beta": self.beta}
        }

    @property
    def status(self) -> Status:
        if self.total_attempts == 0:
            return Status.UNTESTED
        lo, hi = self.credible_interval
        if hi < REAL_EPS:
            return Status.REFUTED
        if lo > REAL_EPS and (hi - lo) <= MAX_SETTLED_WIDTH:
            return Status.CONFIRMED
        
        # Plateau detector
        if len(self.interval_history) == self.interval_history.maxlen:
            oldest = self.interval_history[0]
            newest = self.interval_history[-1]
            if oldest > 0:
                rel_reduction = (oldest - newest) / oldest
                if rel_reduction < 0.05 and lo > REAL_EPS and hi > REAL_EPS:
                    return Status.STUCK
                    
        return Status.PROVISIONAL

    @property
    def is_confirmed(self) -> bool:
        return self.status is Status.CONFIRMED


@dataclass(frozen=True)
class Revision:
    """A belief's status crossed into or out of `CONFIRMED` or `REFUTED`.

    This is the event the "keep only confirmed knowledge, revise on
    contradiction" half of the loop hangs off. Not every new piece of
    evidence produces one -- only evidence that actually changes what a
    downstream consumer of `AffordanceBank.confirmed()` would see.
    """

    key: tuple[int, Action, int | None]
    episode: int
    t: int
    old_status: Status
    new_status: Status


class AffordanceBank:
    """Lifelong belief store over `(concept, verb, tool_concept)` capabilities."""

    def __init__(self) -> None:
        self._beliefs: dict[tuple[int, Action, int | None], Belief] = {}
        os.makedirs("outputs", exist_ok=True)
        self._log_file = "outputs/bank_history.jsonl"

    def observe(
        self,
        target_concept: int,
        action: Action,
        tool_concept: int | None,
        outcome: Outcome,
        *,
        episode: int,
        t: int,
        object_id: str,
        tool_id: str | None = None,
    ) -> Revision | None:
        """Fold one interaction's outcome into the relevant belief.

        Returns a `Revision` iff this observation moved the belief's status
        into or out of `CONFIRMED` or `REFUTED`; returns `None` for evidence
        that does not change what a consumer of `confirmed()` would see.
        """
        key = (target_concept, action, tool_concept)
        belief = self._beliefs.setdefault(key, Belief(key=key))
        old_status = belief.status

        success = outcome.changed_world
        belief.effect_counts[outcome.effect] = (
            belief.effect_counts.get(outcome.effect, 0) + 1
        )
        if success:
            belief.alpha += 1.0
            if outcome.had_remote_effect:
                belief.remote_alpha += 1.0
            else:
                belief.remote_beta += 1.0
        else:
            belief.beta += 1.0
        belief.total_attempts += 1
        belief.evidence.append(
            Evidence(episode, t, object_id, tool_id, outcome.effect, outcome.force_required)
        )
        lo, hi = belief.credible_interval
        belief.interval_history.append(hi - lo)

        new_status = belief.status
        
        self.check_recalibration()
        
        # Logging hook
        lo, hi = belief.credible_interval
        log_entry = {
            "key": (key[0], key[1].name, key[2]),
            "mean": belief.mean,
            "credible_interval": (lo, hi),
            "status": new_status.name,
            "total_attempts": belief.total_attempts,
            "timestamp": time.time(),
        }
        with open(self._log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if new_status == old_status:
            return None
        if old_status not in _SETTLED and new_status not in _SETTLED:
            return None
        return Revision(key, episode, t, old_status, new_status)

    def belief(self, key: tuple[int, Action, int | None]) -> Belief | None:
        return self._beliefs.get(key)

    def beliefs_for_concept(self, concept_id: int) -> tuple[Belief, ...]:
        return tuple(b for b in self._beliefs.values() if b.key[0] == concept_id)

    def derive_composite_belief(
        self, composite_concept: int, part_a_concept: int, part_b_concept: int,
        action: Action, tool_concept: int | None
    ) -> Belief:
        """Derive a starting belief for a composite object from its parts."""
        key = (composite_concept, action, tool_concept)
        if key in self._beliefs:
            return self._beliefs[key]
            
        b_a = self._beliefs.get((part_a_concept, action, tool_concept))
        b_b = self._beliefs.get((part_b_concept, action, tool_concept))
        
        alpha = 1.0
        beta = 1.0
        
        if b_a and b_b:
            mean = (b_a.mean + b_b.mean) / 2.0
            alpha = max(1.0, mean * 2.0)
            beta = max(1.0, (1.0 - mean) * 2.0)
        elif b_a:
            alpha = max(1.0, b_a.mean * 2.0)
            beta = max(1.0, (1.0 - b_a.mean) * 2.0)
        elif b_b:
            alpha = max(1.0, b_b.mean * 2.0)
            beta = max(1.0, (1.0 - b_b.mean) * 2.0)
            
        b_composite = Belief(key=key, alpha=alpha, beta=beta)
        self._beliefs[key] = b_composite
        return b_composite

    def beliefs(self) -> tuple[Belief, ...]:
        """Every belief the bank holds, at any status.

        For inspection and evaluation. Planning must use `confirmed()`
        instead, or it is acting on hypotheses rather than knowledge.
        """
        return tuple(self._beliefs.values())

    def confirmed(self) -> tuple[Belief, ...]:
        """Only the knowledge the loop is allowed to act on.

        Everything else -- provisional, refuted, untested -- stays inside the
        bank as working evidence. This is the enforced form of "remember:
        write only confirmed knowledge": a consumer that wants knowledge
        reaches it through this method, never through `beliefs()`.
        """
        return tuple(b for b in self._beliefs.values() if b.is_confirmed)

    def refuted(self) -> tuple[Belief, ...]:
        return tuple(b for b in self._beliefs.values() if b.status is Status.REFUTED)

    def __len__(self) -> int:
        return len(self._beliefs)

    def check_recalibration(self) -> None:
        """Trigger recalibrate if average credible width of provisional beliefs stays stuck."""
        provisional_beliefs = [b for b in self._beliefs.values() if b.status == Status.PROVISIONAL]
        if len(provisional_beliefs) > 5:
            avg_width = sum((b.credible_interval[1] - b.credible_interval[0]) for b in provisional_beliefs) / len(provisional_beliefs)
            if avg_width > 0.4:
                self.recalibrate()
                
    def recalibrate(self) -> None:
        """Reset or loosen priors for stuck/provisional beliefs to encourage re-exploration."""
        for b in self._beliefs.values():
            if b.status in (Status.PROVISIONAL, Status.STUCK):
                # Loosen the prior by pulling it back towards 1,1
                b.alpha = (b.alpha + 1.0) / 2.0
                b.beta = (b.beta + 1.0) / 2.0
