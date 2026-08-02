"""The lifelong affordance bank: confirmed knowledge, held with calibrated
confidence, and revised when contradicted.

Keys are `(concept_id, action, tool_concept_id | None)` rather than
`(kind, action, tool_kind)` -- the agent never has kind, only the concepts it
has formed from appearance (see `concepts.py`). This has a direct consequence
worth stating plainly: **a belief keyed by a concept that conflates two true
kinds (crate and block look identical, so they land in one concept) will
settle on an intermediate reliability that is correct for the concept and
wrong for either kind alone.** Left alone, the bank cannot fix this -- it
only has what the concept codebook gives it, and that is by design (D2/D5 in
docs/DECISIONS.md). What it CAN do is notice: `force_required` (see
`env/outcomes.py`) gives every belief a second, continuous evidence channel
alongside plain success/failure, and `Belief.status` promotes a belief to
`STUCK` when that channel looks bimodal -- the statistical signature of two
different real kinds sharing one concept. `AffordanceMemory.observe`
(memory.py) reacts to `STUCK` by asking the concept codebook to split.

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

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from ..env import Action, Effect, NULL_EFFECTS, Outcome


class Status(Enum):
    """What the bank currently believes about one `(concept, verb, tool)`."""

    UNTESTED = auto()     # never attempted
    PROVISIONAL = auto()  # attempted, but not enough evidence to settle
    CONFIRMED = auto()    # confident this is a real, working affordance
    REFUTED = auto()       # confident this does not work
    STUCK = auto()         # evidence looks bimodal: probably two blended kinds


#: Posterior mass below which the bank calls an affordance not real.
REAL_EPS: float = 0.10

#: Maximum 95%-credible-interval width to call a belief settled rather than
#: merely provisional. See the module docstring for what this implies in
#: trials.
MAX_SETTLED_WIDTH: float = 0.30

#: z-score for the credible interval (~95%, normal approximation to Beta).
_Z: float = 1.96

#: Evidence pointers kept per belief, for provenance without unbounded memory
#: growth at lifelong interaction volumes (10^4-10^5 interactions). Also the
#: window the bimodality check below draws on.
_EVIDENCE_CAP: int = 20

#: Minimum number of force_required samples before the bimodality check runs
#: at all -- a handful of points cannot distinguish "two populations" from
#: "one population with noise", so below this the belief stays PROVISIONAL
#: rather than risking a false STUCK verdict.
_MIN_EVIDENCE_FOR_SPLIT: int = 6

#: How much bigger the largest gap between adjacent sorted force_required
#: values must be than the spread within either side of that gap, to call the
#: evidence bimodal rather than a single noisy cluster. A classic gap
#: statistic: two real populations leave a wide empty gap between them; noise
#: within one population does not.
_BIMODALITY_GAP_RATIO: float = 2.0

#: Statuses that count as "settled": budget stops going toward them (see
#: `verification/select.py`), and a status change into or out of this set is
#: what makes a `Revision` worth reporting. Public and imported directly by
#: select.py rather than re-declared there, so the two can never quietly
#: diverge on what "settled" means -- exactly the bug that shipped once
#: already (STUCK missing from a separately-maintained copy).
SETTLED_STATUSES = frozenset({Status.CONFIRMED, Status.REFUTED, Status.STUCK})


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
        remote_target_counts: How often each `(remote concept, remote effect)`
            pair was observed, conditioned on a remote effect happening at
            all. See `dominant_remote` -- this is what lets a backward-chaining
            planner treat "press this to open that" as a fact it can chain
            through, not just a probability that something-or-other happens.
        irreversible_alpha, irreversible_beta: Beta posterior, conditioned on
            success, over "this change cannot be undone". Starts at the
            uninformed Beta(1, 1) prior deliberately: a verb nobody has tried
            yet is not assumed safe, so a first attempt is naturally
            discounted rather than treated as free (see
            `verification/select.py`), and the discount relaxes quickly once
            evidence shows it usually is not.
        evidence: A bounded recent sample, for provenance, debugging, and the
            bimodality check.
        total_attempts: True count, even once `evidence` has been truncated.
    """

    key: tuple[int, Action, int | None]
    alpha: float = 1.0
    beta: float = 1.0
    effect_counts: dict[Effect, int] = field(default_factory=dict)
    remote_alpha: float = 1.0
    remote_beta: float = 1.0
    remote_target_counts: dict[tuple[int | None, Effect], int] = field(default_factory=dict)
    irreversible_alpha: float = 1.0
    irreversible_beta: float = 1.0
    evidence: deque = field(default_factory=lambda: deque(maxlen=_EVIDENCE_CAP))
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
    def irreversible_rate(self) -> float:
        """Estimated probability that a success cannot be undone."""
        return self.irreversible_alpha / (self.irreversible_alpha + self.irreversible_beta)

    @property
    def dominant_effect(self) -> Effect | None:
        """The most frequently observed effect, or `None` with no evidence."""
        if not self.effect_counts:
            return None
        return max(self.effect_counts, key=self.effect_counts.get)

    @property
    def dominant_remote(self) -> tuple[int | None, Effect] | None:
        """The most common `(remote concept, remote effect)` this belief's
        successes have caused, or `None` if none ever have."""
        if not self.remote_target_counts:
            return None
        return max(self.remote_target_counts, key=self.remote_target_counts.get)

    def _looks_bimodal(self) -> bool:
        """Whether `force_required` evidence looks like two populations
        rather than one -- the signature of a concept secretly blending two
        real kinds (see the module docstring). A simple gap statistic: find
        the largest gap between adjacent sorted values, and check that gap is
        much wider than the spread on either side of it. Needs
        `_MIN_EVIDENCE_FOR_SPLIT` samples so a handful of noisy trials cannot
        trigger a false split.

        Only successful attempts are considered: an unreliable-but-uniform
        affordance (e.g. a verb that fires 85% of the time but always
        produces the exact same displacement when it does) would otherwise
        look bimodal purely from mixing in failures, whose `force_required`
        is always 0 -- that is aleatoric noise in *whether* it worked, a
        question `alpha`/`beta` already answers, not evidence about *how
        much* it worked, which is the question this check is for.
        """
        forces = sorted(
            e.force_required for e in self.evidence if e.effect not in NULL_EFFECTS
        )
        if len(forces) < _MIN_EVIDENCE_FOR_SPLIT:
            return False
        gaps = [(forces[i + 1] - forces[i], i) for i in range(len(forces) - 1)]
        biggest_gap, split_i = max(gaps)
        if biggest_gap <= 1e-9:
            return False
        low, high = forces[: split_i + 1], forces[split_i + 1:]
        if len(low) < 2 or len(high) < 2:
            return False
        spread = max(low[-1] - low[0], high[-1] - high[0], 1e-6)
        return biggest_gap > _BIMODALITY_GAP_RATIO * spread

    @property
    def status(self) -> Status:
        if self.total_attempts == 0:
            return Status.UNTESTED
        # Checked before CONFIRMED/REFUTED, not after: a belief can be fully
        # confident that a verb "works" (crate/block both always translate
        # under a push) while the MAGNITUDE of what happens is clearly two
        # different populations. That is exactly the case worth flagging --
        # a confident-but-blended CONFIRMED verdict would hide it.
        if self._looks_bimodal():
            return Status.STUCK
        lo, hi = self.credible_interval
        if hi < REAL_EPS:
            return Status.REFUTED
        if lo > REAL_EPS and (hi - lo) <= MAX_SETTLED_WIDTH:
            return Status.CONFIRMED
        return Status.PROVISIONAL

    @property
    def is_confirmed(self) -> bool:
        return self.status is Status.CONFIRMED

    @property
    def operator(self) -> dict | None:
        """A human-readable summary of this belief as a STRIPS-style
        operator, for logging and inspection. `planner.py` does NOT parse
        this -- it builds `planner.Operator` directly from this belief's
        typed fields, since round-tripping through an untyped dict would
        throw away the precondition detail a real backward-chaining search
        needs.
        """
        if not self.is_confirmed or self.dominant_effect is None:
            return None
        remote = self.dominant_remote
        return {
            "action": self.key[1].name,
            "target_concept": self.key[0],
            "tool_concept": self.key[2],
            "effect": self.dominant_effect.name,
            "reliability": round(self.mean, 3),
            "remote_effect": remote[1].name if remote else None,
            "remote_target_concept": remote[0] if remote else None,
        }


@dataclass(frozen=True)
class Revision:
    """A belief's status crossed into or out of a settled status.

    This is the event the "keep only confirmed knowledge, revise on
    contradiction" half of the loop hangs off. Not every new piece of
    evidence produces one -- only evidence that actually changes what a
    downstream consumer of `AffordanceBank.confirmed()` would see, or that
    triggers a concept split (`new_status is Status.STUCK`).
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
        remote_concepts: tuple[int | None, ...] = (),
    ) -> Revision | None:
        """Fold one interaction's outcome into the relevant belief.

        `remote_concepts` gives the concept `AffordanceMemory` resolved for
        each entry in `outcome.remote`, in the same order -- this belongs to
        the caller because resolving an appearance to a concept is a
        `ConceptCodebook` operation, not something the bank can do itself.

        Returns a `Revision` iff this observation moved the belief's status
        into or out of a settled status (`CONFIRMED`, `REFUTED`, `STUCK`);
        returns `None` for evidence that does not change what a consumer of
        `confirmed()` would see.
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
                for i, remote_effect in enumerate(outcome.remote):
                    rc = remote_concepts[i] if i < len(remote_concepts) else None
                    rkey = (rc, remote_effect.effect)
                    belief.remote_target_counts[rkey] = (
                        belief.remote_target_counts.get(rkey, 0) + 1
                    )
            else:
                belief.remote_beta += 1.0
            if outcome.irreversible:
                belief.irreversible_alpha += 1.0
            else:
                belief.irreversible_beta += 1.0
        else:
            belief.beta += 1.0
        belief.total_attempts += 1
        belief.evidence.append(
            Evidence(episode, t, object_id, tool_id, outcome.effect, outcome.force_required)
        )

        new_status = belief.status
        if new_status == old_status:
            return None
        if old_status not in SETTLED_STATUSES and new_status not in SETTLED_STATUSES:
            return None
        return Revision(key, episode, t, old_status, new_status)

    def belief(self, key: tuple[int, Action, int | None]) -> Belief | None:
        return self._beliefs.get(key)

    def beliefs_for_concept(self, concept_id: int) -> tuple[Belief, ...]:
        """Every belief keyed on `concept_id` as the target, any verb/tool.

        Used to estimate how much is still unknown about a concept as a
        whole -- exploration and active-target selection ask this, the bank
        itself never needs to.
        """
        return tuple(b for b in self._beliefs.values() if b.key[0] == concept_id)

    def concept_uncertainty(self, concept_id: int | None) -> float:
        """How much, on average, is still unknown about `concept_id` as a
        whole -- the mean credible-interval width across its unsettled
        beliefs, or 1.0 (maximal) if the concept has never been matched at
        all or has no unsettled beliefs of its own yet.

        The single home for a computation that otherwise gets duplicated
        wherever something wants to prioritise "which object would teach me
        the most right now" -- imagination's relevance scoring and active
        exploration's target selection both ask this, rather than each
        re-deriving it.
        """
        if concept_id is None:
            return 1.0
        widths = [
            b.credible_interval[1] - b.credible_interval[0]
            for b in self.beliefs_for_concept(concept_id)
            if b.status not in SETTLED_STATUSES
        ]
        return sum(widths) / len(widths) if widths else 0.0

    def reopen(self, key: tuple[int, Action, int | None], *, alpha: float = 1.5,
              beta: float = 1.5) -> Belief:
        """Seed a fresh `PROVISIONAL` belief at `key`, overwriting whatever
        was there.

        The public way to do what `AffordanceMemory` needs after a concept
        split: the old, blended belief is no longer trustworthy evidence for
        either resulting concept, so each gets a clean, mildly-informative
        restart rather than reaching into `_beliefs` directly.
        """
        belief = Belief(key=key, alpha=alpha, beta=beta, total_attempts=2)
        self._beliefs[key] = belief
        return belief

    def beliefs(self) -> tuple[Belief, ...]:
        """Every belief the bank holds, at any status.

        For inspection and evaluation. Planning must use `confirmed()`
        instead, or it is acting on hypotheses rather than knowledge.
        """
        return tuple(self._beliefs.values())

    def confirmed(self) -> tuple[Belief, ...]:
        """Only the knowledge the loop is allowed to act on.

        Everything else -- provisional, refuted, stuck, untested -- stays
        inside the bank as working evidence. This is the enforced form of
        "remember: write only confirmed knowledge": a consumer that wants
        knowledge reaches it through this method, never through `beliefs()`.
        """
        return tuple(b for b in self._beliefs.values() if b.is_confirmed)

    def refuted(self) -> tuple[Belief, ...]:
        return tuple(b for b in self._beliefs.values() if b.status is Status.REFUTED)

    def stuck(self) -> tuple[Belief, ...]:
        return tuple(b for b in self._beliefs.values() if b.status is Status.STUCK)

    def __len__(self) -> int:
        return len(self._beliefs)
