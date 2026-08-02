"""Appearance-based concept formation: the agent's own stand-in for "kind".

The agent never observes an object's hidden kind (D2/D5 in
docs/DECISIONS.md). For memory to be *lifelong* it still needs some identity
that survives across episodes, and object ids do not -- they are scoped to a
single episode by design (see `env/types.py`). The only thing that survives
episode boundaries is what an object looked like, so a concept is formed by
grouping similar appearance vectors as they are observed, across the agent's
whole run.

This is deliberately the simplest thing that works: online leader clustering.
Assign each observation to its nearest existing concept if close enough,
otherwise start a new one, and keep a running mean. It is a placeholder for a
learned representation -- once `perception/` exists it may replace this -- but
the appearance descriptor already carries everything currently observable, so
a heavier model would not change what is fundamentally recoverable from it.

**This is where the crate/block trap resurfaces.** Their appearance is
identical by construction (`catalogue.py`), so they are guaranteed to land in
the same concept, forever. Any confirmed belief the bank forms about a block
will therefore be applied to the next crate the agent meets, and be wrong
there. That is not a bug to engineer around in this module -- it is the actual
difficulty a lifelong affordance bank has to cope with, and `bank.py` is
written with it in mind rather than against it.

`split_concept` (called when `bank.py` detects a concept's interaction
evidence looks bimodal) is the exception worth being precise about. It uses
recent appearance evidence to find a genuine separating direction where one
exists -- which helps the OTHER two catalogue traps, lever/switch and
barrel/drum, whose true appearance separation is small (~0.10) but nonzero.
For crate/block specifically, whose true separation is exactly zero, no
algorithm can recover a signal that is not there: `split_concept` still runs
and still helps (it lets each half accumulate independent evidence going
forward), but which physical object lands in which half is then arbitrary.
That is the honest limit of appearance-only recognition, not a defect in this
method -- see docs/DECISIONS.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..env.types import APPEARANCE_DIM

#: Default merge radius, in appearance-descriptor units.
#:
#: Calibrated by simulation against `appearance.DEFAULT_NOISE`. The risk this
#: radius has to manage is at the *start* of a concept's life: the first
#: observation seeds its mean outright, so the second observation is compared
#: against a single noisy sample rather than a stabilised average, and that is
#: when two draws of the same kind are furthest apart in expectation. Over
#: 200000 simulated pairs, two independent same-kind draws exceed 0.48 only
#: 0.0025% of the time (about 0.04% odds of any collision across all 17
#: catalogue kinds). The closest two kinds that are NOT meant to be confused
#: sit 0.57 apart (see `catalogue.py`'s pairwise separation check), so 0.48
#: keeps a real margin on both sides. The deliberate look-alike pairs
#: (crate/block at 0.0 separation; lever/switch and barrel/drum near 0.10) sit
#: far inside this radius and are *expected* to merge -- that is the trap
#: working as designed, not a calibration failure.
DEFAULT_MERGE_RADIUS: float = 0.48

#: Recent raw observations kept per concept, purely to give `split_concept`
#: real data to find a separating direction in. Bounded so this does not grow
#: without limit at lifelong interaction volumes; large enough that the
#: direction-finding SVD below has something meaningful to work with.
_RECENT_CAP: int = 30

#: Minimum recent observations before attempting to find a real separating
#: direction; below this, fall back to a fixed, deterministic pseudo-random
#: direction with a negligible offset (see `split_concept`).
_MIN_FOR_DIRECTION: int = 4


@dataclass
class Concept:
    """One appearance cluster the agent has formed.

    Attributes:
        concept_id: Stable for the lifetime of the codebook -- across every
            episode it has been used for, not just one.
        mean: Running-mean appearance descriptor of every observation merged
            into this concept.
        count: Number of observations merged.
        first_seen: `(episode, t)` of the observation that created it.
        recent: A bounded sample of raw observations, for `split_concept`.
    """

    concept_id: int
    mean: np.ndarray
    count: int
    first_seen: tuple[int, int]
    recent: deque = field(default_factory=lambda: deque(maxlen=_RECENT_CAP))


class ConceptCodebook:
    """Online, appearance-only clustering that survives across episodes.

    This is the closest thing the agent has to a notion of "kind", and it is
    built entirely from what is visible. Two objects merge into the same
    concept exactly when their appearance is close enough that the agent could
    not reliably tell them apart -- nothing more, nothing less.

    Args:
        merge_radius: See `DEFAULT_MERGE_RADIUS`.
        seed: Drives `split_concept`'s fallback direction when there is not
            enough recent evidence to find a real one. Fixed by default so
            that, matching every other stateful piece of this project,
            identical (seed, observation sequence) produces identical
            concepts.
    """

    def __init__(
        self, merge_radius: float = DEFAULT_MERGE_RADIUS, seed: int = 0
    ) -> None:
        self.merge_radius = merge_radius
        self._concepts: list[Concept] = []
        self._rng = np.random.default_rng(seed)

    def assign(self, appearance: np.ndarray, episode: int, t: int) -> int:
        """Return the concept id for one observed appearance descriptor.

        Merges into the nearest existing concept within `merge_radius`, or
        creates a new one. Order-dependent by construction: an online
        clustering cannot revisit a decision once evidence for a better split
        arrives later. That is a real limitation of this first implementation,
        worth knowing about rather than hiding.
        """
        self._check_shape(appearance)
        best_id, best_dist = self._nearest(appearance)
        if best_id is not None and best_dist <= self.merge_radius:
            self._merge(best_id, appearance)
            return best_id
        return self._new_concept(appearance, episode, t)

    def peek(self, appearance: np.ndarray) -> int | None:
        """Nearest existing concept for `appearance`, without creating or
        updating anything.

        The read-only counterpart to `assign`: for asking what is already
        known about an object without the act of looking changing what the
        codebook remembers. `imagination` uses this so that merely observing
        an object -- as opposed to actually interacting with it -- never
        forms or perturbs a concept. Returns `None` when nothing seen so far
        is close enough to match, which is the correct answer for "this looks
        like nothing I have ever interacted with".
        """
        self._check_shape(appearance)
        best_id, best_dist = self._nearest(appearance)
        if best_id is not None and best_dist <= self.merge_radius:
            return best_id
        return None

    def split_concept(self, concept_id: int) -> tuple[int, int]:
        """Split a concept into two, using recent appearance evidence to find
        a genuine separating direction where one exists.

        For a pair with real (if subtle) appearance separation -- lever/switch
        or barrel/drum, both ~0.10 apart, see catalogue.py -- this projects
        recent observations onto their direction of largest spread and
        bisects there, recovering the separation the merge radius smoothed
        over. For a pair with IDENTICAL appearance -- crate/block, the
        flagship trap -- there is no direction to find: recent observations
        are noise around one point in every direction, so the offset degrades
        to negligible and the two resulting concepts start equivalent. That
        is not a bug in this method; it is the honest limit of what
        appearance alone can ever recover (see the module docstring).
        Splitting still helps in that case, because `reopen()`-ing fresh
        beliefs (`memory.py`) lets the two halves accumulate independent
        evidence going forward, even though which physical object lands in
        which half is then arbitrary.
        """
        old_c = self._concepts[concept_id]
        direction, scale = self._separating_direction(old_c)
        offset = direction * scale

        id_a = len(self._concepts)
        self._concepts.append(
            Concept(id_a, old_c.mean + offset, max(1, old_c.count // 2), old_c.first_seen)
        )
        id_b = len(self._concepts)
        self._concepts.append(
            Concept(id_b, old_c.mean - offset, max(1, old_c.count // 2), old_c.first_seen)
        )

        # Retire the old concept rather than removing it, so existing ids
        # elsewhere stay valid indices: moving its mean to infinity means it
        # can never again be `_nearest` to anything.
        old_c.mean = np.full_like(old_c.mean, np.inf)
        return id_a, id_b

    def _separating_direction(self, c: Concept) -> tuple[np.ndarray, float]:
        """Unit vector along the largest spread in `c`'s recent observations,
        and how far to offset along it. Falls back to a fixed pseudo-random
        direction with a negligible offset when there is not enough recent
        data -- deterministic given this codebook's seed, never the global
        numpy random state (see `__init__`'s `seed` argument)."""
        if len(c.recent) >= _MIN_FOR_DIRECTION:
            points = np.stack(c.recent) - c.mean
            try:
                _, singular_values, vt = np.linalg.svd(points, full_matrices=False)
            except np.linalg.LinAlgError:
                singular_values = np.zeros(1)
            if singular_values[0] > 1e-9:
                direction = vt[0].astype(np.float32)
                scale = float(singular_values[0] / np.sqrt(len(points)))
                return direction, max(scale, 1e-3)

        direction = self._rng.normal(size=c.mean.shape[0]).astype(np.float32)
        direction = direction / (float(np.linalg.norm(direction)) + 1e-9)
        return direction, 1e-3

    def _check_shape(self, appearance: np.ndarray) -> None:
        if appearance.shape != (APPEARANCE_DIM,):
            raise ValueError(
                f"appearance must have shape ({APPEARANCE_DIM},), "
                f"got {appearance.shape}"
            )

    def _nearest(self, appearance: np.ndarray) -> tuple[int | None, float]:
        best_id, best_dist = None, np.inf
        for c in self._concepts:
            d = float(np.linalg.norm(c.mean - appearance))
            if d < best_dist:
                best_id, best_dist = c.concept_id, d
        return best_id, best_dist

    def _merge(self, concept_id: int, appearance: np.ndarray) -> None:
        c = self._concepts[concept_id]  # ids are dense indices; see _new_concept
        c.count += 1
        c.mean = c.mean + (appearance - c.mean) / c.count
        c.recent.append(appearance.astype(np.float32).copy())

    def _new_concept(self, appearance: np.ndarray, episode: int, t: int) -> int:
        concept_id = len(self._concepts)
        c = Concept(concept_id, appearance.astype(np.float32).copy(), 1, (episode, t))
        c.recent.append(appearance.astype(np.float32).copy())
        self._concepts.append(c)
        return concept_id

    def __len__(self) -> int:
        return len(self._concepts)

    def concept(self, concept_id: int) -> Concept:
        """The concept currently registered under `concept_id`."""
        return self._concepts[concept_id]

    def concepts(self) -> tuple[Concept, ...]:
        return tuple(self._concepts)
