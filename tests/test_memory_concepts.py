"""Tests for appearance-based concept formation.

The property that matters most is not "clustering works" in the abstract, but
that it reproduces the exact confusability structure the catalogue was built
to have: the three look-alike pairs merge, and nothing else does.
"""

from __future__ import annotations

import numpy as np
import pytest

from lama.env.appearance import DEFAULT_NOISE, describe
from lama.env.catalogue import KINDS
from lama.env.types import APPEARANCE_DIM
from lama.memory.concepts import ConceptCodebook

LOOK_ALIKE_PAIRS = (("crate", "block"), ("lever", "switch"), ("barrel", "drum"))


def _draws(kind: str, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [describe(KINDS[kind], rng, noise=DEFAULT_NOISE) for _ in range(n)]


def test_rejects_wrong_shaped_appearance():
    with pytest.raises(ValueError):
        ConceptCodebook().assign(np.zeros(3, dtype=np.float32), 0, 0)


def test_repeated_identical_observations_share_one_concept():
    cb = ConceptCodebook()
    a = describe(KINDS["cup"], np.random.default_rng(0), noise=0.0)
    ids = {cb.assign(a, 0, t) for t in range(10)}
    assert ids == {cb.assign(a, 0, 0)}
    assert cb.concept(next(iter(ids))).count == 11


def test_first_observation_seeds_the_concept_mean():
    cb = ConceptCodebook()
    a = np.full(APPEARANCE_DIM, 0.3, dtype=np.float32)
    cid = cb.assign(a, episode=2, t=7)
    c = cb.concept(cid)
    assert c.count == 1
    assert c.first_seen == (2, 7)
    assert np.array_equal(c.mean, a)


@pytest.mark.parametrize("kind_a,kind_b", LOOK_ALIKE_PAIRS)
def test_look_alike_pairs_merge_into_one_concept(kind_a, kind_b):
    """The traps the catalogue was built around must actually trap."""
    rng = np.random.default_rng(1)
    cb = ConceptCodebook()
    draws = [(kind_a, d) for d in _draws(kind_a, 60, rng)] + \
            [(kind_b, d) for d in _draws(kind_b, 60, rng)]
    rng.shuffle(draws)
    ids = {cb.assign(d, 0, t) for t, (_, d) in enumerate(draws)}
    assert len(ids) == 1, f"{kind_a}/{kind_b} should be indistinguishable"


def test_all_kinds_together_collapse_to_exactly_the_trap_pairs():
    """End-to-end: every kind, interleaved and noisy, should yield exactly
    three fewer concepts than kinds -- one merge per trap pair."""
    rng = np.random.default_rng(2)
    cb = ConceptCodebook()
    draws = []
    for kind in KINDS:
        draws += [(kind, d) for d in _draws(kind, 80, rng)]
    rng.shuffle(draws)

    assigned: dict[str, set[int]] = {}
    for t, (kind, appearance) in enumerate(draws):
        cid = cb.assign(appearance, 0, t)
        assigned.setdefault(kind, set()).add(cid)

    for kind, ids in assigned.items():
        assert len(ids) == 1, f"{kind} fragmented into {len(ids)} concepts"
    for a, b in LOOK_ALIKE_PAIRS:
        assert assigned[a] == assigned[b], f"{a}/{b} did not merge"
    non_trap_ids = {
        next(iter(ids)) for kind, ids in assigned.items()
        if kind not in {k for pair in LOOK_ALIKE_PAIRS for k in pair}
    }
    trap_ids = {next(iter(assigned[a])) for a, _ in LOOK_ALIKE_PAIRS}
    assert len(non_trap_ids) == len(KINDS) - 2 * len(LOOK_ALIKE_PAIRS)
    assert not non_trap_ids & trap_ids
    assert len(cb) == len(KINDS) - len(LOOK_ALIKE_PAIRS)


def test_distinct_looking_kinds_never_merge():
    rng = np.random.default_rng(3)
    cb = ConceptCodebook()
    ids_a = {cb.assign(a, 0, t) for t, a in enumerate(_draws("cup", 40, rng))}
    ids_b = {cb.assign(b, 0, t) for t, b in enumerate(_draws("pillar", 40, rng))}
    assert ids_a != ids_b
    assert not ids_a & ids_b


def test_concept_ids_are_dense_and_stable_across_episodes():
    """Lifelong identity: unlike object ids, a concept id must keep meaning
    the same thing in episode 5 that it meant in episode 0."""
    cb = ConceptCodebook()
    rng = np.random.default_rng(4)
    cid_ep0 = cb.assign(describe(KINDS["cup"], rng), episode=0, t=3)
    cid_ep5 = cb.assign(describe(KINDS["cup"], rng), episode=5, t=1)
    assert cid_ep0 == cid_ep5


def test_running_mean_converges_toward_the_true_prototype():
    """Converges to within the small residual bias boundary clipping leaves
    (see `appearance.describe`), not to zero -- 500 draws already sit at that
    floor, so more draws should not move the error much further."""
    from lama.env.appearance import prototype

    rng = np.random.default_rng(5)
    cb = ConceptCodebook()
    cid = None
    for t, a in enumerate(_draws("toolbox", 500, rng)):
        cid = cb.assign(a, 0, t)
    error_500 = float(np.linalg.norm(cb.concept(cid).mean - prototype("toolbox")))
    assert error_500 < 0.10

    for t, a in enumerate(_draws("toolbox", 4500, rng), start=500):
        cid = cb.assign(a, 0, t)
    error_5000 = float(np.linalg.norm(cb.concept(cid).mean - prototype("toolbox")))
    assert error_5000 < error_500 + 0.02, "should not keep drifting further"
