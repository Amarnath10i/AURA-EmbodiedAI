"""Building the appearance descriptor an agent actually sees.

The descriptor is the agent's only route to a prior about what an object is
for, so its informativeness is a research parameter, not an implementation
detail. Too informative and the prior becomes a lookup table and there is
nothing to verify; too noisy and every hypothesis is equally uncertain and
selection has nothing to prefer. The catalogue's look-alike pairs sit
deliberately at the edge: distinguishable in principle, unreliable in practice.

Channel layout (`APPEARANCE_DIM` = 16):

===========  ==========================================================
 0 -  2      colour, normalised to `[0, 1]`
 3 -  5      width, depth, height in metres, normalised
 6           visible open configuration, 0 closed / 1 open
 7           visible toppled state, 0 upright / 1 fallen
 8 - 15      surface and shape cues from the kind's texture signature
===========  ==========================================================

Channels 6 and 7 are why preconditions like `TARGET_CLOSED` are learnable at
all: the state is genuinely visible, but it arrives mixed into a vector rather
than as a labelled boolean, so the agent has to work out which channel means
what and that it gates anything. See the module docstring of `types.py`.
"""

from __future__ import annotations

import numpy as np

from .catalogue import KINDS, TEXTURE_DIM, KindSpec
from .types import APPEARANCE_DIM

#: Largest object height in the catalogue, used to normalise size channels.
SIZE_SCALE: float = 2.6

#: Default per-instance observation noise, in descriptor units.
#:
#: Calibrated against the look-alike pairs: at this level `crate` and `block`
#: differ by roughly one standard deviation on a single texture channel, so
#: appearance predicts the kind well above chance but nowhere near reliably.
#: Raising it makes vision useless and every hypothesis equally uncertain;
#: lowering it makes verification unnecessary. Either extreme deletes the
#: research question, so treat this as a tuned quantity and report it.
DEFAULT_NOISE: float = 0.05

_COLOR = slice(0, 3)
_SIZE = slice(3, 6)
_OPEN = 6
_TOPPLED = 7
_TEXTURE = slice(8, 8 + TEXTURE_DIM)


def prototype(kind: str, *, is_open: bool = False, toppled: bool = False) -> np.ndarray:
    """The noise-free descriptor for `kind`.

    Useful for analysis and for measuring how confusable two kinds are; the
    agent never sees this, because every real observation carries noise.
    """
    return _assemble(KINDS[kind], is_open=is_open, toppled=toppled)


def describe(
    spec: KindSpec,
    rng: np.random.Generator,
    *,
    is_open: bool = False,
    toppled: bool = False,
    noise: float = DEFAULT_NOISE,
) -> np.ndarray:
    """One noisy observation of an object of kind `spec`.

    Noise is applied per observation rather than per object, so looking twice
    gives two slightly different readings. That matters: an agent must not be
    able to resolve a look-alike pair just by staring at it, or verification
    becomes optional.

    Channels near 0 or 1 (colours close to black or white, texture cues near
    the extremes) get asymmetrically clipped, which biases their long-run
    average slightly away from the true prototype -- on the order of 0.03 in
    descriptor distance even after thousands of observations. This is small
    relative to `concepts.DEFAULT_MERGE_RADIUS` and does not change which
    kinds merge, so it is left as is rather than replaced with a
    bias-corrected noise model.
    """
    a = _assemble(spec, is_open=is_open, toppled=toppled)
    if noise > 0.0:
        a = a + rng.normal(0.0, noise, size=APPEARANCE_DIM).astype(np.float32)
    return np.clip(a, 0.0, 1.0, out=a)


def _assemble(spec: KindSpec, *, is_open: bool, toppled: bool) -> np.ndarray:
    a = np.zeros(APPEARANCE_DIM, dtype=np.float32)
    a[_COLOR] = np.asarray(spec.color, dtype=np.float32) / 255.0
    a[_SIZE] = (
        np.asarray([*spec.extent, spec.height], dtype=np.float32) / SIZE_SCALE
    )
    a[_OPEN] = float(is_open) if spec.articulated else 0.0
    a[_TOPPLED] = float(toppled)
    a[_TEXTURE] = np.asarray(spec.texture, dtype=np.float32)
    return a


def separation(kind_a: str, kind_b: str) -> float:
    """Noise-free distance between two kinds' descriptors.

    Compare against `DEFAULT_NOISE` to judge how hard a pair is to tell apart:
    a separation near the noise level means appearance alone cannot settle it.
    """
    return float(np.linalg.norm(prototype(kind_a) - prototype(kind_b)))


def confusability(
    kind_a: str, kind_b: str, *, noise: float = DEFAULT_NOISE, trials: int = 4000,
    seed: int = 0,
) -> float:
    """Fraction of noisy observations of `kind_a` closer to `kind_b`'s prototype.

    A direct read on how much work verification has to do for this pair: 0.0
    means appearance settles it, 0.5 means appearance says nothing at all.

    Exact ties count as half. Kinds with identical descriptors are equidistant
    from both prototypes by construction, and a strict comparison would score
    that perfectly separable when it is in fact a coin flip.
    """
    rng = np.random.default_rng(seed)
    spec, pa, pb = KINDS[kind_a], prototype(kind_a), prototype(kind_b)
    obs = np.stack([describe(spec, rng, noise=noise) for _ in range(trials)])
    to_a = np.linalg.norm(obs - pa, axis=1)
    to_b = np.linalg.norm(obs - pb, axis=1)
    ties = np.isclose(to_a, to_b)
    return float((to_b < to_a).mean() + 0.5 * ties.mean())
