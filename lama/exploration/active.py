"""Active exploration: choosing what to walk toward, not just what to test.

`verification/select.py` decides what to test among objects already in
reach. Something still has to decide where to walk when nothing reachable is
worth testing -- the original fallback in `verification/loop.py` was
`_nearest_unreached`, a name honest about its own limit: it has no opinion
about which unreached object is worth the walk, only which one is closest.

`select_exploration_target` replaces that with the same idea `select.py`
already uses for testing -- expected information gain per unit of cost --
applied to the cost of walking there instead of the cost of a verb. An object
whose concept is broadly uncertain (`AffordanceBank.concept_uncertainty`) is
worth a longer walk than one that is merely close.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..env import Observation
from ..memory.memory import AffordanceMemory

#: Trade-off between information gain and distance cost. Higher values make
#: the explorer more willing to walk further for a more uncertain target;
#: lower values make it prefer whatever is closest almost regardless of
#: uncertainty.
DEFAULT_DISTANCE_WEIGHT: float = 1.0


@dataclass(frozen=True)
class ExplorationTarget:
    """One candidate object to walk toward, and why."""

    object_id: str
    expected_info_gain: float
    distance: float
    score: float


def select_exploration_target(
    observation: Observation,
    memory: AffordanceMemory,
    distance_weight: float = DEFAULT_DISTANCE_WEIGHT,
) -> ExplorationTarget | None:
    """The not-yet-reachable object most worth walking toward right now, by
    expected information gain per unit of distance.

    `None` if nothing visible qualifies -- everything is either already
    reachable, currently held, or (rare) there is nothing visible at all.
    This never checks affordability; `verification/loop.py` decides whether
    the walk is affordable with the budget that remains.
    """
    best: ExplorationTarget | None = None
    for obj in observation.objects:
        if obj.held or obj.within_reach:
            continue
        concept_id = memory.concepts.peek(obj.appearance)
        info_gain = memory.bank.concept_uncertainty(concept_id)
        score = info_gain / (distance_weight * obj.distance + 0.01)
        if best is None or score > best.score:
            best = ExplorationTarget(obj.object_id, info_gain, obj.distance, score)
    return best
