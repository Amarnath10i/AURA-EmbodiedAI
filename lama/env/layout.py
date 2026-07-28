"""Shared object-kind planning, used by every backend.

Deciding WHICH kinds populate an episode, and in what shuffled order (so an
id like `obj_07` never correlates with kind), is entirely backend-independent
-- only WHERE those objects end up physically differs between a 2D grid and a
3D Isaac Lab scene. Keeping this one function shared means every backend is
guaranteed to agree on layout composition, and a bug fixed here once (as with
the `bench`/`pallet` confusability fix, or the object-id opacity fix) can
never silently diverge between backends again.
"""

from __future__ import annotations

import numpy as np

from . import catalogue

#: Kinds that must appear in every layout, so the affordances the project is
#: about are always reachable: a locked door, the plate that can open it, and
#: the look-alike pair that decides whether the agent can.
REQUIRED_KINDS: tuple[str, ...] = ("door", "plate", "block", "crate")


#: The four required kinds plus one mandatory actuator (button, lever or
#: valve, so a mechanism is always reachable) -- the true floor on
#: `n_objects`, one more than `len(REQUIRED_KINDS)` alone.
MIN_OBJECTS: int = len(REQUIRED_KINDS) + 1


def plan_kinds(
    rng: np.random.Generator, n_objects: int, include_held_out: bool
) -> list[str]:
    """The kind of each object to place, in shuffled (id-opaque) order.

    Always includes `REQUIRED_KINDS` plus one actuator kind, then fills the
    remainder at random from the catalogue. Returns exactly `n_objects`
    kinds -- `n_objects` is a count, not a floor that a mandatory addition can
    silently exceed.

    Raises:
        ValueError: if `n_objects` is below `MIN_OBJECTS`.
    """
    if n_objects < MIN_OBJECTS:
        raise ValueError(
            f"n_objects must be at least {MIN_OBJECTS} to fit the required "
            f"kinds {REQUIRED_KINDS} plus one mandatory actuator"
        )
    pool = [
        k for k in catalogue.kinds(include_held_out)
        if k not in ("door", "plate")
    ]
    chosen = list(REQUIRED_KINDS)
    chosen.append(str(rng.choice(["button", "lever", "valve"])))
    while len(chosen) < n_objects:
        chosen.append(str(rng.choice(pool)))
    rng.shuffle(chosen)
    return chosen
