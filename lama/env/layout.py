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


def plan_kinds(
    rng: np.random.Generator, n_objects: int, include_held_out: bool
) -> list[str]:
    """The kind of each object to place, in shuffled (id-opaque) order.

    Always includes `REQUIRED_KINDS` plus one actuator kind (button, lever or
    valve, so a mechanism is always reachable), then fills the remainder at
    random from the catalogue.

    Raises:
        ValueError: if `n_objects` cannot fit `REQUIRED_KINDS`.
    """
    if n_objects < len(REQUIRED_KINDS):
        raise ValueError(
            f"n_objects must be at least {len(REQUIRED_KINDS)} to fit the "
            f"required kinds {REQUIRED_KINDS}"
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
