"""Environment layer: the contract, and the backends that implement it.

Downstream modules import from here and never from a concrete backend, so that
a second backend (Isaac Lab) can be added without touching anything that
consumes the environment. See D4 in docs/DECISIONS.md.
"""

from .actions import (
    ALL_ACTIONS,
    INTERACTION_ACTIONS,
    N_ACTIONS,
    Action,
    ActionSpec,
    cost,
    is_relational,
    spec,
)

__all__ = [
    "Action",
    "ActionSpec",
    "ALL_ACTIONS",
    "INTERACTION_ACTIONS",
    "N_ACTIONS",
    "cost",
    "is_relational",
    "spec",
]
