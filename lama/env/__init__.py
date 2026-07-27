"""Environment layer: the contract, and the backends that implement it.

Downstream modules import from here and never from a concrete backend, so that
a second backend (Isaac Lab) can be added without touching anything that
consumes the environment. See D4 in docs/DECISIONS.md.

`AffordanceOracle` is deliberately **not** re-exported here. It is hidden
ground truth for evaluation only, and importing it should require naming
`lama.env.interface` explicitly, so that a call from agent code is obvious in
review rather than buried behind a convenient package import.
"""

from .actions import (
    ALL_ACTIONS,
    INTERACTION_ACTIONS,
    N_ACTIONS,
    SPECS,
    Action,
    ActionSpec,
    cost,
    is_relational,
    spec,
)
from .affordance import Affordance, Precondition, Role
from .interface import Environment
from .outcomes import (
    IRREVERSIBLE_EFFECTS,
    NULL_EFFECTS,
    Effect,
    Outcome,
    RemoteEffect,
    blocked,
    nothing,
)
from .types import (
    APPEARANCE_DIM,
    Interaction,
    InteractionRecord,
    Observation,
    ObjectView,
    StepResult,
)

__all__ = [
    # verbs
    "Action",
    "ActionSpec",
    "ALL_ACTIONS",
    "INTERACTION_ACTIONS",
    "N_ACTIONS",
    "SPECS",
    "cost",
    "is_relational",
    "spec",
    # outcomes
    "Effect",
    "Outcome",
    "RemoteEffect",
    "NULL_EFFECTS",
    "IRREVERSIBLE_EFFECTS",
    "blocked",
    "nothing",
    # hidden ground-truth records
    "Affordance",
    "Role",
    "Precondition",
    # agent-visible data
    "APPEARANCE_DIM",
    "ObjectView",
    "Observation",
    "Interaction",
    "InteractionRecord",
    "StepResult",
    # contract
    "Environment",
]
