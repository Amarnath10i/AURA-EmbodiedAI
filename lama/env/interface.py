"""The backend-agnostic environment contract.

Everything downstream of the environment -- encoder, world model, imagination,
verification, affordance bank, planner, evaluation -- imports from here and
never from a concrete backend. That is what makes a second backend (Isaac Lab)
a swap rather than a rewrite. See D4 in docs/DECISIONS.md.

The module defines two things, and the separation between them is the point:

* `Environment` is what the **agent** may touch. It exposes appearance and
  geometry and nothing semantic.
* `AffordanceOracle` is what **evaluation** may touch. It exposes hidden object
  kinds and the true affordance table. Agent code importing it is a bug, and a
  serious one: it silently invalidates every number the project reports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .affordance import Affordance
from .types import Interaction, Observation, StepResult


class Environment(ABC):
    """A world the agent can perceive and interact with.

    Implementations must guarantee:

    1. **Determinism.** The same `(seed, interaction sequence)` reproduces the
       same trajectory exactly, including stochastic affordance outcomes.
    2. **No semantic leakage.** Nothing returned from `reset` or `step` reveals
       an object's kind or its true affordances -- not in `Observation`, and
       not in `StepResult.info`.
    3. **Total verb coverage.** Every verb in `actions.INTERACTION_ACTIONS` is
       accepted for any target. Attempts that cannot succeed return an outcome
       saying so; they never raise. Learning that a verb fails is knowledge the
       agent paid for, and refusing the attempt would hand it that knowledge
       for free.
    4. **Honest accounting.** Every attempt consumes budget, including failed
       ones, and `StepResult.record.cost` reports what was actually spent.
    """

    #: Identifies the backend in stored results, so a number can always be
    #: traced to the world that produced it.
    backend_name: str = "abstract"

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        """Start a new episode and return the first observation."""

    @abstractmethod
    def step(self, interaction: Interaction) -> StepResult:
        """Attempt `interaction` and return what happened.

        Must not raise for interactions that are merely doomed -- pushing a
        wall, lifting something too heavy, placing with an empty gripper. Those
        return an outcome of `NOTHING` or `BLOCKED`. Raising is reserved for
        malformed input, such as a target that does not exist.
        """

    @property
    @abstractmethod
    def episode(self) -> int:
        """Index of the current episode."""

    @property
    @abstractmethod
    def t(self) -> int:
        """Timestep within the current episode."""

    @property
    @abstractmethod
    def budget_remaining(self) -> float:
        """Interaction-budget units left in this episode."""

    def render(self) -> np.ndarray | None:
        """A top-down RGB view of the world, for videos and debugging.

        Optional. Backends that cannot render return `None` rather than
        raising, so that logging code need not special-case them.
        """
        return None

    def close(self) -> None:
        """Release any backend resources. Safe to call more than once."""

    def __enter__(self) -> "Environment":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


@runtime_checkable
class AffordanceOracle(Protocol):
    """Hidden ground truth, for evaluation only.

    **Agent code must never import or call this.** Every metric the project
    reports depends on the agent having discovered affordances rather than
    read them, and there is no way to detect the difference after the fact
    from the numbers alone.

    A backend that cannot answer these questions can still run an agent; it
    simply cannot score one. Keeping the oracle a separate protocol rather than
    methods on `Environment` is what makes that possible, and makes an
    accidental call visibly wrong at the import line.
    """

    def object_kind(self, object_id: str) -> str:
        """The hidden kind of an object present in the current episode."""
        ...

    def catalogue_affordances(self) -> tuple[Affordance, ...]:
        """Every affordance the world defines, across all object kinds."""
        ...

    def reachable_affordances(self) -> tuple[Affordance, ...]:
        """Affordances that are actually testable in the current episode.

        The recall denominator, and it is not the full catalogue. An affordance
        whose object is absent from this layout, or whose required tool is
        absent, could not have been discovered however good the agent was.
        Scoring against the catalogue instead would penalise agents for the
        layout they were handed.
        """
        ...
