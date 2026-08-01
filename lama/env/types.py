"""The data the environment exchanges with the agent.

Everything the agent is allowed to know flows through these structures, and
nothing else does. In particular there is **no object kind anywhere in an
observation**. The agent sees geometry and appearance and must infer the rest;
kinds exist only in hidden ground truth, behind the evaluation oracle.

Visible state is deliberately not given its own symbolic field. Whether a
cabinet is open, whether a crate has toppled, whether a lamp is lit -- all of
that is genuinely visible, so backends encode it inside `ObjectView.appearance`
rather than exposing a boolean. Otherwise preconditions like
`Precondition.TARGET_CLOSED` would be trivially readable instead of something
the agent has to work out, and the hard half of the problem would disappear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .actions import Action, cost, is_relational, spec
from .outcomes import Outcome

#: Width of the per-object appearance descriptor. Backends must emit exactly
#: this many float32 values. It is a visual summary -- colour, apparent size,
#: shape cues, visible configuration -- not a semantic encoding.
APPEARANCE_DIM: int = 16


@dataclass(frozen=True, eq=False)
class ObjectView:
    """One object as the agent perceives it, with no semantic labels.

    Attributes:
        object_id: Stable within an episode, so interactions can be attributed
            to a specific object. Carries no meaning across episodes.
        position: `(2,)` world-frame position in metres.
        extent: `(2,)` apparent size in metres.
        distance: Metres from the agent.
        bearing: Radians from the agent's heading, in `[-pi, pi]`.
        appearance: `(APPEARANCE_DIM,)` float32 visual descriptor. Informative
            about what the object affords, but never decisive -- two objects
            that look alike may behave differently, which is what leaves the
            agent's prior something to be wrong about.
        within_reach: Whether interaction verbs can be attempted right now.
        held: Whether the agent is currently carrying this object.
    """

    object_id: str
    position: np.ndarray
    extent: np.ndarray
    distance: float
    bearing: float
    appearance: np.ndarray
    within_reach: bool = False
    held: bool = False

    def __post_init__(self) -> None:
        if self.appearance.shape != (APPEARANCE_DIM,):
            raise ValueError(
                f"appearance must have shape ({APPEARANCE_DIM},), "
                f"got {self.appearance.shape}"
            )

@dataclass(frozen=True, eq=False)
class CompositeObject(ObjectView):
    """A combination of two objects treated as a single entity."""
    part_a_id: str = ""
    part_b_id: str = ""
    relation: str = "on"
    
    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.part_a_id, self.part_b_id, self.relation)


@dataclass(frozen=True, eq=False)
class Observation:
    """Everything the agent perceives at one timestep.

    Attributes:
        t: Timestep within the episode.
        agent_position: `(2,)` world-frame position in metres.
        agent_heading: Radians, in `[-pi, pi]`.
        objects: Every object currently perceivable.
        holding: Id of the carried object, or `None`.
        budget_remaining: Interaction-budget units left this episode. The agent
            is allowed to see this: choosing what to test is only meaningful
            when you know what you can still afford.
        rgb: Optional rendered view, for encoders and for debugging. Backends
            may omit it when nothing consumes it.
    """

    t: int
    agent_position: np.ndarray
    agent_heading: float
    objects: tuple[ObjectView, ...] = ()
    holding: str | None = None
    budget_remaining: float = 0.0
    rgb: np.ndarray | None = None

    def view(self, object_id: str) -> ObjectView | None:
        """The view of `object_id`, or `None` if it is not perceivable."""
        for o in self.objects:
            if o.object_id == object_id:
                return o
        return None

    def reachable(self) -> tuple[ObjectView, ...]:
        """Objects the agent could attempt an interaction verb on right now."""
        return tuple(o for o in self.objects if o.within_reach)


@dataclass(frozen=True)
class Interaction:
    """A verb applied to a target.

    For relational verbs the `target` is what is being acted *upon* -- the
    support in `place_on` -- while the tool is whatever the agent happens to be
    holding. That mirrors `Affordance.tool_kind` so a record can be matched
    against ground truth without reinterpretation.
    """

    action: Action
    target: str | None = None

    def __post_init__(self) -> None:
        needs_target = self.action is not Action.RELEASE
        if needs_target and self.target is None:
            raise ValueError(f"{Action(self.action).name} requires a target")
        if not needs_target and self.target is not None:
            raise ValueError(
                f"{Action(self.action).name} does not take a target"
            )

    @property
    def cost(self) -> float:
        """Interaction-budget units this attempt consumes."""
        return cost(self.action)

    @property
    def is_relational(self) -> bool:
        """Whether this verb also depends on what the agent is holding."""
        return is_relational(self.action)

    @property
    def is_affordance_test(self) -> bool:
        """Whether this can reveal an affordance.

        `APPROACH` cannot: it repositions the agent and says nothing about what
        an object is for, so it must not be counted as a verification.
        """
        return self.action is not Action.APPROACH

    def describe(self) -> str:
        """Human-readable form, for logs and evaluation reports."""
        return f"{spec(self.action).name}({self.target or ''})"


@dataclass(frozen=True, eq=False)
class InteractionRecord:
    """The full log of one attempted interaction.

    This is the unit the agent learns from and the unit evaluation counts.
    Failed attempts are recorded too: discovering that a verb does *not* work,
    or does not work yet, is knowledge that was paid for.

    Attributes:
        episode: Episode index.
        t: Timestep within the episode.
        interaction: What was attempted.
        outcome: What actually happened.
        cost: Budget units actually consumed.
        tool_id: What the agent was holding at the time, or `None`. Required to
            match a relational record against ground truth.
        view_before: The target as perceived immediately before.
        view_after: The target immediately after, or `None` if it left the
            scene -- consumed, carried away, or destroyed.
    """

    episode: int
    t: int
    interaction: Interaction
    outcome: Outcome
    cost: float
    tool_id: str | None = None
    view_before: ObjectView | None = None
    view_after: ObjectView | None = None

    @property
    def is_affordance_test(self) -> bool:
        """Whether this attempt could have revealed an affordance."""
        return self.interaction.is_affordance_test


@dataclass(frozen=True, eq=False)
class StepResult:
    """What the environment returns from one `step`.

    Attributes:
        observation: The world after the interaction resolved.
        record: The log of what was attempted.
        done: Whether the episode has ended, by budget exhaustion or task
            completion.
        info: Backend diagnostics. Never load agent-visible signal in here --
            anything the agent may legitimately use belongs in `observation`.
    """

    observation: Observation
    record: InteractionRecord
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
