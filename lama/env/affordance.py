"""The affordance record: what an object kind actually affords.

This is **hidden ground truth**. The environment knows it; the agent never
reads it. It exists so that evaluation can ask the only question that matters:
of the things this object genuinely affords, how many did the agent confirm,
how many did it get wrong, and what did it spend finding out.

An affordance is keyed by `(kind, action, tool_kind)` rather than by `action`
alone, because the interesting uses of an object are relational. "A plate
affords being placed upon" is not a fact about the plate; it is a fact about
the plate *and a heavy enough thing to put on it*. See D5 in
docs/DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .actions import Action, is_relational
from .outcomes import Effect


class Role(IntEnum):
    """How central a use is to the object.

    Ordered by informativeness, so `>` and `max` behave sensibly.
    """

    NONE = 0        # the object does not afford this verb at all
    INCIDENTAL = 1  # technically works, but nearly everything affords it
    SECONDARY = 2   # a real, non-obvious use: the object as a means to an end
    PRIMARY = 3     # what the object is for; its canonical function


class Precondition(IntEnum):
    """A context requirement that must hold for an affordance to fire.

    The agent has to discover these too. An affordance that only fires when a
    precondition holds looks exactly like an unreliable one until the agent
    works out what the condition is -- which is a large part of the problem.

    Values are stable; append rather than renumber.
    """

    GRIPPER_FREE = 0     # the agent must not be carrying anything
    HOLDING_ANY = 1      # the agent must be carrying something
    HOLDING_HEAVY = 2    # the carried object must exceed the weight threshold
    TARGET_OPEN = 3      # the target must currently be open
    TARGET_CLOSED = 4    # the target must currently be closed
    TARGET_UNLOCKED = 5  # the target must not be locked
    TARGET_UPRIGHT = 6   # the target must not already be toppled


@dataclass(frozen=True)
class Affordance:
    """One true `(object kind, verb)` capability of the world.

    Attributes:
        kind: The object kind the verb is applied to.
        action: The verb.
        effect: What happens when it fires.
        role: How central this use is to the object.
        reliability: Probability the effect occurs given the verb is attempted
            and every precondition holds. Values below 1 are *aleatoric*: no
            amount of further testing will make them certain, and an agent that
            keeps testing them is wasting its budget.
        preconditions: Context that must hold for the affordance to fire.
        tool_kind: For relational verbs, the kind that must be held. `None` for
            unary verbs.
        remote_effect: What happens to some *other* object when this fires.
            The signature of a secondary affordance.
    """

    kind: str
    action: Action
    effect: Effect
    role: Role
    reliability: float = 1.0
    preconditions: tuple[Precondition, ...] = ()
    tool_kind: str | None = None
    remote_effect: Effect | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(
                f"reliability must be in [0, 1], got {self.reliability}"
            )
        if is_relational(self.action) and self.tool_kind is None:
            raise ValueError(
                f"{Action(self.action).name} is relational and requires "
                f"tool_kind (kind={self.kind!r})"
            )
        if not is_relational(self.action) and self.tool_kind is not None:
            raise ValueError(
                f"{Action(self.action).name} is not relational but tool_kind "
                f"was given (kind={self.kind!r})"
            )
        if self.role is Role.NONE and self.effect not in (
            Effect.NOTHING,
            Effect.BLOCKED,
        ):
            raise ValueError(
                f"role NONE implies no effect, got {Effect(self.effect).name}"
            )

    @property
    def key(self) -> tuple[str, Action, str | None]:
        """Identity of this capability, for comparison against learned claims.

        Deliberately excludes `effect` and `role`: an agent that confirms the
        right verb on the right object but predicts the wrong effect has made a
        different error than one that never tested the verb, and evaluation
        needs to tell those apart.
        """
        return (self.kind, Action(self.action), self.tool_kind)

    @property
    def is_real(self) -> bool:
        """Whether this capability exists at all and can ever be observed."""
        return self.role is not Role.NONE and self.reliability > 0.0

    @property
    def is_deterministic(self) -> bool:
        """Whether repeated tests under the same conditions always agree."""
        return self.reliability >= 1.0

    @property
    def is_conditional(self) -> bool:
        """Whether context must hold before this can fire."""
        return bool(self.preconditions)

    @property
    def uses_object_as_means(self) -> bool:
        """Whether firing this changes something other than the target.

        The clearest marker of a use that is *about* achieving something else.
        """
        return self.remote_effect is not None

    def describe(self) -> str:
        """Human-readable form, for evaluation reports and debugging."""
        verb = Action(self.action).name.lower()
        head = (
            f"{verb}({self.tool_kind} -> {self.kind})"
            if self.tool_kind
            else f"{verb}({self.kind})"
        )
        parts = [f"{head} => {Effect(self.effect).name.lower()}"]
        if self.remote_effect is not None:
            parts.append(f"remote {Effect(self.remote_effect).name.lower()}")
        if self.preconditions:
            names = ", ".join(p.name.lower() for p in self.preconditions)
            parts.append(f"if {names}")
        if not self.is_deterministic:
            parts.append(f"p={self.reliability:.2f}")
        return f"[{self.role.name.lower()}] " + "; ".join(parts)
