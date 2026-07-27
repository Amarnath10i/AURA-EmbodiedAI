"""What happens when a verb meets an object.

An `Outcome` is the ground truth of a single attempted interaction: the
qualitative effect, the continuous quantities that came with it, anything that
changed elsewhere in the world, and whether the world can be put back.

Three design points carry real weight for LAMA:

* **`NOTHING` and `BLOCKED` are different.** Nothing happened because the object
  does not respond to this verb; blocked means it does respond but something
  prevented it this time. Collapsing them would teach the agent that a locked
  door cannot be opened.
* **`remote` effects are part of the outcome.** Pressing a plate that opens a
  door across the room is not a property of the plate's own motion. Secondary
  affordances -- using one object to achieve something elsewhere -- are
  unrepresentable without this field.
* **`irreversible` is recorded, not inferred.** The agent cannot know in advance
  which tests destroy their own evidence, but the environment can, and
  evaluation needs it to report what a policy spent irrecoverably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Effect(IntEnum):
    """The qualitative result of an interaction.

    Values are stable: they index model outputs and are written into stored
    records, so append new effects at the end rather than renumbering.
    """

    NOTHING = 0      # the object does not respond to this verb
    BLOCKED = 1      # it would respond, but a condition prevented it this time
    TRANSLATED = 2   # moved across its support
    LIFTED = 3       # raised clear of its support
    ROTATED = 4      # turned about the vertical axis
    OPENED = 5       # an articulated part reached its open configuration
    CLOSED = 6       # an articulated part reached its closed configuration
    ACTUATED = 7     # internal state changed (button, lever, switch, valve)
    TOPPLED = 8      # fell past its balance point
    CARRIED = 9      # taken into the gripper
    RELEASED = 10    # let go of
    SUPPORTED = 11   # came to rest on another object
    BROKE = 12       # damaged beyond further use


#: Effects meaning the attempt produced no change in the world. An agent that
#: cannot tell these apart from real effects has learned nothing from the test.
NULL_EFFECTS: frozenset[Effect] = frozenset({Effect.NOTHING, Effect.BLOCKED})

#: Effects that cannot be undone by any sequence of later actions. Attempting a
#: verb that yields one of these permanently changes what remains learnable.
IRREVERSIBLE_EFFECTS: frozenset[Effect] = frozenset({Effect.BROKE, Effect.TOPPLED})


@dataclass(frozen=True)
class RemoteEffect:
    """A change to an object other than the one acted upon.

    This is how the world expresses "that object was useful *for* something":
    the crate did not merely move, it held down the plate, and the door opened.
    """

    object_id: str
    effect: Effect


@dataclass(frozen=True)
class Outcome:
    """The complete, observable-in-principle result of one interaction.

    Continuous quantities are in metres and radians so that a physics backend
    can populate them natively. They are zero whenever the effect does not
    imply them; do not read `displacement` without checking `effect`.
    """

    effect: Effect
    displacement: float = 0.0        # metres moved across the support
    height_gain: float = 0.0         # metres raised clear of the support
    rotation: float = 0.0            # radians turned about the vertical axis
    remote: tuple[RemoteEffect, ...] = field(default_factory=tuple)
    irreversible: bool = False

    @property
    def changed_world(self) -> bool:
        """Whether anything at all happened, here or elsewhere."""
        return self.effect not in NULL_EFFECTS or bool(self.remote)

    @property
    def had_remote_effect(self) -> bool:
        """Whether the interaction changed something other than its target.

        The signature of a secondary affordance: the object was used as a means
        to an end rather than being an end in itself.
        """
        return bool(self.remote)


def nothing() -> Outcome:
    """The object does not respond to this verb."""
    return Outcome(effect=Effect.NOTHING)


def blocked() -> Outcome:
    """The object would respond, but a condition prevented it this time."""
    return Outcome(effect=Effect.BLOCKED)
