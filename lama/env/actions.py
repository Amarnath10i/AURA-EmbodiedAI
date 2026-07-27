"""The interaction verb set.

An affordance is only ever expressed as "this object responds to *this verb*",
so the verb set is the vocabulary the entire system reasons in. It is fixed
here, once, and every backend must implement all of it.

Two properties matter for LAMA specifically:

* **Verbs are not interchangeable in cost.** Looking at an object is nearly
  free; lifting it is not. Hypothesis selection is only interesting when tests
  have different prices, so cost is part of the verb definition rather than a
  detail hidden inside a backend.
* **Some verbs are relational.** `PLACE_ON` needs a second object. Tool use is
  inherently relational, and a purely unary verb set cannot express "use the
  crate to hold down the plate" -- which is the class of affordance this project
  is about.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    """Every interaction the agent can attempt.

    Values are stable: they index model outputs and are written into stored
    records, so append new verbs at the end rather than renumbering.
    """

    APPROACH = 0    # move within reach of a target; not an affordance test
    PUSH = 1        # apply force away from the agent
    PULL = 2        # apply force toward the agent
    LIFT = 3        # raise off its support
    PRESS = 4       # apply downward force in place
    ROTATE = 5      # apply torque about the vertical axis
    OPEN = 6        # move an articulated part to its open configuration
    CLOSE = 7       # move an articulated part to its closed configuration
    GRASP = 8       # take into the gripper and carry
    RELEASE = 9     # let go of whatever is held
    PLACE_ON = 10   # put the held object onto a target (relational)
    TIP = 11        # push past the point of balance


@dataclass(frozen=True)
class ActionSpec:
    """Static properties of a verb, independent of any object or backend."""

    action: Action
    name: str
    relational: bool          # needs a second object (the support)
    needs_reach: bool         # agent must be within reach of the target
    needs_free_gripper: bool  # cannot be attempted while holding something
    needs_held_object: bool   # can only be attempted while holding something
    cost: float               # interaction-budget units consumed per attempt
    description: str


#: Cost units are deliberately coarse. They encode effort and risk, not time:
#: a verb that can damage things or strand the agent costs more, because the
#: point of the budget is to make the agent's choice of test matter.
SPECS: dict[Action, ActionSpec] = {
    Action.APPROACH: ActionSpec(
        Action.APPROACH, "approach", False, False, False, False, 0.2,
        "Move within reach of the target. Never reveals an affordance by "
        "itself; it is the precondition for verbs that do.",
    ),
    Action.PUSH: ActionSpec(
        Action.PUSH, "push", False, True, True, False, 1.0,
        "Force applied away from the agent. The cheapest way to learn whether "
        "an object is free to move at all.",
    ),
    Action.PULL: ActionSpec(
        Action.PULL, "pull", False, True, True, False, 1.0,
        "Force applied toward the agent. Distinguishes objects that are hinged "
        "or handled from ones that merely slide.",
    ),
    Action.LIFT: ActionSpec(
        Action.LIFT, "lift", False, True, True, False, 2.0,
        "Raise the object clear of its support. Expensive, and the main test "
        "for whether an object can serve as a portable tool or weight.",
    ),
    Action.PRESS: ActionSpec(
        Action.PRESS, "press", False, True, False, False, 0.8,
        "Downward force without displacement. The canonical test for "
        "actuators, and the verb most likely to produce a remote effect.",
    ),
    Action.ROTATE: ActionSpec(
        Action.ROTATE, "rotate", False, True, True, False, 1.2,
        "Torque about the vertical axis. Separates handles, valves and dials "
        "from rigidly mounted fixtures.",
    ),
    Action.OPEN: ActionSpec(
        Action.OPEN, "open", False, True, True, False, 1.2,
        "Drive an articulated part toward its open configuration.",
    ),
    Action.CLOSE: ActionSpec(
        Action.CLOSE, "close", False, True, True, False, 1.2,
        "Drive an articulated part toward its closed configuration.",
    ),
    Action.GRASP: ActionSpec(
        Action.GRASP, "grasp", False, True, True, False, 1.5,
        "Take the object into the gripper. Precondition for every relational "
        "verb, so its failure blocks a whole branch of the hypothesis space.",
    ),
    Action.RELEASE: ActionSpec(
        Action.RELEASE, "release", False, False, False, True, 0.3,
        "Let go of the held object where the agent stands.",
    ),
    Action.PLACE_ON: ActionSpec(
        Action.PLACE_ON, "place_on", True, True, False, True, 2.0,
        "Put the held object onto a target. The only verb that can express "
        "using one object on another, and therefore the one that carries most "
        "secondary affordances.",
    ),
    Action.TIP: ActionSpec(
        Action.TIP, "tip", False, True, True, False, 1.6,
        "Push the object past its balance point. Often irreversible, which is "
        "what makes it a genuinely costly hypothesis to test.",
    ),
}

#: Every verb, in stable order. Use this to size model outputs.
ALL_ACTIONS: tuple[Action, ...] = tuple(Action)

#: Verbs that can reveal an affordance. `APPROACH` is excluded: it repositions
#: the agent and tells it nothing about what an object is for.
INTERACTION_ACTIONS: tuple[Action, ...] = tuple(
    a for a in Action if a is not Action.APPROACH
)

N_ACTIONS: int = len(ALL_ACTIONS)


def spec(action: Action) -> ActionSpec:
    """Static properties of `action`."""
    return SPECS[Action(action)]


def cost(action: Action) -> float:
    """Interaction-budget units consumed by attempting `action`."""
    return SPECS[Action(action)].cost


def is_relational(action: Action) -> bool:
    """Whether `action` requires a second object to act upon."""
    return SPECS[Action(action)].relational
