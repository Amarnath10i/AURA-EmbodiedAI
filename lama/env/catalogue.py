"""The object catalogue: hidden physics, and what each object is really for.

This module is the world's ground truth. Two things live here:

* `KINDS` -- the physical and visual properties of each object kind. Mass is
  hidden. Colour and size are visible, because a camera would see them.
* `AFFORDANCES` -- the true `(kind, verb, tool)` capability table, split by
  role into what the object is *for* and what else it can do.

Only non-`NONE` capabilities are declared. Any pair not listed simply does not
work, and `lookup` returns a `NONE` affordance for it. Declaring all 15 kinds
against all 11 verbs would be 165 mostly-empty rows and would make the
interesting entries impossible to see.

Design notes that matter for the result
---------------------------------------

**Look-alike traps.** `crate` and `block` are nearly indistinguishable by
appearance, and behave differently in the two ways that count: a crate opens
and a block does not, and a block is heavy enough to hold a pressure plate down
while a crate is not. `lever` and `switch` are the same trap for verbs: they
look alike, but one is pulled and the other pressed. An agent that trusts
appearance will confidently get these wrong, which is the entire point --
without traps the prior would be a lookup table and verification would be
pointless.

**Cheap verbs partially substitute for right ones.** Shoving a button
sometimes trips it (`PUSH`, p=0.5) even though `PRESS` is what it is for. This
is deliberate: an agent can buy noisy evidence cheaply or clean evidence
expensively, which is a real decision rather than a formality.

**The flagship secondary affordance** is the pressure plate. Its primary use is
being stood on, which stops working the moment the agent walks away. Its
secondary use -- placing something heavy on it -- holds the door open
permanently. That is an object being used as a *means to an end*, it is only
discoverable through a relational verb with the right tool in hand, and no
amount of interacting with the plate alone will reveal it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .actions import Action
from .affordance import Affordance, Precondition, Role
from .outcomes import Effect

#: Carried mass at or above which an object can hold a pressure plate down.
#: Hidden from the agent, and invisible: the crate/block trap is precisely that
#: two objects straddle this threshold while looking the same.
HEAVY_THRESHOLD: float = 5.0

#: Length of the per-kind texture signature inside the appearance descriptor.
TEXTURE_DIM: int = 8


@dataclass(frozen=True)
class KindSpec:
    """Physical and visual properties of one object kind.

    Attributes:
        name: Kind identifier. Hidden from the agent; used by the oracle.
        mass: Kilograms. Hidden -- there is no visual channel for it, which is
            what makes the crate/block trap unsolvable by looking.
        extent: `(width, depth)` footprint in metres. Visible.
        height: Metres. Visible.
        color: RGB, 0-255. Visible.
        texture: `TEXTURE_DIM` surface and shape cues. Visible, and the main
            way appearance carries information about kind.
        fixed: Anchored to the world; cannot be moved by any verb.
        articulated: Has an open/closed configuration that is visible.
        held_out: Reserved for the transfer evaluation. Never appears in
            training layouts.
        look_alike: The kind this one is visually confusable with, if any.
            Documentation for the reader; the confusion is produced by the
            texture and colour values themselves.
    """

    name: str
    mass: float
    extent: tuple[float, float]
    height: float
    color: tuple[int, int, int]
    texture: tuple[float, ...]
    fixed: bool = False
    articulated: bool = False
    held_out: bool = False
    look_alike: str | None = None

    @property
    def is_heavy(self) -> bool:
        """Whether carrying this satisfies `Precondition.HOLDING_HEAVY`."""
        return self.mass >= HEAVY_THRESHOLD

    @property
    def is_portable(self) -> bool:
        """Whether the agent can pick this up at all."""
        return not self.fixed and self.mass <= 15.0


def _k(
    name: str,
    mass: float,
    extent: tuple[float, float],
    height: float,
    color: tuple[int, int, int],
    texture: tuple[float, ...],
    **kw: object,
) -> tuple[str, KindSpec]:
    return name, KindSpec(name, mass, extent, height, color, texture, **kw)


KINDS: dict[str, KindSpec] = dict(
    (
        # -- mechanisms: their effect is somewhere else in the room --------- #
        _k("button", 0.0, (0.12, 0.12), 0.10, (206, 74, 66),
           (0.9, 0.1, 0.0, 0.8, 0.2, 0.1, 0.0, 0.9), fixed=True),
        _k("lever", 0.0, (0.10, 0.10), 0.55, (188, 156, 64),
           (0.2, 0.9, 0.1, 0.1, 0.8, 0.2, 0.7, 0.1), fixed=True,
           look_alike="switch"),
        _k("valve", 0.0, (0.26, 0.26), 0.30, (110, 148, 176),
           (0.4, 0.4, 0.9, 0.3, 0.3, 0.8, 0.2, 0.4), fixed=True),
        _k("plate", 0.0, (0.60, 0.60), 0.04, (128, 106, 168),
           (0.8, 0.2, 0.1, 0.9, 0.1, 0.0, 0.1, 0.8), fixed=True),

        # -- fixtures ------------------------------------------------------- #
        _k("door", 0.0, (0.16, 0.90), 2.10, (176, 82, 72),
           (0.1, 0.7, 0.3, 0.2, 0.9, 0.4, 0.6, 0.2), fixed=True,
           articulated=True),
        _k("shelf", 0.0, (1.60, 0.45), 1.80, (146, 106, 66),
           (0.3, 0.6, 0.6, 0.4, 0.5, 0.7, 0.4, 0.3), fixed=True),
        _k("pillar", 0.0, (0.45, 0.45), 2.60, (96, 100, 108),
           (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5), fixed=True),

        # -- portables: the crate/block trap -------------------------------- #
        # Identical descriptors on purpose. A sealed crate and a solid block do
        # not look different, and the two ways they differ -- mass, and whether
        # there is a lid -- are both invisible until something is tried. This
        # is the one pair appearance cannot settle at any noise level.
        _k("crate", 3.0, (0.45, 0.45), 0.42, (198, 160, 96),
           (0.6, 0.3, 0.7, 0.6, 0.4, 0.3, 0.5, 0.6), articulated=True,
           look_alike="block"),
        _k("block", 12.0, (0.45, 0.45), 0.42, (198, 160, 96),
           (0.6, 0.3, 0.7, 0.6, 0.4, 0.3, 0.5, 0.6), look_alike="crate"),
        _k("toolbox", 9.0, (0.50, 0.28), 0.26, (78, 122, 150),
           (0.7, 0.5, 0.2, 0.3, 0.6, 0.6, 0.3, 0.7), articulated=True),
        _k("cup", 0.3, (0.10, 0.10), 0.14, (232, 228, 220),
           (0.2, 0.2, 0.4, 0.9, 0.7, 0.1, 0.8, 0.2)),

        # -- movables ------------------------------------------------------- #
        _k("cart", 20.0, (0.90, 0.60), 0.95, (196, 128, 48),
           (0.4, 0.8, 0.5, 0.2, 0.3, 0.9, 0.6, 0.4)),
        _k("barrel", 30.0, (0.56, 0.56), 0.88, (86, 138, 96),
           (0.5, 0.7, 0.8, 0.1, 0.2, 0.5, 0.9, 0.3), look_alike="drum"),
        _k("pallet", 8.0, (1.20, 1.00), 0.14, (166, 140, 100),
           (0.3, 0.4, 0.3, 0.7, 0.6, 0.2, 0.2, 0.5)),

        # -- held out for the transfer evaluation --------------------------- #
        _k("switch", 0.0, (0.11, 0.11), 0.52, (184, 152, 70),
           (0.2, 0.9, 0.1, 0.1, 0.8, 0.2, 0.7, 0.2), fixed=True,
           held_out=True, look_alike="lever"),
        _k("drum", 34.0, (0.58, 0.58), 0.90, (90, 134, 100),
           (0.5, 0.7, 0.8, 0.1, 0.2, 0.5, 0.9, 0.4), held_out=True,
           look_alike="barrel"),
        _k("bench", 11.0, (1.40, 0.42), 0.46, (150, 118, 84),
           (0.5, 0.6, 0.4, 0.3, 0.2, 0.6, 0.5, 0.3), held_out=True),
    )
)


def _plate_weight(tool: str) -> Affordance:
    """Placing something heavy on the plate holds its door open for good."""
    return Affordance(
        "plate", Action.PLACE_ON, Effect.SUPPORTED, Role.SECONDARY,
        tool_kind=tool, preconditions=(Precondition.HOLDING_HEAVY,),
        remote_effect=Effect.OPENED,
    )


def _surface(kind: str, tool: str, role: Role = Role.PRIMARY) -> Affordance:
    """`kind` can have `tool` set down on it, and nothing further happens."""
    return Affordance(kind, Action.PLACE_ON, Effect.SUPPORTED, role,
                      tool_kind=tool)


AFFORDANCES: tuple[Affordance, ...] = (
    # ---- button: pressing is what it is for; shoving sometimes works ------ #
    Affordance("button", Action.PRESS, Effect.ACTUATED, Role.PRIMARY,
               remote_effect=Effect.OPENED),
    Affordance("button", Action.PUSH, Effect.ACTUATED, Role.INCIDENTAL,
               reliability=0.5, remote_effect=Effect.OPENED),

    # ---- lever: pulled by design, turns if you insist -------------------- #
    Affordance("lever", Action.PULL, Effect.ACTUATED, Role.PRIMARY,
               remote_effect=Effect.OPENED),
    Affordance("lever", Action.ROTATE, Effect.ACTUATED, Role.SECONDARY,
               reliability=0.45, remote_effect=Effect.OPENED),

    # ---- valve ----------------------------------------------------------- #
    Affordance("valve", Action.ROTATE, Effect.ACTUATED, Role.PRIMARY,
               remote_effect=Effect.OPENED),

    # ---- plate: stand on it, or weigh it down (the flagship secondary) ---- #
    Affordance("plate", Action.PRESS, Effect.ACTUATED, Role.PRIMARY,
               remote_effect=Effect.OPENED),
    _plate_weight("block"),
    _plate_weight("toolbox"),
    _plate_weight("bench"),
    _surface("plate", "crate", Role.INCIDENTAL),   # too light: the trap bites
    _surface("plate", "cup", Role.INCIDENTAL),

    # ---- door ------------------------------------------------------------ #
    Affordance("door", Action.OPEN, Effect.OPENED, Role.PRIMARY,
               preconditions=(Precondition.TARGET_UNLOCKED,)),
    Affordance("door", Action.CLOSE, Effect.CLOSED, Role.PRIMARY,
               preconditions=(Precondition.TARGET_OPEN,)),
    Affordance("door", Action.PUSH, Effect.OPENED, Role.INCIDENTAL,
               reliability=0.7,
               preconditions=(Precondition.TARGET_UNLOCKED,)),

    # ---- shelf: a surface, and nothing else ------------------------------ #
    _surface("shelf", "cup"),
    _surface("shelf", "crate"),
    _surface("shelf", "toolbox"),

    # ---- crate: a container that is also liftable ------------------------ #
    Affordance("crate", Action.OPEN, Effect.OPENED, Role.PRIMARY),
    Affordance("crate", Action.CLOSE, Effect.CLOSED, Role.PRIMARY,
               preconditions=(Precondition.TARGET_OPEN,)),
    Affordance("crate", Action.GRASP, Effect.CARRIED, Role.SECONDARY,
               preconditions=(Precondition.GRIPPER_FREE,)),
    Affordance("crate", Action.LIFT, Effect.LIFTED, Role.SECONDARY),
    Affordance("crate", Action.PUSH, Effect.TRANSLATED, Role.INCIDENTAL),

    # ---- block: looks like a crate, does not open, heavy enough to matter - #
    Affordance("block", Action.GRASP, Effect.CARRIED, Role.PRIMARY,
               preconditions=(Precondition.GRIPPER_FREE,)),
    Affordance("block", Action.LIFT, Effect.LIFTED, Role.PRIMARY),
    Affordance("block", Action.PUSH, Effect.TRANSLATED, Role.INCIDENTAL),

    # ---- toolbox --------------------------------------------------------- #
    Affordance("toolbox", Action.OPEN, Effect.OPENED, Role.PRIMARY),
    Affordance("toolbox", Action.GRASP, Effect.CARRIED, Role.SECONDARY,
               preconditions=(Precondition.GRIPPER_FREE,)),
    Affordance("toolbox", Action.LIFT, Effect.LIFTED, Role.SECONDARY),
    Affordance("toolbox", Action.PUSH, Effect.TRANSLATED, Role.INCIDENTAL),

    # ---- cup ------------------------------------------------------------- #
    Affordance("cup", Action.GRASP, Effect.CARRIED, Role.PRIMARY,
               preconditions=(Precondition.GRIPPER_FREE,)),
    Affordance("cup", Action.LIFT, Effect.LIFTED, Role.PRIMARY),
    Affordance("cup", Action.PUSH, Effect.TRANSLATED, Role.INCIDENTAL),
    Affordance("cup", Action.TIP, Effect.TOPPLED, Role.INCIDENTAL),

    # ---- cart: made to be moved, useful as a moving surface -------------- #
    Affordance("cart", Action.PUSH, Effect.TRANSLATED, Role.PRIMARY),
    Affordance("cart", Action.PULL, Effect.TRANSLATED, Role.PRIMARY),
    _surface("cart", "crate", Role.SECONDARY),
    _surface("cart", "block", Role.SECONDARY),
    _surface("cart", "toolbox", Role.SECONDARY),
    Affordance("cart", Action.TIP, Effect.TOPPLED, Role.INCIDENTAL),

    # ---- barrel: rolls, unpredictably ------------------------------------ #
    Affordance("barrel", Action.PUSH, Effect.TRANSLATED, Role.PRIMARY,
               reliability=0.85),
    Affordance("barrel", Action.ROTATE, Effect.ROTATED, Role.INCIDENTAL),
    Affordance("barrel", Action.TIP, Effect.TOPPLED, Role.SECONDARY),

    # ---- pallet: a surface you can also shove out of the way -------------- #
    _surface("pallet", "crate"),
    _surface("pallet", "block"),
    _surface("pallet", "toolbox"),
    Affordance("pallet", Action.PUSH, Effect.TRANSLATED, Role.INCIDENTAL),

    # ---- held out: switch looks like a lever but must be pressed --------- #
    Affordance("switch", Action.PRESS, Effect.ACTUATED, Role.PRIMARY,
               remote_effect=Effect.OPENED),
    Affordance("drum", Action.PUSH, Effect.TRANSLATED, Role.PRIMARY,
               reliability=0.85),
    Affordance("drum", Action.TIP, Effect.TOPPLED, Role.SECONDARY),
    Affordance("drum", Action.ROTATE, Effect.ROTATED, Role.INCIDENTAL),
    _surface("bench", "crate"),
    _surface("bench", "block"),
    Affordance("bench", Action.PUSH, Effect.TRANSLATED, Role.INCIDENTAL),
)


#: Fast lookup by `(kind, action, tool_kind)`.
_BY_KEY: dict[tuple[str, Action, str | None], Affordance] = {
    a.key: a for a in AFFORDANCES
}


def _validate() -> None:
    """Fail loudly at import if the catalogue is internally inconsistent."""
    if len(_BY_KEY) != len(AFFORDANCES):
        raise ValueError("duplicate affordance keys in the catalogue")
    for name, k in KINDS.items():
        if k.name != name:
            raise ValueError(f"kind {name!r} is registered under {k.name!r}")
        if len(k.texture) != TEXTURE_DIM:
            raise ValueError(
                f"kind {name!r} needs {TEXTURE_DIM} texture values, "
                f"got {len(k.texture)}"
            )
        if k.look_alike is not None and k.look_alike not in KINDS:
            raise ValueError(f"kind {name!r} looks like unknown {k.look_alike!r}")
    for a in AFFORDANCES:
        if a.kind not in KINDS:
            raise ValueError(f"affordance on unknown kind {a.kind!r}")
        if a.tool_kind is not None and a.tool_kind not in KINDS:
            raise ValueError(f"affordance needs unknown tool {a.tool_kind!r}")
        if a.tool_kind is not None and not KINDS[a.tool_kind].is_portable:
            raise ValueError(
                f"tool {a.tool_kind!r} cannot be carried, so "
                f"{a.describe()} is unreachable"
            )
        if Precondition.HOLDING_HEAVY in a.preconditions:
            if a.tool_kind is None or not KINDS[a.tool_kind].is_heavy:
                raise ValueError(
                    f"{a.describe()} requires a heavy tool but "
                    f"{a.tool_kind!r} is not heavy"
                )


_validate()


def lookup(
    kind: str, action: Action, tool_kind: str | None = None
) -> Affordance:
    """The true affordance for a `(kind, verb, tool)` triple.

    Undeclared combinations return a `NONE` affordance rather than raising:
    "this does not work" is a real answer the world must be able to give, and
    the agent has to pay to find it out like any other.
    """
    if kind not in KINDS:
        raise KeyError(f"unknown object kind {kind!r}")
    found = _BY_KEY.get((kind, Action(action), tool_kind))
    if found is not None:
        return found
    return Affordance(kind, action, Effect.NOTHING, Role.NONE,
                      reliability=0.0, tool_kind=tool_kind)


def affordances_for(kind: str) -> tuple[Affordance, ...]:
    """Every real capability of `kind`, most central first."""
    found = [a for a in AFFORDANCES if a.kind == kind]
    return tuple(sorted(found, key=lambda a: (-a.role, a.action)))


def kinds(include_held_out: bool = False) -> tuple[str, ...]:
    """Kind names, excluding the transfer set unless asked."""
    return tuple(
        n for n, k in KINDS.items() if include_held_out or not k.held_out
    )


def by_role(role: Role, include_held_out: bool = False) -> tuple[Affordance, ...]:
    """Every affordance of a given role, for role-split evaluation."""
    allowed = set(kinds(include_held_out))
    return tuple(
        a for a in AFFORDANCES
        if a.role is role
        and a.kind in allowed
        and (a.tool_kind is None or a.tool_kind in allowed)
    )
