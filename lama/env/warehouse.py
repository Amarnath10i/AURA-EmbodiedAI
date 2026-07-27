"""A fast numpy warehouse: the first backend behind the environment contract.

Objects sit on a flat floor with hidden mass and hidden affordance profiles.
The agent perceives geometry and appearance only. Every interaction costs
budget, including the ones that fail.

What this backend deliberately does not simulate
------------------------------------------------

**Navigation.** `APPROACH` places the agent next to its target directly. Path
planning is a solved problem and not what this project is about; simulating it
would add cost to every episode and change none of the numbers that matter.
It still costs budget, so approaching is not free.

**Contact dynamics.** Displacement is a function of hidden mass, not of
integrated forces. The agent cannot tell the difference -- it observes that
heavy things move less -- and a physics backend can reproduce the same
observable relationship later. See D2 in docs/DECISIONS.md.

Mechanisms and why the plate is the interesting one
---------------------------------------------------

Doors start locked, so they can only be opened through a mechanism. Buttons,
levers and valves **latch**: trip one and its door stays open. The pressure
plate does not. Standing on it opens its door only while the agent stands
there, which is useless, because the agent has to walk through the door it is
holding open. Putting something heavy on it opens the door for good.

That asymmetry is the whole reason the catalogue has a plate. It is an
affordance that cannot be discovered by interacting with the plate alone, only
by interacting with the plate *while holding the right kind of thing* -- and
the two candidate things look identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import appearance as appearance_mod
from . import catalogue
from .actions import Action, spec
from .affordance import Affordance, Precondition, Role
from .catalogue import KINDS
from .interface import Environment
from .outcomes import Effect, Outcome, RemoteEffect
from .types import Interaction, InteractionRecord, Observation, ObjectView, StepResult

#: Metres within which interaction verbs may be attempted.
REACH: float = 1.0

#: Object kinds whose actuation opens a linked door.
ACTUATORS: tuple[str, ...] = ("button", "lever", "valve", "plate", "switch")

#: Kinds that must appear in every layout, so that the affordances the project
#: is about are always reachable: a locked door, the plate that can open it,
#: and the look-alike pair that decides whether the agent can.
REQUIRED_KINDS: tuple[str, ...] = ("door", "plate", "block", "crate")


@dataclass
class _Object:
    """Internal world state for one object. Never handed to the agent."""

    object_id: str
    kind: str
    position: np.ndarray
    is_open: bool = False
    locked: bool = False
    latched: bool = False            # opened by a latching mechanism
    toppled: bool = False
    carried: bool = False
    support: str | None = None       # id of the object this rests on
    _spec: catalogue.KindSpec = field(init=False)

    def __post_init__(self) -> None:
        self._spec = KINDS[self.kind]

    @property
    def spec(self) -> catalogue.KindSpec:
        return self._spec


class Warehouse(Environment):
    """A warehouse of objects with hidden affordances.

    Args:
        seed: Drives episode dynamics -- stochastic affordances and appearance
            noise. Same seed and same interactions reproduce a trajectory
            exactly.
        layout_seed: Drives object choice and placement. Fix it to vary
            dynamics on a constant layout, or vary it to generate unseen
            warehouses for the generalisation evaluation.
        budget: Interaction-budget units per episode. The episode ends when it
            runs out, which is what makes the choice of test matter.
        n_objects: Objects placed per layout, including the required kinds.
        include_held_out: Whether the transfer-set kinds may appear. False for
            every training layout.
        appearance_noise: Per-observation descriptor noise. See
            `appearance.DEFAULT_NOISE`.
        floor: `(width, depth)` of the warehouse in metres.
    """

    backend_name = "numpy-warehouse"

    def __init__(
        self,
        seed: int = 0,
        layout_seed: int | None = None,
        budget: float = 60.0,
        n_objects: int = 10,
        include_held_out: bool = False,
        appearance_noise: float = appearance_mod.DEFAULT_NOISE,
        floor: tuple[float, float] = (12.0, 9.0),
    ) -> None:
        if n_objects < len(REQUIRED_KINDS):
            raise ValueError(
                f"n_objects must be at least {len(REQUIRED_KINDS)} to fit the "
                f"required kinds {REQUIRED_KINDS}"
            )
        self._seed = seed
        # A layout_seed given explicitly pins the warehouse across episodes, so
        # dynamics can be varied against a constant world. Left unset, each
        # episode gets a fresh layout -- otherwise every episode would replay
        # the same warehouse and the generalisation numbers would be vacuous.
        self._fixed_layout = layout_seed is not None
        self._layout_seed = seed if layout_seed is None else layout_seed
        self._budget_per_episode = budget
        self._n_objects = n_objects
        self._include_held_out = include_held_out
        self._noise = appearance_noise
        self._floor = np.asarray(floor, dtype=np.float64)

        self._episode = -1
        self._t = 0
        self._budget = 0.0
        self._objects: dict[str, _Object] = {}
        self._links: dict[str, str] = {}     # actuator id -> door id
        self._holding: str | None = None
        self._agent = np.zeros(2)
        self._heading = 0.0
        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # contract
    # ------------------------------------------------------------------ #
    @property
    def episode(self) -> int:
        return self._episode

    @property
    def t(self) -> int:
        return self._t

    @property
    def budget_remaining(self) -> float:
        return self._budget

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed + 7919 * (self._episode + 1))
        self._episode += 1
        self._t = 0
        self._budget = self._budget_per_episode
        self._holding = None
        self._build_layout()
        return self._observe()

    def step(self, interaction: Interaction) -> StepResult:
        if interaction.target is not None and interaction.target not in self._objects:
            raise KeyError(f"no such object: {interaction.target!r}")

        self._t += 1
        self._release_momentary_doors()

        target = self._objects.get(interaction.target) if interaction.target else None
        view_before = self._view(target) if target else None
        tool_id = self._holding

        outcome = self._resolve(interaction, target)
        self._budget = max(0.0, self._budget - interaction.cost)

        after = self._objects.get(interaction.target) if interaction.target else None
        record = InteractionRecord(
            episode=self._episode,
            t=self._t,
            interaction=interaction,
            outcome=outcome,
            cost=interaction.cost,
            tool_id=tool_id,
            view_before=view_before,
            view_after=self._view(after) if after else None,
        )
        return StepResult(
            observation=self._observe(),
            record=record,
            done=self._budget <= 0.0,
            info={"backend": self.backend_name},
        )

    # ------------------------------------------------------------------ #
    # layout
    # ------------------------------------------------------------------ #
    def _build_layout(self) -> None:
        seed = (
            self._layout_seed
            if self._fixed_layout
            else self._layout_seed + 104729 * self._episode
        )
        rng = np.random.default_rng(seed)
        pool = [
            k for k in catalogue.kinds(self._include_held_out)
            if k not in ("door", "plate")
        ]
        chosen = list(REQUIRED_KINDS)
        chosen.append(str(rng.choice(["button", "lever", "valve"])))
        while len(chosen) < self._n_objects:
            chosen.append(str(rng.choice(pool)))
        # Shuffled so that neither an id nor its position in the layout
        # correlates with kind: the required kinds would otherwise always
        # occupy the first few slots.
        rng.shuffle(chosen)

        # Ids are opaque. Naming them after their kind would hand the agent the
        # answer in a field it is allowed to read, and every metric would still
        # look healthy while measuring nothing.
        self._objects = {}
        placed: list[np.ndarray] = []
        for n, kind in enumerate(chosen):
            pos = self._free_position(rng, placed)
            placed.append(pos)
            obj = _Object(f"obj_{n:02d}", kind, pos)
            obj.locked = kind == "door"      # doors need a mechanism
            self._objects[obj.object_id] = obj

        doors = [o.object_id for o in self._objects.values() if o.kind == "door"]
        self._links = {
            o.object_id: doors[i % len(doors)]
            for i, o in enumerate(
                o for o in self._objects.values() if o.kind in ACTUATORS
            )
        }

        self._agent = self._free_position(rng, placed)
        self._heading = float(rng.uniform(-np.pi, np.pi))

    def _free_position(
        self, rng: np.random.Generator, placed: list[np.ndarray], margin: float = 1.0
    ) -> np.ndarray:
        for _ in range(200):
            p = rng.uniform(0.05, 0.95, size=2) * self._floor
            if all(np.linalg.norm(p - q) > margin for q in placed):
                return p
        return rng.uniform(0.05, 0.95, size=2) * self._floor

    # ------------------------------------------------------------------ #
    # resolving an interaction
    # ------------------------------------------------------------------ #
    def _resolve(self, interaction: Interaction, target: _Object | None) -> Outcome:
        action = interaction.action
        s = spec(action)

        if action is Action.APPROACH:
            return self._approach(target)
        if action is Action.RELEASE:
            return self._release()

        assert target is not None  # every other verb requires a target
        if s.needs_reach and self._distance(target) > REACH:
            return Outcome(Effect.BLOCKED)
        if s.needs_free_gripper and self._holding is not None:
            return Outcome(Effect.BLOCKED)
        if s.needs_held_object and self._holding is None:
            return Outcome(Effect.BLOCKED)

        tool_kind = (
            self._objects[self._holding].kind
            if interaction.is_relational and self._holding
            else None
        )
        aff = catalogue.lookup(target.kind, action, tool_kind)
        if not aff.is_real:
            return Outcome(Effect.NOTHING)
        if not self._preconditions_hold(aff, target):
            return Outcome(Effect.BLOCKED)
        if self._rng.random() >= aff.reliability:
            return Outcome(Effect.NOTHING)
        return self._apply(aff, target)

    def _preconditions_hold(self, aff: Affordance, target: _Object) -> bool:
        held = self._objects[self._holding] if self._holding else None
        for p in aff.preconditions:
            if p is Precondition.GRIPPER_FREE and held is not None:
                return False
            if p is Precondition.HOLDING_ANY and held is None:
                return False
            if p is Precondition.HOLDING_HEAVY and (
                held is None or not held.spec.is_heavy
            ):
                return False
            if p is Precondition.TARGET_OPEN and not target.is_open:
                return False
            if p is Precondition.TARGET_CLOSED and target.is_open:
                return False
            if p is Precondition.TARGET_UNLOCKED and target.locked:
                return False
            if p is Precondition.TARGET_UPRIGHT and target.toppled:
                return False
        return True

    def _apply(self, aff: Affordance, target: _Object) -> Outcome:
        effect = Effect(aff.effect)
        displacement = height_gain = rotation = 0.0

        if effect is Effect.TRANSLATED:
            displacement = self._shove(target)
        elif effect is Effect.LIFTED:
            height_gain = 0.30
        elif effect is Effect.ROTATED:
            rotation = float(np.pi / 2)
        elif effect is Effect.CARRIED:
            target.carried = True
            target.support = None
            self._holding = target.object_id
        elif effect is Effect.SUPPORTED:
            displacement = self._place_on(target)
        elif effect is Effect.OPENED:
            target.is_open = True
        elif effect is Effect.CLOSED:
            target.is_open = False
        elif effect is Effect.TOPPLED:
            target.toppled = True

        remote = self._trigger(aff, target, effect)
        return Outcome(
            effect=effect,
            displacement=displacement,
            height_gain=height_gain,
            rotation=rotation,
            remote=remote,
            irreversible=effect in (Effect.TOPPLED, Effect.BROKE),
        )

    def _trigger(
        self, aff: Affordance, target: _Object, effect: Effect
    ) -> tuple[RemoteEffect, ...]:
        """Open the door linked to an actuated mechanism.

        Latching mechanisms stay open. The plate does not: `_release_momentary`
        closes its door again next step unless something heavy is resting on
        it, which is the entire reason the plate's secondary use is worth
        discovering.
        """
        if aff.remote_effect is None:
            return ()
        door_id = self._links.get(target.object_id)
        if door_id is None:
            return ()
        door = self._objects[door_id]
        door.locked = False
        door.is_open = True
        # Everything except the plate latches. The plate is the exception the
        # catalogue is built around.
        door.latched = door.latched or target.kind != "plate"
        return (RemoteEffect(door_id, Effect(aff.remote_effect)),)

    def _release_momentary_doors(self) -> None:
        """Close plate-opened doors that nothing heavy is holding down.

        A door already latched by a button, lever or valve stays open. Several
        mechanisms can share a door, and without this check a plate would close
        a door some other mechanism had permanently opened.
        """
        for actuator_id, door_id in self._links.items():
            plate = self._objects.get(actuator_id)
            if plate is None or plate.kind != "plate":
                continue
            door = self._objects[door_id]
            if door.latched or self._weighted(plate):
                continue
            door.is_open = False
            door.locked = True

    def _weighted(self, plate: _Object) -> bool:
        return any(
            o.support == plate.object_id and o.spec.is_heavy
            for o in self._objects.values()
        )

    # ------------------------------------------------------------------ #
    # effect helpers
    # ------------------------------------------------------------------ #
    def _shove(self, target: _Object) -> float:
        """Push distance falls off with hidden mass.

        The mass itself is never observable, but its consequence is: this is
        the leak through which an agent can learn that two identical-looking
        objects are not the same thing.
        """
        distance = float(np.clip(6.0 / max(target.spec.mass, 0.5), 0.15, 1.60))
        away = target.position - self._agent
        norm = float(np.linalg.norm(away))
        direction = away / norm if norm > 1e-6 else np.array([1.0, 0.0])
        target.position = np.clip(
            target.position + direction * distance, 0.0, self._floor
        )
        return distance

    def _place_on(self, support: _Object) -> float:
        held = self._objects[self._holding]  # guarded by needs_held_object
        moved = float(np.linalg.norm(support.position - held.position))
        held.position = support.position.copy()
        held.carried = False
        held.support = support.object_id
        self._holding = None
        return moved

    def _approach(self, target: _Object | None) -> Outcome:
        if target is None:
            return Outcome(Effect.BLOCKED)
        away = self._agent - target.position
        norm = float(np.linalg.norm(away))
        direction = away / norm if norm > 1e-6 else np.array([1.0, 0.0])
        self._agent = target.position + direction * (REACH * 0.8)
        self._heading = float(np.arctan2(-direction[1], -direction[0]))
        return Outcome(Effect.NOTHING)

    def _release(self) -> Outcome:
        if self._holding is None:
            return Outcome(Effect.BLOCKED)
        held = self._objects[self._holding]
        held.carried = False
        held.position = self._agent.copy()
        self._holding = None
        return Outcome(Effect.RELEASED)

    # ------------------------------------------------------------------ #
    # perception
    # ------------------------------------------------------------------ #
    def _distance(self, obj: _Object) -> float:
        if obj.carried:
            return 0.0
        return float(np.linalg.norm(obj.position - self._agent))

    def _view(self, obj: _Object) -> ObjectView:
        offset = obj.position - self._agent
        bearing = float(np.arctan2(offset[1], offset[0]) - self._heading)
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        distance = self._distance(obj)
        return ObjectView(
            object_id=obj.object_id,
            position=obj.position.copy(),
            extent=np.asarray(obj.spec.extent, dtype=np.float64),
            distance=distance,
            bearing=bearing,
            appearance=appearance_mod.describe(
                obj.spec, self._rng, is_open=obj.is_open,
                toppled=obj.toppled, noise=self._noise,
            ),
            within_reach=distance <= REACH,
            held=obj.carried,
        )

    def _observe(self) -> Observation:
        return Observation(
            t=self._t,
            agent_position=self._agent.copy(),
            agent_heading=self._heading,
            objects=tuple(self._view(o) for o in self._objects.values()),
            holding=self._holding,
            budget_remaining=self._budget,
        )


class WarehouseOracle:
    """Hidden ground truth for a `Warehouse`. **Evaluation only.**

    Agent code must never construct or call this. See `interface.AffordanceOracle`.
    """

    def __init__(self, world: Warehouse) -> None:
        self._world = world

    def object_kind(self, object_id: str) -> str:
        return self._world._objects[object_id].kind

    def ids_of_kind(self, kind: str) -> tuple[str, ...]:
        """Ids of every object of `kind` in the current layout.

        Object ids are opaque by design, so this is the only way to find a
        particular kind -- and it is ground truth, for evaluation and tests.
        """
        return tuple(
            i for i, o in self._world._objects.items() if o.kind == kind
        )

    def catalogue_affordances(self) -> tuple[Affordance, ...]:
        return catalogue.AFFORDANCES

    def present_kinds(self) -> frozenset[str]:
        """Kinds actually placed in the current layout."""
        return frozenset(o.kind for o in self._world._objects.values())

    def reachable_affordances(self) -> tuple[Affordance, ...]:
        """Affordances testable in this layout.

        An affordance whose object is absent, or whose required tool is absent,
        could not have been found however good the agent was. Scoring recall
        against the full catalogue instead would penalise agents for the layout
        they were handed.
        """
        present = self.present_kinds()
        return tuple(
            a for a in catalogue.AFFORDANCES
            if a.kind in present
            and (a.tool_kind is None or a.tool_kind in present)
        )

    def reachable_by_role(self, role: Role) -> tuple[Affordance, ...]:
        """The recall denominator for one role."""
        return tuple(a for a in self.reachable_affordances() if a.role is role)

    def summary(self) -> dict[str, Any]:
        """Layout composition, for reporting alongside results."""
        reachable = self.reachable_affordances()
        return {
            "backend": self._world.backend_name,
            "objects": len(self._world._objects),
            "kinds": sorted(self.present_kinds()),
            "reachable_affordances": len(reachable),
            "by_role": {
                r.name.lower(): sum(1 for a in reachable if a.role is r)
                for r in (Role.PRIMARY, Role.SECONDARY, Role.INCIDENTAL)
            },
        }
