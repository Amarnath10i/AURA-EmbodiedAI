"""IsaacWarehouse -- the Isaac Lab backend for the environment contract (D4).

STATUS: WRITTEN, NEVER RUN. This file was authored entirely from current
Isaac Lab documentation (isaac-sim.github.io/IsaacLab, pinned to the
isaaclab==2.1.0 / isaacsim==4.5.0 generation -- see
docs/ISAAC_LAB_SETUP.md) because the machine that wrote it has a GTX 1650
with 4GB VRAM, which cannot launch Isaac Sim at all. It has not been executed
once. Before trusting anything here beyond "it follows the documented API",
run `scripts/isaac_smoke_test.py` on real hardware and fix whatever breaks
first -- there will almost certainly be something, and fixing it from a real
traceback is worth far more than more speculation from here.

Design: only the PHYSICAL layer is new. Which verbs work on which kinds,
mechanism linking, reliability rolls, remote effects and preconditions are
the exact same `catalogue.lookup` / `affordance.Precondition` reasoning
`warehouse.py` already uses, deliberately kept close to line-for-line
identical so a bug fixed in one is easy to recognise and fix in the other.
The only things this file adds are: real rigid bodies, real forces for
push/pull/rotate/tip (so hidden mass drives displacement through actual
PhysX dynamics rather than the numpy backend's `6.0 / mass` formula), and
reading position back from real simulated poses.

Deliberately NOT modelled (see docs/DECISIONS.md, D6):
  * Grasping/lifting is state bookkeeping, exactly like `warehouse.py` --
    there is no articulated gripper. A held object's prim is teleported to
    follow the agent for visual consistency, nothing more.
  * Doors do not physically swing; `is_open`/`locked` are python state, same
    as the numpy backend. Isaac Lab can do articulated doors
    (`ArticulationCfg` with a revolute joint); that is future work, not
    required to preserve the research semantics (whether a mechanism's
    *effect* is discoverable, not whether it looks like it swings).
  * No collision-blocked navigation: `APPROACH` teleports the agent, exactly
    like `warehouse.py`. Neither backend models walls for movement purposes.
  * No camera rendering. Appearance observations reuse `appearance.describe`
    unchanged -- the same synthetic descriptor the numpy backend uses -- since
    no perception encoder exists yet in this project (see the README status
    table). Real RGB can be added later by replacing the appearance call
    inside `_view` with an actual camera capture, without touching anything
    else in this file or anything downstream of it.
  * Layout composition (which kinds, how many) is fixed for the lifetime of
    one `IsaacWarehouse` instance. Re-randomising it would need runtime prim
    deletion, an API this file's author found no clearly-documented pattern
    for while offline from real hardware. `reset()` re-randomises dynamics
    (door states, agent start position) by teleporting the same objects back
    to their spawn poses -- construct a new instance for a different layout.

The single highest-risk call in this file is `_apply_impulse`, which calls
`RigidObject.set_external_force_and_torque`. Its exact tensor shape and
keyword names have shifted across Isaac Lab releases (even the `main` branch
docs show an in-progress rename to `set_forces_and_torques`). If something
breaks, look there first -- the intended fix is confined to that one method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import appearance as appearance_mod
from . import catalogue
from .actions import Action, spec
from .affordance import Affordance, Precondition, Role
from .catalogue import KINDS, KindSpec
from .interface import Environment
from .layout import plan_kinds
from .outcomes import Effect, Outcome, RemoteEffect
from .types import Interaction, InteractionRecord, Observation, ObjectView, StepResult

#: Metres within which interaction verbs may be attempted. Matches
#: `warehouse.REACH` exactly -- there is no reason for the two backends to
#: disagree about what "in reach" means.
REACH: float = 1.0

#: Object kinds whose actuation opens a linked door. Matches `warehouse.ACTUATORS`.
ACTUATORS: tuple[str, ...] = ("button", "lever", "valve", "plate", "switch")

#: `(width, depth)` of the area objects are scattered across, in metres.
DEFAULT_FLOOR: tuple[float, float] = (6.0, 6.0)

#: Newtons applied for one push/pull/tip attempt, before PhysX and the
#: object's own hidden mass decide how far it actually goes.
DEFAULT_PUSH_FORCE: float = 25.0

#: Newton-metres applied for one rotate attempt.
DEFAULT_TORQUE: float = 4.0

#: Physics steps to let a force act before reading back the result. Higher
#: settles further but costs wall-clock time per interaction; untested, so
#: treat as a first guess to tune once this runs for real.
DEFAULT_SETTLE_STEPS: int = 15

#: Prim path prefix everything in one IsaacWarehouse instance lives under.
_ROOT = "/World/lama_warehouse"


def _require_isaaclab() -> None:
    """Fail with actionable instructions rather than a bare ImportError."""
    try:
        import isaaclab  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "IsaacWarehouse needs Isaac Lab installed and importable, and "
            "this error is as far as this code has ever been run -- see "
            "docs/ISAAC_LAB_SETUP.md before treating anything past this "
            "point as a bug in this file specifically. Install with:\n"
            "  pip install torch==2.5.1 torchvision==0.20.1 "
            "--index-url https://download.pytorch.org/whl/cu121\n"
            "  pip install isaaclab[isaacsim,all]==2.1.0 "
            "--extra-index-url https://pypi.nvidia.com"
        ) from exc


# --------------------------------------------------------------------------- #
# a process-wide singleton: Isaac Sim can only be launched once per process
# --------------------------------------------------------------------------- #
_app_handle: Any = None
_sim_handle: Any = None


def _ensure_app_and_sim(headless: bool, physics_dt: float) -> Any:
    """Launch Isaac Sim on first use; every later call reuses the same app
    and simulation context.

    Constructing a second `IsaacWarehouse` must NOT relaunch the simulator --
    that is expected to fail outright. Callers that want a fresh episode
    should call `reset()` on an existing instance, not construct a new one;
    see the module docstring's note on fixed layouts.
    """
    global _app_handle, _sim_handle
    if _sim_handle is not None:
        return _sim_handle

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=headless)
    _app_handle = launcher.app

    import isaaclab.sim as sim_utils

    sim_cfg = sim_utils.SimulationCfg(dt=physics_dt)
    _sim_handle = sim_utils.SimulationContext(sim_cfg)
    _sim_handle.set_camera_view(eye=[4.0, 4.0, 4.0], target=[0.0, 0.0, 0.0])
    return _sim_handle


@dataclass
class _IsaacObject:
    """Physical + affordance state for one spawned object.

    Mirrors `warehouse._Object` field-for-field on purpose (`is_open`,
    `locked`, `latched`, `toppled`, `carried`, `support`) so the affordance
    resolution logic below reads the same way in both files. `body` and
    `initial_pose` are the only fields the numpy backend has no equivalent
    of.
    """

    object_id: str
    kind: str
    body: Any                 # isaaclab.assets.RigidObject
    initial_pose: "np.ndarray"  # (7,) xyz + wxyz quaternion, spawn pose
    is_open: bool = False
    locked: bool = False
    latched: bool = False
    toppled: bool = False
    carried: bool = False
    support: str | None = None

    @property
    def kind_spec(self) -> KindSpec:
        return KINDS[self.kind]


class IsaacWarehouse(Environment):
    """The same warehouse contract, backed by real PhysX rigid bodies.

    See the module docstring for exactly what is and is not physically
    simulated, and for the caveat that this has never been executed.
    """

    backend_name = "isaac-lab"

    def __init__(
        self,
        seed: int = 0,
        budget: float = 60.0,
        n_objects: int = 10,
        include_held_out: bool = False,
        appearance_noise: float = appearance_mod.DEFAULT_NOISE,
        floor: tuple[float, float] = DEFAULT_FLOOR,
        headless: bool = True,
        physics_dt: float = 1.0 / 60.0,
        push_force: float = DEFAULT_PUSH_FORCE,
        torque: float = DEFAULT_TORQUE,
        settle_steps: int = DEFAULT_SETTLE_STEPS,
    ) -> None:
        _require_isaaclab()

        self._seed = seed
        self._budget_per_episode = budget
        self._n_objects = n_objects
        self._include_held_out = include_held_out
        self._noise = appearance_noise
        self._floor = np.asarray(floor, dtype=np.float64)
        self._push_force = push_force
        self._torque = torque
        self._settle_steps = settle_steps

        self._episode = -1
        self._t = 0
        self._budget = 0.0
        self._holding: str | None = None
        self._agent_pos = np.zeros(2)
        self._agent_heading = 0.0
        self._objects: dict[str, _IsaacObject] = {}
        self._links: dict[str, str] = {}

        self._sim = _ensure_app_and_sim(headless, physics_dt)
        self._rng = np.random.default_rng(seed)
        self._spawn_layout()
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

        import torch

        self._sim.reset()

        for obj in self._objects.values():
            pose = torch.tensor(
                obj.initial_pose, dtype=torch.float32, device=self._sim.device
            ).unsqueeze(0)
            obj.body.write_root_pose_to_sim(pose)
            obj.body.write_root_velocity_to_sim(
                torch.zeros((1, 6), dtype=torch.float32, device=self._sim.device)
            )
            obj.body.write_data_to_sim()
            obj.is_open = False
            obj.locked = obj.kind == "door"
            obj.latched = False
            obj.toppled = False
            obj.carried = False
            obj.support = None

        self._agent_pos = self._rng.uniform(0.1, 0.9, size=2) * self._floor
        self._agent_heading = float(self._rng.uniform(-np.pi, np.pi))

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

    def render(self) -> np.ndarray | None:
        """No camera pipeline is wired up yet; see the module docstring."""
        return None

    def close(self) -> None:
        """No-op: the app is a process-wide singleton (`_ensure_app_and_sim`)
        and is expected to close when the process exits, not mid-run."""

    # ------------------------------------------------------------------ #
    # layout: spawned once, never re-randomised (see module docstring)
    # ------------------------------------------------------------------ #
    def _spawn_layout(self) -> None:
        import isaaclab.sim as sim_utils

        layout_rng = np.random.default_rng(self._seed)
        kinds = plan_kinds(layout_rng, self._n_objects, self._include_held_out)

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func(f"{_ROOT}/ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8))
        light_cfg.func(f"{_ROOT}/light", light_cfg)

        placed: list[np.ndarray] = []
        for i, kind in enumerate(kinds):
            object_id = f"obj_{i:02d}"
            pos = self._free_position(layout_rng, placed)
            placed.append(pos)
            self._objects[object_id] = self._spawn_object(object_id, kind, pos)

        doors = [oid for oid, o in self._objects.items() if o.kind == "door"]
        actuators = [oid for oid, o in self._objects.items() if o.kind in ACTUATORS]
        self._links = {
            oid: doors[i % len(doors)] for i, oid in enumerate(actuators)
        } if doors else {}

    def _spawn_object(self, object_id: str, kind: str, pos: np.ndarray) -> _IsaacObject:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg

        kind_spec = KINDS[kind]
        w, d = kind_spec.extent
        h = max(kind_spec.height, 0.05)
        z = h / 2.0
        color = tuple(c / 255.0 for c in kind_spec.color)

        cfg = RigidObjectCfg(
            prim_path=f"{_ROOT}/{object_id}",
            spawn=sim_utils.CuboidCfg(
                size=(max(w, 0.02), max(d, 0.02), h),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=kind_spec.fixed,
                ),
                mass_props=sim_utils.MassPropertiesCfg(
                    mass=max(kind_spec.mass, 0.05)
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(float(pos[0]), float(pos[1]), z)
            ),
        )
        body = RigidObject(cfg)
        initial_pose = np.array(
            [pos[0], pos[1], z, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        obj = _IsaacObject(object_id=object_id, kind=kind, body=body,
                           initial_pose=initial_pose)
        obj.locked = kind == "door"
        return obj

    def _free_position(
        self, rng: np.random.Generator, placed: list[np.ndarray], margin: float = 1.0
    ) -> np.ndarray:
        for _ in range(200):
            p = rng.uniform(0.05, 0.95, size=2) * self._floor
            if all(np.linalg.norm(p - q) > margin for q in placed):
                return p
        return rng.uniform(0.05, 0.95, size=2) * self._floor

    # ------------------------------------------------------------------ #
    # affordance resolution -- deliberately close to warehouse.py's, see
    # the module docstring for why this is duplicated rather than shared
    # ------------------------------------------------------------------ #
    def _resolve(self, interaction: Interaction, target: _IsaacObject | None) -> Outcome:
        action = interaction.action
        s = spec(action)

        if action is Action.APPROACH:
            return self._approach(target)
        if action is Action.RELEASE:
            return self._release()

        assert target is not None
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

    def _preconditions_hold(self, aff: Affordance, target: _IsaacObject) -> bool:
        held = self._objects[self._holding] if self._holding else None
        for p in aff.preconditions:
            if p is Precondition.GRIPPER_FREE and held is not None:
                return False
            if p is Precondition.HOLDING_ANY and held is None:
                return False
            if p is Precondition.HOLDING_HEAVY and (
                held is None or not held.kind_spec.is_heavy
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

    def _apply(self, aff: Affordance, target: _IsaacObject) -> Outcome:
        effect = Effect(aff.effect)
        displacement = height_gain = rotation = 0.0

        if effect is Effect.TRANSLATED:
            displacement = self._physical_shove(target)
        elif effect is Effect.LIFTED:
            height_gain = 0.30  # state bookkeeping; see module docstring
        elif effect is Effect.ROTATED:
            rotation = self._physical_rotate(target)
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

        remote = self._trigger(aff, target)
        return Outcome(
            effect=effect,
            displacement=displacement,
            height_gain=height_gain,
            rotation=rotation,
            remote=remote,
            irreversible=effect in (Effect.TOPPLED, Effect.BROKE),
        )

    def _trigger(self, aff: Affordance, target: _IsaacObject) -> tuple[RemoteEffect, ...]:
        if aff.remote_effect is None:
            return ()
        door_id = self._links.get(target.object_id)
        if door_id is None:
            return ()
        door = self._objects[door_id]
        door.locked = False
        door.is_open = True
        door.latched = door.latched or target.kind != "plate"
        return (RemoteEffect(door_id, Effect(aff.remote_effect)),)

    def _release_momentary_doors(self) -> None:
        for actuator_id, door_id in self._links.items():
            plate = self._objects.get(actuator_id)
            if plate is None or plate.kind != "plate":
                continue
            door = self._objects[door_id]
            if door.latched or self._weighted(plate):
                continue
            door.is_open = False
            door.locked = True

    def _weighted(self, plate: _IsaacObject) -> bool:
        return any(
            o.support == plate.object_id and o.kind_spec.is_heavy
            for o in self._objects.values()
        )

    # ------------------------------------------------------------------ #
    # the physical layer -- the genuinely new, unverified part
    # ------------------------------------------------------------------ #
    def _apply_impulse(self, obj: _IsaacObject, direction: np.ndarray, magnitude: float) -> None:
        """Apply a horizontal force to `obj` for `self._settle_steps` physics
        steps, then clear it and let the object settle one more step.

        HIGHEST-RISK CALL IN THIS FILE -- see the module docstring.
        """
        import torch

        d = direction / (np.linalg.norm(direction) + 1e-9)
        force = torch.zeros((1, 1, 3), dtype=torch.float32, device=self._sim.device)
        force[0, 0, 0] = float(d[0] * magnitude)
        force[0, 0, 1] = float(d[1] * magnitude)
        zero_torque = torch.zeros((1, 1, 3), dtype=torch.float32, device=self._sim.device)

        obj.body.set_external_force_and_torque(force, zero_torque)
        for _ in range(self._settle_steps):
            obj.body.write_data_to_sim()
            self._sim.step()
            obj.body.update(self._sim.get_physics_dt())
        obj.body.set_external_force_and_torque(zero_torque, zero_torque)
        obj.body.write_data_to_sim()

    def _apply_torque(self, obj: _IsaacObject, magnitude: float) -> None:
        """Vertical-axis torque version of `_apply_impulse`."""
        import torch

        torque = torch.zeros((1, 1, 3), dtype=torch.float32, device=self._sim.device)
        torque[0, 0, 2] = float(magnitude)
        zero_force = torch.zeros((1, 1, 3), dtype=torch.float32, device=self._sim.device)

        obj.body.set_external_force_and_torque(zero_force, torque)
        for _ in range(self._settle_steps):
            obj.body.write_data_to_sim()
            self._sim.step()
            obj.body.update(self._sim.get_physics_dt())
        obj.body.set_external_force_and_torque(zero_force, zero_force)
        obj.body.write_data_to_sim()

    def _physical_shove(self, target: _IsaacObject) -> float:
        """Push `target` away from the agent with a real force; hidden mass
        and PhysX -- not a formula -- decide how far it actually goes."""
        before = self._body_pos(target)
        away = before[:2] - self._agent_pos
        self._apply_impulse(target, away, self._push_force)
        after = self._body_pos(target)
        return float(np.linalg.norm(after[:2] - before[:2]))

    def _physical_rotate(self, target: _IsaacObject) -> float:
        self._apply_torque(target, self._torque)
        return float(np.pi / 2)  # reported nominally; real yaw change is not
                                 # decoded from the quaternion in this pass

    def _place_on(self, support: _IsaacObject) -> float:
        import torch

        held = self._objects[self._holding]
        before = self._body_pos(held)
        support_pos = self._body_pos(support)
        target_pos = support_pos.copy()
        target_pos[2] = support_pos[2] + max(held.kind_spec.height, 0.05)
        pose = torch.tensor(
            [*target_pos, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32,
            device=self._sim.device,
        ).unsqueeze(0)
        held.body.write_root_pose_to_sim(pose)
        held.body.write_root_velocity_to_sim(
            torch.zeros((1, 6), dtype=torch.float32, device=self._sim.device)
        )
        held.carried = False
        held.support = support.object_id
        self._holding = None
        return float(np.linalg.norm(target_pos[:2] - before[:2]))

    def _approach(self, target: _IsaacObject | None) -> Outcome:
        if target is None:
            return Outcome(Effect.BLOCKED)
        pos = self._body_pos(target)[:2]
        away = self._agent_pos - pos
        norm = float(np.linalg.norm(away))
        direction = away / norm if norm > 1e-6 else np.array([1.0, 0.0])
        self._agent_pos = pos + direction * (REACH * 0.8)
        self._agent_heading = float(np.arctan2(-direction[1], -direction[0]))
        return Outcome(Effect.NOTHING)

    def _release(self) -> Outcome:
        import torch

        if self._holding is None:
            return Outcome(Effect.BLOCKED)
        held = self._objects[self._holding]
        pose = torch.tensor(
            [self._agent_pos[0], self._agent_pos[1],
             max(held.kind_spec.height, 0.05) / 2.0, 1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32, device=self._sim.device,
        ).unsqueeze(0)
        held.body.write_root_pose_to_sim(pose)
        held.carried = False
        self._holding = None
        return Outcome(Effect.RELEASED)

    def _body_pos(self, obj: _IsaacObject) -> np.ndarray:
        """Real simulated position, `(3,)`, world frame.

        `.numpy()` on a CPU tensor returns a VIEW sharing the tensor's
        memory, not a copy. Physics backends commonly mutate root state
        buffers in place on every step, so a snapshot taken here for a
        before/after comparison (as `_physical_shove` does) would silently
        track the tensor's live value instead of the moment it was read --
        making every measured displacement 0. `.copy()` is what makes
        `before` actually mean "before".
        """
        return obj.body.data.root_pos_w[0].detach().cpu().numpy().copy()

    def _distance(self, obj: _IsaacObject) -> float:
        if obj.carried:
            return 0.0
        return float(np.linalg.norm(self._body_pos(obj)[:2] - self._agent_pos))

    # ------------------------------------------------------------------ #
    # perception -- unchanged from warehouse.py's model on purpose
    # ------------------------------------------------------------------ #
    def _view(self, obj: _IsaacObject) -> ObjectView:
        pos = self._body_pos(obj)[:2]
        offset = pos - self._agent_pos
        bearing = float(np.arctan2(offset[1], offset[0]) - self._agent_heading)
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        distance = self._distance(obj)
        return ObjectView(
            object_id=obj.object_id,
            position=pos.astype(np.float64),
            extent=np.asarray(obj.kind_spec.extent, dtype=np.float64),
            distance=distance,
            bearing=bearing,
            appearance=appearance_mod.describe(
                obj.kind_spec, self._rng, is_open=obj.is_open,
                toppled=obj.toppled, noise=self._noise,
            ),
            within_reach=distance <= REACH,
            held=obj.carried,
        )

    def _observe(self) -> Observation:
        return Observation(
            t=self._t,
            agent_position=self._agent_pos.copy(),
            agent_heading=self._agent_heading,
            objects=tuple(self._view(o) for o in self._objects.values()),
            holding=self._holding,
            budget_remaining=self._budget,
        )


class IsaacWarehouseOracle:
    """Hidden ground truth for an `IsaacWarehouse`. **Evaluation only.**

    Identical in shape to `warehouse.WarehouseOracle` -- see that class for
    why this split exists (`interface.AffordanceOracle`).
    """

    def __init__(self, world: IsaacWarehouse) -> None:
        self._world = world

    def object_kind(self, object_id: str) -> str:
        return self._world._objects[object_id].kind

    def ids_of_kind(self, kind: str) -> tuple[str, ...]:
        return tuple(
            i for i, o in self._world._objects.items() if o.kind == kind
        )

    def catalogue_affordances(self) -> tuple[Affordance, ...]:
        return catalogue.AFFORDANCES

    def present_kinds(self) -> frozenset[str]:
        return frozenset(o.kind for o in self._world._objects.values())

    def reachable_affordances(self) -> tuple[Affordance, ...]:
        present = self.present_kinds()
        return tuple(
            a for a in catalogue.AFFORDANCES
            if a.kind in present
            and (a.tool_kind is None or a.tool_kind in present)
        )

    def reachable_by_role(self, role: Role) -> tuple[Affordance, ...]:
        return tuple(a for a in self.reachable_affordances() if a.role is role)

    def summary(self) -> dict[str, Any]:
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
