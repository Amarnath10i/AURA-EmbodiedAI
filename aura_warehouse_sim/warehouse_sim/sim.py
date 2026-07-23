"""
AURA Warehouse Simulator (lightweight prototype backend)
========================================================

A fast, dependency-light 2D warehouse world with the same *interface* you will
later back with Isaac Lab. It exists so the entire AURA loop (encoder -> world
model -> planner -> failure memory -> self-improvement) can be prototyped
end-to-end on any machine, cheaply, before touching GPU simulation.

World contents (matching the AURA spec):
  - Walls and shelving racks (static obstacles)
  - Boxes (static clutter, occasionally MOVED by humans -> prediction surprises)
  - Doors that are sometimes open, sometimes closed  -> "door failures"
  - Humans walking between waypoints                 -> "moving human failures"
  - Forklifts patrolling aisles                      -> dynamic obstacles
  - A goal item (the "bottle") to navigate to

API (gymnasium-style):
    sim = WarehouseSim(seed=0)
    obs, info = sim.reset()
    obs, reward, terminated, truncated, info = sim.step(action)

Observations: egocentric RGB uint8 array (obs_size x obs_size x 3), i.e. a
top-down camera centred on and rotated with the robot (a stand-in for the RGB
camera that DINOv3 will encode).

Actions (discrete, 5): 0=forward  1=turn left  2=turn right  3=stay  4=interact

Everything is deterministic given (seed, actions) -> reproducible experiments.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

# ----------------------------- palette (RGB) ------------------------------ #
COLORS = {
    "floor":    (232, 229, 222),
    "wall":     (60, 64, 72),
    "shelf":    (146, 106, 66),
    "box":      (203, 163, 92),
    "door_c":   (178, 68, 60),    # closed door (blocks)
    "door_o":   (120, 190, 120),  # open door (passable, drawn on floor)
    "human":    (66, 120, 200),
    "forklift": (240, 176, 32),
    "goal":     (60, 170, 90),
    "robot":    (40, 40, 46),
    "robot_dir": (250, 250, 250),
}


# ------------------------------- entities --------------------------------- #
@dataclass
class Human:
    pos: np.ndarray                 # (2,) float, world units
    waypoints: list                 # list of (2,) arrays
    wp_index: int = 0
    speed: float = 0.055
    radius: float = 0.32
    carry_cooldown: int = 0         # steps until this human may move a box again


@dataclass
class Forklift:
    pos: np.ndarray
    waypoints: list
    wp_index: int = 0
    speed: float = 0.075
    radius: float = 0.45


@dataclass
class Door:
    cell: tuple                     # (row, col) in the occupancy grid
    open: bool = True
    p_toggle: float = 0.004         # per-step probability of toggling


# ------------------------------ the simulator ------------------------------ #
class WarehouseSim:
    """2D warehouse world with dynamic obstacles, curriculum difficulty, and egocentric RGB rendering."""

    ACTIONS = {0: "forward", 1: "turn_left", 2: "turn_right", 3: "stay", 4: "interact"}
    N_ACTIONS = 5

    def __init__(
        self,
        seed: int = 0,
        size: int = 24,              # world is size x size cells
        obs_size: int = 96,          # egocentric observation resolution (px)
        obs_cells: float = 9.0,      # how many world cells the camera sees across
        px_per_cell: int = 10,       # global render resolution
        max_steps: int = 400,
        n_humans: int = 3,
        n_forklifts: int = 2,
        n_boxes: int = 14,
        n_doors: int = 4,
        difficulty: int = 4,             # 1=Empty, 2=Static Clutter, 3=Dynamic Agents, 4=Interactive (doors/box moving)
        layout_seed: int | None = None,  # fix layout while varying dynamics
    ):
        self.rng = np.random.default_rng(seed)
        self.layout_rng = np.random.default_rng(seed if layout_seed is None else layout_seed)
        self.size = size
        self.obs_size = obs_size
        self.obs_cells = obs_cells
        self.ppc = px_per_cell
        self.max_steps = max_steps
        
        # Difficulty scales the entities
        self.difficulty = difficulty
        self.n_humans = n_humans if difficulty >= 3 else 0
        self.n_forklifts = n_forklifts if difficulty >= 3 else 0
        self.n_boxes = n_boxes if difficulty >= 2 else 0
        self.n_doors = n_doors if difficulty >= 4 else 0

        self.move_step = 0.30        # robot forward distance per step (cells)
        self.turn_step = np.deg2rad(22.5)
        self.robot_radius = 0.36

        self._build_layout()
        self.reset(full=True)

    # ------------------------------ layout -------------------------------- #
    def _build_layout(self):
        """Static occupancy: walls + shelf racks with aisles; doors punched into
        an interior dividing wall. 0=free 1=wall 2=shelf."""
        s = self.size
        g = np.zeros((s, s), dtype=np.int8)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 1

        # interior dividing wall with door gaps (rooms A | B)
        wall_col = s // 2
        g[1:-1, wall_col] = 1

        # shelf racks: vertical racks with aisles, in both rooms
        for col in range(3, s - 3, 4):
            if abs(col - wall_col) <= 1:
                continue
            for row in range(3, s - 3):
                if row % 7 in (5, 6):        # aisle cross-cuts
                    continue
                g[row, col] = 2

        self.grid_static = g
        self.wall_col = wall_col

        # doors: gaps in the dividing wall
        door_rows = self.layout_rng.choice(
            np.arange(2, self.size - 2), size=self.n_doors, replace=False
        )
        self.door_cells = [(int(r), wall_col) for r in sorted(door_rows)]

    # ------------------------------ reset ---------------------------------- #
    def reset(self, seed: int | None = None, full: bool = False):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0

        g = self.grid_static.copy()

        # doors (state re-randomised each episode)
        self.doors = []
        for cell in self.door_cells:
            d = Door(cell=cell, open=bool(self.rng.random() < 0.7))
            g[cell] = 0
            self.doors.append(d)

        # boxes on free floor cells (avoid door cells)
        free = self._free_cells(g)
        self.box_cells = set()
        picks = self.rng.choice(len(free), size=self.n_boxes, replace=False)
        for i in picks:
            self.box_cells.add(tuple(free[i]))

        self.grid = g

        # dynamic agents
        self.humans = [
            Human(pos=self._rand_free_pos(), waypoints=[self._rand_free_pos() for _ in range(3)])
            for _ in range(self.n_humans)
        ]
        self.forklifts = [
            Forklift(pos=self._rand_free_pos(), waypoints=[self._rand_free_pos() for _ in range(3)])
            for _ in range(self.n_forklifts)
        ]

        # robot + goal ("the bottle") in opposite rooms when possible
        self.robot_pos = self._rand_free_pos(room="left")
        self.robot_theta = float(self.rng.uniform(-np.pi, np.pi))
        self.goal_pos = self._rand_free_pos(room="right")

        self.events = []             # world events this step (for failure analysis)
        return self._observe(), self._info()

    # ------------------------------ helpers -------------------------------- #
    def _free_cells(self, g=None):
        g = self.grid_static if g is None else g
        rows, cols = np.where(g == 0)
        return np.stack([rows, cols], axis=1)

    def _rand_free_pos(self, room: str | None = None) -> np.ndarray:
        cells = self._free_cells(self.grid_static)
        if room == "left":
            cells = cells[cells[:, 1] < self.wall_col - 1]
        elif room == "right":
            cells = cells[cells[:, 1] > self.wall_col + 1]
        cells = np.array([c for c in cells if tuple(c) not in getattr(self, "box_cells", set())])
        c = cells[self.rng.integers(len(cells))]
        return c.astype(np.float64) + 0.5

    def _blocked(self, pos: np.ndarray, radius: float) -> bool:
        """Circle vs occupancy collision (walls, shelves, closed doors, boxes)."""
        r0, c0 = int(pos[0]), int(pos[1])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, cc = r0 + dr, c0 + dc
                if not (0 <= r < self.size and 0 <= cc < self.size):
                    return True
                solid = self.grid[r, cc] in (1, 2) or (r, cc) in self.box_cells
                for d in self.doors:
                    if (r, cc) == d.cell and not d.open:
                        solid = True
                if solid:
                    # distance from circle centre to this cell's square
                    nearest = np.clip(pos, [r, cc], [r + 1, cc + 1])
                    if np.linalg.norm(pos - nearest) < radius:
                        return True
        return False

    def _hit_dynamic(self, pos: np.ndarray, radius: float):
        for h in self.humans:
            if np.linalg.norm(pos - h.pos) < radius + h.radius:
                return "human"
        for f in self.forklifts:
            if np.linalg.norm(pos - f.pos) < radius + f.radius:
                return "forklift"
        return None

    # ------------------------------- step ----------------------------------- #
    def step(self, action: int):
        assert action in self.ACTIONS, f"invalid action {action}"
        self.t += 1
        self.events = []
        collided_with = None

        # --- robot kinematics & interaction ---
        if action == 1:
            self.robot_theta += self.turn_step
        elif action == 2:
            self.robot_theta -= self.turn_step
        elif action == 4: # INTERACT
            direction = np.array([np.sin(self.robot_theta), np.cos(self.robot_theta)])
            target_cell = tuple((self.robot_pos + direction * 1.0).astype(int))
            # Test hypothesis: what happens if I interact?
            if target_cell in self.box_cells:
                # Push the box if the cell behind it is free
                push_target = tuple((self.robot_pos + direction * 2.0).astype(int))
                if self.grid[push_target] == 0 and push_target not in self.box_cells:
                    self.box_cells.remove(target_cell)
                    self.box_cells.add(push_target)
                    self.events.append("robot_pushed_box")
            for d in self.doors:
                if target_cell == d.cell:
                    d.open = not d.open
                    self.events.append(f"robot_toggled_door")
        elif action == 0:
            direction = np.array([np.sin(self.robot_theta), np.cos(self.robot_theta)])
            target = self.robot_pos + direction * self.move_step
            if self._blocked(target, self.robot_radius):
                collided_with = "static"
                self.events.append("collision_static")
            else:
                hit = self._hit_dynamic(target, self.robot_radius)
                if hit:
                    collided_with = hit
                    self.events.append(f"collision_{hit}")
                else:
                    self.robot_pos = target
        self.robot_theta = (self.robot_theta + np.pi) % (2 * np.pi) - np.pi

        # --- world dynamics ---
        self._step_doors()
        self._step_humans()
        self._step_forklifts()

        # --- task logic ---
        dist_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        success = dist_goal < 0.8
        reward = 1.0 if success else (-0.1 if collided_with else -0.005)
        terminated = bool(success)
        truncated = self.t >= self.max_steps

        info = self._info()
        info.update(collision=collided_with, success=success, events=list(self.events))
        return self._observe(), reward, terminated, truncated, info

    def _step_doors(self):
        for d in self.doors:
            if self.rng.random() < d.p_toggle:
                d.open = not d.open
                self.events.append(f"door_{'opened' if d.open else 'closed'}")

    def _step_humans(self):
        for h in self.humans:
            wp = h.waypoints[h.wp_index]
            v = wp - h.pos
            dist = np.linalg.norm(v)
            if dist < 0.3:
                h.wp_index = (h.wp_index + 1) % len(h.waypoints)
                # occasionally pick a brand-new waypoint (unpredictability)
                if self.rng.random() < 0.3:
                    h.waypoints[h.wp_index] = self._rand_free_pos()
            else:
                step = h.pos + (v / dist) * h.speed
                if not self._blocked(step, h.radius):
                    h.pos = step
                else:
                    h.waypoints[h.wp_index] = self._rand_free_pos()

            # humans sometimes MOVE A NEARBY BOX -> classic AURA surprise event
            # only happens on max difficulty
            if self.difficulty >= 4:
                if h.carry_cooldown > 0:
                    h.carry_cooldown -= 1
                elif self.rng.random() < 0.01 and self.box_cells:
                    cell = min(self.box_cells,
                               key=lambda c: np.linalg.norm(np.array(c) + 0.5 - h.pos))
                if np.linalg.norm(np.array(cell) + 0.5 - h.pos) < 2.5:
                    free = [tuple(c) for c in self._free_cells(self.grid)
                            if tuple(c) not in self.box_cells]
                    new_cell = free[self.rng.integers(len(free))]
                    self.box_cells.remove(cell)
                    self.box_cells.add(new_cell)
                    h.carry_cooldown = 80
                    self.events.append("human_moved_box")

    def _step_forklifts(self):
        for f in self.forklifts:
            wp = f.waypoints[f.wp_index]
            v = wp - f.pos
            dist = np.linalg.norm(v)
            if dist < 0.4:
                f.wp_index = (f.wp_index + 1) % len(f.waypoints)
            else:
                step = f.pos + (v / dist) * f.speed
                if not self._blocked(step, f.radius):
                    f.pos = step
                else:
                    f.waypoints[f.wp_index] = self._rand_free_pos()

    # ----------------------------- rendering -------------------------------- #
    def render_global(self) -> np.ndarray:
        """Full top-down RGB view of the warehouse (H, W, 3) uint8."""
        ppc = self.ppc
        img = np.empty((self.size * ppc, self.size * ppc, 3), dtype=np.uint8)
        img[:] = COLORS["floor"]

        def fill_cell(r, c, color, inset=0):
            img[r * ppc + inset:(r + 1) * ppc - inset,
                c * ppc + inset:(c + 1) * ppc - inset] = color

        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == 1:
                    fill_cell(r, c, COLORS["wall"])
                elif self.grid[r, c] == 2:
                    fill_cell(r, c, COLORS["shelf"], inset=1)
        for (r, c) in self.box_cells:
            fill_cell(r, c, COLORS["box"], inset=2)
        for d in self.doors:
            fill_cell(*d.cell, COLORS["door_o"] if d.open else COLORS["door_c"], inset=1)

        self._disc(img, self.goal_pos, 0.34, COLORS["goal"])
        for h in self.humans:
            self._disc(img, h.pos, h.radius, COLORS["human"])
        for f in self.forklifts:
            self._disc(img, f.pos, f.radius, COLORS["forklift"])

        # robot with heading tick
        self._disc(img, self.robot_pos, self.robot_radius, COLORS["robot"])
        tip = self.robot_pos + np.array([np.sin(self.robot_theta),
                                         np.cos(self.robot_theta)]) * self.robot_radius * 0.9
        self._disc(img, tip, 0.10, COLORS["robot_dir"])
        return img

    def _disc(self, img, pos, radius, color):
        ppc = self.ppc
        cy, cx = pos[0] * ppc, pos[1] * ppc
        rr = radius * ppc
        y0, y1 = int(max(0, cy - rr)), int(min(img.shape[0], cy + rr + 1))
        x0, x1 = int(max(0, cx - rr)), int(min(img.shape[1], cx + rr + 1))
        if y1 <= y0 or x1 <= x0:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1]
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= rr ** 2
        img[y0:y1, x0:x1][mask] = color

    def _observe(self) -> np.ndarray:
        """Egocentric RGB: crop around the robot, rotated so 'up' = heading."""
        global_img = self.render_global()
        ppc = self.ppc
        half = int(self.obs_cells * ppc / 2)

        # take a larger crop, rotate, then centre-crop -> no corner artefacts
        big = int(half * 1.5)
        pad = big + 2
        padded = np.pad(global_img, ((pad, pad), (pad, pad), (0, 0)),
                        constant_values=20)
        cy = int(self.robot_pos[0] * ppc) + pad
        cx = int(self.robot_pos[1] * ppc) + pad
        crop = padded[cy - big:cy + big, cx - big:cx + big]
        crop = _rotate_rgb(crop, np.degrees(self.robot_theta))
        c = crop.shape[0] // 2
        ego = crop[c - half:c + half, c - half:c + half]

        return _resize_rgb(ego, self.obs_size)

    # ------------------------------- misc ----------------------------------- #
    def _info(self):
        return {
            "t": self.t,
            "robot_pos": self.robot_pos.copy(),
            "robot_theta": float(self.robot_theta),
            "goal_pos": self.goal_pos.copy(),
            "goal_dist": float(np.linalg.norm(self.robot_pos - self.goal_pos)),
            "doors_open": [d.open for d in self.doors],
        }

    def state_summary(self) -> dict:
        """Ground-truth context tags — used later to *validate* discovered
        failure clusters (the sim knows the truth; the robot must discover it)."""
        near = lambda p, r: np.linalg.norm(self.robot_pos - p) < r
        return {
            "near_human": any(near(h.pos, 2.5) for h in self.humans),
            "near_forklift": any(near(f.pos, 3.0) for f in self.forklifts),
            "near_door": any(near(np.array(d.cell) + 0.5, 2.5) for d in self.doors),
            "near_box": any(near(np.array(c) + 0.5, 2.0) for c in self.box_cells),
        }


# ---------------------- tiny image ops (numpy only) ------------------------ #
def _resize_rgb(img: np.ndarray, out: int) -> np.ndarray:
    ys = (np.arange(out) * img.shape[0] / out).astype(int)
    xs = (np.arange(out) * img.shape[1] / out).astype(int)
    return img[ys][:, xs]


def _rotate_rgb(img: np.ndarray, deg: float) -> np.ndarray:
    """Nearest-neighbour rotation about the image centre."""
    th = np.deg2rad(deg)
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    ys, xs = np.mgrid[0:h, 0:w]
    ys = ys - cy
    xs = xs - cx
    src_y = (np.cos(th) * ys - np.sin(th) * xs + cy).round().astype(int)
    src_x = (np.sin(th) * ys + np.cos(th) * xs + cx).round().astype(int)
    valid = (src_y >= 0) & (src_y < h) & (src_x >= 0) & (src_x < w)
    out = np.full_like(img, 20)
    out[valid] = img[src_y[valid], src_x[valid]]
    return out
