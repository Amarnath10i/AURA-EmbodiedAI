# AURA Warehouse Simulator (prototype backend)

A fast, numpy-only 2D warehouse world implementing the AURA environment spec, so the
**entire AURA loop can be prototyped end-to-end on any machine** before moving to
Isaac Lab. Same interface, cheaper physics.

## What's in the world
| Element | Behaviour | AURA failure cluster it feeds |
|---|---|---|
| Walls + shelf racks | Static obstacles with aisles | baseline navigation |
| Boxes | Static clutter; **humans occasionally move them** | "moved box" surprises |
| Doors | Randomly open/close over time | door failures |
| Humans | Walk between waypoints, re-plan unpredictably | moving-human failures |
| Forklifts | Patrol aisles, larger & faster | dynamic-obstacle failures |
| Goal ("the bottle") | Navigation target in the far room | task success metric |

## API (gymnasium-style)
```python
from sim import WarehouseSim

sim = WarehouseSim(seed=0)                 # deterministic given seed
obs, info = sim.reset()                    # obs: (96, 96, 3) uint8 egocentric RGB
obs, reward, term, trunc, info = sim.step(a)   # a in {0:fwd 1:left 2:right 3:stay}
sim.render_global()                        # full top-down RGB for videos/debug
sim.state_summary()                        # ground-truth context tags (validation only!)
```

`info` includes `collision`, `success`, `events` (e.g. `human_moved_box`,
`door_closed`) and `goal_dist`.

**Important:** `state_summary()` / `event_tags` are ground truth the *robot must not
see during training*. They exist to **validate** that the failure clusters AURA
discovers by itself line up with real situation types — that's your evaluation table.

## Files
- `sim.py` — the simulator (world, dynamics, collision, egocentric rendering)
- `collect.py` — Stage-0 random-walk data collection → `transitions_*.npz`
- `demo.py` — renders an episode GIF + observation sheet

## Quick start
```bash
python demo.py                       # warehouse_demo.gif + warehouse_frames.png
python collect.py --episodes 50 --seed 0 --out data
```

Dataset shards contain `obs, action, next_obs, event_tags, episode, t` — exactly
what the world-model training stage consumes.

## Throughput
~300 transitions/s per process (96×96 obs, single core). 2M transitions ≈ 2 h on
one core, or minutes across parallel processes with different `--seed`s.

## Migrating to Isaac Lab (Month 1–2)
Keep this file layout and API. Write an `IsaacWarehouseSim` exposing the same
`reset/step/render_global/state_summary` signature backed by Isaac Lab's warehouse
assets; `collect.py`, the encoder, world model, planner and failure loop all keep
working unchanged. Requirements on your GPU machine: NVIDIA RTX-class GPU, Isaac Sim
+ Isaac Lab install (see NVIDIA's docs), PyTorch with CUDA.

## Design notes
- Deterministic: identical (seed, action sequence) ⇒ identical trajectories.
- Egocentric observation = top-down crop rotated with the robot's heading — a
  stand-in for the RGB camera DINOv3 will encode; swap freely later.
- `layout_seed` lets you fix the map while varying dynamics (or generate **new
  warehouses** for the generalization metric).
