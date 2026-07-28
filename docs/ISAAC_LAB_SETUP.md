# Isaac Lab backend: setup and status

## Status: written, never run

`lama/env/isaac_warehouse.py` was authored entirely from current Isaac Lab
documentation. The machine that wrote it has a GTX 1650 with 4GB VRAM, which
cannot launch Isaac Sim at all -- no RTX cores, below the minimum VRAM for
every generation of Isaac Sim, including old ones. It has never been executed.

It has been verified as far as that constraint allows: `tests/test_isaac_warehouse_logic.py`
injects a fake `isaaclab` package (built from real `torch` tensors, not
mocks that accept anything) into `sys.modules`, so every line of
`IsaacWarehouse` actually runs -- every method call, every attribute access,
every tensor shape assumption -- against something that enforces shapes and
keyword names the way the real library would. That caught one real bug (a
numpy/torch memory-aliasing issue that silently zeroed every displacement
measurement) before it could reach real hardware. What it cannot verify is
whether the fake's guessed API surface (particularly
`RigidObject.set_external_force_and_torque`'s exact signature) matches the
real one -- that has genuinely shifted across Isaac Lab releases, and only
running against real Isaac Sim settles it.

**Run `scripts/isaac_smoke_test.py` before anything else.** It touches
nothing from this project -- it is close to Isaac Lab's own tutorial code,
adapted to spawn one cuboid, drop it, push it, and report whether it moved.
If that fails, the problem is the install or this exact API surface having
moved, not `IsaacWarehouse` specifically. Fix that script until it passes,
*then* try `scripts/run_lama_isaac.py`.

## Why this exact version is pinned

Isaac Sim's minimum GPU spec has climbed release over release:

| Isaac Sim generation | Stated minimum GPU |
|---|---|
| 4.5.0 (this pin) | RTX 3070, 8GB VRAM |
| current `main` docs | RTX 4080, 16GB VRAM |

An RTX 4060 (8GB) clears the 4.5.0-era minimum but not the current one.
Installing "latest" Isaac Lab will very likely refuse to run well, or at all,
on this hardware. Pin the versions below; do not `pip install --upgrade`
inside the Isaac environment without re-checking the GPU requirement first.

## Setup

Use a **separate** Python 3.10 virtual environment from whatever this repo's
main test suite runs under -- Isaac Sim 4.5.0 requires Python 3.10
specifically, and this project's own code has no reason to avoid it, but do
not assume your existing environment happens to already be 3.10.

```powershell
py -3.10 -m venv .venv-isaac
.venv-isaac\Scripts\Activate.ps1
pip install --upgrade pip

# CUDA 12 build, matching a modern driver:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Isaac Lab, with Isaac Sim as a bundled dependency:
pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com

# This project's own dependencies, in the same environment:
pip install -r requirements.txt
```

The first Isaac Sim launch pulls additional extensions from NVIDIA's
registry and can take upwards of ten minutes. That is expected, not a hang.

Driver: recommended 537.58+ (Windows) for this generation. Newer drivers
within the same major series are normally fine; if the smoke test reports a
driver mismatch, that is the first thing to check.

## Order of operations on the real machine

1. `python scripts/isaac_smoke_test.py` -- confirms Isaac Lab itself works
   here. Fix this first if it fails; do not debug `IsaacWarehouse` before this
   passes.
2. `python scripts/run_lama_isaac.py --episodes 1` -- constructs an
   `IsaacWarehouse`, runs one episode of the exact same verification loop
   used against the numpy backend, and reports what memory learned. Compare
   its printed layout summary and any confirmed beliefs against what
   `python -c "from lama.env.warehouse import Warehouse; ..."` produces on the
   numpy backend for a sanity check that both backends are telling a
   consistent story.
3. Report back exactly what broke, with the full traceback. Given the
   verification already done, failures are most likely narrow: a tensor
   shape or keyword argument at the one call site flagged in
   `isaac_warehouse.py`'s module docstring as highest-risk
   (`_apply_impulse`), or a renamed method if the installed version drifts
   from the 2.1.0 pin above.

## What this backend deliberately does not do (yet)

See `lama/env/isaac_warehouse.py`'s module docstring for the full list and
the reasoning behind each: no articulated grasping (state bookkeeping,
matching the numpy backend), no physically-swinging doors, no
collision-blocked navigation, no camera rendering, and layout composition is
fixed for the lifetime of one instance rather than re-randomised every
episode. None of these are required for the research question LAMA asks; all
are documented, deliberate scope cuts, not oversights.
