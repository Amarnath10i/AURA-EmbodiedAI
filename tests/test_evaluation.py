"""Tests for the evaluation harness.

The property that matters most: a confirmed belief (keyed by the agent's own
concept ids) has to be resolved back to real kind names before it can be
compared against ground truth at all. Get that resolution wrong and every
precision/recall number is meaningless while still looking like a number.
"""

from __future__ import annotations

from lama.env import Action
from lama.env.warehouse import Warehouse, WarehouseOracle
from lama.evaluation.eval import ConceptKindTracker, Evaluator, _make_oracle
from lama.memory.memory import AffordanceMemory
from lama.verification.loop import run_episode


def _env_factory(seed=0, **kw):
    return Warehouse(seed=seed, budget=60.0, **kw)


# --------------------------------------------------------------------------- #
# oracle construction
# --------------------------------------------------------------------------- #
def test_make_oracle_returns_the_right_type_for_the_numpy_backend():
    env = Warehouse(seed=0)
    oracle = _make_oracle(env)
    assert isinstance(oracle, WarehouseOracle)


def test_make_oracle_rejects_an_unknown_backend():
    class _FakeEnv:
        backend_name = "not-a-real-backend"

    try:
        _make_oracle(_FakeEnv())
        assert False, "should have raised"
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# ConceptKindTracker: the core fix
# --------------------------------------------------------------------------- #
def test_tracker_resolves_a_concept_to_its_true_kind():
    env = Warehouse(seed=0, layout_seed=1, budget=60.0)
    mem = AffordanceMemory()
    tracker = ConceptKindTracker()

    door = WarehouseOracle(env).ids_of_kind("door")[0]
    log_start = len(mem.log)
    from lama.env import Interaction
    before = env.reset()
    before = env.step(Interaction(Action.APPROACH, door)).observation
    step = env.step(Interaction(Action.PUSH, door))
    mem.observe(before, step.record)

    tracker.observe_episode(env, mem, log_start)
    concept = next(b.key[0] for b in mem.bank.beliefs())
    resolution = tracker.resolve(concept)
    assert resolution.dominant_kind == "door"
    assert not resolution.is_blended


def test_unresolved_concept_reports_none():
    tracker = ConceptKindTracker()
    resolution = tracker.resolve(999)
    assert resolution.dominant_kind is None
    assert resolution.kind_counts == {}


def test_tracker_detects_a_blended_concept():
    """The crate/block trap, forced deterministically: interact with both
    under the SAME concept id (guaranteed, since their appearance is
    identical) and confirm the tracker reports it as blended."""
    from lama.env import Interaction

    env = Warehouse(seed=0, layout_seed=1, budget=60.0)
    mem = AffordanceMemory()
    tracker = ConceptKindTracker()
    oracle = WarehouseOracle(env)
    crate = oracle.ids_of_kind("crate")[0]
    block = oracle.ids_of_kind("block")[0]

    log_start = len(mem.log)
    before = env.reset()
    for target in (crate, block):
        before = env.step(Interaction(Action.APPROACH, target)).observation
        step = env.step(Interaction(Action.PUSH, target))
        mem.observe(before, step.record)
        before = step.observation

    tracker.observe_episode(env, mem, log_start)
    concept_ids = {b.key[0] for b in mem.bank.beliefs() if b.key[1] is Action.PUSH}
    assert len(concept_ids) == 1, "crate and block must share one concept"
    resolution = tracker.resolve(next(iter(concept_ids)))
    assert resolution.is_blended
    assert set(resolution.kind_counts) == {"crate", "block"}

    blended = tracker.blended_concepts()
    assert len(blended) == 1


# --------------------------------------------------------------------------- #
# Evaluator: runs end to end without crashing, sane bounds
# --------------------------------------------------------------------------- #
def test_evaluate_efficiency_runs_and_returns_sane_values():
    ev = Evaluator(_env_factory, AffordanceMemory, run_episode)
    result = ev.evaluate_efficiency(n_episodes=5, seed=0)
    assert result["total_interactions"] >= 0
    assert result["confirmed_count"] >= 0
    assert result["interactions_per_confirmed"] > 0


def test_evaluate_precision_recall_runs_and_returns_bounded_values():
    ev = Evaluator(_env_factory, AffordanceMemory, run_episode)
    result = ev.evaluate_precision_recall(n_episodes=8, seed=1)
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0
    assert result["tp"] >= 0 and result["fp"] >= 0 and result["fn"] >= 0


def test_confirmed_beliefs_are_never_false_positives_by_construction():
    """A meaningful sanity check on the bank's own calibration, not just the
    evaluator's plumbing: with enough episodes, everything CONFIRMED should
    genuinely be real -- the bank's confirmation threshold is supposed to
    mean something."""
    ev = Evaluator(_env_factory, AffordanceMemory, run_episode)
    result = ev.evaluate_precision_recall(n_episodes=15, seed=0)
    assert result["fp"] == 0, "a confirmed belief that is not real ground truth"


def test_evaluate_retention_runs_and_returns_bounded_values():
    ev = Evaluator(_env_factory, AffordanceMemory, run_episode)
    result = ev.evaluate_retention(task_a_episodes=4, task_b_episodes=4, seed=2)
    assert 0.0 <= result["retention"] <= 1.0


def test_evaluate_transfer_runs_and_uses_the_real_include_held_out_flag():
    """Regression guard for the imported version's held_out=[...] kwarg,
    which did not match either backend's actual constructor parameter."""
    ev = Evaluator(_env_factory, AffordanceMemory, run_episode)
    result = ev.evaluate_transfer(train_episodes=3, test_episodes=3, seed=3)
    assert set(result["held_out_kinds"]) == {"switch", "drum", "bench"}
    assert 0.0 <= result["transfer_precision"] <= 1.0
    assert 0.0 <= result["transfer_recall"] <= 1.0
