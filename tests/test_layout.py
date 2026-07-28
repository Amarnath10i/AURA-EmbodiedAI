"""Tests for the backend-shared kind-planning logic.

Thin on purpose: `test_warehouse.py` already exercises this thoroughly
end-to-end through `Warehouse`. These pin the function directly so a second
backend can rely on it without re-deriving these guarantees itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from lama.env.layout import MIN_OBJECTS, REQUIRED_KINDS, plan_kinds


def test_required_kinds_are_always_present():
    rng = np.random.default_rng(0)
    chosen = plan_kinds(rng, n_objects=10, include_held_out=False)
    assert set(REQUIRED_KINDS) <= set(chosen)


def test_at_least_one_actuator_kind_is_added():
    """The guarantee is "a mechanism is always reachable" -- at least one,
    not exactly one. The random fill can draw a second actuator kind too,
    same as any other kind; that is fine and not specially excluded."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        chosen = plan_kinds(rng, n_objects=10, include_held_out=False)
        actuators = [k for k in chosen if k in ("button", "lever", "valve")]
        assert len(actuators) >= 1


def test_returns_exactly_n_objects():
    rng = np.random.default_rng(0)
    assert len(plan_kinds(rng, n_objects=15, include_held_out=False)) == 15


def test_rejects_too_few_objects_for_the_required_kinds():
    with pytest.raises(ValueError):
        plan_kinds(np.random.default_rng(0), n_objects=2, include_held_out=False)


def test_min_objects_accounts_for_the_mandatory_actuator():
    """REQUIRED_KINDS alone is not enough room: one actuator kind is always
    added on top, so the true floor is one more than len(REQUIRED_KINDS)."""
    assert MIN_OBJECTS == len(REQUIRED_KINDS) + 1
    with pytest.raises(ValueError):
        plan_kinds(np.random.default_rng(0), n_objects=len(REQUIRED_KINDS),
                  include_held_out=False)


def test_returns_exactly_n_objects_at_the_minimum():
    rng = np.random.default_rng(0)
    chosen = plan_kinds(rng, n_objects=MIN_OBJECTS, include_held_out=False)
    assert len(chosen) == MIN_OBJECTS


def test_held_out_kinds_are_excluded_by_default():
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(200):
        seen.update(plan_kinds(rng, n_objects=10, include_held_out=False))
    assert not seen & {"switch", "drum", "bench"}


def test_held_out_kinds_can_appear_when_enabled():
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(200):
        seen.update(plan_kinds(rng, n_objects=10, include_held_out=True))
    assert seen & {"switch", "drum", "bench"}


def test_order_is_shuffled_not_required_kinds_first():
    """Regression guard: if this ever comes back sorted with REQUIRED_KINDS
    first, object ids (assigned by position) would correlate with kind again."""
    rng = np.random.default_rng(2)
    first_slots = [plan_kinds(rng, 10, False)[0] for _ in range(30)]
    assert len(set(first_slots)) > 1, "first slot should vary across calls"
