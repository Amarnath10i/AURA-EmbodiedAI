"""Tests for the lifelong affordance bank.

These pin the Bayesian thresholds documented in `bank.py` (calibrated by
simulation, not guessed), the revision-on-contradiction behaviour, and the
"only confirmed knowledge leaves the bank" boundary the rest of the loop will
depend on.
"""

from __future__ import annotations

from lama.env import Action, Effect, Outcome, RemoteEffect
from lama.memory.bank import AffordanceBank, Status

KEY = (0, Action.LIFT, None)


def observe_many(bank, key, outcomes, episode=0, object_id="x"):
    revisions = []
    for t, outcome in enumerate(outcomes):
        rev = bank.observe(*key, outcome, episode=episode, t=t, object_id=object_id)
        if rev is not None:
            revisions.append(rev)
    return revisions


def success(effect=Effect.LIFTED, **kw):
    return Outcome(effect, **kw)


def failure(effect=Effect.NOTHING):
    return Outcome(effect)


# --------------------------------------------------------------------------- #
# status lifecycle
# --------------------------------------------------------------------------- #
def test_unobserved_key_is_untested():
    bank = AffordanceBank()
    assert bank.belief(KEY) is None


def test_single_success_is_provisional_not_confirmed():
    """One trial must never look like knowledge."""
    bank = AffordanceBank()
    bank.observe(*KEY, success(), episode=0, t=0, object_id="x")
    assert bank.belief(KEY).status is Status.PROVISIONAL


def test_eight_clean_successes_confirm_a_deterministic_affordance():
    bank = AffordanceBank()
    revisions = observe_many(bank, KEY, [success()] * 8)
    assert bank.belief(KEY).status is Status.CONFIRMED
    assert len(revisions) == 1
    assert revisions[0].old_status is Status.PROVISIONAL
    assert revisions[0].new_status is Status.CONFIRMED


def test_twentyseven_clean_failures_refute_a_fresh_belief():
    bank = AffordanceBank()
    revisions = observe_many(bank, KEY, [failure()] * 27)
    assert bank.belief(KEY).status is Status.REFUTED
    assert revisions[-1].new_status is Status.REFUTED


def test_a_truly_unreliable_affordance_confirms_at_its_real_rate():
    """The point of Beta-Bernoulli: p=0.5 should end up CONFIRMED, not stuck
    PROVISIONAL forever, once there is enough evidence to pin the rate down."""
    bank = AffordanceBank()
    outcomes = [success(Effect.ACTUATED) if i % 2 == 0 else failure()
                for i in range(40)]
    observe_many(bank, KEY, outcomes)
    belief = bank.belief(KEY)
    assert belief.status is Status.CONFIRMED
    assert 0.4 < belief.mean < 0.6


def test_revision_only_fires_on_a_settled_status_boundary():
    """Ordinary evidence accumulation (PROVISIONAL -> PROVISIONAL) must not
    spam revisions; only crossing into or out of CONFIRMED/REFUTED counts."""
    bank = AffordanceBank()
    revisions = observe_many(bank, KEY, [success()] * 3)
    assert bank.belief(KEY).status is Status.PROVISIONAL
    assert revisions == []


def test_confirmed_belief_can_be_overturned_by_contradiction():
    """The crate/block scenario in miniature: strong confirming evidence,
    then strong contradicting evidence, must be able to change the bank's
    mind -- it is not allowed to freeze a belief once CONFIRMED."""
    bank = AffordanceBank()
    observe_many(bank, KEY, [success()] * 8)
    assert bank.belief(KEY).status is Status.CONFIRMED

    revisions = observe_many(bank, KEY, [failure()] * 60, episode=1)
    assert bank.belief(KEY).status is not Status.CONFIRMED
    assert any(r.old_status is Status.CONFIRMED for r in revisions)


def test_a_conflated_concept_settles_at_its_blended_rate():
    """If a concept merges two kinds that behave differently half the time
    each (crate/block sharing a concept, one holds the plate, one does not),
    the bank should settle on the blended rate rather than get stuck -- this
    is the documented cost of appearance-based generalisation, not a failure
    of the bank."""
    bank = AffordanceBank()
    # Both placements succeed (SUPPORTED) either way -- the concept cannot
    # tell block from crate by whether placing works. What differs is the
    # remote effect: only the block's half opens the door.
    outcomes = [
        Outcome(Effect.SUPPORTED, remote=(RemoteEffect("door", Effect.OPENED),))
        if i % 2 == 0 else Outcome(Effect.SUPPORTED)
        for i in range(60)
    ]
    observe_many(bank, KEY, outcomes)
    belief = bank.belief(KEY)
    assert belief.status is Status.CONFIRMED, "placing something always works"
    assert 0.35 < belief.remote_rate < 0.65, "but only half of it opens the door"


# --------------------------------------------------------------------------- #
# what a belief remembers
# --------------------------------------------------------------------------- #
def test_dominant_effect_tracks_the_most_common_outcome():
    bank = AffordanceBank()
    observe_many(bank, KEY, [success(Effect.LIFTED)] * 5 + [failure()] * 2)
    assert bank.belief(KEY).dominant_effect is Effect.LIFTED


def test_dominant_effect_is_none_before_any_evidence():
    from lama.memory.bank import Belief

    assert Belief(key=KEY).dominant_effect is None


def test_remote_rate_is_conditioned_on_success_only():
    """Failures must not dilute the remote-effect estimate: a verb that
    fails does not get a chance to have a remote effect at all."""
    bank = AffordanceBank()
    outcomes = (
        [Outcome(Effect.SUPPORTED, remote=(RemoteEffect("door", Effect.OPENED),))] * 5
        + [failure()] * 50   # many failures; must not move remote_rate
    )
    observe_many(bank, KEY, outcomes)
    assert bank.belief(KEY).remote_rate > 0.8


def test_evidence_is_capped_but_total_attempts_is_not():
    bank = AffordanceBank()
    observe_many(bank, KEY, [success()] * 50)
    belief = bank.belief(KEY)
    assert belief.total_attempts == 50
    assert len(belief.evidence) <= 20


def test_evidence_records_provenance():
    bank = AffordanceBank()
    bank.observe(0, Action.GRASP, None, success(Effect.CARRIED),
                episode=3, t=17, object_id="crate_0")
    ev = bank.belief((0, Action.GRASP, None)).evidence[-1]
    assert (ev.episode, ev.t, ev.object_id, ev.effect) == (3, 17, "crate_0", Effect.CARRIED)


# --------------------------------------------------------------------------- #
# key identity and the confirmed()/beliefs() boundary
# --------------------------------------------------------------------------- #
def test_different_keys_are_independent():
    bank = AffordanceBank()
    observe_many(bank, (0, Action.PUSH, None), [success(Effect.TRANSLATED)] * 8)
    assert bank.belief((0, Action.PULL, None)) is None
    assert bank.belief((1, Action.PUSH, None)) is None
    assert bank.belief((0, Action.PUSH, 2)) is None


def test_relational_key_includes_the_tool_concept():
    bank = AffordanceBank()
    bank.observe(0, Action.PLACE_ON, 5, success(Effect.SUPPORTED), episode=0, t=0, object_id="p")
    bank.observe(0, Action.PLACE_ON, 6, failure(), episode=0, t=1, object_id="p")
    assert bank.belief((0, Action.PLACE_ON, 5)).mean > 0.5
    assert bank.belief((0, Action.PLACE_ON, 6)).mean < 0.5


def test_confirmed_excludes_provisional_and_refuted():
    bank = AffordanceBank()
    observe_many(bank, (0, Action.PUSH, None), [success(Effect.TRANSLATED)] * 8)   # confirmed
    observe_many(bank, (1, Action.PUSH, None), [failure()] * 27)                    # refuted
    observe_many(bank, (2, Action.PUSH, None), [success(Effect.TRANSLATED)] * 2)    # provisional

    confirmed = bank.confirmed()
    assert len(confirmed) == 1
    assert confirmed[0].key == (0, Action.PUSH, None)
    assert len(bank.refuted()) == 1
    assert len(bank.beliefs()) == 3


def test_len_counts_distinct_keys_not_observations():
    bank = AffordanceBank()
    observe_many(bank, KEY, [success()] * 5)
    assert len(bank) == 1
    bank.observe(1, Action.PUSH, None, success(Effect.TRANSLATED), episode=0, t=0, object_id="y")
    assert len(bank) == 2
