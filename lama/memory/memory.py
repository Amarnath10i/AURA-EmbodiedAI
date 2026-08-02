"""`AffordanceMemory`: the single object the rest of the system talks to.

Concepts, the bank, and the ledger are each independently simple; the real
jobs left are (1) translating between the environment's vocabulary (opaque,
episode-scoped object ids) and memory's vocabulary (concept ids that persist
for the agent's whole lifetime), and (2) reacting when the bank notices a
concept looks like it secretly blends two real kinds.
"""

from __future__ import annotations

from ..env import InteractionRecord, Observation
from .bank import AffordanceBank, Revision, Status
from .concepts import ConceptCodebook
from .log import InteractionLog

#: How many times one appearance lineage may be split. Splitting a concept
#: whose kinds are genuinely appearance-identical (crate/block, the flagship
#: trap) cannot succeed -- there is no signal to find -- but nothing in
#: split_concept's own data can tell that apart in advance from a split that
#: DOES work (see concepts.py's docstring: the measured appearance-spread
#: difference between a real and a fake trap is negligible). Left uncapped,
#: a lifelong run keeps encountering fresh crate/block instances across new
#: episodes, and each fresh STUCK verdict re-splits an already-split
#: descendant, cascading indefinitely -- confirmed in a real 150-episode run,
#: which produced 27 STUCK beliefs, all of them retired (already re-split)
#: crate/block lineage. One split per lineage keeps the real benefit (a
#: lineage that CAN be separated, like lever/switch, is validated to improve
#: from one split) while bounding the pointless case to doubling once, not
#: growing without limit.
MAX_SPLIT_GENERATIONS: int = 1


class AffordanceMemory:
    """Turns a stream of `(Observation, InteractionRecord)` pairs into
    lifelong knowledge.

    Args:
        concepts: A codebook to use, or `None` to create one with the default
            merge radius. Passing one in lets several memories share a
            codebook, or a caller inspect it directly.
        bank: A bank to use, or `None` to create an empty one.
    """

    def __init__(
        self,
        concepts: ConceptCodebook | None = None,
        bank: AffordanceBank | None = None,
    ) -> None:
        self.concepts = concepts if concepts is not None else ConceptCodebook()
        self.bank = bank if bank is not None else AffordanceBank()
        self.log = InteractionLog()

    def observe(
        self, before: Observation, record: InteractionRecord
    ) -> Revision | None:
        """Fold one interaction into memory.

        `before` must be the observation returned immediately prior to the
        step that produced `record` -- it is where the tool's appearance is
        looked up, since `InteractionRecord` only carries a view of the
        target. Interactions that cannot reveal an affordance (`APPROACH`, or
        anything with no target) are ignored and this returns `None`.

        Returns the `Revision` if this observation changed what
        `bank.confirmed()` would return, else `None`.
        """
        interaction = record.interaction
        if not record.is_affordance_test or interaction.target is None:
            return None

        target_view = record.view_before or before.view(interaction.target)
        if target_view is None:
            return None
        target_concept = self.concepts.assign(
            target_view.appearance, record.episode, record.t
        )

        tool_concept = None
        if interaction.is_relational and record.tool_id is not None:
            tool_view = before.view(record.tool_id)
            if tool_view is not None:
                tool_concept = self.concepts.assign(
                    tool_view.appearance, record.episode, record.t
                )

        # A remote effect names an object elsewhere in the world (see
        # outcomes.RemoteEffect); resolving it to a concept is what lets the
        # bank later say "pressing this tends to open THAT KIND of thing",
        # not just "tends to open something". peek(), not assign(): merely
        # noticing a remote object changed must not be treated as having
        # interacted with it.
        remote_concepts = tuple(
            self._peek_concept(before, re.object_id)
            for re in record.outcome.remote
        )

        self.log.append(
            episode=record.episode, t=record.t, target_concept=target_concept,
            action=interaction.action, tool_concept=tool_concept,
            outcome=record.outcome, cost=record.cost,
            object_id=interaction.target, tool_id=record.tool_id,
        )

        revision = self.bank.observe(
            target_concept, interaction.action, tool_concept, record.outcome,
            episode=record.episode, t=record.t,
            object_id=interaction.target, tool_id=record.tool_id,
            remote_concepts=remote_concepts,
        )

        if revision is not None and revision.new_status is Status.STUCK:
            self._split_concept(revision.key, target_concept, record.episode)

        return revision

    def _peek_concept(self, before: Observation, object_id: str) -> int | None:
        view = before.view(object_id)
        if view is None:
            return None
        return self.concepts.peek(view.appearance)

    def _split_concept(
        self, key: tuple, target_concept: int, episode: int
    ) -> None:
        """A belief just turned STUCK: its evidence looks bimodal, meaning
        `target_concept` probably blends two real kinds (see bank.py's
        module docstring). Ask the codebook to split it, and give each
        resulting concept a fresh, mildly-informative belief at the same key
        -- the old, blended evidence is not trustworthy for either half, so
        restarting cleanly is correct, not wasteful.

        Does nothing once `target_concept` has already reached
        `MAX_SPLIT_GENERATIONS`: the belief simply stays `STUCK`, which
        already stops further budget going to this key (see `bank.
        SETTLED_STATUSES`) without cascading the concept space further.
        """
        if self.concepts.concept(target_concept).generation >= MAX_SPLIT_GENERATIONS:
            return
        action, tool_concept = key[1], key[2]
        id_a, id_b = self.concepts.split_concept(target_concept)
        for new_id in (id_a, id_b):
            self.bank.reopen((new_id, action, tool_concept))
