"""`AffordanceMemory`: the single object the rest of the system talks to.

Concepts, the bank, and the ledger are each independently simple; the only
real job left is translating between the environment's vocabulary (opaque,
episode-scoped object ids) and memory's vocabulary (concept ids that persist
for the agent's whole lifetime). That translation happens exactly once, here,
so nothing downstream has to know that object ids are not what they are keyed
on.
"""

from __future__ import annotations

from ..env import InteractionRecord, Observation
from .bank import AffordanceBank, Revision
from .concepts import ConceptCodebook
from .log import InteractionLog


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

        self.log.append(
            episode=record.episode, t=record.t, target_concept=target_concept,
            action=interaction.action, tool_concept=tool_concept,
            outcome=record.outcome, cost=record.cost,
            object_id=interaction.target, tool_id=record.tool_id,
        )
        return self.bank.observe(
            target_concept, interaction.action, tool_concept, record.outcome,
            episode=record.episode, t=record.t,
            object_id=interaction.target, tool_id=record.tool_id,
        )
