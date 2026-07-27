"""The raw interaction ledger.

Every attempted interaction that could reveal an affordance is appended here,
tagged with the concept ids `AffordanceMemory` assigned to its target and
tool. This is deliberately dumb storage: it does not decide what is true, the
`AffordanceBank` does. What it gives the rest of the project is a full
evidence trail -- what was actually spent, in what order, to reach any given
belief -- which is what makes "interactions per confirmed affordance" and
similar efficiency metrics computable after the fact, and what a `Revision`
event can point back into.

In-memory and list-backed for now. Swapping in SQLite for persistence across
process runs is a contained change if the project ever needs a run to survive
a restart; nothing downstream should need to know the difference, so nothing
here exposes storage details beyond the query methods below.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..env import Action, Outcome


@dataclass(frozen=True)
class LoggedInteraction:
    """One ledger entry: what was attempted, on what concepts, and what happened."""

    episode: int
    t: int
    target_concept: int
    action: Action
    tool_concept: int | None
    outcome: Outcome
    cost: float
    object_id: str
    tool_id: str | None


class InteractionLog:
    """Append-only ledger of attempted interactions, queryable by concept."""

    def __init__(self) -> None:
        self._entries: list[LoggedInteraction] = []

    def append(
        self,
        *,
        episode: int,
        t: int,
        target_concept: int,
        action: Action,
        tool_concept: int | None,
        outcome: Outcome,
        cost: float,
        object_id: str,
        tool_id: str | None,
    ) -> LoggedInteraction:
        entry = LoggedInteraction(
            episode=episode, t=t, target_concept=target_concept, action=action,
            tool_concept=tool_concept, outcome=outcome, cost=cost,
            object_id=object_id, tool_id=tool_id,
        )
        self._entries.append(entry)
        return entry

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> tuple[LoggedInteraction, ...]:
        return tuple(self._entries)

    def for_concept(self, concept_id: int) -> tuple[LoggedInteraction, ...]:
        """Every attempt whose target was this concept."""
        return tuple(e for e in self._entries if e.target_concept == concept_id)

    def for_key(
        self, target_concept: int, action: Action, tool_concept: int | None
    ) -> tuple[LoggedInteraction, ...]:
        """Every attempt matching one `AffordanceBank` key, in order.

        The evidence trail for a single belief -- what a `Revision` on that
        key is a summary of.
        """
        return tuple(
            e for e in self._entries
            if e.target_concept == target_concept
            and e.action == action
            and e.tool_concept == tool_concept
        )

    def total_cost(self) -> float:
        """Total budget spent across every logged attempt."""
        return sum(e.cost for e in self._entries)
