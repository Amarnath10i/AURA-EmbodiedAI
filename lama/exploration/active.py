"""Active exploration: replaces dummy _nearest_unreached with information-gain planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env import Observation, ObjectView, Environment
from ..memory.memory import AffordanceMemory
from ..memory.concepts import ConceptMemory
from ..verification.select import select_next
from ..imagination.hypothesis import imagine, Hypothesis


@dataclass
class ExplorationTarget:
    """Target for exploration with expected info gain."""
    object_id: str
    expected_info_gain: float
    distance: float
    reachable: bool


class ActiveExplorer(nn.Module):
    """
    Active exploration using expected information gain (EIG).
    Replaces the dummy nearest-unreached heuristic.
    """

    def __init__(
        self,
        concept_memory: ConceptMemory,
        affordance_memory: AffordanceMemory,
        beta: float = 1.0,  # trade-off: info_gain vs distance cost
    ):
        super().__init__()
        self.concepts = concept_memory
        self.memory = affordance_memory
        self.beta = beta

    def compute_info_gain(
        self,
        obs: Observation,
        object_view: ObjectView,
    ) -> float:
        """
        Expected information gain from interacting with this object.
        Uses the bank's uncertainty over all verbs applicable to this object's concept.
        """
        concept_id = self.concepts.peek(object_view.appearance)
        if concept_id is None:
            # Unseen concept: maximum uncertainty
            return 1.0

        # Get all beliefs for this concept
        total_uncertainty = 0.0
        count = 0
        from ..memory.bank import _SETTLED
        for belief in self.memory.bank.beliefs_for_concept(concept_id):
            if belief.status not in _SETTLED:
                lo, hi = belief.credible_interval
                total_uncertainty += (hi - lo)
                count += 1

        if count == 0:
            return 0.0
        return total_uncertainty / count

    def select_exploration_target(
        self,
        obs: Observation,
    ) -> Optional[ExplorationTarget]:
        """Select best object to explore based on EIG / distance."""
        best = None
        best_score = -float("inf")

        for obj in obs.objects:
            if obj.held:
                continue

            eig = self.compute_info_gain(obs, obj)
            distance = obj.distance

            # Score: info_gain / (distance_cost + epsilon)
            score = eig / (self.beta * distance + 0.01)

            if score > best_score:
                best_score = score
                best = ExplorationTarget(
                    object_id=obj.object_id,
                    expected_info_gain=eig,
                    distance=distance,
                    reachable=obj.within_reach,
                )

        return best


class FrontierExplorer(nn.Module):
    """
    Frontier-based exploration: find boundary between known/unknown.
    More structured than random walk.
    """

    def __init__(
        self,
        concept_memory: ConceptMemory,
        affordance_memory: AffordanceMemory,
    ):
        super().__init__()
        self.concepts = concept_memory
        self.memory = affordance_memory

    def find_frontiers(self, obs: Observation) -> List[Tuple[ObjectView, float]]:
        """
        Frontiers: objects whose concept has unsettled beliefs.
        Returns (object, uncertainty_score).
        """
        frontiers = []
        for obj in obs.objects:
            if obj.held:
                continue
            concept_id = self.concepts.peek(obj.appearance)
            if concept_id is None:
                frontiers.append((obj, 1.0))  # completely unknown
            else:
                # Check if concept has unsettled beliefs
                from ..memory.bank import _SETTLED
                unsettled = sum(
                    1 for b in self.memory.bank.beliefs_for_concept(concept_id)
                    if b.status not in _SETTLED
                )
                if unsettled > 0:
                    frontiers.append((obj, min(1.0, unsettled / 5.0)))
        return frontiers

    def select_frontier(self, obs: Observation) -> Optional[ObjectView]:
        frontiers = self.find_frontiers(obs)
        if not frontiers:
            return None
        # Prefer closer frontiers with higher uncertainty
        frontiers.sort(key=lambda x: -x[1] / (x[0].distance + 0.01))
        return frontiers[0][0]


def active_verify_once(
    env: Environment,
    memory: AffordanceMemory,
    observation: Observation,
    explorer: ActiveExplorer,
) -> VerificationStep:
    """
    Enhanced verify_once with active exploration.
    1. Try verification hypotheses (imagine -> select)
    2. If nothing worth testing, explore frontiers
    3. If no frontiers, approach nearest unexplored
    """
    from ..verification.loop import VerificationStep, _nearest_unreached

    # Standard verification loop
    hypotheses = imagine(observation, memory)
    chosen = select_next(hypotheses, env.budget_remaining)

    if chosen is not None:
        step = env.step(chosen.action, chosen.target_id)
        revision = memory.observe(observation, step.record)
        return VerificationStep(chosen, None, step, revision)

    # No testable hypothesis -> active exploration
    frontier_obj = explorer.select_exploration_target(observation)
    if frontier_obj is not None and env.budget_remaining >= action_cost(Action.APPROACH):
        step = env.step(Action.APPROACH, frontier_obj.object_id)
        return VerificationStep(None, frontier_obj.object_id, step, None)

    # Fallback: original dummy heuristic
    target = _nearest_unreached(observation)
    if target is not None and env.budget_remaining >= action_cost(Action.APPROACH):
        step = env.step(Action.APPROACH, target.object_id)
        return VerificationStep(None, target.object_id, step, None)

    return VerificationStep(None, None, None, None)