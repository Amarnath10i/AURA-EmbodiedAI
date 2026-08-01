"""Cross-Entropy Method (CEM) Planner with World Model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Optional
from lama.world_model.rssm import WorldModel, RSSMState
from lama.memory.bank import AffordanceBank
from collections import deque


@dataclass
class PlanResult:
    actions: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor


class CEMPlanner(nn.Module):
    """Model-Predictive Control with Cross-Entropy Method."""

    def __init__(
        self,
        world_model: WorldModel,
        action_dim: int = 12,
        horizon: int = 12,
        n_candidates: int = 512,
        n_elite: int = 64,
        n_iterations: int = 8,
        temperature: float = 0.5,
        discount: float = 0.99,
    ):
        super().__init__()
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.n_iterations = n_iterations
        self.temperature = temperature
        self.discount = discount

    @torch.no_grad()
    def plan(
        self,
        init_state: RSSMState,
        n_steps: int = 1,
    ) -> PlanResult:
        """Plan action sequence using CEM."""
        batch_size = init_state.deter.shape[0]
        device = init_state.deter.device

        # Expand initial state for candidates
        init_state_expanded = RSSMState(
            deter=init_state.deter.repeat(self.n_candidates, 1),
            stoch=init_state.stoch.repeat(self.n_candidates, 1),
            logits=init_state.logits.repeat(self.n_candidates, 1),
        )

        # Initialize action distribution
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device)

        for _ in range(self.n_iterations):
            # Sample candidates
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                self.n_candidates, self.horizon, self.action_dim, device=device
            )
            # Action space: one-hot for discrete verbs
            actions = F.gumbel_softmax(actions, tau=self.temperature, hard=True, dim=-1)

            # Imagine trajectory
            stochs, rewards, values, continues = self.world_model.imagine_trajectory(
                init_state_expanded, actions
            )

            # Compute returns (discounted sum)
            discounts = self.discount ** torch.arange(self.horizon, device=device)
            returns = (rewards * discounts).sum(dim=1)  # [n_candidates]

            # Elite selection
            elite_idx = returns.topk(self.n_elite).indices
            elite_actions = actions[elite_idx]

            # Update distribution
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=0.1)

        # Return best action sequence
        best_idx = returns.argmax()
        best_actions = actions[best_idx:best_idx+1]

        return PlanResult(
            actions=best_actions[0],
            values=values[best_idx],
            rewards=rewards[best_idx],
        )


class MPPIPlanner(nn.Module):
    """Model-Predictive Path Integral (MPPI) - gradient-free, sample-efficient."""

    def __init__(
        self,
        world_model: WorldModel,
        action_dim: int = 12,
        horizon: int = 12,
        n_samples: int = 256,
        lambda_: float = 1.0,
        noise_sigma: float = 0.5,
    ):
        super().__init__()
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_samples = n_samples
        self.lambda_ = lambda_
        self.noise_sigma = noise_sigma

    @torch.no_grad()
    def plan(self, init_state: RSSMState) -> PlanResult:
        batch_size = init_state.deter.shape[0]
        device = init_state.deter.device

        # Expand for samples
        init_state_expanded = RSSMState(
            deter=init_state.deter.repeat(self.n_samples, 1),
            stoch=init_state.stoch.repeat(self.n_samples, 1),
            logits=init_state.logits.repeat(self.n_samples, 1),
        )

        # Sample action sequences
        actions = torch.randn(
            self.n_samples, self.horizon, self.action_dim, device=device
        ) * self.noise_sigma
        actions = F.gumbel_softmax(actions, tau=1.0, hard=True, dim=-1)

        # Imagine trajectories
        stochs, rewards, values, continues = self.world_model.imagine_trajectory(
            init_state_expanded, actions
        )

        # Returns with soft-value (MPPI)
        discounts = self.discount ** torch.arange(self.horizon, device=device)
        returns = (rewards * discounts).sum(dim=1)

        # MPPI weight: softmax over returns
        weights = F.softmax(returns / self.lambda_, dim=0)
        weighted_actions = (weights.view(-1, 1, 1) * actions).sum(dim=0)

        return PlanResult(
            actions=weighted_actions,
            values=values,
            rewards=rewards,
        )


def make_planner(
    world_model: WorldModel,
    planner_type: str = "cem",
    **kwargs,
) -> nn.Module:
    """Factory for planners."""
    if planner_type == "cem":
        return CEMPlanner(world_model, **kwargs)
    elif planner_type == "mppi":
        return MPPIPlanner(world_model, **kwargs)
    elif planner_type == "regression":
        return RegressionPlanner(kwargs.get("bank"))
    else:
        raise ValueError(f"Unknown planner: {planner_type}")

class RegressionPlanner:
    """Symbolic backward-chaining regression planner over confirmed operators."""
    
    def __init__(self, bank: AffordanceBank):
        self.bank = bank
        
    def plan(self, goal_predicate: str, max_depth: int = 5) -> list[dict] | None:
        """
        Search backward from goal_predicate.
        Returns a sequence of operators to achieve the goal, or None if no plan found.
        """
        # Get all confirmed operators
        operators = [b.operator for b in self.bank.confirmed() if b.operator is not None]
        
        # Simple BFS regression search
        # Queue stores tuples of (current_goals, plan_so_far)
        queue = deque([({goal_predicate}, [])])
        visited = set()
        
        while queue:
            current_goals, plan = queue.popleft()
            
            # If no unsatisfied goals, plan is complete
            if not current_goals:
                return plan
                
            goals_tuple = frozenset(current_goals)
            if goals_tuple in visited or len(plan) >= max_depth:
                continue
            visited.add(goals_tuple)
            
            # Pick a goal to resolve (for simplicity, we just take one)
            goal = next(iter(current_goals))
            remaining_goals = current_goals - {goal}
            
            # Find operators whose effect matches the goal
            for op in operators:
                effect_state = op["effect"].get("state")
                if effect_state == goal:
                    # New subgoals are the preconditions of this operator
                    new_goals = remaining_goals.union(set(op["precondition"]))
                    new_plan = [op] + plan
                    queue.append((new_goals, new_plan))
                    
        return None