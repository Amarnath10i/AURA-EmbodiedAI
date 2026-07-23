"""
Cross-Entropy Method (CEM) Planner.
Uses the DreamerWorldModel to imagine future rollouts and select actions
that maximize predicted intrinsic/extrinsic reward.
"""
import torch
import torch.nn as nn
from typing import Tuple

from lama.models.world_model import DreamerWorldModel

class CEMPlanner:
    def __init__(
        self, 
        world_model: DreamerWorldModel, 
        action_dim: int,
        horizon: int = 15,
        num_samples: int = 1000,
        num_elites: int = 100,
        num_iters: int = 5,
        device: str = "cuda"
    ):
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iters = num_iters
        self.device = device
        
    @torch.no_grad()
    def plan(self, h_current: torch.Tensor, z_current: torch.Tensor) -> torch.Tensor:
        """
        Plans a sequence of actions using CEM.
        h_current: (1, deter_dim)
        z_current: (1, stoch_dim)
        Returns: (action_dim,)
        """
        # Expand current state for num_samples parallel imaginations
        h_expand = h_current.repeat(self.num_samples, 1)
        z_expand = z_current.repeat(self.num_samples, 1)
        
        # Initial action distribution parameters
        action_mean = torch.zeros((self.horizon, self.action_dim), device=self.device)
        action_std = torch.ones((self.horizon, self.action_dim), device=self.device)
        
        best_action = action_mean[0]
        
        for i in range(self.num_iters):
            # 1. Sample action sequences: (num_samples, horizon, action_dim)
            actions = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * torch.randn(
                (self.num_samples, self.horizon, self.action_dim), device=self.device
            )
            
            # Clip actions to valid ranges (e.g. [-1, 1])
            actions = torch.clamp(actions, -1.0, 1.0)
            
            # 2. Imagine future states and get reward predictions
            h_seq, z_seq, preds = self.world_model.imagine(h_expand, z_expand, actions)
            
            # 3. Compute returns (Sum of predicted rewards)
            # preds["reward"] is (num_samples, horizon)
            returns = preds["reward"].sum(dim=1)
            
            # 4. Find Elites
            elite_idxs = torch.argsort(returns, descending=True)[:self.num_elites]
            elite_actions = actions[elite_idxs] # (num_elites, horizon, action_dim)
            
            # 5. Update Distribution
            action_mean = elite_actions.mean(dim=0)
            action_std = elite_actions.std(dim=0)
            
            # Save the best action just in case it's the final iteration
            best_action = elite_actions[0, 0]
            
        return best_action
