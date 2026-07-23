"""
Model Predictive Path Integral (MPPI) Planner.
An advanced alternative to CEM for World Model planning.
"""
import torch
import torch.nn as nn
from typing import Tuple

from lama.models.world_model import DreamerWorldModel

class MPPIPlanner:
    def __init__(
        self, 
        world_model: DreamerWorldModel, 
        action_dim: int,
        horizon: int = 15,
        num_samples: int = 1000,
        temperature: float = 0.5,
        noise_sigma: float = 0.5,
        device: str = "cuda"
    ):
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.noise_sigma = noise_sigma
        self.device = device
        
        # Warm-start action sequence
        self.action_mean = torch.zeros((self.horizon, self.action_dim), device=self.device)
        
    @torch.no_grad()
    def plan(self, h_current: torch.Tensor, z_current: torch.Tensor) -> torch.Tensor:
        """
        Plans using MPPI.
        """
        h_expand = h_current.repeat(self.num_samples, 1)
        z_expand = z_current.repeat(self.num_samples, 1)
        
        # 1. Sample action noise
        noise = torch.randn((self.num_samples, self.horizon, self.action_dim), device=self.device) * self.noise_sigma
        
        # 2. Add to current mean sequence
        actions = self.action_mean.unsqueeze(0) + noise
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # 3. Imagine futures
        h_seq, z_seq, preds = self.world_model.imagine(h_expand, z_expand, actions)
        
        # 4. Compute trajectory costs (Negative Reward)
        returns = preds["reward"].sum(dim=1) # (num_samples,)
        costs = -returns
        
        # 5. MPPI Weighting
        beta = torch.min(costs)
        weights = torch.exp(-(costs - beta) / self.temperature)
        weights = weights / (weights.sum() + 1e-8)
        
        # 6. Update action mean using weighted noise
        weighted_noise = (weights.unsqueeze(1).unsqueeze(2) * noise).sum(dim=0)
        self.action_mean = self.action_mean + weighted_noise
        
        best_action = self.action_mean[0].clone()
        
        # 7. Shift action mean forward for the next time step (warm start)
        self.action_mean[:-1] = self.action_mean[1:].clone()
        self.action_mean[-1] *= 0.0
        
        return best_action
