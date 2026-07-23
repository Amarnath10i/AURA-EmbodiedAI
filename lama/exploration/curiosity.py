"""
Curiosity and Intrinsic Motivation Module.
Computes Intrinsic Reward = Prediction Error + Ensemble Variance + Latent Novelty.
"""
import torch
import torch.nn as nn
from typing import Dict

class DynamicsEnsemble(nn.Module):
    """A lightweight ensemble of MLPs to estimate model uncertainty."""
    def __init__(self, deter_dim: int, stoch_dim: int, action_dim: int, num_models: int = 5):
        super().__init__()
        self.num_models = num_models
        in_dim = deter_dim + stoch_dim + action_dim
        
        # We use a single shared network predicting multiple outputs for efficiency
        # instead of 5 separate PyTorch modules
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, stoch_dim * num_models)
        )
        self.stoch_dim = stoch_dim

    def forward(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns ensemble predictions: (B, num_models, stoch_dim)"""
        x = torch.cat([h, z, action], dim=-1)
        preds = self.net(x) # (B, stoch_dim * num_models)
        B = x.shape[0]
        return preds.view(B, self.num_models, self.stoch_dim)

class CuriosityModule(nn.Module):
    def __init__(self, deter_dim: int, stoch_dim: int, action_dim: int):
        super().__init__()
        self.ensemble = DynamicsEnsemble(deter_dim, stoch_dim, action_dim)
        
    def compute_intrinsic_reward(
        self, 
        h_t: torch.Tensor, 
        z_t: torch.Tensor, 
        a_t: torch.Tensor, 
        z_next_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the intrinsic exploration reward for a transition.
        1. Ensemble Variance: Disagreement between ensemble models (Epistemic uncertainty)
        2. Prediction Error: Difference between mean ensemble prediction and actual z_{t+1}
        """
        # (B, num_models, stoch_dim)
        preds = self.ensemble(h_t, z_t, a_t) 
        
        # 1. Ensemble Variance (Disagreement)
        variance = preds.var(dim=1).mean(dim=-1) # (B,)
        
        # 2. Prediction Error
        mean_pred = preds.mean(dim=1) # (B, stoch_dim)
        pred_error = torch.nn.functional.mse_loss(mean_pred, z_next_true, reduction='none').mean(dim=-1) # (B,)
        
        # Total Intrinsic Reward
        # We scale variance higher because it drops off as the model learns, driving exploration to new frontiers
        intrinsic_reward = 0.5 * pred_error + 1.0 * variance
        return intrinsic_reward
