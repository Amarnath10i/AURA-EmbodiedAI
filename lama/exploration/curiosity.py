"""
Curiosity-Driven Exploration (Phase 7)
======================================
Provides intrinsic reward based on prediction error of the world model.
The robot is rewarded for encountering states it cannot yet predict.
"""
import torch
import numpy as np
from lama.models.forward_model import DualWorldModel

class CuriosityModule:
    """
    Computes intrinsic reward as the prediction error between
    the world model's prior (predicted) and the actual observation.
    """
    def __init__(self, curiosity_scale: float = 0.1):
        self.curiosity_scale = curiosity_scale
        self.prediction_errors = []
        
    def compute_intrinsic_reward(
        self, 
        z_predicted: torch.Tensor, 
        z_actual: torch.Tensor
    ) -> float:
        """
        intrinsic_reward = λ * ||z_predicted - z_actual||^2
        """
        with torch.no_grad():
            error = torch.norm(z_predicted - z_actual, dim=-1).mean().item()
            self.prediction_errors.append(error)
            return self.curiosity_scale * error
    
    def combined_reward(self, extrinsic: float, z_pred: torch.Tensor, z_actual: torch.Tensor) -> float:
        """
        r_total = r_extrinsic + λ * r_intrinsic
        """
        intrinsic = self.compute_intrinsic_reward(z_pred, z_actual)
        return extrinsic + intrinsic
    
    def get_stats(self):
        if not self.prediction_errors:
            return {"mean_error": 0.0, "max_error": 0.0}
        return {
            "mean_error": float(np.mean(self.prediction_errors[-100:])),
            "max_error": float(np.max(self.prediction_errors[-100:])),
        }
