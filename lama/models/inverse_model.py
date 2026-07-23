import torch
import torch.nn as nn

class InverseDynamicsModel(nn.Module):
    """
    Predicts the action that was taken to transition from z_t to z_{t+1}.
    This grounds the latent space, forcing it to capture controllable features.
    """
    def __init__(self, action_dim: int, latent_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            z_t: (B, latent_dim)
            z_next: (B, latent_dim)
        Returns:
            action_logits: (B, action_dim)
        """
        x = torch.cat([z_t, z_next], dim=-1)
        return self.net(x)
