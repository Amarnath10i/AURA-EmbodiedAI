"""
Affordance Predictor Head.
Predicts the likelihood of each discovered affordance cluster given the current latent state.
"""
import torch
import torch.nn as nn

class AffordancePredictor(nn.Module):
    def __init__(self, deter_dim: int, stoch_dim: int, num_clusters: int):
        super().__init__()
        in_dim = deter_dim + stoch_dim
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_clusters) # Outputs logits for CrossEntropyLoss
        )
        
    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Predicts affordance cluster logits.
        h: (B, deter_dim)
        z: (B, stoch_dim)
        Returns: (B, num_clusters)
        """
        x = torch.cat([h, z], dim=-1)
        return self.net(x)
