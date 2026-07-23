"""
Affordance Predictor Head (Phase 6)
===================================
Once affordances have been discovered via clustering, this module
trains a lightweight MLP to predict affordance probabilities
directly from the visual latent z_t.

Architecture:
    Image -> Encoder -> Latent z -> MLP -> Affordance Probabilities
"""
import torch
import torch.nn as nn

class AffordancePredictor(nn.Module):
    """
    Predicts which affordances are possible for the current observation.
    Trained on the pseudo-labels produced by AffordanceDiscovery.
    """
    def __init__(self, latent_dim: int = 768, n_affordances: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_affordances)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim) — encoded observation
        Returns: (B, n_affordances) — logits for each affordance
        """
        return self.head(z)
    
    def predict_affordances(self, z: torch.Tensor, threshold: float = 0.5):
        """
        Returns a binary mask of active affordances and their probabilities.
        """
        with torch.no_grad():
            logits = self.forward(z)
            probs = torch.sigmoid(logits)
            active = probs > threshold
            return active, probs
