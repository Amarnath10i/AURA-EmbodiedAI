import torch
import torch.nn as nn

class LatentDecoder(nn.Module):
    """
    Reconstructs RGB observations from the combined latent state of both world models.
    Used for training the world model (reconstruction loss) and for visualization.
    """
    def __init__(self, state_dim: int, output_size: int = 256):
        """
        Args:
            state_dim: Total dimension of the concatenated dual world model state
                       (h_p + s_p + h_b + s_b = 2 * (det_state_dim + stoch_state_dim))
            output_size: Spatial resolution of the reconstructed image.
        """
        super().__init__()
        self.output_size = output_size
        
        # Project state into spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 4 * 4) # -> (256, 4, 4) spatial
        )
        
        # Upsample to output_size x output_size x 3
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 4->8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8->16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16->32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),   # 32->64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),    # 64->128
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1),     # 128->256
            nn.Sigmoid() # Output in [0, 1]
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, state_dim) concatenated dual model state
        Returns: (B, 3, output_size, output_size) reconstructed image
        """
        x = self.fc(state)
        x = x.reshape(x.size(0), 256, 4, 4)
        x = self.deconv(x)
        return x


class RewardModel(nn.Module):
    """
    Predicts the scalar reward from the combined dual world model state.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, state_dim)
        Returns: (B, 1) predicted reward
        """
        return self.net(state)
