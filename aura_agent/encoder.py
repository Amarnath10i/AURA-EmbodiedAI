import torch
import torch.nn as nn
from .config import AgentConfig

class Encoder(nn.Module):
    """
    Visual Encoder: Compresses 96x96x3 RGB observations into a compact latent vector.
    Architecture: 4 Conv2d layers with stride 2, followed by a linear layer.
    """
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        self.conv = nn.Sequential(
            nn.Conv2d(config.obs_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Calculate flattened size:
        # 96 -> 48 -> 24 -> 12 -> 6
        # 256 channels * 6 * 6 = 9216
        self.flatten_dim = 256 * 6 * 6
        
        self.fc = nn.Linear(self.flatten_dim, config.latent_dim)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, C, H, W) tensor, normalized to [0, 1]
        Returns: (B, latent_dim) latent vector
        """
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    """
    Visual Decoder: Reconstructs 96x96x3 RGB observations from the latent vector.
    Actually, in Dreamer/RSSM, it usually reconstructs from the combined state (h_t, s_t).
    We take a combined feature vector as input.
    """
    def __init__(self, in_features: int, config: AgentConfig):
        super().__init__()
        self.config = config
        self.flatten_dim = 256 * 6 * 6
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, self.flatten_dim),
            nn.ReLU()
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, config.obs_channels, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, in_features) tensor (usually h_t + s_t)
        Returns: (B, C, H, W) reconstructed observation
        """
        x = self.fc(features)
        x = x.view(x.size(0), 256, 6, 6)
        obs_hat = self.deconv(x)
        return obs_hat
