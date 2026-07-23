"""
Multi-modal Decoder Heads for the Dreamer World Model.
Predicts RGB, Depth, Joint States, Reward, and Termination from the RSSM latent state (h + z).
"""
import torch
import torch.nn as nn
from typing import Tuple

class ConvDecoder(nn.Module):
    """Decodes latent state back to RGB and Depth images (256x256)."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 256 * 4 * 4)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.ELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.ELU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)      # 256x256 (3 for RGB, 1 for Depth)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is (B, feature_dim) or (B, T, feature_dim)
        original_shape = x.shape
        if len(original_shape) == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            
        feat = self.fc(x)
        feat = feat.view(-1, 256, 4, 4)
        out = self.deconv(feat) # (B*T, 4, 256, 256)
        
        rgb = out[:, :3, :, :]
        depth = out[:, 3:, :, :]
        
        if len(original_shape) == 3:
            rgb = rgb.view(B, T, 3, 256, 256)
            depth = depth.view(B, T, 1, 256, 256)
            
        return rgb, depth

class DenseDecoder(nn.Module):
    """Generic MLP decoder for dense vectors (Joint states, reward, termination)"""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, layers: int = 3):
        super().__init__()
        modules = []
        curr_dim = in_dim
        for _ in range(layers - 1):
            modules.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.ELU()
            ])
            curr_dim = hidden_dim
        modules.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*modules)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiModalDecoder(nn.Module):
    """Wraps all prediction heads."""
    def __init__(self, deter_dim: int, stoch_dim: int, state_dim: int):
        super().__init__()
        feature_dim = deter_dim + stoch_dim
        
        self.image_head = ConvDecoder(feature_dim)
        self.state_head = DenseDecoder(feature_dim, state_dim)
        self.reward_head = DenseDecoder(feature_dim, 1)
        self.done_head = DenseDecoder(feature_dim, 1) # Outputs logits

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = torch.cat([h, z], dim=-1)
        
        rgb, depth = self.image_head(feat)
        state_pred = self.state_head(feat)
        reward_pred = self.reward_head(feat).squeeze(-1)
        done_logits = self.done_head(feat).squeeze(-1)
        
        return {
            "rgb": rgb,
            "depth": depth,
            "state": state_pred,
            "reward": reward_pred,
            "done": done_logits
        }
