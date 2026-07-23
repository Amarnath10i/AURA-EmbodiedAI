import torch
import torch.nn as nn
import torchvision.models as models

class VisionRobotEncoder(nn.Module):
    """
    Fuses high-dimensional RGB input with low-dimensional robot proprioception
    into a single latent representation z.
    
    Uses ResNet18 as a budget stand-in for DINOv2 to allow fast local prototyping.
    """
    def __init__(self, robot_state_dim: int, vision_latent_dim: int = 512, final_latent_dim: int = 768):
        super().__init__()
        
        # Vision Backbone: ResNet18 (Budget DINOv2)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final classification layer to get the 512-dim feature vector
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Proprioception Encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Fusion Layer
        # Concatenate Vision (512) + Proprioception (128) -> 640
        self.fusion = nn.Sequential(
            nn.Linear(vision_latent_dim + 128, final_latent_dim),
            nn.LayerNorm(final_latent_dim),
            nn.ReLU()
        )
        
    def forward(self, rgb: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """
        rgb: (B, C, H, W) normalized to [0, 1]
        robot_state: (B, robot_state_dim)
        Returns: (B, final_latent_dim)
        """
        # Encode Vision
        v = self.vision_backbone(rgb) # (B, 512, 1, 1)
        v = v.view(v.size(0), -1)     # (B, 512)
        
        # Encode Proprioception
        p = self.proprio_encoder(robot_state) # (B, 128)
        
        # Fuse
        z = torch.cat([v, p], dim=-1) # (B, 640)
        z_out = self.fusion(z)        # (B, final_latent_dim)
        
        return z_out
