"""
Vision + Proprioception Encoder for LAMA.
Now upgraded to use a frozen DINOv2 ViT-S Foundation Model for robust visual representations.
"""
import torch
import torch.nn as nn
import torchvision.transforms as T

class VisionRobotEncoder(nn.Module):
    def __init__(self, latent_dim: int = 512, robot_state_dim: int = 14, freeze_vision: bool = True):
        super().__init__()
        
        # 1. Vision Backbone (DINOv2)
        # Using the small version for speed/memory efficiency (384 dim output)
        self.vision_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.vision_dim = 384 
        
        if freeze_vision:
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
                
        # DINOv2 requires 224x224 inputs normalized with ImageNet stats
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # 2. Proprioception MLP
        self.robot_mlp = nn.Sequential(
            nn.Linear(robot_state_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64)
        )
        self.proprio_dim = 64

        # 3. Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(self.vision_dim + self.proprio_dim, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, rgb: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """
        rgb: (B, 3, H, W) in [0, 1]
        robot_state: (B, robot_state_dim)
        Returns: (B, latent_dim)
        """
        # Encode Vision
        rgb_transformed = self.transform(rgb)
        
        if not self.vision_backbone.parameters().__next__().requires_grad:
            with torch.no_grad():
                vision_features = self.vision_backbone(rgb_transformed)
        else:
            vision_features = self.vision_backbone(rgb_transformed)
            
        # Encode Proprioception
        robot_features = self.robot_mlp(robot_state)
        
        # Fuse
        combined = torch.cat([vision_features, robot_features], dim=-1)
        latent = self.fusion(combined)
        
        return latent
