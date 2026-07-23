"""
The full Dreamer-style World Model for LAMA.
Wraps the Encoder, RSSM (Prior/Posterior), and Multi-Modal Decoders.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

from lama.models.encoder import VisionRobotEncoder
from lama.models.forward_model import RSSM
from lama.models.decoder import MultiModalDecoder
from lama.envs.task_config import TabletopTaskConfig

class DreamerWorldModel(nn.Module):
    def __init__(self, config: TabletopTaskConfig):
        super().__init__()
        self.config = config
        
        # Hyperparams
        self.stoch_dim = 32
        self.deter_dim = 256
        self.embed_dim = 512
        
        # Modalities
        self.encoder = VisionRobotEncoder(latent_dim=self.embed_dim, robot_state_dim=config.robot_state_dim)
        self.rssM = RSSM(action_dim=config.action_space, embed_dim=self.embed_dim, deter_dim=self.deter_dim, stoch_dim=self.stoch_dim)
        self.decoder = MultiModalDecoder(deter_dim=self.deter_dim, stoch_dim=self.stoch_dim, state_dim=config.robot_state_dim)

    def forward(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor):
        """
        Rolls out the Posterior model over a sequence of observations and actions.
        Used during training.
        """
        B, T = actions.shape[:2]
        device = actions.device
        
        # 1. Encode observations
        # Encode expects (B, ...) not (B, T, ...) so we flatten
        rgb_flat = obs["rgb"].reshape(B * T, 3, 256, 256)
        state_flat = obs["state"].reshape(B * T, -1)
        embed_flat = self.encoder(rgb_flat, state_flat)
        embed = embed_flat.reshape(B, T, self.embed_dim)
        
        # 2. Rollout RSSM
        h, z = self.rssM.initial_state(B, device)
        
        h_seq, z_seq = [], []
        prior_mean_seq, prior_std_seq = [], []
        post_mean_seq, post_std_seq = [], []
        
        for t in range(T):
            a_t = actions[:, t] if t > 0 else torch.zeros_like(actions[:, 0]) # Action previous
            e_t = embed[:, t]
            
            h, z, prior_stats, post_stats = self.rssM.step_posterior(h, z, a_t, e_t)
            
            h_seq.append(h)
            z_seq.append(z)
            prior_mean_seq.append(prior_stats["prior_mean"])
            prior_std_seq.append(prior_stats["prior_std"])
            post_mean_seq.append(post_stats["post_mean"])
            post_std_seq.append(post_stats["post_std"])
            
        h_seq = torch.stack(h_seq, dim=1)
        z_seq = torch.stack(z_seq, dim=1)
        
        # 3. Decode
        preds = self.decoder(h_seq, z_seq)
        
        # 4. Pack stats for KL loss
        kl_stats = {
            "prior_mean": torch.stack(prior_mean_seq, dim=1),
            "prior_std": torch.stack(prior_std_seq, dim=1),
            "post_mean": torch.stack(post_mean_seq, dim=1),
            "post_std": torch.stack(post_std_seq, dim=1),
        }
        
        return preds, kl_stats

    def imagine(self, h_prev: torch.Tensor, z_prev: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Rolls out the Prior model into the future given a sequence of actions.
        Used by the Planner (CEM/MPPI).
        """
        B, H = actions.shape[:2] # Horizon
        
        h, z = h_prev, z_prev
        h_seq, z_seq = [], []
        
        for t in range(H):
            h, z, _ = self.rssM.step_prior(h, z, actions[:, t])
            h_seq.append(h)
            z_seq.append(z)
            
        h_seq = torch.stack(h_seq, dim=1)
        z_seq = torch.stack(z_seq, dim=1)
        
        # Predict rewards and terminations for the imagined trajectory
        preds = self.decoder(h_seq, z_seq)
        return h_seq, z_seq, preds
