"""
Dreamer-style Recurrent State Space Model (RSSM) for LAMA.
Includes the Representation Model (Posterior) and Dynamics Model (Prior).
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict

class RSSM(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int = 512, deter_dim: int = 256, stoch_dim: int = 32):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        
        # RNN Cell for deterministic state (h)
        self.cell = nn.GRUCell(self.stoch_dim + action_dim, self.deter_dim)
        
        # Prior Model: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_dim, 256),
            nn.ELU(),
            nn.Linear(256, 2 * self.stoch_dim) # Outputs mean and std
        )
        
        # Posterior (Representation) Model: q(z_t | h_t, e_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(self.deter_dim + embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, 2 * self.stoch_dim)
        )

    def initial_state(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns initial (h_0, z_0)"""
        h = torch.zeros((batch_size, self.deter_dim), device=device)
        z = torch.zeros((batch_size, self.stoch_dim), device=device)
        return h, z

    def step_prior(self, h_prev: torch.Tensor, z_prev: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Rolls forward the dynamics model without observation.
        h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        z_t ~ p(z_t | h_t)
        """
        rnn_in = torch.cat([z_prev, action], dim=-1)
        h_t = self.cell(rnn_in, h_prev)
        
        prior_stats = self.prior_net(h_t)
        prior_mean, prior_std = torch.chunk(prior_stats, 2, dim=-1)
        prior_std = torch.nn.functional.softplus(prior_std) + 0.1 # Min std
        
        # Sample prior
        prior_z = prior_mean + prior_std * torch.randn_like(prior_mean)
        
        stats = {"prior_mean": prior_mean, "prior_std": prior_std}
        return h_t, prior_z, stats

    def step_posterior(self, h_prev: torch.Tensor, z_prev: torch.Tensor, action: torch.Tensor, obs_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
        """
        Rolls forward the representation model with observation.
        h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        z_t ~ q(z_t | h_t, e_t)
        """
        # First compute deterministic state and prior
        h_t, prior_z, prior_stats = self.step_prior(h_prev, z_prev, action)
        
        # Then compute posterior
        post_in = torch.cat([h_t, obs_embed], dim=-1)
        post_stats = self.posterior_net(post_in)
        post_mean, post_std = torch.chunk(post_stats, 2, dim=-1)
        post_std = torch.nn.functional.softplus(post_std) + 0.1
        
        # Sample posterior
        post_z = post_mean + post_std * torch.randn_like(post_mean)
        
        p_stats = {"post_mean": post_mean, "post_std": post_std}
        return h_t, post_z, prior_stats, p_stats
