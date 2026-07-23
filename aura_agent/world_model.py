import torch
import torch.nn as nn
from torch.distributions import Normal
from .config import AgentConfig

class TransitionModel(nn.Module):
    """
    Advances the deterministic state: h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
    """
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        # input: stoch_state + action_dim
        self.fc_in = nn.Sequential(
            nn.Linear(config.stoch_state_dim + config.action_dim, 256),
            nn.ReLU()
        )
        self.gru = nn.GRUCell(256, config.det_state_dim)
        
    def forward(self, h_prev: torch.Tensor, s_prev: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """
        h_prev: (B, det_state_dim)
        s_prev: (B, stoch_state_dim)
        a_prev: (B, action_dim) one-hot
        Returns: h_t (B, det_state_dim)
        """
        x = torch.cat([s_prev, a_prev], dim=-1)
        x = self.fc_in(x)
        h_t = self.gru(x, h_prev)
        return h_t


class PriorModel(nn.Module):
    """
    Predicts the next stochastic state from the deterministic state: s_t_hat = f(h_t)
    Outputs parameters of a Normal distribution.
    """
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.fc = nn.Sequential(
            nn.Linear(config.det_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * config.stoch_state_dim) # mean and log_std
        )
        
    def forward(self, h_t: torch.Tensor):
        x = self.fc(h_t)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        # constrain std for stability
        std = torch.exp(torch.clamp(log_std, min=-5.0, max=2.0))
        return mean, std


class PosteriorModel(nn.Module):
    """
    Corrects the stochastic state using the actual observation: s_t = f(h_t, z_t)
    Outputs parameters of a Normal distribution.
    """
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.fc = nn.Sequential(
            nn.Linear(config.det_state_dim + config.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * config.stoch_state_dim) # mean and log_std
        )
        
    def forward(self, h_t: torch.Tensor, z_t: torch.Tensor):
        x = torch.cat([h_t, z_t], dim=-1)
        x = self.fc(x)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        std = torch.exp(torch.clamp(log_std, min=-5.0, max=2.0))
        return mean, std


class RewardModel(nn.Module):
    """
    Predicts the reward from the combined state: r_t = f(h_t, s_t)
    """
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.det_state_dim + config.stoch_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_t, s_t], dim=-1)
        return self.fc(x)


class WorldModel(nn.Module):
    """
    The full Recurrent State-Space Model (RSSM).
    Contains: Transition, Prior, Posterior, and Reward models.
    (Encoder and Decoder are typically trained jointly but defined separately for modularity).
    """
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.transition = TransitionModel(config)
        self.prior = PriorModel(config)
        self.posterior = PosteriorModel(config)
        self.reward = RewardModel(config)
        
    def step_prior(self, h_prev: torch.Tensor, s_prev: torch.Tensor, a_prev: torch.Tensor):
        """Imagination step: without observation"""
        h_t = self.transition(h_prev, s_prev, a_prev)
        mean, std = self.prior(h_t)
        dist = Normal(mean, std)
        s_t = dist.rsample()
        return h_t, s_t, dist
        
    def step_posterior(self, h_prev: torch.Tensor, s_prev: torch.Tensor, a_prev: torch.Tensor, z_t: torch.Tensor):
        """Reality step: with observation"""
        h_t = self.transition(h_prev, s_prev, a_prev)
        
        # prior distribution (for KL loss)
        prior_mean, prior_std = self.prior(h_t)
        prior_dist = Normal(prior_mean, prior_std)
        
        # posterior distribution
        post_mean, post_std = self.posterior(h_t, z_t)
        post_dist = Normal(post_mean, post_std)
        s_t = post_dist.rsample()
        
        return h_t, s_t, prior_dist, post_dist
