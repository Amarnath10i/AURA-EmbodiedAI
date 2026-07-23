import torch
import torch.nn as nn
from torch.distributions import Normal
from .config import AgentConfig

class CoreRSSM(nn.Module):
    """
    The core recurrent state-space model used as a building block.
    """
    def __init__(self, config: AgentConfig, name: str = "rssm"):
        super().__init__()
        self.config = config
        self.name = name
        
        # Transition Model: h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
        self.fc_in = nn.Sequential(
            nn.Linear(config.stoch_state_dim + config.action_dim, 256),
            nn.ReLU()
        )
        self.gru = nn.GRUCell(256, config.det_state_dim)
        
        # Prior Model: s_t_hat = f(h_t)
        self.prior_fc = nn.Sequential(
            nn.Linear(config.det_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * config.stoch_state_dim) # mean and log_std
        )
        
        # Posterior Model: s_t = f(h_t, z_t)
        self.posterior_fc = nn.Sequential(
            nn.Linear(config.det_state_dim + config.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * config.stoch_state_dim) # mean and log_std
        )
        
    def step_prior(self, h_prev: torch.Tensor, s_prev: torch.Tensor, a_prev: torch.Tensor):
        x = torch.cat([s_prev, a_prev], dim=-1)
        x = self.fc_in(x)
        h_t = self.gru(x, h_prev)
        
        x_prior = self.prior_fc(h_t)
        mean, log_std = torch.chunk(x_prior, 2, dim=-1)
        std = torch.exp(torch.clamp(log_std, min=-5.0, max=2.0))
        dist = Normal(mean, std)
        s_t = dist.rsample()
        return h_t, s_t, dist
        
    def step_posterior(self, h_prev: torch.Tensor, s_prev: torch.Tensor, a_prev: torch.Tensor, z_t: torch.Tensor):
        x = torch.cat([s_prev, a_prev], dim=-1)
        x = self.fc_in(x)
        h_t = self.gru(x, h_prev)
        
        # Prior
        x_prior = self.prior_fc(h_t)
        prior_mean, prior_log_std = torch.chunk(x_prior, 2, dim=-1)
        prior_std = torch.exp(torch.clamp(prior_log_std, min=-5.0, max=2.0))
        prior_dist = Normal(prior_mean, prior_std)
        
        # Posterior
        x_post = torch.cat([h_t, z_t], dim=-1)
        x_post = self.posterior_fc(x_post)
        post_mean, post_log_std = torch.chunk(x_post, 2, dim=-1)
        post_std = torch.exp(torch.clamp(post_log_std, min=-5.0, max=2.0))
        post_dist = Normal(post_mean, post_std)
        s_t = post_dist.rsample()
        
        return h_t, s_t, prior_dist, post_dist


class DualWorldModel(nn.Module):
    """
    Implements the Dual World Model architecture.
    PWM: Physical World Model (models deterministic/inanimate physics)
    BWM: Behavior World Model (models stochastic/agent behavior)
    """
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        self.pwm = CoreRSSM(config, name="physical")
        self.bwm = CoreRSSM(config, name="behavior")
        
        # Reward is predicted from the combined state of both models
        self.reward_model = nn.Sequential(
            nn.Linear(2 * (config.det_state_dim + config.stoch_state_dim), 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def step_prior(self, h_prev_p, s_prev_p, h_prev_b, s_prev_b, a_prev):
        h_t_p, s_t_p, prior_p = self.pwm.step_prior(h_prev_p, s_prev_p, a_prev)
        h_t_b, s_t_b, prior_b = self.bwm.step_prior(h_prev_b, s_prev_b, a_prev)
        return (h_t_p, s_t_p, prior_p), (h_t_b, s_t_b, prior_b)
        
    def step_posterior(self, h_prev_p, s_prev_p, h_prev_b, s_prev_b, a_prev, z_t):
        h_t_p, s_t_p, prior_p, post_p = self.pwm.step_posterior(h_prev_p, s_prev_p, a_prev, z_t)
        h_t_b, s_t_b, prior_b, post_b = self.bwm.step_posterior(h_prev_b, s_prev_b, a_prev, z_t)
        return (h_t_p, s_t_p, prior_p, post_p), (h_t_b, s_t_b, prior_b, post_b)
        
    def reward(self, h_t_p, s_t_p, h_t_b, s_t_b):
        x = torch.cat([h_t_p, s_t_p, h_t_b, s_t_b], dim=-1)
        return self.reward_model(x)

class UncertaintyEstimator:
    """
    Estimates uncertainty based on the variance of the Dual World Models.
    High uncertainty = intrinsic reward (Hypothesis generation trigger).
    """
    @staticmethod
    def compute(prior_p, prior_b):
        # We define uncertainty as the volume (entropy/scale) of the behavior and physical priors.
        # Spikes when the models are unsure about what will happen next.
        u_p = prior_p.scale.mean(dim=-1)
        u_b = prior_b.scale.mean(dim=-1)
        return u_p + u_b
