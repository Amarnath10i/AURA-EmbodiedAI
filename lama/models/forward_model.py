import torch
import torch.nn as nn
from torch.distributions import Normal

class CoreRSSM(nn.Module):
    """
    The core recurrent state-space model used for LAMA.
    Takes the fused vision+proprioception latent (z_t).
    """
    def __init__(self, action_dim: int, latent_dim: int = 768, det_state_dim: int = 256, stoch_state_dim: int = 32):
        super().__init__()
        self.det_state_dim = det_state_dim
        self.stoch_state_dim = stoch_state_dim
        
        # Transition Model
        self.fc_in = nn.Sequential(
            nn.Linear(stoch_state_dim + action_dim, 256),
            nn.ReLU()
        )
        self.gru = nn.GRUCell(256, det_state_dim)
        
        # Prior Model: s_t_hat = f(h_t)
        self.prior_fc = nn.Sequential(
            nn.Linear(det_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * stoch_state_dim)
        )
        
        # Posterior Model: s_t = f(h_t, z_t)
        self.posterior_fc = nn.Sequential(
            nn.Linear(det_state_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * stoch_state_dim)
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
        
        x_prior = self.prior_fc(h_t)
        prior_mean, prior_log_std = torch.chunk(x_prior, 2, dim=-1)
        prior_std = torch.exp(torch.clamp(prior_log_std, min=-5.0, max=2.0))
        prior_dist = Normal(prior_mean, prior_std)
        
        x_post = torch.cat([h_t, z_t], dim=-1)
        x_post = self.posterior_fc(x_post)
        post_mean, post_log_std = torch.chunk(x_post, 2, dim=-1)
        post_std = torch.exp(torch.clamp(post_log_std, min=-5.0, max=2.0))
        post_dist = Normal(post_mean, post_std)
        s_t = post_dist.rsample()
        
        return h_t, s_t, prior_dist, post_dist


class DualWorldModel(nn.Module):
    """
    LAMA Dual World Model for Isaac Lab.
    PWM: Physical World Model (objects/blocks)
    BWM: Behavior World Model (agents/robot dynamics)
    """
    def __init__(self, action_dim: int, latent_dim: int = 768):
        super().__init__()
        self.pwm = CoreRSSM(action_dim, latent_dim)
        self.bwm = CoreRSSM(action_dim, latent_dim)
        
    def step_prior(self, h_prev_p, s_prev_p, h_prev_b, s_prev_b, a_prev):
        h_t_p, s_t_p, prior_p = self.pwm.step_prior(h_prev_p, s_prev_p, a_prev)
        h_t_b, s_t_b, prior_b = self.bwm.step_prior(h_prev_b, s_prev_b, a_prev)
        return (h_t_p, s_t_p, prior_p), (h_t_b, s_t_b, prior_b)
        
    def step_posterior(self, h_prev_p, s_prev_p, h_prev_b, s_prev_b, a_prev, z_t):
        h_t_p, s_t_p, prior_p, post_p = self.pwm.step_posterior(h_prev_p, s_prev_p, a_prev, z_t)
        h_t_b, s_t_b, prior_b, post_b = self.bwm.step_posterior(h_prev_b, s_prev_b, a_prev, z_t)
        return (h_t_p, s_t_p, prior_p, post_p), (h_t_b, s_t_b, prior_b, post_b)
