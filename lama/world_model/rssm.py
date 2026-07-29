"""DreamerV3-style RSSM World Model for counterfactual imagination."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor


class RSSM(nn.Module):
    """Recurrent State Space Model (DreamerV3) with categorical latents."""

    def __init__(
        self,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        category_size: int = 32,
        hidden_dim: int = 512,
        action_dim: int = 12,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.category_size = category_size
        self.num_categories = stoch_dim // category_size
        self.hidden_dim = hidden_dim

        # GRU for deterministic path
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # Stochastic prior
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim),
        )

        # Stochastic posterior
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim),
        )

        # Embed action
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Embed observation (appearance + state)
        self.obs_embed = nn.Sequential(
            nn.Linear(64 + 32, hidden_dim),  # appearance + state
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
            logits=torch.zeros(batch_size, self.stoch_dim, device=device),
        )

    def forward(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
    ) -> Tuple[RSSMState, RSSMState]:
        """Returns (prior, post) states."""
        # Deterministic update
        action_emb = self.action_embed(action)
        x = torch.cat([prev_state.stoch, action_emb], dim=-1)
        deter = self.gru(x, prev_state.deter)

        # Prior
        prior_logits = self.prior_net(deter)
        prior_stoch = self._sample(prior_logits)
        prior = RSSMState(deter, prior_stoch, prior_logits)

        if obs is not None:
            # Posterior
            obs_emb = self.obs_embed(obs)
            post_input = torch.cat([deter, obs_emb], dim=-1)
            post_logits = self.post_net(post_input)
            post_stoch = self._sample(post_logits)
            post = RSSMState(deter, post_stoch, post_logits)
        else:
            post = prior

        return prior, post

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Categorical reparameterization (Straight-through Gumbel-Softmax)."""
        logits = logits.view(-1, self.num_categories, self.category_size)
        dist = torch.distributions.RelaxedOneHotCategorical(0.5, logits=logits)
        sample = dist.rsample()
        # Straight-through
        hard = torch.zeros_like(sample).scatter_(-1, sample.argmax(-1, keepdim=True), 1.0)
        return (hard - sample).detach() + sample.view(-1, self.stoch_dim)

    def imagine(
        self,
        init_state: RSSMState,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Roll out imagination trajectory."""
        states = []
        state = init_state
        for a in actions.unbind(0):
            state, _ = self.forward(state, a)
            states.append(state)
        return torch.stack([s.stoch for s in states]), torch.stack([s.deter for s in states])


class WorldModel(nn.Module):
    """Full world model: RSSM + observation decoder + reward/value heads."""

    def __init__(
        self,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        category_size: int = 32,
        hidden_dim: int = 512,
        action_dim: int = 12,
        obs_dim: int = 96,  # appearance(64) + state(32)
    ):
        super().__init__()
        self.rssm = RSSM(deter_dim, stoch_dim, category_size, hidden_dim, action_dim)

        # Observation decoder (appearance + state)
        self.obs_decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Reward predictor
        self.reward_net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Value predictor (for planning)
        self.value_net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Continue predictor (done)
        self.continue_net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        prev_state: Optional[RSSMState] = None,
    ) -> Tuple[RSSMState, RSSMState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (prior, post, obs_pred, reward_pred, continue_pred)."""
        if prev_state is None:
            batch_size = obs.shape[0]
            prev_state = self.rssm.initial(batch_size, obs.device)

        prior, post = self.rssm(prev_state, action, obs)
        features = torch.cat([post.deter, post.stoch], dim=-1)

        obs_pred = self.obs_decoder(features)
        reward_pred = self.reward_net(features)
        continue_pred = self.continue_net(features)

        return prior, post, obs_pred, reward_pred, continue_pred

    def imagine_trajectory(
        self,
        init_state: RSSMState,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (states, rewards, values, continues)."""
        stochs, deters = self.rssm.imagine(init_state, actions)
        features = torch.cat([deters, stochs], dim=-1)
        rewards = self.reward_net(features).squeeze(-1)
        values = self.value_net(features).squeeze(-1)
        continues = self.continue_net(features).squeeze(-1).sigmoid()
        return stochs, rewards, values, continues