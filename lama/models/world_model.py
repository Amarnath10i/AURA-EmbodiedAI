"""DreamerV3-style RSSM World Model for LAMA.

Provides latent dynamics with calibrated uncertainty for imagination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env.types import Observation, Action


@dataclass
class RSSMState:
    """RSSM latent state."""
    deter: torch.Tensor      # deterministic GRU state
    stoch: torch.Tensor      # stochastic discrete latent (categorical)
    logits: torch.Tensor     # logits for stochastic

    def get_features(self) -> torch.Tensor:
        """Concatenated features for decoder/predictor."""
        return torch.cat([self.deter, self.stoch], dim=-1)


class RSSM(nn.Module):
    """Recurrent State-Space Model (DreamerV3)."""

    def __init__(
        self,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        action_dim: int = 12,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes

        # GRU for deterministic path
        self.gru = nn.GRUCell(
            stoch_dim + action_dim,
            deter_dim,
        )

        # Prior network: p(z_t | h_t)
        self.prior = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )

        # Posterior network: q(z_t | h_t, x_t)
        self.posterior = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )

        # Input encoder
        self.encoder = nn.Sequential(
            nn.Linear(64 + 128, hidden_dim),  # appearance + state
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim),
        )

    def init_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
            logits=torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device),
        )

    def discrete_stochastic(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Straight-through categorical sampling (Gumbel-Softmax)."""
        # logits: [..., stoch_dim, stoch_classes]
        shape = logits.shape
        logits = logits.view(-1, self.stoch_classes)
        stoch = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        stoch = stoch.view(*shape[:-1], self.stoch_dim)
        return stoch, logits

    def forward(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
    ) -> RSSMState:
        """One step of RSSM."""
        # GRU update: h_t = GRU(h_{t-1}, [z_{t-1}, a_t])
        gru_in = torch.cat([prev_state.stoch, action], dim=-1)
        deter = self.gru(gru_in, prev_state.deter)

        # Prior: p(z_t | h_t)
        prior_logits = self.prior(deter).view(-1, self.stoch_dim, self.stoch_classes)
        prior_stoch, _ = self.discrete_stochastic(prior_logits)

        if obs is not None:
            # Posterior: q(z_t | h_t, x_t)
            embedded = self.encoder(obs)
            post_in = torch.cat([deter, embedded], dim=-1)
            post_logits = self.posterior(post_in).view(-1, self.stoch_dim, self.stoch_classes)
            post_stoch, _ = self.discrete_stochastic(post_logits)

            return RSSMState(
                deter=deter,
                stoch=post_stoch,
                logits=post_logits,
            )
        else:
            # Imagination mode: use prior
            return RSSMState(
                deter=deter,
                stoch=prior_stoch,
                logits=prior_logits,
            )

    def imagine_trajectory(
        self,
        init_state: RSSMState,
        actions: torch.Tensor,  # [n_candidates, horizon, action_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Imagine trajectory for planning."""
        n_candidates, horizon, _ = actions.shape
        device = actions.device

        # Expand init state
        state = RSSMState(
            deter=init_state.deter.repeat(n_candidates, 1),
            stoch=init_state.stoch.repeat(n_candidates, 1),
            logits=init_state.logits.repeat(n_candidates, 1, 1),
        )

        stochs = []
        for t in range(horizon):
            state = self.forward(state, actions[:, t])
            stochs.append(state.stoch)

        stochs = torch.stack(stochs, dim=1)  # [n, horizon, stoch_dim]
        return stochs


class WorldModel(nn.Module):
    """Full world model: RSSM + decoders + reward/value heads."""

    def __init__(
        self,
        obs_dim: int = 192,  # appearance + state
        action_dim: int = 12,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.rssm = RSSM(deter_dim, stoch_dim, stoch_classes, action_dim, hidden_dim)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        # Observation decoder
        self.decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Reward predictor
        self.reward = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Continue (discount) predictor
        self.continue_ = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Value predictor
        self.value = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[RSSMState] = None,
    ) -> Tuple[RSSMState, dict]:
        """Encode sequence, return final state and predictions."""
        batch, seq_len = actions.shape[:2]
        device = actions.device

        if state is None:
            state = self.rssm.init_state(batch, device)

        post_states = []
        prior_states = []
        rewards = []
        values = []
        continues = []

        for t in range(seq_len):
            # Posterior step (with observation)
            post_state = self.rssm.forward(state, actions[:, t], obs[:, t])
            post_states.append(post_state)

            # Prior step (without observation)
            prior_state = self.rssm.forward(state, actions[:, t], obs=None)
            prior_states.append(prior_state)

            # Predictions from posterior
            feat = post_state.get_features()
            rewards.append(self.reward(feat))
            values.append(self.value(feat))
            continues.append(torch.sigmoid(self.continue_(feat)))

            state = post_state

        return post_states[-1], {
            "post_states": post_states,
            "prior_states": prior_states,
            "rewards": torch.stack(rewards, dim=1),
            "values": torch.stack(values, dim=1),
            "continues": torch.stack(continues, dim=1),
        }

    def imagine_trajectory(
        self,
        init_state: RSSMState,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Imagine for planning."""
        stochs = self.rssm.imagine_trajectory(init_state, actions)

        # Decode predictions
        batch, horizon = stochs.shape[:2]
        feats = []
        for t in range(horizon):
            deter = self.rssm.gru(
                torch.cat([stochs[:, t], actions[:, t]], dim=-1),
                init_state.deter if t == 0 else feats[-1][:, :self.deter_dim]
            )
            feat = torch.cat([deter, stochs[:, t]], dim=-1)
            feats.append(feat)

        feats = torch.stack(feats, dim=1)
        rewards = self.reward(feats).squeeze(-1)
        values = self.value(feats).squeeze(-1)
        continues = torch.sigmoid(self.continue_(feats)).squeeze(-1)

        return stochs, rewards, values, continues

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor,
        state: Optional[RSSMState] = None,
        kl_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """World model loss: reconstruction + reward + continue + KL."""
        final_state, outputs = self.forward(obs, actions, state)

        post_states = outputs["post_states"]
        prior_states = outputs["prior_states"]

        # Reconstruction
        feat = torch.stack([s.get_features() for s in post_states], dim=1)
        recon = self.decoder(feat)
        recon_loss = F.mse_loss(recon, obs)

        # Reward
        reward_loss = F.mse_loss(outputs["rewards"].squeeze(-1), rewards)

        # Continue
        continue_loss = F.binary_cross_entropy(outputs["continues"].squeeze(-1), continues)

        # KL divergence (posterior || prior)
        kl_loss = 0
        for post, prior in zip(post_states, prior_states):
            kl = torch.distributions.kl.kl_divergence(
                torch.distributions.OneHotCategorical(logits=post.logits),
                torch.distributions.OneHotCategorical(logits=prior.logits),
            ).mean()
            kl_loss += kl
        kl_loss = kl_loss / len(post_states)

        total = recon_loss + reward_loss + continue_loss + kl_weight * kl_loss

        return total, {
            "recon": recon_loss.item(),
            "reward": reward_loss.item(),
            "continue": continue_loss.item(),
            "kl": kl_loss.item(),
        }