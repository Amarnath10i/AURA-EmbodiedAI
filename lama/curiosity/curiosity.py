"""Intrinsic Curiosity Module (ICM) + Random Network Distillation (RND)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CuriosityOutput:
    intrinsic_reward: torch.Tensor
    forward_loss: torch.Tensor
    inverse_loss: torch.Tensor
    rnd_loss: torch.Tensor


class RND(nn.Module):
    """Random Network Distillation for novelty-based exploration."""

    def __init__(self, obs_dim: int = 96, hidden_dim: int = 512, out_dim: int = 512):
        super().__init__()
        # Target network (fixed, random weights)
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Predictor network (trained)
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Freeze target
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns prediction error (novelty score)."""
        with torch.no_grad():
            target_out = self.target(obs)
        pred_out = self.predictor(obs)
        error = (pred_out - target_out).pow(2).mean(dim=-1)
        return error

    def loss(self, obs: torch.Tensor) -> torch.Tensor:
        """MSE loss for training predictor."""
        with torch.no_grad():
            target_out = self.target(obs)
        pred_out = self.predictor(obs)
        return F.mse_loss(pred_out, target_out)


class ICM(nn.Module):
    """Intrinsic Curiosity Module: forward + inverse dynamics."""

    def __init__(
        self,
        obs_dim: int = 96,
        action_dim: int = 12,
        hidden_dim: int = 512,
        feature_dim: int = 256,
    ):
        super().__init__()
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Inverse dynamics: (phi(s), phi(s')) -> a
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Forward dynamics: (phi(s), a) -> phi(s')
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> CuriosityOutput:
        phi_s = self.encoder(obs)
        phi_s_next = self.encoder(next_obs)

        # Inverse: predict action from (s, s')
        inverse_pred = self.inverse_net(torch.cat([phi_s, phi_s_next], dim=-1))
        inverse_loss = F.cross_entropy(inverse_pred, action.argmax(-1))

        # Forward: predict phi(s') from (phi(s), a)
        forward_pred = self.forward_net(torch.cat([phi_s, action], dim=-1))
        forward_loss = F.mse_loss(forward_pred, phi_s_next.detach())

        # Intrinsic reward = forward prediction error
        intrinsic_reward = (forward_pred - phi_s_next.detach()).pow(2).mean(dim=-1)

        return CuriosityOutput(
            intrinsic_reward=intrinsic_reward,
            forward_loss=forward_loss,
            inverse_loss=inverse_loss,
            rnd_loss=torch.tensor(0.0, device=obs.device),
        )


class CuriosityEnsemble(nn.Module):
    """Ensemble of RND + ICM for robust exploration (SOTA)."""

    def __init__(
        self,
        obs_dim: int = 96,
        action_dim: int = 12,
        hidden_dim: int = 512,
        n_ensemble: int = 5,
    ):
        super().__init__()
        self.rnd = nn.ModuleList([RND(obs_dim, hidden_dim) for _ in range(n_ensemble)])
        self.icm = ICM(obs_dim, action_dim, hidden_dim)
        self.n_ensemble = n_ensemble

    def intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, CuriosityOutput]:
        """Returns (total_intrinsic_reward, icm_output)."""
        # RND ensemble: mean prediction error
        rnd_errors = [rnd(obs) for rnd in self.rnd]
        rnd_reward = torch.stack(rnd_errors).mean(0)

        # ICM
        icm_out = self.icm(obs, next_obs, action)

        # Combined intrinsic reward
        total = 0.5 * rnd_reward + 0.5 * icm_out.intrinsic_reward
        return total, icm_out

    def loss(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor):
        # RND losses
        rnd_loss = sum(rnd.loss(obs) for rnd in self.rnd) / self.n_ensemble
        # ICM losses
        icm_out = self.icm(obs, next_obs, action)
        icm_loss = icm_out.forward_loss + 0.1 * icm_out.inverse_loss
        return rnd_loss + icm_loss, {"rnd": rnd_loss.item(), "icm": icm_loss.item()}


class ContrastiveConceptLearner(nn.Module):
    """SimCLR-style contrastive learning for visual concepts."""

    def __init__(
        self,
        appearance_dim: int = 64,
        hidden_dim: int = 512,
        proj_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Linear(appearance_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns projected embeddings."""
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=-1)

    def loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """NT-Xent contrastive loss."""
        z1 = self.forward(x1)
        z2 = self.forward(x2)

        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        sim = z @ z.t() / self.temperature  # [2B, 2B]

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -9e15)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.arange(2 * batch_size, device=z.device)
        labels[:batch_size] += batch_size
        labels[batch_size:] -= batch_size

        return F.cross_entropy(sim, labels)