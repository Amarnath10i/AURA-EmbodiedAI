"""
LAMA: Train the Dual World Model (Forward Model) + Decoder + Reward Model + Inverse Dynamics.
Loads per-episode data from the dataset/ directory.
"""
import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.distributions.kl import kl_divergence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lama.models.encoder import VisionRobotEncoder
from lama.models.forward_model import DualWorldModel
from lama.models.decoder import LatentDecoder, RewardModel
from lama.models.inverse_model import InverseDynamicsModel
from lama.envs.task_config import TabletopTaskConfig

# ---- Hyperparameters ----
LATENT_DIM = 768
DET_STATE_DIM = 256
STOCH_STATE_DIM = 32
EPOCHS = 50
BATCH_SIZE = 8
LR = 3e-4
KL_SCALE = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_episodes(dataset_dir: str):
    """Load all episodes from the dataset directory."""
    episodes = []
    ep_dirs = sorted(glob.glob(os.path.join(dataset_dir, "episode*")))
    for ep_dir in ep_dirs:
        ep = {
            "rgb": np.load(os.path.join(ep_dir, "rgb.npy")),
            "state": np.load(os.path.join(ep_dir, "state.npy")),
            "action": np.load(os.path.join(ep_dir, "action.npy")),
            "reward": np.load(os.path.join(ep_dir, "reward.npy")),
        }
        episodes.append(ep)
    return episodes

def train():
    config = TabletopTaskConfig()
    state_dim = 2 * (DET_STATE_DIM + STOCH_STATE_DIM)
    
    # ---- Initialize Models ----
    encoder = VisionRobotEncoder(
        robot_state_dim=config.robot_state_dim,
        final_latent_dim=LATENT_DIM
    ).to(DEVICE)
    
    world_model = DualWorldModel(
        action_dim=config.action_space,
        latent_dim=LATENT_DIM
    ).to(DEVICE)
    
    decoder = LatentDecoder(state_dim=state_dim, output_size=config.sensors.resolution[0]).to(DEVICE)
    reward_model = RewardModel(state_dim=state_dim).to(DEVICE)
    inverse_model = InverseDynamicsModel(action_dim=config.action_space, latent_dim=LATENT_DIM).to(DEVICE)
    
    # ---- Optimizer ----
    all_params = (
        list(encoder.parameters()) +
        list(world_model.parameters()) +
        list(decoder.parameters()) +
        list(reward_model.parameters()) +
        list(inverse_model.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=LR)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # ---- Load Data ----
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
    episodes = load_episodes(dataset_dir)
    if not episodes:
        print(f"ERROR: No episodes found in {dataset_dir}. Run collect_demonstrations.py first.")
        return
    print(f"Loaded {len(episodes)} episodes from {dataset_dir}")
    print(f"Training on device: {DEVICE}")
    
    # ---- Training Loop ----
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        np.random.shuffle(episodes)
        
        for ep in tqdm(episodes, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            T = len(ep["action"])
            if T < 2:
                continue
                
            # Prepare tensors
            rgb = torch.FloatTensor(ep["rgb"]).permute(0, 3, 1, 2).to(DEVICE) / 255.0  # (T, 3, H, W)
            robot_state = torch.FloatTensor(ep["state"]).to(DEVICE)                      # (T, state_dim)
            actions = torch.FloatTensor(ep["action"]).to(DEVICE)                         # (T, action_dim)
            rewards = torch.FloatTensor(ep["reward"]).to(DEVICE)                         # (T,)
            
            # Encode all timesteps
            z_all = encoder(rgb, robot_state)  # (T, LATENT_DIM)
            
            # Initialize dual RSSM states
            h_p = torch.zeros(1, DET_STATE_DIM, device=DEVICE)
            s_p = torch.zeros(1, STOCH_STATE_DIM, device=DEVICE)
            h_b = torch.zeros(1, DET_STATE_DIM, device=DEVICE)
            s_b = torch.zeros(1, STOCH_STATE_DIM, device=DEVICE)
            
            recon_loss_total = 0.0
            kl_loss_total = 0.0
            reward_loss_total = 0.0
            inverse_loss_total = 0.0
            
            for t in range(T):
                a_prev = actions[t-1:t] if t > 0 else torch.zeros(1, actions.size(-1), device=DEVICE)
                z_t = z_all[t:t+1]
                
                # Posterior step (reality)
                (h_p, s_p, prior_p, post_p), (h_b, s_b, prior_b, post_b) = world_model.step_posterior(
                    h_p, s_p, h_b, s_b, a_prev, z_t
                )
                
                # Decode from posterior
                features = torch.cat([h_p, s_p, h_b, s_b], dim=-1)
                rgb_hat = decoder(features)
                reward_hat = reward_model(features)
                
                # Reconstruction loss
                recon_loss_total += mse_loss(rgb_hat, rgb[t:t+1])
                
                # KL divergence (both models, free bits)
                kl_p = kl_divergence(post_p, prior_p).mean()
                kl_b = kl_divergence(post_b, prior_b).mean()
                kl_loss_total += torch.max(kl_p, torch.tensor(1.0, device=DEVICE))
                kl_loss_total += torch.max(kl_b, torch.tensor(1.0, device=DEVICE))
                
                # Reward loss
                reward_loss_total += mse_loss(reward_hat.squeeze(-1), rewards[t:t+1])
                
                # Inverse dynamics loss (predict action from z_t -> z_{t+1})
                if t < T - 1:
                    z_next = z_all[t+1:t+2]
                    action_pred = inverse_model(z_t, z_next)
                    # For continuous actions, use MSE
                    inverse_loss_total += mse_loss(action_pred, actions[t:t+1])
            
            # Total loss
            loss = recon_loss_total + KL_SCALE * kl_loss_total + reward_loss_total + inverse_loss_total
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 100.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Detach states for next episode
            h_p = h_p.detach()
            s_p = s_p.detach()
            h_b = h_b.detach()
            s_b = s_b.detach()
        
        avg_loss = epoch_loss / max(len(episodes), 1)
        print(f"  Epoch {epoch+1}/{EPOCHS}  |  Avg Loss: {avg_loss:.4f}")
    
    # ---- Save Models ----
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lama_world_model.pth")
    torch.save({
        "encoder": encoder.state_dict(),
        "world_model": world_model.state_dict(),
        "decoder": decoder.state_dict(),
        "reward_model": reward_model.state_dict(),
        "inverse_model": inverse_model.state_dict(),
    }, save_path)
    print(f"Saved models to {save_path}")


if __name__ == "__main__":
    train()
