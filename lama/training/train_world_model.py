"""
Training Loop for the Dreamer World Model.
Samples batches from the HDF5 Replay Buffer and computes RSSM + Reconstruction losses.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, kl_divergence
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lama.models.world_model import DreamerWorldModel
from lama.envs.task_config import TabletopTaskConfig
from lama.data.replay_buffer import ReplayBuffer

def compute_kl_loss(prior_mean, prior_std, post_mean, post_std):
    """
    Computes KL(Posterior || Prior) to train the prior.
    Includes KL balancing (DreamerV2/V3 trick) to prevent the posterior from collapsing.
    """
    post_dist = Normal(post_mean, post_std)
    prior_dist = Normal(prior_mean, prior_std)
    
    # KL Balancing: scale down the gradient toward the posterior
    kl_prior = kl_divergence(post_dist.detach(), prior_dist).mean()
    kl_post = kl_divergence(post_dist, prior_dist.detach()).mean()
    
    kl_loss = 0.8 * kl_prior + 0.2 * kl_post
    return torch.clamp(kl_loss, min=1.0) # Free Nats

def train_world_model(epochs: int, batch_size: int, seq_len: int, lr: float = 3e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = TabletopTaskConfig()
    model = DreamerWorldModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    storage_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'hdf5')
    buffer = ReplayBuffer(storage_dir=storage_dir)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    print(f"Training Dreamer World Model on {device}...")
    
    for epoch in range(1, epochs + 1):
        try:
            batch = buffer.sample_batch(batch_size, seq_len, split="train")
        except ValueError:
            print("Buffer is empty. Run collect_demonstrations.py first.")
            return
            
        # Move to device and format shapes
        # rgb in buffer is (B, T, H, W, C), PyTorch needs (B, T, C, H, W)
        rgb = torch.tensor(batch["rgb"], dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3) / 255.0
        depth = torch.tensor(batch["depth"], dtype=torch.float32, device=device).unsqueeze(2) # (B, T, 1, H, W)
        state = torch.tensor(batch["state"], dtype=torch.float32, device=device)
        action = torch.tensor(batch["action"], dtype=torch.float32, device=device)
        reward_target = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
        done_target = torch.tensor(batch["done"], dtype=torch.float32, device=device)
        
        obs = {"rgb": rgb, "state": state}
        
        optimizer.zero_grad()
        
        # Forward pass (Posterior rollout)
        preds, kl_stats = model(obs, action)
        
        # 1. Reconstruction Losses
        loss_rgb = mse_loss(preds["rgb"], rgb)
        # Depth might have NaN if unnormalized in real sensor, assuming normalized here
        loss_depth = mse_loss(preds["depth"], depth) 
        loss_state = mse_loss(preds["state"], state)
        
        # 2. RL Task Losses
        loss_reward = mse_loss(preds["reward"], reward_target)
        loss_done = bce_loss(preds["done"], done_target)
        
        # 3. RSSM KL Loss
        loss_kl = compute_kl_loss(
            kl_stats["prior_mean"], kl_stats["prior_std"],
            kl_stats["post_mean"], kl_stats["post_std"]
        )
        
        # Total Objective
        # Scale factors are typical in Dreamer architectures
        loss = 1.0 * loss_rgb + 1.0 * loss_depth + 1.0 * loss_state + 1.0 * loss_reward + 1.0 * loss_done + 1.0 * loss_kl
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        
        print(f"Epoch {epoch:03d} | Total: {loss.item():.2f} | RGB: {loss_rgb.item():.2f} | KL: {loss_kl.item():.2f}")
        
    # Save Model
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "dreamer_world_model.pth"))
    print("Saved Dreamer World Model.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=50)
    args = p.parse_args()
    
    train_world_model(args.epochs, args.batch, args.seq)
