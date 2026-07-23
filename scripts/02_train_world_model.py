import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from aura_agent.config import AgentConfig
from aura_agent.encoder import Encoder, Decoder
from aura_agent.world_model import DualWorldModel
from aura_agent.memory import ReplayBuffer

def train():
    config = AgentConfig()
    print(f"Training on device: {config.device}")
    
    # Initialize networks
    encoder = Encoder(config).to(config.device)
    # Decoder now takes input from both world models: PWM and BWM (each has h + s)
    decoder = Decoder(2 * (config.det_state_dim + config.stoch_state_dim), config).to(config.device)
    rssm = DualWorldModel(config).to(config.device)
    
    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(rssm.parameters())
    optimizer = optim.Adam(params, lr=config.lr)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    
    # Load data
    buffer = ReplayBuffer(config, max_size=100_000) # smaller size for prototyping
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    buffer.load_shards(data_dir)
    
    if buffer.size < config.batch_size * 2:
        print("Not enough data to train. Please run 01_collect_data.py first.")
        return
        
    epochs = 100 # For prototyping
    steps_per_epoch = 100
    
    print("Starting training loop...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_recon = 0.0
        epoch_reward = 0.0
        
        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}"):
            # Sample batch: (B, T, C, H, W)
            batch = buffer.sample_sequence_batch(config.batch_size, config.seq_length)
            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            
            B, T, C, H, W = obs.shape
            
            # Embed all observations
            # (B*T, C, H, W) -> (B*T, latent_dim) -> (B, T, latent_dim)
            obs_flat = obs.reshape(B * T, C, H, W)
            z = encoder(obs_flat).reshape(B, T, -1)
            
            # Initial states for Dual RSSM
            h_p = torch.zeros(B, config.det_state_dim, device=config.device)
            s_p = torch.zeros(B, config.stoch_state_dim, device=config.device)
            h_b = torch.zeros(B, config.det_state_dim, device=config.device)
            s_b = torch.zeros(B, config.stoch_state_dim, device=config.device)
            
            total_loss = 0
            recon_loss = 0
            kl_loss = 0
            reward_loss = 0
            
            # Unroll RSSM over time
            for t in range(T):
                # a_prev is actions[:, t-1] except for t=0 where it's zeros
                a_prev = actions[:, t-1] if t > 0 else torch.zeros(B, config.action_dim, device=config.device)
                
                # Reality step (posterior)
                (h_p, s_p, prior_p, post_p), (h_b, s_b, prior_b, post_b) = rssm.step_posterior(
                    h_p, s_p, h_b, s_b, a_prev, z[:, t]
                )
                
                # Decode from posterior states of both models
                features = torch.cat([h_p, s_p, h_b, s_b], dim=-1)
                obs_hat = decoder(features)
                
                # Predict reward
                reward_hat = rssm.reward(h_p, s_p, h_b, s_b).squeeze(-1)
                
                # Accumulate losses
                recon_loss += mse_loss(obs_hat, obs[:, t])
                
                # KL divergence for both models
                kl_p = kl_divergence(post_p, prior_p).mean()
                kl_b = kl_divergence(post_b, prior_b).mean()
                
                # Free bits optimization to prevent mode collapse (KL balancing)
                kl_loss += torch.max(kl_p, torch.tensor(1.0, device=config.device))
                kl_loss += torch.max(kl_b, torch.tensor(1.0, device=config.device))
                
                reward_loss += mse_loss(reward_hat, rewards[:, t])
                
            loss = recon_loss + config.kl_scale * kl_loss + reward_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 100.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            epoch_reward += reward_loss.item()
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss/steps_per_epoch:.4f} "
              f"(Recon: {epoch_recon/steps_per_epoch:.4f}, KL: {epoch_kl/steps_per_epoch:.4f}, Rew: {epoch_reward/steps_per_epoch:.4f})")
              
    # Save models
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'rssm': rssm.state_dict()
    }, os.path.join(os.path.dirname(__file__), '..', 'models', 'world_model.pth'))
    print("Models saved.")

if __name__ == "__main__":
    train()
