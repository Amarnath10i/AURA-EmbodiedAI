"""
Trains the Affordance Predictor.
1. Uses the trained World Model to encode the Replay Buffer into latent states.
2. Extracts interaction outcomes and runs KMeans to discover true physical affordances.
3. Trains the Affordance Predictor head to map (h_t, z_t) -> affordance cluster.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lama.envs.task_config import TabletopTaskConfig
from lama.data.replay_buffer import ReplayBuffer
from lama.models.world_model import DreamerWorldModel
from lama.affordance.discovery import AffordanceDiscovery
from lama.affordance.predictor import AffordancePredictor

def train_affordances(epochs: int = 30, batch_size: int = 32, seq_len: int = 50, num_clusters: int = 8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TabletopTaskConfig()
    
    # Load World Model (Frozen)
    world_model = DreamerWorldModel(config).to(device)
    wm_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'dreamer_world_model.pth')
    world_model.load_state_dict(torch.load(wm_path, map_location=device, weights_only=True))
    world_model.eval()
    
    # Init Discovery & Predictor
    discovery = AffordanceDiscovery(num_clusters=num_clusters)
    predictor = AffordancePredictor(world_model.deter_dim, world_model.stoch_dim, num_clusters).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    storage_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'hdf5')
    buffer = ReplayBuffer(storage_dir=storage_dir)
    
    # Phase 1: Affordance Discovery (KMeans)
    print("--- Phase 1: Extracting Outcomes & Discovering Affordances ---")
    all_outcomes = []
    
    # We sample a large batch to fit KMeans
    try:
        discovery_batch = buffer.sample_batch(100, seq_len, split="train")
    except ValueError:
        print("Buffer is empty.")
        return
        
    with torch.no_grad():
        rgb = torch.tensor(discovery_batch["rgb"], dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3) / 255.0
        state = torch.tensor(discovery_batch["state"], dtype=torch.float32, device=device)
        actions = torch.tensor(discovery_batch["action"], dtype=torch.float32, device=device)
        rewards = torch.tensor(discovery_batch["reward"], dtype=torch.float32, device=device)
        
        obs = {"rgb": rgb, "state": state}
        _, kl_stats = world_model(obs, actions)
        
        # We use the posterior mean as the latent state representation
        z_seq = kl_stats["post_mean"] 
        
        outcomes_np = discovery.extract_outcomes(actions, z_seq, rewards)
        discovery.fit(outcomes_np)
        
    # Phase 2: Train Predictor Head
    print("--- Phase 2: Training Affordance Predictor ---")
    for epoch in range(1, epochs + 1):
        batch = buffer.sample_batch(batch_size, seq_len, split="train")
        
        rgb = torch.tensor(batch["rgb"], dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3) / 255.0
        state = torch.tensor(batch["state"], dtype=torch.float32, device=device)
        actions = torch.tensor(batch["action"], dtype=torch.float32, device=device)
        rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            obs = {"rgb": rgb, "state": state}
            _, kl_stats = world_model(obs, actions)
            h_seq = kl_stats["post_mean"] # Simplified: should be actual deterministic h, using z for now
            z_seq = kl_stats["post_mean"]
            
            outcomes_np = discovery.extract_outcomes(actions, z_seq, rewards)
            labels_np = discovery.predict(outcomes_np)
            labels = torch.tensor(labels_np, dtype=torch.long, device=device)
            
            # Re-shape h and z to align with outcomes (T-1)
            h_in = h_seq[:, :-1].reshape(-1, world_model.deter_dim) # Actually using z_seq size here, fix below
            z_in = z_seq[:, :-1].reshape(-1, world_model.stoch_dim)
            
            # Correcting dummy h extraction:
            # We need to extract the exact h_seq from the forward pass.
            # For simplicity in this script, we'll run a quick loop to get true h_seq
            B, T = actions.shape[:2]
            embed_flat = world_model.encoder(rgb.reshape(B*T, 3, 256, 256), state.reshape(B*T, -1))
            embed = embed_flat.reshape(B, T, world_model.embed_dim)
            h_0, z_0 = world_model.rssM.initial_state(B, device)
            
            h_true_seq = []
            for t in range(T):
                a_t = actions[:, t] if t > 0 else torch.zeros_like(actions[:, 0])
                h_0, z_0, _, _ = world_model.rssM.step_posterior(h_0, z_0, a_t, embed[:, t])
                h_true_seq.append(h_0)
            
            h_true_seq = torch.stack(h_true_seq, dim=1)
            h_in = h_true_seq[:, :-1].reshape(-1, world_model.deter_dim)
            
        # Train Predictor
        optimizer.zero_grad()
        logits = predictor(h_in, z_in)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Accuracy: {acc.item():.2f}")
        
    save_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'affordance_predictor.pth')
    torch.save(predictor.state_dict(), save_path)
    print(f"Saved Affordance Predictor to {save_path}")

if __name__ == "__main__":
    train_affordances()
