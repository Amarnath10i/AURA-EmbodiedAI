import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from aura_agent.config import AgentConfig
from aura_agent.encoder import Encoder
from aura_agent.world_model import WorldModel
from aura_agent.affordance import AffordanceLearner
from aura_warehouse_sim.warehouse_sim.sim import WarehouseSim

def discover_affordances(episodes: int = 10, seed_offset: int = 20000):
    config = AgentConfig()
    
    encoder = Encoder(config).to(config.device)
    rssm = WorldModel(config).to(config.device)
    
    # Load trained models
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'world_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=config.device)
        encoder.load_state_dict(checkpoint['encoder'])
        rssm.load_state_dict(checkpoint['rssm'])
        print("Loaded trained world model for affordance discovery.")
    else:
        print("WARNING: No trained model found. Affordance discovery will use random weights (meaningless results).")
        
    affordance_learner = AffordanceLearner(config.latent_dim, surprise_threshold=0.15)
    
    print(f"\n--- Running Affordance Discovery ({episodes} episodes) ---")
    
    with torch.no_grad():
        for ep in tqdm(range(episodes), desc="Episodes"):
            sim = WarehouseSim(seed=seed_offset + ep, obs_size=config.obs_size)
            obs, info = sim.reset()
            
            # initial state
            h_t = torch.zeros(1, config.det_state_dim, device=config.device)
            s_t = torch.zeros(1, config.stoch_state_dim, device=config.device)
            a_prev = torch.zeros(1, config.action_dim, device=config.device)
            
            while True:
                # Encode obs
                obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(config.device) / 255.0
                z_t = encoder(obs_tensor)
                
                # Update belief (posterior)
                h_t, s_t, _, _ = rssm.step_posterior(h_t, s_t, a_prev, z_t)
                
                # Take random action for exploration
                action = np.random.randint(0, config.action_dim)
                a_prev = torch.nn.functional.one_hot(torch.tensor([action]), num_classes=config.action_dim).float().to(config.device)
                
                # Imagine next state (prior)
                h_next_prior, s_next_prior, _ = rssm.step_prior(h_t, s_t, a_prev)
                z_next_pred = torch.cat([h_next_prior, s_next_prior], dim=-1) # Simplified: using state as proxy for z
                reward_pred = rssm.reward(h_next_prior, s_next_prior).item()
                
                # Take real step
                next_obs, reward, term, trunc, _ = sim.step(action)
                
                # Encode real next obs
                next_obs_tensor = torch.FloatTensor(next_obs).permute(2, 0, 1).unsqueeze(0).to(config.device) / 255.0
                z_next_actual = encoder(next_obs_tensor)
                
                # Add to affordance learner (it handles surprise thresholding)
                # Note: passing states instead of direct z for simplification
                affordance_learner.add_transition(
                    obs_latent=z_t.squeeze(0),
                    action=action,
                    next_obs_actual=z_next_actual.squeeze(0),
                    next_obs_predicted=z_next_actual.squeeze(0), # Dummy for now, actual implementation would compare decoded obs or state vectors
                    reward_actual=reward,
                    reward_predicted=reward_pred
                )
                
                obs = next_obs
                if term or trunc:
                    break
                    
    # Cluster surprises into discrete affordances
    print("\nClustering surprising transitions to discover object affordances...")
    affordance_learner.build_memory(n_clusters=5)
    
if __name__ == "__main__":
    discover_affordances(episodes=10)
