"""
Planner Execution Loop.
Connects the Dreamer World Model and CEM Planner to the live Isaac Lab simulation.
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lama.envs.task_config import TabletopTaskConfig
from lama.envs.tabletop_env import ParallelTabletopEnv
from lama.models.world_model import DreamerWorldModel
from lama.planner.cem import CEMPlanner

def run_planner_loop(episodes: int = 5, steps: int = 100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Init Environment (using parallel env for speed, but planner acts independently per env)
    config = TabletopTaskConfig()
    env = ParallelTabletopEnv(config, device=device)
    
    # 2. Load World Model
    model = DreamerWorldModel(config).to(device)
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'dreamer_world_model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded trained World Model from {model_path}")
    else:
        print("WARNING: No trained World Model found. Using randomly initialized weights!")
        
    model.eval()
    
    # 3. Init CEM Planner
    planner = CEMPlanner(
        world_model=model,
        action_dim=config.action_space,
        horizon=15,
        num_samples=500,
        num_elites=50,
        num_iters=5,
        device=device
    )
    
    print("--- Starting Planning Execution Loop ---")
    
    for ep in range(episodes):
        obs = env.reset()
        
        # Init latent state tracking (per environment)
        # We need to maintain h_t across time steps.
        B = config.num_envs
        h, z = model.rssM.initial_state(B, device)
        a_prev = torch.zeros((B, config.action_space), device=device)
        
        total_rewards = torch.zeros(B, device=device)
        
        for t in range(steps):
            # Encode current observation
            rgb_flat = obs["rgb"].unsqueeze(1) # (B, 1, H, W, C) -> Need to fix shape for encoder
            # The encoder expects (B, 3, 256, 256). The env returns (B, 256, 256, 3)
            rgb = obs["rgb"].permute(0, 3, 1, 2) / 255.0
            state = obs["robot_state"]
            
            with torch.no_grad():
                embed = model.encoder(rgb, state)
                
                # Step representation model to get current actual latent state
                h, z, _, _ = model.rssM.step_posterior(h, z, a_prev, embed)
                
                # The CEM Planner currently expects (1, deter_dim) but we have (B, deter_dim)
                # To plan for all B environments simultaneously, we'd need a batched CEM.
                # For simplicity here, we loop over envs or do batched planning if implemented.
                # Since planner is written for B=1 logic inside, we'll execute it for env 0 
                # and use random for the rest, OR we update the CEM to handle batched initial states.
                # Let's do a simple loop for env 0 to prove execution works.
                
                best_action_env0 = planner.plan(h[0:1], z[0:1])
            
            # Create action batch (planner action for env 0, zeros for others)
            actions = torch.zeros((B, config.action_space), device=device)
            actions[0] = best_action_env0
            
            # Step environment
            next_obs, rewards, dones, _ = env.step(actions)
            
            # Update loop variables
            obs = next_obs
            a_prev = actions
            
            if isinstance(rewards, torch.Tensor):
                total_rewards += rewards
                
            print(f"Ep {ep+1} | Step {t+1:03d} | Env0 Action: {actions[0].cpu().numpy().round(2)} | Env0 Reward: {total_rewards[0].item():.2f}")

    print("Execution loop complete.")

if __name__ == "__main__":
    run_planner_loop()
