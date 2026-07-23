import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from aura_agent.config import AgentConfig
from aura_agent.agent import AURAAgent
from aura_warehouse_sim.warehouse_sim.sim import WarehouseSim

def evaluate_agent(episodes: int = 10, seed_offset: int = 42000):
    config = AgentConfig()
    agent = AURAAgent(config)
    
    successes = 0
    collisions = 0
    steps_taken = []
    
    print(f"\n--- Starting Evaluation ({episodes} episodes) ---")
    
    for ep in tqdm(range(episodes), desc="Evaluating"):
        sim = WarehouseSim(seed=seed_offset + ep, obs_size=config.obs_size)
        obs, info = sim.reset()
        agent.reset()
        
        while True:
            # Agent picks action
            action = agent.step(obs)
            
            # Env steps
            next_obs, reward, term, trunc, info = sim.step(action)
            obs = next_obs
            
            if term or trunc:
                if info.get('success', False):
                    successes += 1
                if info.get('collision') is not None:
                    collisions += 1
                steps_taken.append(info['t'])
                break
                
    success_rate = (successes / episodes) * 100
    collision_rate = (collisions / episodes) * 100
    avg_steps = np.mean(steps_taken) if steps_taken else 0
    
    print("\n--- Evaluation Results ---")
    print(f"Success Rate:   {success_rate:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}%")
    print(f"Avg Steps/Ep:   {avg_steps:.1f}")
    
    # We could also trigger Failure Analysis here if we collected errors
    
if __name__ == "__main__":
    evaluate_agent(episodes=10) # Quick eval for prototyping
