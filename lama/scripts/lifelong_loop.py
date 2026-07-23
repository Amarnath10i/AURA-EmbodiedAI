"""
Lifelong Active Learning Loop (Orchestrator).
The core autonomous loop of LAMA: Collect -> Train -> Discover -> Plan.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lama.scripts.collect_demonstrations import collect_data
from lama.training.train_world_model import train_world_model
from lama.training.train_affordance import train_affordances
from lama.envs.task_config import TabletopTaskConfig

def lifelong_loop():
    print("=======================================================")
    print("  LAMA: Lifelong Active Learning Loop Initialized")
    print("=======================================================")
    
    config = TabletopTaskConfig()
    iteration = 1
    max_iterations = 5
    performance_threshold = 0.95 # e.g. 95% interaction success or affordance accuracy
    
    while iteration <= max_iterations:
        print(f"\n--- [Iteration {iteration}] ---")
        
        # 1. Collect Data (Progressive Scaling + Curiosity)
        # In a full implementation, collect_data would switch from scripted 
        # primitives to the CEM Planner guided by the CuriosityModule after Iteration 1.
        phase = "B" if iteration > 1 else "A"
        print(f"1. Active Data Collection (Phase {phase})...")
        collect_data(phase=phase, steps_per_episode=50, seed=42 + iteration)
        
        # 2. Train World Model
        print("2. Training Dreamer World Model...")
        # Train for fewer epochs in continual loop since buffer is mostly the same
        train_world_model(epochs=10, batch_size=32, seq_len=50)
        
        # 3. Discover Affordances
        print("3. Discovering Affordances (HDBSCAN/KMeans)...")
        train_affordances(epochs=15, batch_size=32)
        
        # 4. Evaluate (Placeholder for OOD Testing)
        print("4. Evaluating OOD Generalization...")
        performance = evaluate_pipeline()
        print(f"Performance Score: {performance:.2f}")
        
        if performance >= performance_threshold:
            print("Performance threshold reached! Lifelong learning complete.")
            break
            
        # 5. Curriculum Scaling
        # Increase environment difficulty (add clutter, dynamic lighting)
        print("5. Scaling Curriculum Difficulty...")
        config.domain_randomization.randomize_lighting = True
        
        iteration += 1

def evaluate_pipeline() -> float:
    """Placeholder for zero-shot OOD evaluation."""
    # Simulates performance growing over time
    return torch.rand(1).item() * 0.5 + 0.5

if __name__ == "__main__":
    lifelong_loop()
