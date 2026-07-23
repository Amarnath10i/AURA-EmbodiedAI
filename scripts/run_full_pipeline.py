import os
import sys

def run_pipeline():
    print("========================================")
    print("AURA Embodied AI Agent - Full Pipeline")
    print("========================================\n")
    
    print("1a. Active Exploration (Difficulty Level 1: Empty)...")
    os.system(f"{sys.executable} scripts/01_active_exploration.py --episodes 5 --difficulty 1")
    
    print("\n1b. Active Exploration (Difficulty Level 2: Static Clutter)...")
    os.system(f"{sys.executable} scripts/01_active_exploration.py --episodes 5 --difficulty 2")
    
    print("\n2. Dual World Model Training...")
    os.system(f"{sys.executable} scripts/02_train_world_model.py")
    
    print("\n[Milestone 1 Complete] Pipeline pauses here.")
    print("Affordance Discovery and Evaluation will be added in Milestone 2/3.")

if __name__ == "__main__":
    run_pipeline()
