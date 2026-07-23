import os
import sys

def run_pipeline():
    print("========================================")
    print("AURA Embodied AI Agent - Full Pipeline")
    print("========================================\n")
    
    print("1. Data Collection...")
    os.system(f"{sys.executable} scripts/01_collect_data.py --episodes 10")
    
    print("\n2. World Model Training...")
    os.system(f"{sys.executable} scripts/02_train_world_model.py")
    
    print("\n3. Affordance Discovery...")
    os.system(f"{sys.executable} scripts/03_discover_affordances.py")
    
    print("\n4. Agent Evaluation (Planner + World Model)...")
    os.system(f"{sys.executable} scripts/05_evaluate.py")
    
    print("\nPipeline complete!")

if __name__ == "__main__":
    run_pipeline()
