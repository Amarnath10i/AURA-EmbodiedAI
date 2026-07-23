"""
LAMA: Full Pipeline Runner
==========================
Orchestrates all 10 phases sequentially.
"""
import os
import sys

def run_lama_pipeline():
    print("=" * 60)
    print("  LAMA: Learning Affordances via Active Exploration")
    print("       and Dual World Models in NVIDIA Isaac Lab")
    print("=" * 60)
    
    # Phase 2 — Collect Demonstrations
    print("\n[Phase 2] Collecting Demonstrations (Mock Mode)...")
    os.system(f"{sys.executable} lama/scripts/collect_demonstrations.py --episodes 10 --steps 50")
    
    # Phase 5 — Train Dual World Model
    print("\n[Phase 5] Training Dual World Model...")
    os.system(f"{sys.executable} lama/training/train_world_model.py")
    
    # Phase 6 — Discover & Train Affordances
    print("\n[Phase 6] Discovering Affordances & Training Predictor...")
    os.system(f"{sys.executable} lama/training/train_affordance.py")
    
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("  Models saved to: models/")
    print("=" * 60)

if __name__ == "__main__":
    run_lama_pipeline()
