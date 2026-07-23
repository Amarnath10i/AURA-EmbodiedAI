"""
Advanced Outcome-Based Affordance Discovery.
Uses HDBSCAN to discover latent prototypes dynamically instead of fixed KMeans clusters.
"""
import torch
import numpy as np
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    from sklearn.cluster import KMeans
    HDBSCAN_AVAILABLE = False
    print("WARNING: hdbscan not installed. Falling back to KMeans. Run: pip install hdbscan")

from typing import List

class AffordanceDiscovery:
    def __init__(self, min_cluster_size: int = 15):
        self.min_cluster_size = min_cluster_size
        
        if HDBSCAN_AVAILABLE:
            self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, prediction_data=True)
        else:
            self.clusterer = KMeans(n_clusters=8, random_state=42, n_init=10)
            
        self.is_fitted = False
        
    def extract_outcomes(
        self, 
        actions: torch.Tensor, 
        z_seq: torch.Tensor, 
        rewards: torch.Tensor
    ) -> np.ndarray:
        """Extracts [Action, Delta Z, Reward] vectors."""
        B, T = actions.shape[:2]
        outcomes = []
        
        for t in range(T - 1):
            a_t = actions[:, t]
            z_t = z_seq[:, t]
            z_next = z_seq[:, t+1]
            r_t = rewards[:, t].unsqueeze(-1)
            
            delta_z = z_next - z_t
            outcome = torch.cat([a_t, delta_z, r_t], dim=-1)
            outcomes.append(outcome)
            
        outcomes_tensor = torch.stack(outcomes, dim=1)
        return outcomes_tensor.detach().cpu().numpy().reshape(-1, outcomes_tensor.shape[-1])
        
    def fit(self, outcomes_np: np.ndarray):
        """Fits HDBSCAN to dynamically discover affordance prototypes."""
        print(f"Discovering affordances from {len(outcomes_np)} interaction outcomes...")
        self.clusterer.fit(outcomes_np)
        self.is_fitted = True
        
        if HDBSCAN_AVAILABLE:
            num_clusters = len(set(self.clusterer.labels_)) - (1 if -1 in self.clusterer.labels_ else 0)
            print(f"HDBSCAN discovered {num_clusters} dynamic affordance prototypes (excluding noise).")
        else:
            print(f"KMeans fitted with 8 clusters.")
            
        # In a full implementation of Prototype Learning, we would extract the cluster exemplars
        # (the densest core points from HDBSCAN) and use them as learnable embeddings.
        
    def predict(self, outcomes_np: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Clusterer is not fitted yet.")
            
        if HDBSCAN_AVAILABLE:
            labels, _ = hdbscan.approximate_predict(self.clusterer, outcomes_np)
            return labels
        else:
            return self.clusterer.predict(outcomes_np)
