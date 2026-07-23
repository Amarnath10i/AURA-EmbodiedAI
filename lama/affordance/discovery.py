"""
Affordance Discovery Module (Phase 6 — Core Research Contribution)
==================================================================
Discovers what objects DO (affordances) without any human labels.

Method:
    For every interaction transition (z_t, action, z_{t+1}):
        1. Compute Δz = z_{t+1} - z_t
        2. Collect all Δz vectors.
        3. Cluster them using DBSCAN or KMeans.
        4. Each cluster = one discovered affordance (e.g., pushable, openable, liftable).

The clusters are pseudo-labels for the Affordance Predictor Head.
"""
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DiscoveredAffordance:
    cluster_id: int
    centroid: np.ndarray         # Mean Δz for this cluster
    label: str                    # Auto-assigned or human-readable name
    count: int                    # Number of transitions in this cluster
    example_actions: List[int]    # Most common actions that produced this effect

class AffordanceDiscovery:
    """
    Discovers affordances from a buffer of latent transitions.
    """
    def __init__(self, method: str = "dbscan", n_clusters: int = 6):
        self.method = method
        self.n_clusters = n_clusters
        
        self.delta_z_buffer: List[np.ndarray] = []
        self.action_buffer: List[int] = []
        self.z_t_buffer: List[np.ndarray] = []
        
        self.affordances: List[DiscoveredAffordance] = []
        self.labels: Optional[np.ndarray] = None
        
    def add_transition(self, z_t: np.ndarray, action: int, z_next: np.ndarray):
        """Add a single interaction transition to the buffer."""
        delta_z = z_next - z_t
        self.delta_z_buffer.append(delta_z)
        self.action_buffer.append(action)
        self.z_t_buffer.append(z_t)
        
    def add_batch(self, z_all: np.ndarray, actions: np.ndarray):
        """
        Add a batch of sequential transitions.
        z_all: (T, latent_dim)
        actions: (T-1,) or (T-1, action_dim)
        """
        for t in range(len(actions)):
            action_id = int(actions[t]) if actions[t].ndim == 0 else int(np.argmax(actions[t]))
            self.add_transition(z_all[t], action_id, z_all[t+1])
    
    def discover(self, min_transitions: int = 20) -> List[DiscoveredAffordance]:
        """
        Run clustering on the collected Δz vectors to discover affordances.
        """
        if len(self.delta_z_buffer) < min_transitions:
            print(f"Not enough transitions ({len(self.delta_z_buffer)}/{min_transitions}). Skipping discovery.")
            return []
            
        X = np.array(self.delta_z_buffer)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if self.method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            self.labels = clusterer.fit_predict(X_scaled)
        elif self.method == "kmeans":
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.labels = clusterer.fit_predict(X_scaled)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        unique_labels = set(self.labels)
        unique_labels.discard(-1)  # Remove noise label from DBSCAN
        
        # Auto-assign human-readable names based on cluster characteristics
        AFFORDANCE_NAMES = [
            "pushable", "pullable", "openable", "rotatable",
            "liftable", "pressable", "slidable", "graspable",
            "toggleable", "stackable"
        ]
        
        self.affordances = []
        for i, label in enumerate(sorted(unique_labels)):
            mask = self.labels == label
            cluster_deltas = X[mask]
            cluster_actions = np.array(self.action_buffer)[mask]
            
            # Find the most common action in this cluster
            from collections import Counter
            action_counts = Counter(cluster_actions.tolist())
            top_actions = [a for a, _ in action_counts.most_common(3)]
            
            name = AFFORDANCE_NAMES[i] if i < len(AFFORDANCE_NAMES) else f"affordance_{label}"
            
            aff = DiscoveredAffordance(
                cluster_id=int(label),
                centroid=cluster_deltas.mean(axis=0),
                label=name,
                count=int(mask.sum()),
                example_actions=top_actions
            )
            self.affordances.append(aff)
            
        print(f"\n=== Discovered {len(self.affordances)} Affordances ===")
        for aff in self.affordances:
            print(f"  [{aff.label}] cluster={aff.cluster_id}, count={aff.count}, "
                  f"top_actions={aff.example_actions}, "
                  f"|centroid|={np.linalg.norm(aff.centroid):.3f}")
        
        return self.affordances
    
    def get_pseudo_labels(self) -> np.ndarray:
        """Returns the cluster assignments for all buffered transitions."""
        if self.labels is None:
            raise RuntimeError("Call discover() first.")
        return self.labels
