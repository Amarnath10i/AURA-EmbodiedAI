from dataclasses import dataclass, field
import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any

@dataclass
class Affordance:
    visual_signature: np.ndarray   # latent centroid (dim: latent_dim)
    action_context: int            # which action triggered this
    effect_type: str               # "BLOCKS", "CHANGES", "ENABLES", etc.
    confidence: float = 0.0        # frequency of this effect
    surprise_mean: float = 0.0     # avg prediction error
    examples: List[Dict] = field(default_factory=list) # stored transitions

class AffordanceLearner:
    """
    Discovers affordances (what objects do) by analyzing prediction surprises
    from the world model.
    """
    def __init__(self, latent_dim: int, surprise_threshold: float = 0.1):
        self.latent_dim = latent_dim
        self.surprise_threshold = surprise_threshold
        self.memory: List[Affordance] = []
        
        # Buffer for clustering
        self.surprising_transitions = []
        
    def add_transition(self, obs_latent: torch.Tensor, action: int, next_obs_actual: torch.Tensor, next_obs_predicted: torch.Tensor, reward_actual: float, reward_predicted: float):
        """
        Evaluate a transition for surprise and store if it exceeds threshold.
        All inputs are single elements (not batched).
        """
        # Calculate surprise (MSE of latent reconstruction or reward)
        # Here we use latent difference as a proxy for visual surprise
        surprise = torch.nn.functional.mse_loss(next_obs_predicted, next_obs_actual).item()
        reward_surprise = abs(reward_predicted - reward_actual)
        
        total_surprise = surprise + 0.5 * reward_surprise
        
        if total_surprise > self.surprise_threshold:
            self.surprising_transitions.append({
                'signature': obs_latent.detach().cpu().numpy(),
                'action': action,
                'surprise': total_surprise,
                'reward_surprise': reward_surprise
            })
            
    def build_memory(self, n_clusters: int = 5):
        """
        Cluster the surprising transitions to form discrete affordances.
        """
        if len(self.surprising_transitions) < n_clusters:
            print("Not enough surprising transitions to cluster affordances.")
            return
            
        signatures = np.stack([t['signature'] for t in self.surprising_transitions])
        
        # Cluster visual signatures to find "object types"
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(signatures)
        
        self.memory = []
        for c in range(n_clusters):
            idx = np.where(clusters == c)[0]
            if len(idx) == 0: continue
            
            cluster_transitions = [self.surprising_transitions[i] for i in idx]
            
            # Analyze dominant action and effect
            actions = [t['action'] for t in cluster_transitions]
            action_context = max(set(actions), key=actions.count)
            
            avg_surprise = np.mean([t['surprise'] for t in cluster_transitions])
            avg_rew_surprise = np.mean([t['reward_surprise'] for t in cluster_transitions])
            
            # Heuristic rule for effect type (in a real system, we'd use latent state differences)
            if avg_rew_surprise > 0.1:
                effect = "BLOCKS"  # Large reward surprise usually means unexpected collision
            else:
                effect = "CHANGES" # Visual surprise without reward penalty
                
            aff = Affordance(
                visual_signature=kmeans.cluster_centers_[c],
                action_context=action_context,
                effect_type=effect,
                confidence=len(idx) / len(self.surprising_transitions),
                surprise_mean=avg_surprise,
                examples=cluster_transitions[:5]
            )
            self.memory.append(aff)
            
        print(f"Built affordance memory with {len(self.memory)} affordances.")
        
    def find_similar(self, visual_signature: np.ndarray, threshold: float = 0.5) -> List[Affordance]:
        """
        Retrieve affordances that match the current visual signature.
        """
        matches = []
        for aff in self.memory:
            dist = np.linalg.norm(aff.visual_signature - visual_signature)
            if dist < threshold:
                matches.append(aff)
        return matches
