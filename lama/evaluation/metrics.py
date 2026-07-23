"""
Evaluation Metrics (Phase 10)
=============================
Comprehensive metrics for World Model, Affordance Discovery, Planning, and Exploration.
"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, normalized_mutual_info_score
from typing import Dict, List, Optional

class WorldModelMetrics:
    """Evaluates the accuracy of the latent world model."""
    
    @staticmethod
    def reconstruction_mse(rgb_pred: np.ndarray, rgb_actual: np.ndarray) -> float:
        return float(np.mean((rgb_pred - rgb_actual) ** 2))
    
    @staticmethod
    def latent_prediction_mse(z_pred: np.ndarray, z_actual: np.ndarray) -> float:
        return float(np.mean((z_pred - z_actual) ** 2))
    
    @staticmethod
    def reward_mae(r_pred: np.ndarray, r_actual: np.ndarray) -> float:
        return float(np.mean(np.abs(r_pred - r_actual)))
    
    @staticmethod
    def multistep_rollout_error(z_preds: List[np.ndarray], z_actuals: List[np.ndarray]) -> Dict[str, float]:
        """Compute MSE at each step of a multi-step rollout."""
        errors = {}
        for t, (zp, za) in enumerate(zip(z_preds, z_actuals)):
            errors[f"step_{t+1}_mse"] = float(np.mean((zp - za) ** 2))
        errors["mean_mse"] = float(np.mean(list(errors.values())))
        return errors


class AffordanceMetrics:
    """Evaluates the quality of discovered affordances against ground truth (if available)."""
    
    @staticmethod
    def clustering_purity(predicted_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """Fraction of samples in each cluster that belong to the majority class."""
        total = 0
        correct = 0
        for cluster_id in np.unique(predicted_labels):
            if cluster_id == -1:
                continue
            mask = predicted_labels == cluster_id
            cluster_true = true_labels[mask]
            most_common_count = np.bincount(cluster_true).max()
            total += mask.sum()
            correct += most_common_count
        return correct / max(total, 1)
    
    @staticmethod
    def nmi(predicted_labels: np.ndarray, true_labels: np.ndarray) -> float:
        valid = predicted_labels >= 0
        return float(normalized_mutual_info_score(true_labels[valid], predicted_labels[valid]))
    
    @staticmethod
    def precision_recall_f1(predicted_labels: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        valid = predicted_labels >= 0
        p, r, f1, _ = precision_recall_fscore_support(
            true_labels[valid], predicted_labels[valid], average='macro', zero_division=0
        )
        return {"precision": float(p), "recall": float(r), "f1": float(f1)}


class PlanningMetrics:
    """Tracks planning success over evaluation episodes."""
    
    def __init__(self):
        self.successes = 0
        self.total = 0
        self.steps_list = []
        self.replans = 0
        
    def log_episode(self, success: bool, steps: int, n_replans: int = 0):
        self.total += 1
        if success:
            self.successes += 1
        self.steps_list.append(steps)
        self.replans += n_replans
        
    def summary(self) -> Dict[str, float]:
        return {
            "success_rate": self.successes / max(self.total, 1),
            "avg_steps": float(np.mean(self.steps_list)) if self.steps_list else 0.0,
            "total_replans": self.replans,
        }


class ExplorationMetrics:
    """Measures exploration coverage and efficiency."""
    
    def __init__(self):
        self.visited_states: set = set()
        self.collisions = 0
        self.total_steps = 0
        
    def log_step(self, state_hash: int, collided: bool):
        self.visited_states.add(state_hash)
        self.total_steps += 1
        if collided:
            self.collisions += 1
            
    def summary(self) -> Dict[str, float]:
        return {
            "unique_states_visited": len(self.visited_states),
            "collision_rate": self.collisions / max(self.total_steps, 1),
            "exploration_efficiency": len(self.visited_states) / max(self.total_steps, 1),
        }
