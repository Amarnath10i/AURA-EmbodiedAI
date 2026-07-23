import torch
import numpy as np
from dataclasses import dataclass
from .config import AgentConfig
from .world_model import DualWorldModel, UncertaintyEstimator

@dataclass
class Hypothesis:
    condition_latent: np.ndarray
    tested_action: int
    expected_uncertainty_reduction: float
    confidence: float = 0.0
    confirmed: bool = False

class HypothesisGenerator:
    """
    Monitors uncertainty. When a highly uncertain state is reached, it generates
    a hypothesis to test an action that might resolve it.
    """
    def __init__(self, uncertainty_threshold: float = 0.5):
        self.threshold = uncertainty_threshold
        self.active_hypotheses = []
        
    def check_state(self, z_t: torch.Tensor, uncertainty: float) -> bool:
        """Returns True if the current state warrants hypothesis testing."""
        return uncertainty > self.threshold
        
    def generate(self, z_t: torch.Tensor, action: int, expected_reduction: float) -> Hypothesis:
        hyp = Hypothesis(
            condition_latent=z_t.detach().cpu().numpy(),
            tested_action=action,
            expected_uncertainty_reduction=expected_reduction
        )
        self.active_hypotheses.append(hyp)
        return hyp

class ActiveExperimentPlanner:
    """
    Plans actions specifically to maximize intrinsic reward (uncertainty),
    acting as a scientist designing an experiment rather than a goal-seeker.
    """
    def __init__(self, config: AgentConfig, world_model: DualWorldModel):
        self.config = config
        self.world_model = world_model
        
    def plan_experiment(self, h_p: torch.Tensor, s_p: torch.Tensor, h_b: torch.Tensor, s_b: torch.Tensor) -> int:
        """
        Uses CEM to find an action sequence that maximizes Uncertainty (information gain).
        """
        B = self.config.cem_candidates
        H = min(5, self.config.cem_horizon) # Shorter horizon for pure exploration
        
        action_probs = torch.ones((H, self.config.action_dim), device=self.config.device) / self.config.action_dim
        
        for iteration in range(self.config.cem_iterations):
            dist = torch.distributions.Categorical(probs=action_probs)
            actions = dist.sample((B,))
            actions_onehot = torch.nn.functional.one_hot(actions, num_classes=self.config.action_dim).float()
            
            h_curr_p = h_p.repeat(B, 1)
            s_curr_p = s_p.repeat(B, 1)
            h_curr_b = h_b.repeat(B, 1)
            s_curr_b = s_b.repeat(B, 1)
            
            cumulative_uncertainty = torch.zeros(B, device=self.config.device)
            
            for t in range(H):
                a_t = actions_onehot[:, t, :]
                
                # Imagine step
                (h_curr_p, s_curr_p, prior_p), (h_curr_b, s_curr_b, prior_b) = self.world_model.step_prior(
                    h_curr_p, s_curr_p, h_curr_b, s_curr_b, a_t
                )
                
                # Objective is to maximize uncertainty
                u_t = UncertaintyEstimator.compute(prior_p, prior_b)
                
                # If we test 'interact' (action 4), we give a small bonus to encourage testing
                interact_bonus = (actions[:, t] == 4).float() * 0.2
                
                cumulative_uncertainty += u_t + interact_bonus
                
            _, topk_indices = torch.topk(cumulative_uncertainty, self.config.cem_top_k)
            topk_actions = actions[topk_indices]
            
            new_probs = torch.zeros_like(action_probs)
            for t in range(H):
                counts = torch.bincount(topk_actions[:, t], minlength=self.config.action_dim).float()
                new_probs[t] = counts / self.config.cem_top_k
                
            action_probs = 0.8 * new_probs + 0.2 * action_probs
            
        return torch.argmax(action_probs[0]).item()
