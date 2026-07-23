"""
Cross-Entropy Method (CEM) Planner (Phase 8)
=============================================
Uses the trained Dual World Model to imagine future trajectories,
then selects the action sequence that maximizes expected reward.

    Current state -> Encode -> Latent z
        -> World model rollout (10-20 steps)
            -> Evaluate cumulative reward for each candidate sequence
        -> Select best sequence
        -> Execute only the first action
        -> Repeat (Model Predictive Control / MPC)
"""
import torch
import numpy as np
from lama.models.forward_model import DualWorldModel
from lama.models.decoder import RewardModel

class CEMPlanner:
    def __init__(
        self, 
        action_dim: int, 
        world_model: DualWorldModel,
        reward_model: RewardModel,
        horizon: int = 15,
        candidates: int = 256,
        top_k: int = 32,
        iterations: int = 5,
        device: str = "cuda"
    ):
        self.action_dim = action_dim
        self.world_model = world_model
        self.reward_model = reward_model
        self.horizon = horizon
        self.candidates = candidates
        self.top_k = top_k
        self.iterations = iterations
        self.device = device
        
    @torch.no_grad()
    def plan(self, h_p, s_p, h_b, s_b) -> np.ndarray:
        """
        Run CEM optimization to find the best action sequence.
        Returns the first action of the best sequence.
        """
        # Initialize action distribution (continuous)
        action_mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        action_std = torch.ones(self.horizon, self.action_dim, device=self.device) * 0.5
        
        det_dim = h_p.size(-1)
        stoch_dim = s_p.size(-1)
        
        for iteration in range(self.iterations):
            # Sample candidate action sequences
            noise = torch.randn(self.candidates, self.horizon, self.action_dim, device=self.device)
            actions = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * noise
            actions = torch.clamp(actions, -1.0, 1.0)
            
            # Expand initial states for all candidates
            h_p_exp = h_p.expand(self.candidates, -1)
            s_p_exp = s_p.expand(self.candidates, -1)
            h_b_exp = h_b.expand(self.candidates, -1)
            s_b_exp = s_b.expand(self.candidates, -1)
            
            # Imagine rollouts
            cumulative_reward = torch.zeros(self.candidates, device=self.device)
            
            for t in range(self.horizon):
                a_t = actions[:, t]
                
                (h_p_exp, s_p_exp, _), (h_b_exp, s_b_exp, _) = self.world_model.step_prior(
                    h_p_exp, s_p_exp, h_b_exp, s_b_exp, a_t
                )
                
                features = torch.cat([h_p_exp, s_p_exp, h_b_exp, s_b_exp], dim=-1)
                reward = self.reward_model(features).squeeze(-1)
                cumulative_reward += reward * (0.99 ** t) # Discounting
            
            # Select top-k sequences
            _, topk_idx = torch.topk(cumulative_reward, self.top_k)
            elite_actions = actions[topk_idx]
            
            # Refit the distribution
            action_mean = elite_actions.mean(dim=0)
            action_std = elite_actions.std(dim=0) + 1e-6
        
        # Return the first action of the optimized sequence
        return action_mean[0].cpu().numpy()
