import torch
from .config import AgentConfig
from .world_model import WorldModel
from .affordance import AffordanceLearner

class CreativePlanner:
    """
    Model-based planner using the Cross-Entropy Method (CEM) combined with Affordance Search
    for creative reuse in novel situations.
    """
    def __init__(self, config: AgentConfig, world_model: WorldModel, affordances: AffordanceLearner):
        self.config = config
        self.world_model = world_model
        self.affordances = affordances
        
    def plan(self, obs_latent: torch.Tensor, h_t: torch.Tensor, s_t: torch.Tensor) -> int:
        """
        Plans the best immediate action using CEM over imagined rollouts.
        obs_latent: (1, latent_dim)
        h_t, s_t: current world model state (1, dim)
        """
        B = self.config.cem_candidates
        H = self.config.cem_horizon
        
        # Initialize action distribution (mean and std for continuous space, but we have discrete actions)
        # We will use CEM to optimize a continuous policy parameterization, or just sample discrete sequences.
        # For discrete actions, it's easier to sample uniform random sequences, evaluate, and pick the best,
        # or maintain a categorical distribution per step.
        
        # We maintain a Categorical distribution over actions at each step of the horizon
        action_probs = torch.ones((H, self.config.action_dim), device=self.config.device) / self.config.action_dim
        
        for iteration in range(self.config.cem_iterations):
            # 1. Sample sequences of actions from current distribution
            # shape: (B, H)
            dist = torch.distributions.Categorical(probs=action_probs)
            actions = dist.sample((B,))
            actions_onehot = torch.nn.functional.one_hot(actions, num_classes=self.config.action_dim).float()
            
            # 2. Imagine rollouts
            # Duplicate start state B times
            h_curr = h_t.repeat(B, 1)
            s_curr = s_t.repeat(B, 1)
            
            cumulative_rewards = torch.zeros(B, device=self.config.device)
            
            for t in range(H):
                a_t = actions_onehot[:, t, :]
                
                # Step prior (imagination)
                h_curr, s_curr, _ = self.world_model.step_prior(h_curr, s_curr, a_t)
                
                # Predict reward
                r_t = self.world_model.reward(h_curr, s_curr).squeeze(-1)
                
                # Discount or just sum
                cumulative_rewards += r_t
                
            # 3. Affordance search (Creative reuse injection)
            # If all rewards are terrible, we might be stuck. Query affordances.
            if cumulative_rewards.max() < 0.1 and iteration == 0 and len(self.affordances.memory) > 0:
                # Find similar affordances to current observation
                obs_np = obs_latent.squeeze(0).detach().cpu().numpy()
                matches = self.affordances.find_similar(obs_np)
                
                if matches:
                    # Inject affordance-guided actions into the first few candidates
                    for i, match in enumerate(matches[:min(B//5, len(matches))]):
                        # Force the first action to be the affordance action context
                        actions[i, 0] = match.action_context
                        actions_onehot[i, 0] = torch.nn.functional.one_hot(torch.tensor(match.action_context), num_classes=self.config.action_dim).float().to(self.config.device)
                        # We would re-evaluate these injected sequences if doing properly,
                        # for now we rely on the next iterations to pick up on it if it's good.
            
            # 4. Update distribution (Cross-Entropy)
            # Keep top K
            _, topk_indices = torch.topk(cumulative_rewards, self.config.cem_top_k)
            topk_actions = actions[topk_indices] # (K, H)
            
            # Update action_probs
            new_probs = torch.zeros_like(action_probs)
            for t in range(H):
                counts = torch.bincount(topk_actions[:, t], minlength=self.config.action_dim).float()
                new_probs[t] = counts / self.config.cem_top_k
                
            # Mix with previous to avoid premature convergence
            action_probs = 0.8 * new_probs + 0.2 * action_probs
            
        # Return the first action of the best sequence from the final distribution
        # Or simply the argmax of the first step's probabilities
        best_action = torch.argmax(action_probs[0]).item()
        return best_action
