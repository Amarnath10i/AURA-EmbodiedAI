import torch
from .config import AgentConfig
from .world_model import WorldModel

class CuriosityExplorer:
    """
    Intrinsic motivation module that computes an intrinsic reward based on
    prediction error or state uncertainty (Plan2Explore style).
    """
    def __init__(self, config: AgentConfig, world_model: WorldModel):
        self.config = config
        self.world_model = world_model
        
    def compute_intrinsic_reward(self, h_t: torch.Tensor, s_t: torch.Tensor, a_t: torch.Tensor) -> float:
        """
        Computes intrinsic reward for an imagined action based on world model uncertainty.
        For simplicity, we use the variance of the prior distribution (or ensemble disagreement
        in a full Plan2Explore setup).
        """
        # Step the prior to get the imagined distribution of the next state
        with torch.no_grad():
            _, _, prior_dist = self.world_model.step_prior(h_t, s_t, a_t)
            
            # Use the scale (std dev) of the predicted state distribution as uncertainty
            uncertainty = prior_dist.scale.mean().item()
            
        return uncertainty
