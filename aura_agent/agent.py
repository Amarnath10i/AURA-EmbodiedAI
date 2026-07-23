import torch
import numpy as np
from .config import AgentConfig
from .encoder import Encoder, Decoder
from .world_model import DualWorldModel, UncertaintyEstimator
from .hypothesis import HypothesisGenerator, ActiveExperimentPlanner
import os

class AURAAgent:
    """
    Main Orchestrator for the AURA Embodied AI Agent.
    Ties together Perception, World Model, Affordances, Planning, and Self-Improvement.
    """
    def __init__(self, config: AgentConfig):
        self.config = config
        
        # Subsystems
        self.encoder = Encoder(config).to(config.device)
        self.decoder = Decoder(2 * (config.det_state_dim + config.stoch_state_dim), config).to(config.device)
        self.world_model = DualWorldModel(config).to(config.device)
        
        self.hypothesis_gen = HypothesisGenerator()
        self.active_planner = ActiveExperimentPlanner(config, self.world_model)
        
        # State tracking (Dual)
        self.h_p = torch.zeros(1, config.det_state_dim, device=config.device)
        self.s_p = torch.zeros(1, config.stoch_state_dim, device=config.device)
        self.h_b = torch.zeros(1, config.det_state_dim, device=config.device)
        self.s_b = torch.zeros(1, config.stoch_state_dim, device=config.device)
        self.a_prev = torch.zeros(1, config.action_dim, device=config.device)
        
        self.load_models()
        
    def load_models(self):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'world_model.pth')
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.config.device)
                self.encoder.load_state_dict(checkpoint['encoder'])
                self.decoder.load_state_dict(checkpoint['decoder'])
                self.world_model.load_state_dict(checkpoint['rssm'])
                print("Loaded trained models.")
            except Exception as e:
                print(f"Failed to load old models (architecture likely changed). Using fresh initialized weights.")
        else:
            print("No trained models found. Using initialized weights.")
            
    def reset(self):
        self.h_p = torch.zeros(1, self.config.det_state_dim, device=self.config.device)
        self.s_p = torch.zeros(1, self.config.stoch_state_dim, device=self.config.device)
        self.h_b = torch.zeros(1, self.config.det_state_dim, device=self.config.device)
        self.s_b = torch.zeros(1, self.config.stoch_state_dim, device=self.config.device)
        self.a_prev = torch.zeros(1, self.config.action_dim, device=self.config.device)
        
    def step_active_exploration(self, obs_np: np.ndarray) -> int:
        """
        Execute one active exploration step (hunting for hypotheses).
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_np).permute(2, 0, 1).unsqueeze(0).to(self.config.device) / 255.0
            z_t = self.encoder(obs_tensor)
            
            # Posterior update
            (self.h_p, self.s_p, prior_p, _), (self.h_b, self.s_b, prior_b, _) = self.world_model.step_posterior(
                self.h_p, self.s_p, self.h_b, self.s_b, self.a_prev, z_t
            )
            
            # Uncertainty estimation
            u_t = UncertaintyEstimator.compute(prior_p, prior_b).item()
            
            if self.hypothesis_gen.check_state(z_t, u_t):
                # We reached an interesting state, plan an experiment to maximize info
                action = self.active_planner.plan_experiment(self.h_p, self.s_p, self.h_b, self.s_b)
                self.hypothesis_gen.generate(z_t, action, expected_reduction=u_t * 0.5)
            else:
                # If not highly uncertain, still explore to find uncertainty
                action = self.active_planner.plan_experiment(self.h_p, self.s_p, self.h_b, self.s_b)
                
            self.a_prev = torch.nn.functional.one_hot(torch.tensor([action]), num_classes=self.config.action_dim).float().to(self.config.device)
            return action
