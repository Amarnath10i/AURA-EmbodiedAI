import torch
import numpy as np
from .config import AgentConfig
from .encoder import Encoder, Decoder
from .world_model import WorldModel
from .affordance import AffordanceLearner
from .planner import CreativePlanner
from .failure_analyzer import FailureAnalyzer
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
        self.decoder = Decoder(config.det_state_dim + config.stoch_state_dim, config).to(config.device)
        self.world_model = WorldModel(config).to(config.device)
        
        self.affordances = AffordanceLearner(config.latent_dim)
        self.planner = CreativePlanner(config, self.world_model, self.affordances)
        self.failure_analyzer = FailureAnalyzer()
        
        # State tracking
        self.h_t = torch.zeros(1, config.det_state_dim, device=config.device)
        self.s_t = torch.zeros(1, config.stoch_state_dim, device=config.device)
        self.a_prev = torch.zeros(1, config.action_dim, device=config.device)
        
        self.load_models()
        
    def load_models(self):
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'world_model.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.config.device)
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.decoder.load_state_dict(checkpoint['decoder'])
            self.world_model.load_state_dict(checkpoint['rssm'])
            print("Loaded trained models.")
        else:
            print("No trained models found. Using initialized weights.")
            
    def reset(self):
        self.h_t = torch.zeros(1, self.config.det_state_dim, device=self.config.device)
        self.s_t = torch.zeros(1, self.config.stoch_state_dim, device=self.config.device)
        self.a_prev = torch.zeros(1, self.config.action_dim, device=self.config.device)
        
    def step(self, obs_np: np.ndarray) -> int:
        """
        Execute one agent step.
        """
        with torch.no_grad():
            # 1. Encode Observation
            obs_tensor = torch.FloatTensor(obs_np).permute(2, 0, 1).unsqueeze(0).to(self.config.device) / 255.0
            z_t = self.encoder(obs_tensor)
            
            # 2. Update World Model state (Posterior)
            self.h_t, self.s_t, prior_dist, post_dist = self.world_model.step_posterior(
                self.h_t, self.s_t, self.a_prev, z_t
            )
            
            # 3. Plan next action
            action = self.planner.plan(z_t, self.h_t, self.s_t)
            
            # 4. Update previous action tracking
            self.a_prev = torch.nn.functional.one_hot(torch.tensor([action]), num_classes=self.config.action_dim).float().to(self.config.device)
            
            return action
