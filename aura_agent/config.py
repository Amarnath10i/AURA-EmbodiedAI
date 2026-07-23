from dataclasses import dataclass
import torch

@dataclass
class AgentConfig:
    # Environment
    obs_size: int = 96
    obs_channels: int = 3
    action_dim: int = 4
    
    # Encoder / Decoder
    latent_dim: int = 64
    
    # RSSM (World Model)
    det_state_dim: int = 256  # GRU hidden state
    stoch_state_dim: int = 32 # Stochastic state (sampled)
    
    # Training
    batch_size: int = 64
    seq_length: int = 15      # Truncated BPTT length
    lr: float = 3e-4
    kl_scale: float = 0.1
    
    # Planning (CEM)
    cem_horizon: int = 15
    cem_iterations: int = 5
    cem_candidates: int = 50
    cem_top_k: int = 5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
