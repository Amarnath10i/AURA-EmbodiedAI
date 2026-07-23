import numpy as np
import torch
import os
import glob
from typing import Tuple, Dict
from .config import AgentConfig

class ReplayBuffer:
    """
    Experience replay memory designed to load .npz shards from the collector
    and sample sequences for RSSM training.
    """
    def __init__(self, config: AgentConfig, max_size: int = 500_000):
        self.config = config
        self.max_size = max_size
        
        # Pre-allocate memory
        S = config.obs_size
        self.obs = np.zeros((max_size, S, S, 3), dtype=np.uint8)
        self.actions = np.zeros((max_size,), dtype=np.int8)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=bool)
        self.episodes = np.zeros((max_size,), dtype=np.int32)
        self.event_tags = np.zeros((max_size, 4), dtype=bool)
        
        self.size = 0
        self.ptr = 0
        
    def load_shards(self, data_dir: str):
        """Load transitions from .npz files collected by collect.py"""
        files = glob.glob(os.path.join(data_dir, "*.npz"))
        print(f"Found {len(files)} dataset shards in {data_dir}")
        for f in files:
            data = np.load(f)
            N = len(data["obs"])
            
            # If exceeding capacity, wrap around
            if self.ptr + N > self.max_size:
                rem = self.max_size - self.ptr
                self._insert(data, slice(0, rem), slice(self.ptr, self.max_size))
                self.ptr = 0
                N -= rem
                self.size = self.max_size
                
                if N > 0:
                    self._insert(data, slice(rem, rem + N), slice(self.ptr, self.ptr + N))
                    self.ptr += N
            else:
                self._insert(data, slice(0, N), slice(self.ptr, self.ptr + N))
                self.ptr = (self.ptr + N) % self.max_size
                self.size = min(self.size + N, self.max_size)
                
        print(f"Loaded {self.size} transitions into ReplayBuffer.")
        
    def _insert(self, data, src_slice, dst_slice):
        self.obs[dst_slice] = data["obs"][src_slice]
        self.actions[dst_slice] = data["action"][src_slice]
        self.episodes[dst_slice] = data["episode"][src_slice]
        if "event_tags" in data:
            self.event_tags[dst_slice] = data["event_tags"][src_slice]
            
        # Reconstruct pseudo-rewards / dones based on basic tags if they don't exist
        # In a full run, the collector should save true rewards.
        # Here we just set dummy values since the collector currently only saves obs/act/tags/ep
        if "reward" in data:
            self.rewards[dst_slice] = data["reward"][src_slice]
        if "term" in data:
            self.dones[dst_slice] = data["term"][src_slice]
            
    def sample_sequence_batch(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of contiguous sequences of length seq_len.
        Ensures sequences don't cross episode boundaries.
        Returns tensors on config.device
        """
        assert self.size > seq_len, "Not enough data to sample a sequence"
        
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        for _ in range(batch_size):
            valid = False
            while not valid:
                # Pick a random starting index
                start_idx = np.random.randint(0, self.size - seq_len)
                end_idx = start_idx + seq_len
                
                # Check if this sequence crosses the current write pointer
                if start_idx < self.ptr <= end_idx and self.size == self.max_size:
                    continue
                    
                # Check if this sequence crosses an episode boundary
                if len(np.unique(self.episodes[start_idx:end_idx])) > 1:
                    continue
                    
                valid = True
                batch_obs.append(self.obs[start_idx:end_idx])
                batch_actions.append(self.actions[start_idx:end_idx])
                batch_rewards.append(self.rewards[start_idx:end_idx])
                batch_dones.append(self.dones[start_idx:end_idx])
                
        # Convert to numpy then torch
        # Shape: (batch_size, seq_len, ...)
        obs = torch.FloatTensor(np.stack(batch_obs)).permute(0, 1, 4, 2, 3) / 255.0 # (B, T, C, H, W)
        actions = torch.LongTensor(np.stack(batch_actions)) # (B, T)
        actions_onehot = torch.nn.functional.one_hot(actions, num_classes=self.config.action_dim).float()
        rewards = torch.FloatTensor(np.stack(batch_rewards)) # (B, T)
        dones = torch.BoolTensor(np.stack(batch_dones)) # (B, T)
        
        return {
            "obs": obs.to(self.config.device),
            "actions": actions_onehot.to(self.config.device),
            "rewards": rewards.to(self.config.device),
            "dones": dones.to(self.config.device)
        }
