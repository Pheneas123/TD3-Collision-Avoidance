from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import random
import numpy as np
import torch

DEFAULT_BUFFER_SIZE: int = 1_000_000

class ReplayBuffer:
    def __init__(
        self,
        max_size: int = DEFAULT_BUFFER_SIZE,
        device: torch.device | str = "cpu",
    ):
        """
        Replay buffer for off-policy RL algorithms (TD3).

        Args:
            max_size: Maximum number of transitions stored.
            device: Torch device to move sampled tensors to.
        """
        self.buffer: Deque[
            Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]
        ] = deque(maxlen=max_size)

        self.device = torch.device(device)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition in the buffer.
        """
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions.

        Returns:
            state, action, reward, next_state, done tensors on self.device
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.from_numpy(np.asarray(state)).float().to(self.device)
        action = torch.from_numpy(np.asarray(action)).float().to(self.device)
        reward = (torch.from_numpy(np.asarray(reward)).float().unsqueeze(1).to(self.device))
        next_state = torch.from_numpy(np.asarray(next_state)).float().to(self.device)
        done = torch.from_numpy(np.asarray(done)).float().unsqueeze(1).to(self.device)

        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)
