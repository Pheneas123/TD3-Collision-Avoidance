from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR

from td3.networks import Actor, td3_Critic
from td3.replay_buffer import ReplayBuffer


# TD3 constants

DEFAULT_GAMMA: float = 0.99
DEFAULT_TAU: float = 0.005

DEFAULT_POLICY_NOISE: float = 0.2
DEFAULT_NOISE_CLIP: float = 0.5
DEFAULT_POLICY_FREQ: int = 2

DEFAULT_EXPL_NOISE_DECAY_START_EP: int = 500
DEFAULT_EXPL_NOISE_DECAY_RATE: float = 0.999
DEFAULT_EXPL_NOISE_STD_INIT: float = 0.3
DEFAULT_EXPL_NOISE_STD_MIN: float = 0.05

DEFAULT_GRAD_CLIP_NORM: float = 1.0


@dataclass
class TD3HyperParams:
    gamma: float = DEFAULT_GAMMA
    tau: float = DEFAULT_TAU
    policy_noise: float = DEFAULT_POLICY_NOISE
    noise_clip: float = DEFAULT_NOISE_CLIP
    policy_freq: int = DEFAULT_POLICY_FREQ

    expl_noise_decay_start_ep: int = DEFAULT_EXPL_NOISE_DECAY_START_EP
    expl_noise_decay_rate: float = DEFAULT_EXPL_NOISE_DECAY_RATE
    expl_noise_std_init: float = DEFAULT_EXPL_NOISE_STD_INIT
    expl_noise_std_min: float = DEFAULT_EXPL_NOISE_STD_MIN

    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM


class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action,
        *,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        device: Optional[str | torch.device] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        hparams: TD3HyperParams = TD3HyperParams(),
        actor_scheduler_kwargs: Optional[dict] = None,
        critic_scheduler_kwargs: Optional[dict] = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = td3_Critic(state_dim, action_dim).to(self.device)
        self.critic_target = td3_Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimisers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # schedulers
        self.actor_scheduler = None
        self.critic_scheduler = None
        if actor_scheduler_kwargs is not None:
            self.actor_scheduler = CyclicLR(self.actor_optimizer, **actor_scheduler_kwargs)
        if critic_scheduler_kwargs is not None:
            self.critic_scheduler = CyclicLR(self.critic_optimizer, **critic_scheduler_kwargs)

        # Replay buffer
        self.replay_buffer = replay_buffer if replay_buffer is not None else ReplayBuffer(device=self.device)

        # Action bounds on device (for clamping noisy target actions)
        self.max_action_tensor = torch.as_tensor(max_action, dtype=torch.float32, device=self.device)

        # Hyperparameters
        self.hparams = hparams

        # TD3 bookkeeping
        self.total_it = 0

        # Exploration noise state
        self.noise_std = self.hparams.expl_noise_std_init

    def select_action(self, state: np.ndarray, *, noise: bool = True, current_episode: int = 0) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if noise:
            if current_episode >= self.hparams.expl_noise_decay_start_ep:
                self.noise_std = max(
                    self.hparams.expl_noise_std_min,
                    self.noise_std * self.hparams.expl_noise_decay_rate,
                )
            exploration_noise = np.random.normal(0.0, self.noise_std, size=action.shape)

            max_action_np = self.max_action_tensor.detach().cpu().numpy()
            action = np.clip(action + exploration_noise, -max_action_np, max_action_np)

        return action

    def train(self, batch_size: int) -> Optional[Tuple[Optional[float], float]]:
        if len(self.replay_buffer) < batch_size:
            return None

        self.total_it += 1

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        if reward.ndim == 1:
            reward = reward.unsqueeze(1)
        if done.ndim == 1:
            done = done.unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.hparams.policy_noise).clamp(
                -self.hparams.noise_clip, self.hparams.noise_clip
            )
            next_action = self.actor_target(next_state)
            next_action = (next_action + noise).clamp(-self.max_action_tensor, self.max_action_tensor)

            q1_t, q2_t = self.critic_target(next_state, next_action)
            q_min = torch.min(q1_t, q2_t)
            td_target = reward + (1.0 - done) * self.hparams.gamma * q_min

        q1, q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(q1, td_target) + nn.MSELoss()(q2, td_target)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.hparams.grad_clip_norm)
        self.critic_optimizer.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        actor_loss_val: Optional[float] = None

        if self.total_it % self.hparams.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.hparams.grad_clip_norm)
            self.actor_optimizer.step()
            if self.actor_scheduler is not None:
                self.actor_scheduler.step()

            actor_loss_val = float(actor_loss.item())

            with torch.no_grad():
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.data.mul_(1.0 - self.hparams.tau).add_(self.hparams.tau * p.data)
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.mul_(1.0 - self.hparams.tau).add_(self.hparams.tau * p.data)

        return actor_loss_val, float(critic_loss.item())

    def save_actor(self, path: str) -> None:
        torch.save(self.actor.state_dict(), path)

    def load_actor(self, path: str) -> None:
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
