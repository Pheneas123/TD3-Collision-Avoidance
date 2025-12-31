from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from td3.env import FishingVesselEnv
from td3.td3 import TD3Agent, TD3HyperParams


N_OTHER_VESSELS: int = 5
EPISODES: int = 10_000
MAX_STEPS: int = 500
BATCH_SIZE: int = 128

ACTOR_LR: float = 1e-4
CRITIC_LR: float = 1e-3

GAMMA: float = 0.99
TAU: float = 0.005
POLICY_NOISE: float = 0.2
NOISE_CLIP: float = 0.5
POLICY_FREQ: int = 2

SAVE_GIF_EVERY_EP: int = 100
REWARD_PLOT_FILENAME: str = "episode_rewards_td3.png"


def train_td3(
    agent: TD3Agent,
    env: FishingVesselEnv,
    *,
    episodes: int,
    max_steps: int,
    batch_size: int,
    run_dir: Path,
    gif_dir: Path,
) -> List[float]:
    """
    Train the TD3 agent and log to TensorBoard.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    returns: List[float] = []
    outcome_counts: Dict[str, int] = {"goal": 0, "collision": 0, "max steps": 0}
    moving_avg_window = 50

    global_step = 0

    for episode in range(episodes):
        state, _info = env.reset()
        total_reward = 0.0
        reason = "max steps"

        speeds: List[float] = []
        heading_changes: List[float] = []
        distances_to_goal: List[float] = []
        min_dist_to_vessel: List[float] = []
        avg_dist_to_vessels: List[float] = []

        prev_heading = float(env.heading)

        done = False
        step = 0

        while not done and step < max_steps:
            action = agent.select_action(state, noise=True, current_episode=episode)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # store transition
            agent.replay_buffer.push(state, action, float(reward), next_state, done)

            # update
            result = agent.train(batch_size=batch_size)

            if result is not None:
                actor_loss, critic_loss = result
                if actor_loss is not None:
                    writer.add_scalar("Loss/Actor", actor_loss, global_step)
                writer.add_scalar("Loss/Critic", critic_loss, global_step)

            # diagnostics
            speeds.append(float(env.velocity))

            heading_change = abs(float(env.heading) - prev_heading)
            heading_changes.append(heading_change)
            prev_heading = float(env.heading)

            dist_to_goal = float(np.linalg.norm(env.goal - env.position))
            distances_to_goal.append(dist_to_goal)

            vessel_distances = [float(np.linalg.norm(env.position - v[0])) for v in env.other_vessels]
            if vessel_distances:
                min_dist_to_vessel.append(float(np.min(vessel_distances)))
                avg_dist_to_vessels.append(float(np.mean(vessel_distances)))

            state = next_state
            total_reward += float(reward)

            if done:
                reason = info.get("done_reason", "max steps")
                break

            step += 1
            global_step += 1

        outcome_counts[reason] += 1
        returns.append(total_reward)

        # TensorBoard per-episode
        writer.add_scalar("Reward/EpisodeTotal", total_reward, episode)

        if len(returns) >= moving_avg_window:
            moving_avg = float(np.mean(returns[-moving_avg_window:]))
            writer.add_scalar("Reward/MovingAverage", moving_avg, episode)

        writer.add_scalar("Outcome/GoalReached", 1 if reason == "goal" else 0, episode)
        writer.add_scalar("Outcome/Collision", 1 if reason == "collision" else 0, episode)
        writer.add_scalar("Outcome/Timeout", 1 if reason == "max steps" else 0, episode)

        if speeds:
            writer.add_scalar("Agent/MeanSpeed", float(np.mean(speeds)), episode)
        if heading_changes:
            writer.add_scalar("Agent/MeanHeadingChange", float(np.mean(heading_changes)), episode)
        if distances_to_goal:
            writer.add_scalar("Agent/MeanDistanceToGoal", float(np.mean(distances_to_goal)), episode)
            writer.add_scalar("Agent/FinalDistanceToGoal", float(distances_to_goal[-1]), episode)
        if min_dist_to_vessel:
            writer.add_scalar("Proximity/MinDistanceToVessel", float(np.min(min_dist_to_vessel)), episode)
        if avg_dist_to_vessels:
            writer.add_scalar("Proximity/AvgDistanceToVessels", float(np.mean(avg_dist_to_vessels)), episode)

        writer.add_scalar("Exploration/ActionNoiseStd", float(agent.noise_std), episode)
        writer.add_scalar("Environment/NumberOfVessels", int(env.n_other_vessels), episode)

        print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Reason={reason}, Steps={step}")

        if (episode + 1) % SAVE_GIF_EVERY_EP == 0:
            gif_name = f"episode_{episode + 1}.gif"
            env.save_gif(gif_name, gif_dir=str(gif_dir))

    print("\n=== Training Summary ===")
    for k, v in outcome_counts.items():
        print(f"{k}: {v}")

    writer.close()

    agent.save_actor(str(run_dir / "td3_actor.pth"))

    return returns


def main() -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"fishing_td3_{timestamp}"
    gif_dir = Path("gifs") / f"fishing_td3_{timestamp}"


    env = FishingVesselEnv(n_other_vessels=N_OTHER_VESSELS)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = np.array([env.max_turn, 1.0], dtype=np.float32)  # keep consistent with your action space

    hparams = TD3HyperParams(
        gamma=GAMMA,
        tau=TAU,
        policy_noise=POLICY_NOISE,
        noise_clip=NOISE_CLIP,
        policy_freq=POLICY_FREQ,
    )

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        hparams=hparams,
        device=None,
        replay_buffer=None,
        actor_scheduler_kwargs=None,
        critic_scheduler_kwargs=None,
    )

    returns = train_td3(
        agent,
        env,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        run_dir=run_dir,
        gif_dir=gif_dir,
    )

    plt.figure(figsize=(12, 6))
    plt.plot(returns, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("TD3 Collision Avoidance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / REWARD_PLOT_FILENAME)
    print(f"Saved reward curve to {run_dir / REWARD_PLOT_FILENAME}")


if __name__ == "__main__":
    main()
