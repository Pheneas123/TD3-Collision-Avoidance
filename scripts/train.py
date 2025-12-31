from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import yaml

from td3.env import FishingVesselEnv
from td3.td3 import TD3Agent, TD3HyperParams


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")
    return cfg


def train_td3(
    agent: TD3Agent,
    env: FishingVesselEnv,
    *,
    episodes: int,
    max_steps: int,
    batch_size: int,
    run_dir: Path,
    gif_dir: Path,
    save_gif_every_ep: int,
) -> List[float]:
    """
    Train the TD3 agent and log to TensorBoard.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    returns: List[float] = []
    outcome_counts: Dict[str, int] = {
        "goal": 0,
        "collision": 0,
        "max steps": 0
    }
    moving_avg_window = 50

    global_step = 0

    DIAG_EVERY = 10
    LOG_LOSS_EVERY = 50
    UPDATE_EVERY = 2

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
            action = agent.select_action(state,
                                         noise=True,
                                         current_episode=episode)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # store transition
            agent.replay_buffer.push(state, action, float(reward), next_state,
                                     done)

            # update
            result = None
            if global_step % UPDATE_EVERY == 0:
                result = agent.train(batch_size=batch_size)
            if result is not None and (global_step % LOG_LOSS_EVERY == 0):
                actor_loss, critic_loss = result
                if actor_loss is not None:
                    writer.add_scalar("Loss/Actor", actor_loss, global_step)
                writer.add_scalar("Loss/Critic", critic_loss, global_step)

            # diagnostics
            if step % DIAG_EVERY == 0:
                speeds.append(float(env.velocity))

                heading_change = abs(float(env.heading) - prev_heading)
                heading_changes.append(heading_change)
                prev_heading = float(env.heading)

                dist_to_goal = float(np.linalg.norm(env.goal - env.position))
                distances_to_goal.append(dist_to_goal)

                vessel_distances = [
                    float(np.linalg.norm(env.position - v[0]))
                    for v in env.other_vessels
                ]
                if vessel_distances:
                    min_dist_to_vessel.append(float(np.min(vessel_distances)))
                    avg_dist_to_vessels.append(float(np.mean(vessel_distances)))

            state = next_state
            total_reward += float(reward)

            step += 1
            global_step += 1

            if done:
                reason = info.get("done_reason", "max steps")
                break

        # episode accounting
        if reason not in outcome_counts:
            outcome_counts[reason] = 0
        outcome_counts[reason] += 1
        returns.append(total_reward)

        # TensorBoard per-episode
        writer.add_scalar("Reward/EpisodeTotal", total_reward, episode)

        if len(returns) >= moving_avg_window:
            moving_avg = float(np.mean(returns[-moving_avg_window:]))
            writer.add_scalar("Reward/MovingAverage", moving_avg, episode)

        writer.add_scalar("Outcome/GoalReached", 1 if reason == "goal" else 0,
                          episode)
        writer.add_scalar("Outcome/Collision",
                          1 if reason == "collision" else 0, episode)
        writer.add_scalar("Outcome/Timeout", 1 if reason == "max steps" else 0,
                          episode)

        if speeds:
            writer.add_scalar("Agent/MeanSpeed", float(np.mean(speeds)),
                              episode)
        if heading_changes:
            writer.add_scalar("Agent/MeanHeadingChange",
                              float(np.mean(heading_changes)), episode)
        if distances_to_goal:
            writer.add_scalar("Agent/MeanDistanceToGoal",
                              float(np.mean(distances_to_goal)), episode)
            writer.add_scalar("Agent/FinalDistanceToGoal",
                              float(distances_to_goal[-1]), episode)
        if min_dist_to_vessel:
            writer.add_scalar("Proximity/MinDistanceToVessel",
                              float(np.min(min_dist_to_vessel)), episode)
        if avg_dist_to_vessels:
            writer.add_scalar("Proximity/AvgDistanceToVessels",
                              float(np.mean(avg_dist_to_vessels)), episode)

        writer.add_scalar("Exploration/ActionNoiseStd", float(agent.noise_std),
                          episode)
        writer.add_scalar("Environment/NumberOfVessels",
                          int(env.n_other_vessels), episode)

        print(
            f"Episode {episode + 1}: Reward={total_reward:.2f}, Reason={reason}, Steps={step}"
        )

        if save_gif_every_ep > 0 and (episode + 1) % save_gif_every_ep == 0:
            gif_name = f"episode_{episode + 1}.gif"
            env.save_gif(gif_name, gif_dir=str(gif_dir))

    print("\n=== Training Summary ===")
    for k, v in outcome_counts.items():
        print(f"{k}: {v}")

    writer.close()

    agent.save_actor(str(run_dir / "td3_actor.pth"))

    return returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    log_cfg = cfg.get("logging", {})

    n_other_vessels = int(env_cfg.get("n_other_vessels", 5))

    episodes = int(train_cfg.get("episodes", 10_000))
    max_steps = int(train_cfg.get("max_steps", 500))
    batch_size = int(train_cfg.get("batch_size", 128))

    actor_lr = float(train_cfg.get("actor_lr", 1e-4))
    critic_lr = float(train_cfg.get("critic_lr", 1e-3))

    hparams = TD3HyperParams(
        gamma=float(train_cfg.get("gamma", 0.99)),
        tau=float(train_cfg.get("tau", 0.005)),
        policy_noise=float(train_cfg.get("policy_noise", 0.2)),
        noise_clip=float(train_cfg.get("noise_clip", 0.5)),
        policy_freq=int(train_cfg.get("policy_freq", 2)),
    )

    save_gif_every_ep = int(log_cfg.get("save_gif_every_ep", 10))
    reward_plot_filename = str(
        log_cfg.get("reward_plot_filename", "episode_rewards_td3.png"))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"fishing_td3_{timestamp}"
    gif_dir = Path("gifs") / f"fishing_td3_{timestamp}"

    max_episode_steps = int(env_cfg.get("max_episode_steps", 500))
    env = FishingVesselEnv(n_other_vessels=n_other_vessels, max_episode_steps=max_episode_steps)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = np.array([env.max_turn, 1.0], dtype=np.float32)

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        hparams=hparams,
        device=None,  # auto
        replay_buffer=None,
        actor_scheduler_kwargs=None,
        critic_scheduler_kwargs=None,
    )

    returns = train_td3(
        agent,
        env,
        episodes=episodes,
        max_steps=max_steps,
        batch_size=batch_size,
        run_dir=run_dir,
        gif_dir=gif_dir,
        save_gif_every_ep=save_gif_every_ep,
    )

    plt.figure(figsize=(12, 6))
    plt.plot(returns, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("TD3 Collision Avoidance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / reward_plot_filename)
    print(f"Saved reward curve to {run_dir / reward_plot_filename}")


if __name__ == "__main__":
    main()
