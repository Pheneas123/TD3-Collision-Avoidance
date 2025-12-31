from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from td3.env import FishingVesselEnv
from td3.td3 import TD3Agent


# Default
N_OTHER_VESSELS: int = 5
N_EVAL_EPISODES: int = 100
MAX_EVAL_STEPS: int = 500
SAVE_DIR: str = "eval_outputs"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def policy_evaluation(
    agent: TD3Agent,
    env: FishingVesselEnv,
    *,
    n_eval_episodes: int,
    max_eval_steps: int,
    seed: int,
    name: str = "",
    save_dir: str | Path = SAVE_DIR,
    save_gif: bool = False,
) -> List[float]:
    success_count = 0
    collision_count = 0
    timeout_count = 0
    episode_rewards: List[float] = []

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    desc = f"Evaluating {name}".strip() if name else "Evaluating"
    for ep in tqdm(range(n_eval_episodes), desc=desc):
        state, _info = env.reset(seed=seed + ep)
        total_reward = 0.0
        reason = "max steps"

        for _step in range(max_eval_steps):
            action = agent.select_action(state, noise=False, current_episode=ep)

            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            state = next_state

            if terminated or truncated:
                reason = info.get("done_reason", "max steps")
                break

        if reason == "goal":
            success_count += 1
        elif reason == "collision":
            collision_count += 1
        else:
            timeout_count += 1

        episode_rewards.append(total_reward)

        if save_gif and ep == 0:
            prefix = name.lower().replace(" ", "_") if name else "eval"
            gif_name = f"{prefix}_ep1_{reason}.gif"
            env.save_gif(gif_name, gif_dir=str(save_dir))
            print(f"Saved eval GIF to '{save_dir / gif_name}'")

    success_rate = success_count / n_eval_episodes * 100.0
    print(f"\n[{name}] Evaluated {n_eval_episodes} episodes.")
    print(f"[{name}] Goals: {success_count}")
    print(f"[{name}] Collisions: {collision_count}")
    print(f"[{name}] Timeouts: {timeout_count}")
    print(f"[{name}] Success Rate: {success_rate:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, alpha=0.6, label="Episode Reward")
    plt.axhline(float(np.mean(episode_rewards)), linestyle="--", label="Mean Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{name} Evaluation Rewards".strip() or "Evaluation Rewards")
    plt.legend()
    plt.tight_layout()

    prefix = name.lower().replace(" ", "_") if name else "eval"
    plot_path = save_dir / f"{prefix}_rewards.png"
    plt.savefig(plot_path)
    print(f"Saved evaluation plot as '{plot_path}'")

    return episode_rewards

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=str, default=None, help="Path to td3_actor.pth (optional)")
    parser.add_argument("--episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_EVAL_STEPS)
    parser.add_argument("--vessels", type=int, default=N_OTHER_VESSELS)
    parser.add_argument("--name", type=str, default="TD3")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR)
    parser.add_argument("--gif", action="store_true", help="Save a GIF for the first eval episode")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for deterministic evaluation")
    args = parser.parse_args()

    seed_everything(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(args.save_dir) / f"{args.name.lower().replace(' ', '_')}_{timestamp}"

    env = FishingVesselEnv(n_other_vessels=args.vessels)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = np.array([env.max_turn, 1.0], dtype=np.float32)

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
    )

    if args.actor is not None:
        agent.load_actor(args.actor)
        print(f"Loaded actor from: {args.actor}")

    policy_evaluation(
        agent,
        env,
        n_eval_episodes=args.episodes,
        max_eval_steps=args.max_steps,
        seed=args.seed,
        name=args.name,
        save_dir=save_dir,
        save_gif=args.gif,
    )


if __name__ == "__main__":
    main()