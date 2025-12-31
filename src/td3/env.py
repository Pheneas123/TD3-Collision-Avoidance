from __future__ import annotations

import os
import io
from typing import Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

# Environment constants

# World / bounds
X_BOUNDS: Tuple[float, float] = (-200.0, 1200.0)
Y_BOUNDS: Tuple[float, float] = (-200.0, 1200.0)
OBS_NORM_DIV: float = 1200.0

# Episode control
MAX_EPISODE_STEPS: int = 500

# Agent / goal
DEFAULT_GOAL: np.ndarray = np.array([1000.0, 1000.0], dtype=np.float32)
DEFAULT_MAX_SPEED: float = 7.0
DEFAULT_MAX_TURN: float = np.pi / 18  # 10 degrees
DEFAULT_DSAFE: float = 40.0
DEFAULT_DGOAL: float = 40.0

# Vessel generation
VESSEL_L_RANGE: Tuple[float, float] = (5.0, 20.0)
VESSEL_B_RANGE: Tuple[float, float] = (2.0, 8.0)
VESSEL_POS_RANGE: Tuple[float, float] = (100.0, 1000.0)
VESSEL_PLACEMENT_ATTEMPTS: int = 100
PLACEMENT_EXTRA_GAP: float = 10.0

# Other vessel dynamics
VESSEL_HEADING_NOISE_RANGE: Tuple[float, float] = (
    -np.pi / 180,
    np.pi / 180,
)  # +/- 1 degree
VESSEL_SPEED_NOISE_RANGE: Tuple[float, float] = (-0.05, 0.05)
WALL_AVOID_MARGIN: float = 150.0
BOUNDARY_TURN_RATE: float = np.pi / 18  # 10 degrees
COLLISION_TURN_RATE: float = np.pi / 30  # 6 degrees

# Reward parameters
K1_PROGRESS: float = 1.65
K2_COLLISION_PENALTY: float = 500.0
K3_PROXIMITY_PENALTY: float = 75.0
K4_SMOOTHNESS_PENALTY: float = 0.011
K5_TIME_PENALTY: float = 0.1
K6_GOAL_REWARD: float = 500.0
K7_BOUNDARY_PENALTY: float = 5.0

BOUNDARY_PENALTY_ZONE: float = 50.0
PROXIMITY_EXTRA_BUFFER: float = 50.0
TIME_PENALTY_AFTER_STEPS: int = 300
TIME_PENALTY_GROWTH_SCALE: float = 100.0
TIME_PENALTY_CAP: float = 10.0

# Rendering / GIFs
RENDER_XLIM: Tuple[float, float] = (-60.0, 1200.0)
RENDER_YLIM: Tuple[float, float] = (-60.0, 1200.0)
GIF_XLIM: Tuple[float, float] = (-250.0, 1250.0)
GIF_YLIM: Tuple[float, float] = (-250.0, 1250.0)
GIF_MAX_FRAMES: int = 100
GIF_FRAME_DURATION_MS: int = 100


def compute_domain_radius(
    L: float = 10.0, B: float = 3.0, safety_margin: float = 5.0
) -> float:
    """
    Based on the length L and width B of the ship, as well as the safety boundary
    safety_margin, calculate the radius of the collision circle used for
    simplified collision detection.
    """
    return float(np.sqrt((L / 2) ** 2 + (B / 2) ** 2) + safety_margin)


class FishingVesselEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n_other_vessels: int, *, max_episode_steps: int = MAX_EPISODE_STEPS):
        super().__init__()

        self.max_episode_steps = int(max_episode_steps)

        self.max_speed: float = DEFAULT_MAX_SPEED
        self.max_turn: float = DEFAULT_MAX_TURN
        self.n_other_vessels: int = n_other_vessels
        self.goal: np.ndarray = DEFAULT_GOAL.copy()
        self.dsafe: float = DEFAULT_DSAFE
        self.dgoal: float = DEFAULT_DGOAL

        self.agent_radius: float = compute_domain_radius(
            L=10.0, B=3.0, safety_margin=5.0
        )

        state_dim = 10 + 5 * self.n_other_vessels
        low_state = np.full(state_dim, -np.inf, dtype=np.float32)
        high_state = np.full(state_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_state, high=high_state, dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-self.max_turn, -1.0], dtype=np.float32),
            high=np.array([self.max_turn, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.fig = None
        self.ax = None
        self.agent_line = None
        self.agent_buffer_patch = None
        self.vessel_lines: List = []
        self.vessel_buffer_patches: List = []

        self.reset()

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.position = np.array([0.0, 0.0])
        self.velocity = 1.0
        self.heading = 0.0

        self.other_vessels = self._init_other_vessels()
        self.prev_distance = np.linalg.norm(self.goal - self.position)

        self.trajectory = [self.position.copy()]
        self.other_trajectories = [[] for _ in range(self.n_other_vessels)]

        self.prev_heading = self.heading
        self.prev_velocity = self.velocity
        self.current_step = 0

        obs = self._get_state()
        info = {}
        return obs, info

    def _init_other_vessels(self):
        vessels = []
        self.vessel_radii = []
        self.vessel_sizes = []

        for _ in range(self.n_other_vessels):
            L = self.np_random.uniform(*VESSEL_L_RANGE)
            B = self.np_random.uniform(*VESSEL_B_RANGE)
            radius = compute_domain_radius(L, B, safety_margin=5.0)

            for _attempt in range(VESSEL_PLACEMENT_ATTEMPTS):
                pos = self.np_random.uniform(*VESSEL_POS_RANGE, size=2)

                safe_from_agent = np.linalg.norm(pos - np.array([0.0, 0.0])) > (
                    radius + self.agent_radius + PLACEMENT_EXTRA_GAP
                )

                safe_from_others = True
                for j, v in enumerate(vessels):
                    dist_others = np.linalg.norm(pos - np.array(v[0]))
                    if dist_others <= (
                        radius + self.vessel_radii[j] + PLACEMENT_EXTRA_GAP
                    ):
                        safe_from_others = False
                        break

                if safe_from_agent and safe_from_others:
                    break
            else:
                raise RuntimeError(
                    "Could not place vessel — reduce count or expand area."
                )

            speed = self.np_random.uniform(0.0, self.max_speed)
            heading = self.np_random.uniform(-np.pi, np.pi)

            vessels.append([pos, speed, heading])
            self.vessel_radii.append(radius)
            self.vessel_sizes.append((L, B))

        return vessels

    def step(self, action: np.ndarray):
        delta_heading, delta_speed = action
        self.heading += float(delta_heading)
        self.velocity = np.clip(self.velocity + float(delta_speed), 0.0, self.max_speed)

        dx = self.velocity * np.cos(self.heading)
        dy = self.velocity * np.sin(self.heading)
        self.position += np.array([dx, dy])

        self.position[0] = np.clip(self.position[0], *X_BOUNDS)
        self.position[1] = np.clip(self.position[1], *Y_BOUNDS)

        for i in range(self.n_other_vessels):
            pos, speed, heading = self.other_vessels[i]

            delta_heading_ = self.np_random.uniform(*VESSEL_HEADING_NOISE_RANGE)
            delta_speed_ = self.np_random.uniform(*VESSEL_SPEED_NOISE_RANGE)
            speed = np.clip(speed + delta_speed_, 0.0, self.max_speed)

            turn_command = 0.0

            new_pos = pos + np.array([speed * np.cos(heading), speed * np.sin(heading)])
            new_pos[0] = np.clip(new_pos[0], *X_BOUNDS)
            new_pos[1] = np.clip(new_pos[1], *Y_BOUNDS)

            # Wall avoidance
            if new_pos[0] - X_BOUNDS[0] < WALL_AVOID_MARGIN:
                turn_command += BOUNDARY_TURN_RATE
            elif X_BOUNDS[1] - new_pos[0] < WALL_AVOID_MARGIN:
                turn_command -= BOUNDARY_TURN_RATE
            if new_pos[1] - Y_BOUNDS[0] < WALL_AVOID_MARGIN:
                turn_command += BOUNDARY_TURN_RATE
            elif Y_BOUNDS[1] - new_pos[1] < WALL_AVOID_MARGIN:
                turn_command -= BOUNDARY_TURN_RATE

            # Vessel collision avoidance
            for j, (other_pos, _, _) in enumerate(self.other_vessels):
                if i != j:
                    dist = np.linalg.norm(new_pos - other_pos)
                    min_dist = self.vessel_radii[i] + self.vessel_radii[j] + self.dsafe
                    if dist < min_dist:
                        closeness = 1.0 - (dist / min_dist)
                        turn_command -= closeness * COLLISION_TURN_RATE
                        break

            heading = heading + delta_heading_ + turn_command
            self.other_trajectories[i].append(new_pos.copy())
            self.other_vessels[i] = [new_pos, speed, heading]

        reward, collision, goal_reached = self._calculate_reward()
        terminated = collision or goal_reached

        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps


        done_reason = "running"
        if collision:
            done_reason = "collision"
        elif goal_reached:
            done_reason = "goal"
        elif truncated:
            done_reason = "max steps"


        self.trajectory.append(self.position.copy())

        obs = self._get_state()
        info = {"done_reason": done_reason}
        return obs, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        xt, yt = self.position
        vt = self.velocity
        thetat = self.heading
        deltax, deltay = self.goal - self.position

        state = [
            xt / OBS_NORM_DIV,
            yt / OBS_NORM_DIV,
            vt / self.max_speed,
            thetat / np.pi,
            deltax / OBS_NORM_DIV,
            deltay / OBS_NORM_DIV,
            (xt - X_BOUNDS[0]) / OBS_NORM_DIV,
            (X_BOUNDS[1] - xt) / OBS_NORM_DIV,
            (yt - Y_BOUNDS[0]) / OBS_NORM_DIV,
            (Y_BOUNDS[1] - yt) / OBS_NORM_DIV,
        ]

        vessels_with_distance = [
            (i, np.linalg.norm(v[0] - self.position))
            for i, v in enumerate(self.other_vessels)
        ]
        sorted_indices = [
            idx for idx, _ in sorted(vessels_with_distance, key=lambda x: x[1])
        ]

        for i in range(self.n_other_vessels):
            if i < len(sorted_indices):
                idx = sorted_indices[i]
                pos, v_, theta_ = self.other_vessels[idx]
                deltaxi, deltayi = pos - self.position
                di = np.linalg.norm([deltaxi, deltayi])
                state += [
                    deltaxi / OBS_NORM_DIV,
                    deltayi / OBS_NORM_DIV,
                    v_ / self.max_speed,
                    theta_ / np.pi,
                    di / OBS_NORM_DIV,
                ]
            else:
                state += [0.0] * 5

        return np.array(state, dtype=np.float32)

    def _calculate_reward(self):
        dcurrent = np.linalg.norm(self.goal - self.position)

        reward = 0.0
        collision = False
        goal_reached = False

        # 1) Distance progress
        reward += K1_PROGRESS * (self.prev_distance - dcurrent)

        # 2) Collision + proximity
        for i, (pos, _, _) in enumerate(self.other_vessels):
            d = np.linalg.norm(pos - self.position)
            collision_dist = self.agent_radius + self.vessel_radii[i]
            if d < collision_dist:
                reward -= K2_COLLISION_PENALTY
                collision = True
                break
            else:
                buffer_zone = collision_dist + self.dsafe + PROXIMITY_EXTRA_BUFFER
                if d < buffer_zone:
                    norm_dist = (buffer_zone - d) / buffer_zone
                    reward -= K3_PROXIMITY_PENALTY * (norm_dist**2)

        # 3) Boundary penalty
        for bound, pos in [
            (X_BOUNDS[0], self.position[0]),
            (X_BOUNDS[1], self.position[0]),
            (Y_BOUNDS[0], self.position[1]),
            (Y_BOUNDS[1], self.position[1]),
        ]:
            dist_to_bound = abs(pos - bound)
            if dist_to_bound < BOUNDARY_PENALTY_ZONE:
                reward -= (
                    K7_BOUNDARY_PENALTY
                    * (BOUNDARY_PENALTY_ZONE - dist_to_bound)
                    / BOUNDARY_PENALTY_ZONE
                )

        # 4) Smoothness penalty
        reward -= K4_SMOOTHNESS_PENALTY * (
            abs(self.heading - self.prev_heading)
            + abs(self.velocity - self.prev_velocity)
        )

        # 5) Time penalty
        if self.current_step > TIME_PENALTY_AFTER_STEPS:
            growth = np.exp(
                (self.current_step - TIME_PENALTY_AFTER_STEPS)
                / TIME_PENALTY_GROWTH_SCALE
            )
            time_penalty = min(K5_TIME_PENALTY * growth, TIME_PENALTY_CAP)
        else:
            time_penalty = K5_TIME_PENALTY
        reward -= time_penalty

        # 6) Goal bonus
        if dcurrent < self.dgoal:
            reward += K6_GOAL_REWARD
            goal_reached = True

        self.prev_distance = dcurrent
        self.prev_heading = self.heading
        self.prev_velocity = self.velocity

        return reward, collision, goal_reached

    def render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.ax.set_xlim(*RENDER_XLIM)
            self.ax.set_ylim(*RENDER_YLIM)
            self.ax.set_aspect("equal")
            self.ax.grid(True)

            (self.agent_line,) = self.ax.plot([], [], "b-", linewidth=2)
            self.agent_buffer_patch = Circle(
                (0, 0),
                self.agent_radius + self.dsafe,
                fill=False,
                linestyle="--",
                alpha=0.4,
            )
            self.ax.add_patch(self.agent_buffer_patch)

            self.ax.plot(self.goal[0], self.goal[1], "g*", markersize=15)
            plt.show(block=False)

        xs = [p[0] for p in self.trajectory]
        ys = [p[1] for p in self.trajectory]
        self.agent_line.set_data(xs, ys)
        self.agent_buffer_patch.center = tuple(self.position)

        plt.draw()
        plt.pause(0.01)

    def save_gif(self, filename: str, gif_dir: str = "gifs") -> None:
        """
        Saves a visualisation of the episode as an animated GIF.
        """
        if len(self.trajectory) < 2:
            return

        import matplotlib
        matplotlib.use("Agg", force=True)

        os.makedirs(gif_dir, exist_ok=True)
        full_path = os.path.join(gif_dir, filename)

        frames: List[Image.Image] = []

        stride = max(1, len(self.trajectory) // GIF_MAX_FRAMES)
        frame_indices = range(0, len(self.trajectory), stride)

        for t in frame_indices:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_xlim(*GIF_XLIM)
            ax.set_ylim(*GIF_YLIM)
            ax.set_aspect("equal")
            ax.axis("off")

            # Agent trajectory + current position (blue)
            traj = np.array(self.trajectory[: t + 1], dtype=np.float32)
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=2)
            ax.plot(traj[-1, 0], traj[-1, 1], "bo", markersize=4)

            # Agent safety buffer (blue)
            ax.add_patch(
                Circle(
                    (traj[-1, 0], traj[-1, 1]),
                    self.agent_radius + self.dsafe,
                    edgecolor="blue",
                    fill=False,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.5,
                )
            )

            # Goal marker + goal zone (green)
            ax.plot(self.goal[0], self.goal[1], "g*", markersize=10)
            ax.add_patch(
                Circle(
                    (self.goal[0], self.goal[1]),
                    self.dgoal,
                    edgecolor="green",
                    fill=False,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.6,
                )
            )

            # Other vessels: trajectories + current position + buffer (red)
            for i, vessel_traj in enumerate(self.other_trajectories):
                if len(vessel_traj) > t:
                    vtraj = np.array(vessel_traj[: t + 1], dtype=np.float32)
                    if len(vtraj) > 1:
                        ax.plot(vtraj[:, 0], vtraj[:, 1], "r--", linewidth=1)
                    ax.plot(vtraj[-1, 0], vtraj[-1, 1], "rx", markersize=4)

                    buffer_radius = self.vessel_radii[i] + self.dsafe
                    ax.add_patch(
                        Circle(
                            (vtraj[-1, 0], vtraj[-1, 1]),
                            buffer_radius,
                            edgecolor="red",
                            fill=False,
                            linestyle=":",
                            linewidth=1.2,
                            alpha=0.3,
                        )
                    )

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            buf.seek(0)

            with Image.open(buf) as im:
                frames.append(im.convert("RGB").copy())
            buf.close()

        if len(frames) < 2:
            return

        frames[0].save(
            full_path,
            save_all=True,
            append_images=frames[1:],
            duration=GIF_FRAME_DURATION_MS,
            loop=0,
            optimise=False,
        )
