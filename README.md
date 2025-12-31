# TD3-Collision-Avoidance

Building from DDPG Collision Avoidance.

This project implements a 2D collision-avoidance task for a fishing vessel navigating towards a goal while avoiding moving vessels. A TD3 agent is trained in a custom Gymnasium environment.

## Project structure

- `src/td3/`
  - `env.py`  
    Custom Gymnasium environment (`FishingVesselEnv`) including reward, termination logic, rendering, and GIF saving.
  - `td3.py`  
    TD3 agent implementation (actor/critic networks, target updates, noise, training step).
  - `networks.py`  
    Neural network definitions for actor and critics.
  - `replay_buffer.py`  
    Replay buffer used for off-policy learning.
- `scripts/`
  - `train.py`  
    Training entry-point, TensorBoard logging, reward curve saving, periodic GIF export.
  - `evaluate.py`  
    (If present) deterministic evaluation / rollouts.
- `runs/`  
  TensorBoard logs + saved models (gitignored).
- `gifs/`  
  Episode GIF exports (gitignored).

## Setup

Create a virtual environment and install dependencies.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -e .
