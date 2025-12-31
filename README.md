# TD3-Collision-Avoidance

Building from [DDPG Collision Avoidance](https://github.com/Pheneas123/DDPG-Collision-Avoidance).

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
    Deterministic evaluation / rollouts.
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
```

Run training with 
```
python scripts/train.py --config configs/default.yaml
```

Run evaluation with 
```
python scripts/evaluate.py --config configs/default.yaml --actor runs/<*name of run*>/td3_actor.pth --gif
```

## Plans

- Vectorise calculating reward to try speed up training
- Improve other vessels
  - Make predefined classes of vessels that are called, rather than randomly generated
  - Improve their movement pattern and the way they avoid colliding with eachother
- Curriculum learning, start with fewer vessels and increase over time
- Fix PyTorch so it selects GPU if compatible one is available
- Add some randomness to where agent spawns and where goal is
