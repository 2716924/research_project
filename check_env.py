"""
Run this once on the cluster (login node or a quick interactive job) after
setting up your environment, BEFORE submitting the real SLURM jobs. It
checks the three things that have actually broken so far: MinAtar's
Gymnasium registration, and whether torch can see a GPU.

    python check_env.py
"""
import torch
import gymnasium as gym
import minatar  # noqa: F401
from minatar.gym import register_envs

register_envs()

print(f"torch version:      {torch.__version__}")
print(f"CUDA available:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:                {torch.cuda.get_device_name(0)}")
else:
    print("No GPU visible -- if you expected one, check your --gres=gpu:N "
          "request and that you loaded a CUDA-matched torch build.")

registered = sorted(k for k in gym.envs.registry.keys() if "MinAtar" in k)
print(f"Registered MinAtar envs ({len(registered)}): {registered}")
if not registered:
    raise SystemExit(
        "No MinAtar envs registered -- register_envs() did not work. "
        "Check `pip show minatar` (need >= 1.0.13) and that you're not "
        "shadowing the package with a local file/folder named minatar."
    )

env = gym.make("MinAtar/Breakout-v1")
obs, _ = env.reset(seed=0)
print(f"MinAtar/Breakout-v1 observation shape: {obs.shape}")
env.close()

try:
    import wandb  # noqa: F401

    print(f"wandb installed:    yes ({wandb.__version__})")
except ImportError:
    print("wandb installed:    no  (pip install -r requirements.txt)")

print("\nEnvironment check passed.")
