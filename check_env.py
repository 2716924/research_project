"""
Run this once on the cluster login node before submitting SLURM jobs.
Checks MinAtar registration and CPU-only torch setup for bigbatch jobs.

    python check_env.py
"""
import os

os.environ.setdefault("RCT_DEVICE", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from rct.torch_setup import DEVICE

import gymnasium as gym
import minatar  # noqa: F401
from minatar.gym import register_envs

register_envs()

print(f"Selected device:    {DEVICE}")
print(f"RCT_DEVICE:         {os.environ.get('RCT_DEVICE')}")
print(f"CUDA visible:       {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

registered = sorted(k for k in gym.envs.registry.keys() if "MinAtar" in k)
print(f"Registered MinAtar envs ({len(registered)}): {registered}")
if not registered:
    raise SystemExit(
        "No MinAtar envs registered -- register_envs() did not work. "
        "Check `pip show minatar` (need >= 1.0.13)."
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
