"""Evaluate trained agents and compute sample-efficiency (AUC)."""

import numpy as np
import torch
import gymnasium as gym

from rct.constants import EVAL_SEEDS
from rct.torch_setup import DEVICE


def evaluate_agent(q_network, env_name, num_episodes=10, eval_seeds=EVAL_SEEDS):
    """
    Run greedy evaluation episodes (no exploration).

    Uses a fixed set of environment seeds so scores are comparable
    across individuals and algorithm variants.
    """
    if num_episodes > len(eval_seeds):
        raise ValueError(
            f"Requested {num_episodes} eval episodes but only "
            f"{len(eval_seeds)} fixed eval seeds are defined."
        )

    eval_env = gym.make(env_name)
    total_rewards = []

    for i in range(num_episodes):
        state, _ = eval_env.reset(seed=eval_seeds[i])
        episode_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                action = q_network(state_tensor).argmax(dim=1).item()

            state, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    eval_env.close()
    return np.mean(total_rewards)


def compute_auc(eval_trajectory, eval_frequency):
    """Area under the learning curve, integrated over training frames."""
    if not eval_trajectory:
        return 0.0

    x = np.arange(1, len(eval_trajectory) + 1, dtype=float) * eval_frequency
    y = np.asarray(eval_trajectory, dtype=float)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x=x))
    return float(np.trapz(y, x=x))
