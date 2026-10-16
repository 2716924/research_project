"""Evaluate trained agents and compute sample-efficiency (AUC)."""

import numpy as np
import torch
import gymnasium as gym

from rct.constants import EVAL_SEEDS, TRAINING_SETTINGS


def evaluate_agent(q_network, env_name, num_episodes=10, eval_seeds=EVAL_SEEDS,
                   max_episode_steps=TRAINING_SETTINGS["evaluation_max_episode_steps"]):
    """
    Run greedy evaluation episodes (no exploration).

    Uses a fixed set of environment seeds so scores are comparable
    across individuals and algorithm variants.
    """
    if not 1 <= num_episodes <= len(eval_seeds):
        raise ValueError(
            f"Requested {num_episodes} eval episodes but only "
            f"{len(eval_seeds)} fixed eval seeds are defined."
        )
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")

    eval_env = gym.make(env_name)
    total_rewards = []
    _state_buf = None  # Lazily initialised, reused across episodes.

    was_training = q_network.training
    q_network.eval()
    try:
        for i in range(num_episodes):
            state, _ = eval_env.reset(seed=eval_seeds[i])
            if _state_buf is None:
                _state_buf = torch.empty(1, *state.shape, dtype=torch.float32)
            episode_reward = 0.0
            done = False
            episode_steps = 0

            while not done:
                if episode_steps >= max_episode_steps:
                    raise RuntimeError(
                        f"Evaluation episode exceeded {max_episode_steps} steps in {env_name} "
                        f"(seed {eval_seeds[i]}). No partial return was recorded. "
                        "A finite-horizon evaluation would require an explicit protocol change."
                    )
                with torch.inference_mode():
                    _state_buf[0] = torch.from_numpy(np.ascontiguousarray(state))
                    q_values = q_network(_state_buf)
                    if not torch.isfinite(q_values).all():
                        raise FloatingPointError("Nonfinite Q-values during evaluation")
                    action = q_values.argmax(dim=1).item()

                state, reward, terminated, truncated, _ = eval_env.step(action)
                episode_steps += 1
                episode_reward += reward
                done = terminated or truncated

            if not np.isfinite(episode_reward):
                raise FloatingPointError("Nonfinite evaluation episode return")
            total_rewards.append(episode_reward)
    finally:
        eval_env.close()
        q_network.train(was_training)
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
