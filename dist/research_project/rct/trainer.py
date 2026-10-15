"""
Train one RCT individual under a chosen algorithmic variant.

All five arms share the same training loop structure. Only the parts
listed in Table 4.2 of the proposal differ between variants:
  - control : uniform replay + 1-step max-Q target
  - ddqn    : uniform replay + decoupled action selection/evaluation
  - per     : prioritised replay + 1-step max-Q target
  - nstep   : uniform replay + n-step return target
  - averaged: uniform replay + averaged max-Q target across K snapshots
"""

import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym

from rct.constants import TREATMENT_DEFAULTS, TRAINING_SETTINGS, VARIANTS
from rct.evaluation import evaluate_agent
from rct.network import MinAtarQNetwork
from rct.replay import NStepCollector, PrioritizedReplayBuffer, UniformReplayBuffer
from rct.targets import (
    AveragedTargetEnsemble,
    control_target,
    double_dqn_target,
    nstep_target,
)
from rct.slurm_utils import should_stop_for_time_limit
from rct.torch_setup import DEVICE

# Check the SLURM time limit every N frames to avoid per-frame syscall overhead.
_TIME_CHECK_INTERVAL = 10_000


def _np_to_tensor(arr, dtype=torch.float32):
    """Zero-copy numpy → torch conversion when dtypes already match."""
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t if t.dtype == dtype else t.to(dtype)


def train_individual(
    variant,
    game_id,
    individual,
    in_channels,
    num_actions,
    total_frames,
    eval_frequency,
    num_eval_episodes,
    gamma=TRAINING_SETTINGS["gamma"],
    log_every=1000,
    metrics_callback=None,
):
    """Train a single hyperparameter individual and return its eval trajectory."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if total_frames < 1 or eval_frequency < 1 or log_every < 1:
        raise ValueError("Frame budget and evaluation/logging intervals must be positive")
    config = individual["config"]
    seed = individual["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_env = gym.make(game_id)

    q_network = MinAtarQNetwork(
        in_channels, num_actions, num_layers=config["num_layers"]
    ).to(DEVICE)
    target_network = MinAtarQNetwork(
        in_channels, num_actions, num_layers=config["num_layers"]
    ).to(DEVICE)
    q_network.train()
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.RMSprop(q_network.parameters(), lr=config["lr"],
                             **TRAINING_SETTINGS["optimizer_options"])

    n_step_collector = None
    target_ensemble = None

    if variant == "per":
        per_defaults = TREATMENT_DEFAULTS["per"]
        replay_buffer = PrioritizedReplayBuffer(
            capacity=config["buffer_capacity"],
            alpha=per_defaults["alpha"],
            beta_start=per_defaults["beta_start"],
            priority_eps=per_defaults["priority_eps"],
        )
    else:
        replay_buffer = UniformReplayBuffer(config["buffer_capacity"])

    if variant == "nstep":
        n_step_collector = NStepCollector(
            replay_buffer, n_step=TREATMENT_DEFAULTS["nstep"]["n_step"], gamma=gamma
        )
    if variant == "averaged":
        avg_defaults = TREATMENT_DEFAULTS["averaged"]
        target_ensemble = AveragedTargetEnsemble(
            ensemble_size=avg_defaults["ensemble_size"],
            in_channels=in_channels,
            num_actions=num_actions,
            num_layers=config["num_layers"],
            device=DEVICE,
        )

    state, _ = train_env.reset(seed=seed)
    eval_trajectory = []

    epsilon = TRAINING_SETTINGS["epsilon_initial"]
    epsilon_min = TRAINING_SETTINGS["epsilon_final"]
    epsilon_decay = (epsilon - epsilon_min) / config["epsilon_decay_steps"]
    replay_start_size = config["replay_start_size"]

    # Pre-allocate a reusable tensor for single-state action selection.
    # MinAtar obs shape is fixed for a given game, so this is safe.
    _state_buf = torch.empty(1, *state.shape, dtype=torch.float32)

    completed = False
    for frame in range(total_frames):
        # Check SLURM time limit periodically, not every frame.
        if frame % _TIME_CHECK_INTERVAL == 0 and should_stop_for_time_limit():
            print("    Approaching SLURM time limit — stopping mid-individual for resume.")
            break

        if random.random() < epsilon or len(replay_buffer) < replay_start_size:
            action = random.randrange(num_actions)
        else:
            with torch.inference_mode():
                _state_buf[0] = torch.from_numpy(np.ascontiguousarray(state))
                action = q_network(_state_buf).argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = train_env.step(action)
        if truncated and not terminated:
            train_env.close()
            raise ValueError("External episode truncation is unsupported by this MinAtar protocol; "
                             "it requires continuation bootstraps and stored n-step tail horizons")
        done = terminated or truncated

        if variant == "nstep":
            n_step_collector.push(state, action, reward, next_state, done)
        else:
            replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        if done:
            if variant == "nstep":
                n_step_collector.flush_episode()
            state, _ = train_env.reset()

        if len(replay_buffer) >= replay_start_size:
            epsilon = max(epsilon_min, epsilon - epsilon_decay)

            batch_size = config["batch_size"]
            can_train = len(replay_buffer) > batch_size

            if can_train:
                is_weights = None
                per_indices = None

                if variant == "per":
                    replay_buffer.set_beta(frame, total_frames)
                    per_indices, states, actions, rewards, next_states, dones, is_weights = (
                        replay_buffer.sample(batch_size)
                    )
                else:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Zero-copy numpy→tensor (buffer stores float32/int64 natively).
                states_t = _np_to_tensor(states)
                actions_t = _np_to_tensor(actions, dtype=torch.long).unsqueeze(1)
                rewards_t = _np_to_tensor(rewards).unsqueeze(1)
                next_states_t = _np_to_tensor(next_states)
                dones_t = _np_to_tensor(dones).unsqueeze(1)

                current_q_values = q_network(states_t).gather(1, actions_t)

                if variant == "ddqn":
                    target_q_values = double_dqn_target(
                        next_states_t, rewards_t, dones_t, q_network, target_network, gamma
                    )
                elif variant == "averaged":
                    target_q_values = target_ensemble.compute_target(
                        next_states_t, rewards_t, dones_t, target_network, gamma
                    )
                elif variant == "nstep":
                    target_q_values = nstep_target(
                        next_states_t,
                        rewards_t,
                        dones_t,
                        target_network,
                        gamma,
                        TREATMENT_DEFAULTS["nstep"]["n_step"],
                    )
                else:
                    # control and per: uniform 1-step max-Q bootstrap
                    target_q_values = control_target(
                        next_states_t, rewards_t, dones_t, target_network, gamma
                    )

                td_errors = (current_q_values - target_q_values).detach()
                elementwise_loss = F.mse_loss(current_q_values, target_q_values, reduction="none")

                if variant == "per" and is_weights is not None:
                    weight_tensor = _np_to_tensor(is_weights).unsqueeze(1)
                    loss = (elementwise_loss * weight_tensor).mean()
                else:
                    loss = elementwise_loss.mean()

                if not torch.isfinite(loss):
                    train_env.close()
                    raise FloatingPointError(f"Nonfinite loss in {variant} at frame {frame + 1}")
                if per_indices is not None:
                    replay_buffer.update_priorities(per_indices, td_errors.squeeze(1).numpy())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in q_network.parameters() if p.grad is not None):
                    train_env.close()
                    raise FloatingPointError(f"Nonfinite gradient in {variant} at frame {frame + 1}")
                optimizer.step()

                if (frame + 1) % log_every == 0:
                    diagnostics = f" | sampled_mse: {elementwise_loss.mean().item():.6g}"
                    diagnostics += f" | sampled_abs_td: {td_errors.abs().mean().item():.6g}"
                    if is_weights is not None:
                        diagnostics += (
                            f" | is_weight_mean: {is_weights.mean():.6g}"
                            f" | is_weight_min: {is_weights.min():.6g}"
                            f" | beta: {replay_buffer.beta:.6f}"
                        )
                    print(
                        f"    [{variant}] frame {frame + 1}/{total_frames} | "
                        f"loss: {loss.item():.6f} | epsilon: {epsilon:.4f}"
                        f"{diagnostics}"
                    )
                    if metrics_callback is not None:
                        metrics = {
                            "train/loss": loss.item(), "train/epsilon": epsilon,
                            "train/sampled_mse": elementwise_loss.mean().item(),
                            "train/sampled_abs_td": td_errors.abs().mean().item(),
                        }
                        if is_weights is not None:
                            metrics.update({
                                "train/is_weight_mean": float(is_weights.mean()),
                                "train/is_weight_min": float(is_weights.min()),
                                "train/beta": replay_buffer.beta,
                            })
                        metrics_callback(frame + 1, metrics)

            if (frame + 1) % config["target_update_freq"] == 0:
                target_network.load_state_dict(q_network.state_dict())
                if variant == "averaged":
                    target_ensemble.snapshot(target_network)

        if (frame + 1) % eval_frequency == 0:
            try:
                mean_score = evaluate_agent(q_network, game_id, num_episodes=num_eval_episodes)
            except BaseException:
                train_env.close()
                raise
            if not np.isfinite(mean_score):
                train_env.close()
                raise FloatingPointError(f"Nonfinite evaluation return in {variant} at frame {frame + 1}")
            eval_trajectory.append(mean_score)
            print(
                f"    [{variant}] frame {frame + 1}/{total_frames} | "
                f"eval mean return: {mean_score:.3f}"
            )
            if metrics_callback is not None:
                metrics_callback(frame + 1, {"eval/mean_return": mean_score})

    else:
        # A time-budget break before the last step is still incomplete, even
        # when its zero-based frame index is total_frames - 1.
        completed = True
    train_env.close()
    return eval_trajectory, completed
