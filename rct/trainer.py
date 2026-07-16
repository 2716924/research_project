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

from rct.constants import TREATMENT_DEFAULTS
from rct.evaluation import evaluate_agent
from rct.network import MinAtarQNetwork
from rct.replay import NStepCollector, PrioritizedReplayBuffer, UniformReplayBuffer
from rct.targets import AveragedTargetEnsemble, control_target, double_dqn_target
from rct.torch_setup import DEVICE


def train_individual(
    variant,
    game_id,
    individual,
    in_channels,
    num_actions,
    total_frames,
    eval_frequency,
    num_eval_episodes,
    gamma=0.99,
):
    """Train a single hyperparameter individual and return its eval trajectory."""
    config = individual["config"]
    seed = individual["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_env = gym.make(game_id)

    q_network = MinAtarQNetwork(
        in_channels, num_actions, num_layers=config["num_layers"]
    ).to(DEVICE)
    target_network = MinAtarQNetwork(
        in_channels, num_actions, num_layers=config["num_layers"]
    ).to(DEVICE)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.RMSprop(q_network.parameters(), lr=config["lr"])

    uniform_buffer = UniformReplayBuffer(config["buffer_capacity"])
    per_buffer = None
    n_step_collector = None
    target_ensemble = None

    if variant == "per":
        per_defaults = TREATMENT_DEFAULTS["per"]
        per_buffer = PrioritizedReplayBuffer(
            capacity=config["buffer_capacity"],
            alpha=per_defaults["alpha"],
            beta_start=per_defaults["beta_start"],
            priority_eps=per_defaults["priority_eps"],
        )
    elif variant == "nstep":
        n_step = TREATMENT_DEFAULTS["nstep"]["n_step"]
        n_step_collector = NStepCollector(uniform_buffer, n_step=n_step, gamma=gamma)
    elif variant == "averaged":
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

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = (1.0 - epsilon_min) / config["epsilon_decay_steps"]
    replay_start_size = config["replay_start_size"]

    for frame in range(total_frames):
        memory_len = len(per_buffer) if variant == "per" else len(uniform_buffer)

        if random.random() < epsilon or memory_len < replay_start_size:
            action = random.randrange(num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                action = q_network(state_tensor).argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated

        if variant == "nstep":
            n_step_collector.push(state, action, reward, next_state, done)
        elif variant == "per":
            per_buffer.push(state, action, reward, next_state, done)
        else:
            uniform_buffer.push(state, action, reward, next_state, done)

        state = next_state
        if done:
            if variant == "nstep":
                n_step_collector.flush_episode()
            state, _ = train_env.reset()

        if memory_len >= replay_start_size:
            epsilon = max(epsilon_min, epsilon - epsilon_decay)

            batch_size = config["batch_size"]
            can_train = memory_len > batch_size

            if can_train:
                is_weights = None
                per_indices = None

                if variant == "per":
                    per_buffer.set_beta(frame, total_frames)
                    per_indices, states, actions, rewards, next_states, dones, is_weights = (
                        per_buffer.sample(batch_size)
                    )
                else:
                    states, actions, rewards, next_states, dones = uniform_buffer.sample(batch_size)

                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

                current_q_values = q_network(states).gather(1, actions)

                if variant == "ddqn":
                    target_q_values = double_dqn_target(
                        next_states, rewards, dones, q_network, target_network, gamma
                    )
                elif variant == "averaged":
                    target_q_values = target_ensemble.compute_target(
                        next_states, rewards, dones, target_network, gamma
                    )
                else:
                    target_q_values = control_target(
                        next_states, rewards, dones, target_network, gamma
                    )

                td_errors = (current_q_values - target_q_values).detach()
                elementwise_loss = F.mse_loss(current_q_values, target_q_values, reduction="none")

                if variant == "per" and is_weights is not None:
                    weight_tensor = torch.FloatTensor(is_weights).unsqueeze(1).to(DEVICE)
                    loss = (elementwise_loss * weight_tensor).mean()
                    per_buffer.update_priorities(per_indices, td_errors.squeeze(1).cpu().numpy())
                else:
                    loss = elementwise_loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (frame + 1) % config["target_update_freq"] == 0:
                target_network.load_state_dict(q_network.state_dict())
                if variant == "averaged":
                    target_ensemble.snapshot(target_network)

        if (frame + 1) % eval_frequency == 0:
            mean_score = evaluate_agent(q_network, game_id, num_episodes=num_eval_episodes)
            eval_trajectory.append(mean_score)
            print(
                f"    [{variant}] frame {frame + 1}/{total_frames} | "
                f"eval mean return: {mean_score:.3f}"
            )

    train_env.close()
    return eval_trajectory
