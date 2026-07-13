"""
Control Group (Canonical Nature-DQN) Trainer
=============================================
Implements the "control group" arm of the Randomised Control Trial (RCT)
described in the research proposal: a canonical Nature-DQN
[Mnih et al. 2015] trained under hyperparameter configurations sampled
from the shared population space.

Design decisions:
  * Individuals (hyperparameter configs) are sampled ONCE up front from a
    dedicated RNG, independent of the RNG used during training.
  * Each individual gets its own training seed (TRAINING_SEED_BASE + j).
  * Evaluation uses a fixed SET of distinct deterministic seeds.
  * Incorporates replay_start_size to match Mnih et al. (2015) specifications.
"""

import argparse
import json
import os
import random
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import minatar  # noqa: F401
from minatar.gym import register_envs

register_envs()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
torch.set_num_threads(_num_threads)

# ---------------------------------------------------------
# RCT-level constants
# ---------------------------------------------------------
GAMES = ["MinAtar/Breakout-v1", "MinAtar/Asterix-v1", "MinAtar/Freeway-v1"]

POPULATION_SPACE = {
    "lr": [0.0001, 0.0005, 0.001],
    "buffer_capacity": [10000, 50000, 100000],
    "batch_size": [32, 64],
    "target_update_freq": [500, 1000, 2000],
    "epsilon_decay_steps": [50000, 100000, 200000],
    "num_layers": [1, 2],
    "replay_start_size": [5000] # Pure exploration phase before learning begins
}

POPULATION_SEED = 2024          
TRAINING_SEED_BASE = 1337       
EVAL_SEEDS = [1000 + i for i in range(10)]  


# ---------------------------------------------------------
# 1. THE BRAIN OF THE AGENT (NEURAL NETWORK)
# ---------------------------------------------------------
class MinAtarQNetwork(nn.Module):
    def __init__(self, in_channels, num_actions, num_layers=2, hidden_dim=128):
        super(MinAtarQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        flat_features = 16 * 8 * 8

        layers = []
        current_dim = flat_features
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())  
            current_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, num_actions)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = self.hidden_layers(x)
        return self.output_layer(x)


# ---------------------------------------------------------
# 2. THE MEMORY BANK (EXPERIENCE REPLAY)
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------
# 3. POPULATION SAMPLING
# ---------------------------------------------------------
def sample_individuals(num_individuals, population_seed=POPULATION_SEED):
    rng = random.Random(population_seed)
    individuals = []
    for j in range(num_individuals):
        config = {param: rng.choice(values) for param, values in POPULATION_SPACE.items()}
        individuals.append({
            "individual_id": j,
            "seed": TRAINING_SEED_BASE + j,
            "config": config,
        })
    return individuals


# ---------------------------------------------------------
# 4. EVALUATION
# ---------------------------------------------------------
def evaluate_agent(q_network, env_name, num_episodes=10, eval_seeds=EVAL_SEEDS):
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


# ---------------------------------------------------------
# 5. TRAINING A SINGLE INDIVIDUAL 
# ---------------------------------------------------------
def train_control_individual(game_id, individual, in_channels, num_actions,
                              total_frames, eval_frequency, num_eval_episodes,
                              gamma=0.99):
    config = individual["config"]
    seed = individual["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_env = gym.make(game_id)

    q_network = MinAtarQNetwork(in_channels, num_actions, num_layers=config["num_layers"]).to(DEVICE)
    target_network = MinAtarQNetwork(in_channels, num_actions, num_layers=config["num_layers"]).to(DEVICE)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.RMSprop(q_network.parameters(), lr=config["lr"])
    memory = ReplayBuffer(config["buffer_capacity"])

    state, _ = train_env.reset(seed=seed)
    eval_trajectory = []

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = (1.0 - epsilon_min) / config["epsilon_decay_steps"]
    
    replay_start_size = config["replay_start_size"]

    for frame in range(total_frames):
        # Exploration logic
        if random.random() > epsilon or len(memory) < replay_start_size:
            action = random.randrange(num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                action = q_network(state_tensor).argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated

        memory.push(state, action, reward, next_state, done)
        state = next_state

        if done:
            state, _ = train_env.reset()

        # Only decay epsilon after the initial buffer generation period is complete
        if len(memory) >= replay_start_size:
            epsilon = max(epsilon_min, epsilon - epsilon_decay)

            if len(memory) > config["batch_size"]:
                states, actions, rewards, next_states, dones = memory.sample(config["batch_size"])

                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

                current_q_values = q_network(states).gather(1, actions)

                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
                    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

                loss = F.mse_loss(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (frame + 1) % config["target_update_freq"] == 0:
                target_network.load_state_dict(q_network.state_dict())

        # Testing the agent 
        if (frame + 1) % eval_frequency == 0:
            mean_score = evaluate_agent(q_network, game_id, num_episodes=num_eval_episodes)
            eval_trajectory.append(mean_score)
            print(f"    frame {frame + 1}/{total_frames} | eval mean return: {mean_score:.3f}")

    train_env.close()
    return eval_trajectory


def _result_path(output_dir, game_id):
    return os.path.join(output_dir, f"control_results_{game_id.replace('/', '_')}.json")


def _load_existing_results(out_path):
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
        results = data.get("results", [])
        completed_ids = {r["individual_id"] for r in results}
        return results, completed_ids
    return [], set()


def _save_results(out_path, game_id, total_frames, eval_frequency, num_eval_episodes, game_results):
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({
            "game_id": game_id,
            "total_frames": total_frames,
            "eval_frequency": eval_frequency,
            "num_eval_episodes": num_eval_episodes,
            "eval_seeds": EVAL_SEEDS[:num_eval_episodes],
            "results": game_results,
        }, f, indent=2)
    os.replace(tmp_path, out_path)


# ---------------------------------------------------------
# 6. THE MAIN TRIAL LOOP
# ---------------------------------------------------------
def execute_rct_trial(num_individuals=30, total_frames=5_000_000,
                       eval_frequency=50_000, num_eval_episodes=10,
                       output_dir=".", games=None):

    os.makedirs(output_dir, exist_ok=True)
    games = games if games is not None else GAMES

    individuals = sample_individuals(num_individuals)

    print(f"Using device: {DEVICE}")

    for game_id in games:
        print(f"\n{'=' * 50}\nStarting Randomised Control Trial for {game_id}\n{'=' * 50}")

        init_env = gym.make(game_id)
        in_channels = init_env.observation_space.shape[-1]
        num_actions = init_env.action_space.n
        init_env.close()

        out_path = _result_path(output_dir, game_id)
        game_results, completed_ids = _load_existing_results(out_path)
        if completed_ids:
            print(f"Resuming: {len(completed_ids)}/{num_individuals} individuals already completed for {game_id}")

        for individual in individuals:
            j = individual["individual_id"]

            if j in completed_ids:
                print(f"--- Skipping Individual {j + 1}/{num_individuals} (already completed) ---")
                continue

            print(f"\n--- Testing Individual {j + 1}/{num_individuals} ---")
            print(f"Seed: {individual['seed']} | Settings: {individual['config']}")

            eval_trajectory = train_control_individual(
                game_id=game_id,
                individual=individual,
                in_channels=in_channels,
                num_actions=num_actions,
                total_frames=total_frames,
                eval_frequency=eval_frequency,
                num_eval_episodes=num_eval_episodes,
            )

            if hasattr(np, 'trapezoid'):
                integrate_func = np.trapezoid
            else:
                integrate_func = np.trapz

            computed_auc = float(integrate_func(eval_trajectory)) if len(eval_trajectory) > 1 else float(np.sum(eval_trajectory))

            game_results.append({
                "individual_id": j,
                "seed": individual["seed"],
                "config": individual["config"],
                "eval_trajectory": eval_trajectory,
                "auc": computed_auc,
            })
            print(f"Completed Control Group. Total Performance Score (AUC): {computed_auc:.4f}")

            _save_results(out_path, game_id, total_frames, eval_frequency, num_eval_episodes, game_results)

        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control-group (canonical Nature-DQN) RCT trainer")
    parser.add_argument("--num-individuals", type=int, default=30,
                         help="N sampled hyperparameter configurations")
    parser.add_argument("--total-frames", type=int, default=5_000_000,
                         help="Frames per individual")
    parser.add_argument("--eval-frequency", type=int, default=50_000,
                         help="Evaluate every this many frames")
    parser.add_argument("--num-eval-episodes", type=int, default=10,
                         help="Greedy evaluation episodes per checkpoint")
    parser.add_argument("--output-dir", type=str, default="./results",
                         help="Where to write per-game JSON result files")
    parser.add_argument("--game", type=str, default=None, choices=GAMES,
                         help="Run only this game")
    parser.add_argument("--pilot", action="store_true",
                         help="Quick smoke-test scale instead of full spec-scale run")
    args = parser.parse_args()

    games = [args.game] if args.game is not None else None  

    if args.pilot:
        execute_rct_trial(num_individuals=3, total_frames=15_000,
                           eval_frequency=3_000, num_eval_episodes=5,
                           output_dir=args.output_dir, games=games)
    else:
        execute_rct_trial(num_individuals=args.num_individuals,
                           total_frames=args.total_frames,
                           eval_frequency=args.eval_frequency,
                           num_eval_episodes=args.num_eval_episodes,
                           output_dir=args.output_dir, games=games)