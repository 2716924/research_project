"""
Control Group (Canonical Nature-DQN) Trainer
=============================================
Implements the "control group" arm of the Randomised Control Trial (RCT)
described in the research proposal (Chapter 4): a canonical Nature-DQN
[Mnih et al. 2015] trained under hyperparameter configurations sampled
from the shared population space in Table 4.1.

This script is deliberately scoped to the CONTROL GROUP only. The four
treatment arms (Double DQN, Prioritised Experience Replay, n-step
learning, Averaged DQN) are not implemented here -- see the note at the
bottom of the file for how this scaffolding is meant to support them.

Design decisions worth flagging to a reader/reviewer:
  * Individuals (hyperparameter configs) are sampled ONCE up front from a
    dedicated RNG, independent of the RNG used during training. This
    keeps the sampled population reproducible regardless of how many
    random draws training happens to make.
  * Each individual gets its own training seed (TRAINING_SEED_BASE + j),
    rather than every individual sharing one fixed seed. This preserves
    genuine across-individual variation while still being reproducible.
  * Individuals (id, seed, config) are persisted to disk per game, so a
    future treatment-arm script can load the *same* individuals and train
    under the *same* configuration + seed -- the paired design the
    proposal's statistical analysis (paired t-tests, Sec 4.4.2) depends on.
  * Evaluation uses a fixed SET of distinct deterministic seeds (not one
    seed repeated), since a fully greedy policy would otherwise produce
    identical episodes every time.
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

# Gymnasium is supposed to auto-discover this via the 'gymnasium.envs' entry
# point declared in MinAtar's setup.py, but that plugin-discovery mechanism
# can silently fail to fire depending on the Gymnasium version/environment.
# Calling it explicitly is safe (idempotent) and removes that dependency.
register_envs()

# Lock the randomness so the experiment is exactly repeatable
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------
# RCT-level constants (see proposal Sec 4.1, 4.3)
# ---------------------------------------------------------
GAMES = ["MinAtar/Breakout-v1", "MinAtar/Asterix-v1", "MinAtar/Freeway-v1"]

POPULATION_SPACE = {
    "lr": [0.0001, 0.0005, 0.001],
    "buffer_capacity": [10000, 50000, 100000],
    "batch_size": [32, 64],
    "target_update_freq": [500, 1000, 2000],
    "epsilon_decay_steps": [50000, 100000, 200000],
    "num_layers": [1, 2],
}

POPULATION_SEED = 2024          # Controls which configs get sampled
TRAINING_SEED_BASE = 1337       # Individual j trains with seed BASE + j
EVAL_SEEDS = [1000 + i for i in range(10)]  # Fixed set of distinct eval seeds


# ---------------------------------------------------------
# 1. THE BRAIN OF THE AGENT (NEURAL NETWORK)
# ---------------------------------------------------------
class MinAtarQNetwork(nn.Module):
    """
    This is the agent's brain. It looks at the game screen and estimates
    how many points it will get for taking each possible action.
    """
    def __init__(self, in_channels, num_actions, num_layers=2, hidden_dim=128):
        super(MinAtarQNetwork, self).__init__()
        # The agent's "eyes" - looking at the 10x10 game grid to spot patterns
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Flattening the grid into a single line of information.
        # MinAtar is 10x10, so a 3x3 convolution leaves an 8x8 grid.
        flat_features = 16 * 8 * 8

        # The deeper thinking part of the brain.
        # Mathematically, this applies the row-vector format: z = xW_1 + b_1
        layers = []
        current_dim = flat_features
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())  # A mathematical filter that only lets positive signals through
            current_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # The final output: one score for every possible joystick move
        self.output_layer = nn.Linear(current_dim, num_actions)

    def forward(self, x):
        # Rearrange the game screen data so the PyTorch brain can read it
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))

        # Flatten the data
        x = x.reshape(x.size(0), -1)

        # Think about the data and make a final prediction
        x = self.hidden_layers(x)
        return self.output_layer(x)


# ---------------------------------------------------------
# 2. THE MEMORY BANK (EXPERIENCE REPLAY)
# ---------------------------------------------------------
class ReplayBuffer:
    """
    The agent's memory. It stores past experiences so the agent can
    look back and learn from past successes and mistakes.
    """
    def __init__(self, capacity):
        # A memory box that forgets the oldest memories when it gets full
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Save what just happened into the memory box
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Pick a random handful of past memories to study
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------
# 3. POPULATION SAMPLING (Table 4.1)
# ---------------------------------------------------------
def sample_individuals(num_individuals, population_seed=POPULATION_SEED):
    """
    Draws `num_individuals` hyperparameter configurations from the shared
    population space (Table 4.1), using a dedicated RNG that is completely
    independent of whatever randomness training itself consumes. This is
    what makes the sampled population reproducible on its own terms.

    Each individual also gets a fixed training seed, so that a future
    treatment-arm script can reproduce the exact same "participant" by
    reusing (config, seed) -- the crux of the paired RCT design.
    """
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
    """
    Tests the agent without any random guessing (fully greedy, epsilon=0)
    to see how good it really is. Uses a fixed SET of distinct deterministic
    seeds -- with a greedy policy, reusing one seed for every episode would
    just replay the same trajectory num_episodes times.
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
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = q_network(state_tensor).argmax(dim=1).item()

            state, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    eval_env.close()
    return np.mean(total_rewards)


# ---------------------------------------------------------
# 5. TRAINING A SINGLE INDIVIDUAL (ONE CONTROL-GROUP RUN)
# ---------------------------------------------------------
def train_control_individual(game_id, individual, in_channels, num_actions,
                              total_frames, eval_frequency, num_eval_episodes,
                              gamma=0.99):
    """
    Trains one canonical Nature-DQN agent for a single sampled individual
    (hyperparameter configuration) and returns its evaluation trajectory.
    """
    config = individual["config"]
    seed = individual["seed"]

    # Seed everything for this individual's training run specifically.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_env = gym.make(game_id)

    # Setup the active brain and the slow, stable target brain
    q_network = MinAtarQNetwork(in_channels, num_actions, num_layers=config["num_layers"])
    target_network = MinAtarQNetwork(in_channels, num_actions, num_layers=config["num_layers"])
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.RMSprop(q_network.parameters(), lr=config["lr"])
    memory = ReplayBuffer(config["buffer_capacity"])

    state, _ = train_env.reset(seed=seed)
    eval_trajectory = []

    # Exploration settings: start by guessing, slowly rely on the brain.
    # epsilon_decay_steps is itself a sampled hyperparameter (Table 4.1).
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = (1.0 - epsilon_min) / config["epsilon_decay_steps"]

    for frame in range(total_frames):
        # Decide action: Guess randomly or use the brain
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = q_network(state_tensor).argmax(dim=1).item()
        else:
            action = random.randrange(num_actions)

        # See what happens in the game
        next_state, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated

        # Remember what happened
        memory.push(state, action, reward, next_state, done)
        state = next_state

        if done:
            state, _ = train_env.reset()

        epsilon = max(epsilon_min, epsilon - epsilon_decay)

        # Learn from past memories
        if len(memory) > config["batch_size"]:
            states, actions, rewards, next_states, dones = memory.sample(config["batch_size"])

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            # Ask the active brain for the current score guess
            current_q_values = q_network(states).gather(1, actions)

            # Ask the stable brain for the best possible future score
            # (canonical 1-step target -- this is the control group, so no
            # Double-DQN decoupling, no n-step return, no averaging.)
            with torch.no_grad():
                next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + (gamma * next_q_values * (1 - dones))

            # Calculate the mistake and update the brain
            loss = F.mse_loss(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the stable brain occasionally
        if (frame + 1) % config["target_update_freq"] == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Periodically test the agent (spec: every 5e4 frames, Sec 4.3)
        if (frame + 1) % eval_frequency == 0:
            mean_score = evaluate_agent(q_network, game_id, num_episodes=num_eval_episodes)
            eval_trajectory.append(mean_score)
            print(f"    frame {frame + 1}/{total_frames} | eval mean return: {mean_score:.3f}")

    train_env.close()
    return eval_trajectory


# ---------------------------------------------------------
# 6. THE MAIN TRIAL LOOP
# ---------------------------------------------------------
def execute_rct_trial(num_individuals=30, total_frames=5_000_000,
                       eval_frequency=50_000, num_eval_episodes=10,
                       output_dir="."):
    """
    Runs the control-group arm of the RCT across all three MinAtar games,
    for N sampled individuals, and writes results (+ the individuals
    themselves) to disk so treatment arms can later reuse them.

    NOTE: the proposal specifies N >= 30 individuals and 5,000,000 frames
    per run (Sec 4.1, 4.3). Those are the defaults here. For a fast local
    smoke test, override both to much smaller values -- see --pilot in
    the CLI below.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sample the population ONCE and reuse it across all three games, so
    # that (for a given individual_id) the exact same hyperparameter
    # configuration and seed is used everywhere -- consistent with the
    # population being defined independently of any single environment.
    individuals = sample_individuals(num_individuals)

    for game_id in GAMES:
        print(f"\n{'=' * 50}\nStarting Randomised Control Trial for {game_id}\n{'=' * 50}")

        init_env = gym.make(game_id)
        in_channels = init_env.observation_space.shape[-1]
        num_actions = init_env.action_space.n
        init_env.close()

        game_results = []

        for individual in individuals:
            j = individual["individual_id"]
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

            # AUC as per Sec 4.4.1: area under the evaluation-reward curve.
            # Evaluation points are equally spaced, so trapezoidal
            # integration is the direct implementation of "area under
            # the curve" (a plain sum is a monotonic proxy for this but
            # isn't literally an integral).
            computed_auc = float(np.trapz(eval_trajectory)) if len(eval_trajectory) > 1 else float(np.sum(eval_trajectory))

            game_results.append({
                "individual_id": j,
                "seed": individual["seed"],
                "config": individual["config"],
                "eval_trajectory": eval_trajectory,
                "auc": computed_auc,
            })
            print(f"Completed Control Group. Total Performance Score (AUC): {computed_auc:.4f}")

        # Persist per-game results + the individuals themselves so a
        # future treatment-arm script can reuse the exact same population.
        out_path = os.path.join(output_dir, f"control_results_{game_id.replace('/', '_')}.json")
        with open(out_path, "w") as f:
            json.dump({
                "game_id": game_id,
                "total_frames": total_frames,
                "eval_frequency": eval_frequency,
                "num_eval_episodes": num_eval_episodes,
                "eval_seeds": EVAL_SEEDS[:num_eval_episodes],
                "results": game_results,
            }, f, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control-group (canonical Nature-DQN) RCT trainer")
    parser.add_argument("--num-individuals", type=int, default=30,
                         help="N sampled hyperparameter configurations (proposal requires N >= 30)")
    parser.add_argument("--total-frames", type=int, default=5_000_000,
                         help="Frames per individual (proposal spec: 5,000,000)")
    parser.add_argument("--eval-frequency", type=int, default=50_000,
                         help="Evaluate every this many frames (proposal spec: 50,000)")
    parser.add_argument("--num-eval-episodes", type=int, default=10,
                         help="Greedy evaluation episodes per checkpoint (proposal spec: 10)")
    parser.add_argument("--output-dir", type=str, default="./results",
                         help="Where to write per-game JSON result files")
    parser.add_argument("--pilot", action="store_true",
                         help="Quick smoke-test scale instead of full spec-scale run")
    args = parser.parse_args()

    if args.pilot:
        # Fast local sanity check only -- NOT the values used for the
        # actual dissertation results (see proposal Sec 4.1, 4.3).
        execute_rct_trial(num_individuals=3, total_frames=15_000,
                           eval_frequency=3_000, num_eval_episodes=5,
                           output_dir=args.output_dir)
    else:
        execute_rct_trial(num_individuals=args.num_individuals,
                           total_frames=args.total_frames,
                           eval_frequency=args.eval_frequency,
                           num_eval_episodes=args.num_eval_episodes,
                           output_dir=args.output_dir)

# ---------------------------------------------------------
# Where the treatment arms plug in later
# ---------------------------------------------------------
# Each treatment arm (Double DQN, PER, n-step, Averaged DQN) should:
#   1. Call sample_individuals(num_individuals) with the SAME
#      num_individuals and POPULATION_SEED used here, so it reproduces
#      the identical set of (config, seed) individuals.
#   2. Reuse train_control_individual's structure but swap out exactly
#      one mechanism per arm (e.g. the TD target formula for DDQN /
#      Averaged DQN, the buffer's sampling distribution for PER, the
#      return horizon for n-step) -- keeping everything else identical,
#      per the proposal's Sec 2.4 definitions.
#   3. Write results to a matching JSON file (e.g.
#      ddqn_results_<game>.json) with the same individual_id/seed keys,
#      so the two files can be paired by individual_id for the paired
#      t-tests in Sec 4.4.2.