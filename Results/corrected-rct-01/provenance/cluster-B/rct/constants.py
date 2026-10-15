"""Shared experiment constants from the research proposal."""

# First three are proposal games; the last two are an explicit extension.
# This order also defines the SLURM task mapping; keep it stable within a cohort.
GAMES = [
    "MinAtar/Breakout-v1", "MinAtar/Asterix-v1", "MinAtar/Freeway-v1",
    "MinAtar/SpaceInvaders-v1", "MinAtar/Seaquest-v1",
]

# Table 4.1 in the proposal: shared hyperparameter population space.
POPULATION_SPACE = {
    "lr": [0.0001, 0.0005, 0.001],
    "buffer_capacity": [10000, 50000, 100000],
    "batch_size": [32, 64],
    "target_update_freq": [500, 1000, 2000],
    "epsilon_decay_steps": [50000, 100000, 200000],
    "num_layers": [1, 2],
    "replay_start_size": [5000],
}

POPULATION_SEED = 2024
TRAINING_SEED_BASE = 1337
EVAL_SEEDS = [1000 + i for i in range(10)]

# Explicitly preserve the existing shared training recipe. These are not a
# retuning to the MinAtar reference implementation.
TRAINING_SETTINGS = {
    "gamma": 0.99,
    "epsilon_initial": 1.0,
    "epsilon_final": 0.1,
    "loss": "mse",
    "optimizer": "RMSprop",
    "optimizer_options": {"alpha": 0.99, "eps": 1e-8, "momentum": 0.0, "centered": False},
    "evaluation_policy": "greedy_online_network",
    "evaluation_max_episode_steps": 100_000,
    "evaluation_limit_behavior": "raise_without_recording_partial_score",
    "averaged_snapshot_schedule": "target_sync_after_optimizer_step",
}

# Five algorithmic arms in the RCT (Table 4.2).
VARIANTS = ["control", "ddqn", "per", "nstep", "averaged"]

VARIANT_LABELS = {
    "control": "Baseline DQN (control group)",
    "ddqn": "Double DQN",
    "per": "Prioritised Experience Replay (PER)",
    "nstep": "n-step DQN",
    "averaged": "Averaged DQN",
}

# Treatment-specific settings (fixed; not part of the shared population).
# Values follow widely used references:
#   PER  -> Schaul et al. 2016, proportional variant (alpha=.6, beta0=.4)
#   n-step -> Hessel et al. 2018 (Rainbow n=3); Hernandez-Garcia & Sutton 2019
#   Averaged DQN -> Anschel et al. 2017 (K=10)
TREATMENT_DEFAULTS = {
    "per": {
        "alpha": 0.6,
        "beta_start": 0.4,
        "priority_eps": 1e-6,
    },
    "nstep": {
        "n_step": 3,
    },
    "averaged": {
        "ensemble_size": 10,
    },
}
