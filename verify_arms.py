#!/usr/bin/env python3
"""
Verify RCT algorithmic arms match the proposal design (Table 4.2).

Run: python verify_arms.py
"""

import inspect
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from rct.constants import GAMES, TREATMENT_DEFAULTS, VARIANTS
from rct.replay import NStepCollector, PrioritizedReplayBuffer, UniformReplayBuffer
from rct.sampling import sample_individuals
from rct.targets import (
    AveragedTargetEnsemble,
    control_target,
    double_dqn_target,
    nstep_target,
)
from rct.trainer import train_individual


class DummyNet(torch.nn.Module):
    """Fixed-Q network for target math checks."""

    def __init__(self, q_values):
        super().__init__()
        self.q_values = torch.tensor(q_values, dtype=torch.float32)

    def forward(self, x):
        batch = x.size(0)
        return self.q_values.unsqueeze(0).expand(batch, -1)


def assert_close(name, actual, expected, tol=1e-5):
    diff = abs(float(actual) - float(expected))
    if diff > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def test_control_target():
    target_net = DummyNet([0.0, 5.0, 2.0])
    rewards = torch.tensor([[1.0]])
    dones = torch.tensor([[0.0]])
    next_states = torch.zeros(1, 10, 10, 1)
    gamma = 0.99

    out = control_target(next_states, rewards, dones, target_net, gamma)
    assert_close("control bootstrap", out.item(), 1.0 + 0.99 * 5.0)


def test_double_dqn_target():
    online = DummyNet([10.0, 1.0, 3.0])   # argmax -> action 0
    target = DummyNet([2.0, 8.0, 4.0])     # Q(s', a_online=0) = 2.0
    rewards = torch.tensor([[0.5]])
    dones = torch.tensor([[0.0]])
    next_states = torch.zeros(1, 10, 10, 1)
    gamma = 0.99

    out = double_dqn_target(next_states, rewards, dones, online, target, gamma)
    assert_close("ddqn bootstrap", out.item(), 0.5 + 0.99 * 2.0)


def test_nstep_target():
    target_net = DummyNet([0.0, 4.0, 1.0])
    n_step_return = torch.tensor([[1.0 + 0.99 * 2.0 + (0.99 ** 2) * 3.0]])
    dones = torch.tensor([[0.0]])
    next_states = torch.zeros(1, 10, 10, 1)
    gamma = 0.99
    n_step = 3

    out = nstep_target(next_states, n_step_return, dones, target_net, gamma, n_step)
    expected = n_step_return.item() + (gamma ** n_step) * 4.0
    assert_close("nstep bootstrap uses gamma^n", out.item(), expected)

    terminal = nstep_target(
        next_states, n_step_return, torch.tensor([[1.0]]), target_net, gamma, n_step
    )
    assert_close("nstep terminal has no bootstrap", terminal.item(), n_step_return.item())


def test_nstep_collector_returns():
    buffer = UniformReplayBuffer(100)
    collector = NStepCollector(buffer, n_step=3, gamma=0.9)

    states = [np.zeros((10, 10, 1)) for _ in range(4)]
    for i in range(3):
        collector.push(states[i], 0, float(i + 1), states[i + 1], False)

    assert len(buffer) == 1
    reward, done = buffer._rewards[0], buffer._dones[0]
    expected = 1.0 + 0.9 * 2.0 + (0.9 ** 2) * 3.0
    assert_close("n-step stored return", reward, expected)
    assert done == 0


def test_per_large_capacity_small_fill():
    """Reproduces the cluster PER crash: capacity >> stored transitions."""
    random.seed(0)
    buffer = PrioritizedReplayBuffer(capacity=100_000)
    state = np.zeros((10, 10), dtype=np.float32)
    for _ in range(5_000):
        buffer.push(state, 0, 1.0, state, False)

    indices, states, actions, rewards, next_states, dones, weights = buffer.sample(64)
    assert len(indices) == 64
    assert all(idx < 5_000 for idx in indices)
    assert all(w > 0 for w in weights)


def test_per_buffer_priorities():
    buf = PrioritizedReplayBuffer(capacity=32, alpha=0.6, beta_start=0.4)
    state = np.zeros((10, 10, 1))
    for _ in range(16):
        buf.push(state, 0, 1.0, state, False)

    indices, _, _, _, _, _, weights = buf.sample(8)
    assert len(indices) == 8
    assert len(weights) == 8
    assert all(w > 0 for w in weights)

    buf.update_priorities(indices[:4], np.array([0.1, 0.5, 1.0, 2.0]))
    assert buf.max_priority > 1.0


def test_averaged_ensemble():
    from rct.network import MinAtarQNetwork

    device = torch.device("cpu")
    ensemble = AveragedTargetEnsemble(
        ensemble_size=3, in_channels=1, num_actions=2, num_layers=1, device=device
    )
    nets = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        net = MinAtarQNetwork(1, 2, num_layers=1).to(device)
        nets.append(net)
        ensemble.snapshot(net)

    rewards = torch.tensor([[0.0]])
    dones = torch.tensor([[0.0]])
    next_states = torch.zeros(1, 10, 10, 1)
    fallback = nets[0]
    gamma = 0.99

    out = ensemble.compute_target(next_states, rewards, dones, fallback, gamma)
    q_sum = sum(net(next_states) for net in nets)
    avg_q = q_sum / len(nets)
    expected_next_q = avg_q.max(1)[0].item()
    assert_close("averaged max-of-avg-Q", out.item(), gamma * expected_next_q)


def test_trainer_variant_isolation():
    source = inspect.getsource(train_individual)

    checks = {
        "control constructs uniform buffer": 'UniformReplayBuffer(config["buffer_capacity"])' in source,
        "per constructs prioritised buffer": 'PrioritizedReplayBuffer(' in source,
        "nstep uses n-step collector": 'n_step_collector.push' in source,
        "ddqn uses double target": "double_dqn_target" in source,
        "nstep uses nstep target": "nstep_target" in source,
        "per uses control 1-step target": "control_target" in source,
        "averaged uses ensemble": "target_ensemble.compute_target" in source,
        "averaged snapshots on sync": "target_ensemble.snapshot" in source,
    }
    for name, ok in checks.items():
        if not ok:
            raise AssertionError(f"Trainer isolation check failed: {name}")


def test_paired_population():
    cohort_a = sample_individuals(30)
    cohort_b = sample_individuals(30)
    if cohort_a != cohort_b:
        raise AssertionError("Population cohort is not reproducible across arms")


def test_slurm_array_mapping():
    variants = ["control", "ddqn", "per", "nstep", "averaged"]
    games = GAMES
    num_games = len(games)
    num_tasks = len(variants) * num_games
    mapped = []
    for task_id in range(num_tasks):
        variant = variants[task_id // num_games]
        game = games[task_id % num_games]
        mapped.append((variant, game))

    if len(mapped) != num_tasks:
        raise AssertionError(f"Expected {num_tasks} array tasks")
    if len(set(mapped)) != num_tasks:
        raise AssertionError("Array mapping has collisions")
    if mapped[0] != ("control", games[0]):
        raise AssertionError("Task 0 should be control × first game")
    if mapped[-1] != ("averaged", games[-1]):
        raise AssertionError("Last task should be averaged × last game")


def test_constants_coverage():
    if set(VARIANTS) != {"control", "ddqn", "per", "nstep", "averaged"}:
        raise AssertionError("VARIANTS list is incomplete")
    for key in ("per", "nstep", "averaged"):
        if key not in TREATMENT_DEFAULTS:
            raise AssertionError(f"Missing treatment defaults for {key}")


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    tests = [
        test_constants_coverage,
        test_control_target,
        test_double_dqn_target,
        test_nstep_target,
        test_nstep_collector_returns,
        test_per_buffer_priorities,
        test_per_large_capacity_small_fill,
        test_averaged_ensemble,
        test_trainer_variant_isolation,
        test_paired_population,
        test_slurm_array_mapping,
    ]

    for test in tests:
        test()
        print(f"  OK  {test.__name__}")

    print(f"\nAll {len(tests)} arm verification checks passed.")
    # Include the regression that exposed PER's double exponent, paper-derived
    # target oracles, real MinAtar dispatch checks and strict resume protection.
    suite = unittest.TestSuite()
    for pattern in ("test_algorithm_audit.py", "test_remaining_arms.py", "test_run_integrity.py", "test_cluster_tracking.py",
                    "test_followup.py"):
        suite.addTests(unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
