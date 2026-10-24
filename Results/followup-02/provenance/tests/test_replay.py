import random

import numpy as np

from rct.replay import PrioritizedReplayBuffer


def test_prioritized_replay_does_not_sample_uninitialized_entries():
    buffer = PrioritizedReplayBuffer(capacity=8)

    for i in range(3):
        buffer.push(np.array([i]), 0, 1.0, np.array([i + 1]), False)

    indices, states, actions, rewards, next_states, dones, weights = buffer.sample(2)

    assert len(indices) == 2
    assert len(states) == 2
    assert all(idx in range(3) for idx in indices)
    assert weights.shape == (2,)


def test_prioritized_replay_large_capacity_small_fill():
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


def test_prioritized_replay_handles_small_buffer_without_crashing():
    random.seed(0)
    buffer = PrioritizedReplayBuffer(capacity=8)

    for i in range(2):
        buffer.push(np.array([i]), 0, 1.0, np.array([i + 1]), False)

    indices, states, actions, rewards, next_states, dones, weights = buffer.sample(2)

    assert len(indices) == 2
    assert len(states) == 2
