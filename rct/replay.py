"""
Experience replay buffers for the RCT arms.

Implementations follow trusted community references:
  - Uniform replay: Mnih et al. 2015 (Nature DQN)
  - Prioritised replay: Schaul et al. 2015; segment-tree pattern from
    OpenAI Baselines / CleanRL rainbow implementations
  - n-step collector: Hernandez-Garcia & Sutton 2019
"""

import collections
import random

import numpy as np

from rct.segment_tree import MinSegmentTree, SumSegmentTree


class UniformReplayBuffer:
    """Standard experience replay with uniform random sampling."""

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


class PrioritizedReplayBuffer:
    """
    Prioritised Experience Replay (PER) — Schaul et al. 2015.

    Transitions with larger TD errors are sampled more often.
    Importance-sampling weights correct for the biased sampling.
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, priority_eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.priority_eps = priority_eps
        self.pos = 0
        self.size = 0
        self.data = [None] * capacity
        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        idx = self.pos
        self.data[idx] = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.sum_tree.update(idx, priority)
        self.min_tree.update(idx, priority)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _leaf_priority(self, idx):
        return float(self.sum_tree.tree[idx + self.capacity - 1])

    def _retrieve_valid_tree(self, value, total):
        """Tree lookup with resampling guard for ring-buffer slots not yet written."""
        upper = max(total, np.nextafter(0.0, 1.0))
        sample_value = min(max(value, np.nextafter(0.0, upper)), upper)

        for _ in range(32):
            idx = self.sum_tree.retrieve(sample_value)
            if idx < self.size and self.data[idx] is not None:
                return idx, self._leaf_priority(idx)
            sample_value = random.uniform(0.0, upper)

        idx = random.randrange(self.size)
        return idx, self._leaf_priority(idx)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        sample_size = min(batch_size, self.size)
        indices = []
        priorities = []
        total = self.sum_tree.total()
        if total <= 0:
            raise ValueError("Replay buffer priorities must be positive before sampling")

        segment = total / sample_size

        for i in range(sample_size):
            low = segment * i
            high = segment * (i + 1)
            value = random.uniform(low, high)
            idx, priority = self._retrieve_valid_tree(value, total)
            indices.append(idx)
            priorities.append(priority)

        probs = np.array(priorities, dtype=np.float64) / total
        min_prob = self.min_tree.min() / total
        min_prob = max(min_prob, 1e-12)
        max_weight = (min_prob * self.size) ** (-self.beta)
        weights = (probs * self.size) ** (-self.beta)
        weights /= max_weight

        batch = [self.data[i] for i in indices]
        if any(item is None for item in batch):
            raise RuntimeError("PER sampled an uninitialized transition")

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return indices, state, action, reward, next_state, done, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self.priority_eps) ** self.alpha
            self.sum_tree.update(idx, priority)
            self.min_tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def set_beta(self, frame, total_frames, beta_end=1.0):
        """Linearly anneal beta from beta_start to 1.0 (Rainbow / CleanRL pattern)."""
        fraction = min(1.0, frame / max(total_frames, 1))
        self.beta = self.beta_start + fraction * (beta_end - self.beta_start)

    def __len__(self):
        return self.size


class NStepCollector:
    """
    Wraps a replay buffer and stores n-step returns before pushing.

    n-step target (Hernandez-Garcia & Sutton 2019):
        G = sum_{k=0}^{n-1} gamma^k * r_{t+k+1} + gamma^n * max_a Q(s_{t+n}, a; theta-)
    """

    def __init__(self, replay_buffer, n_step, gamma):
        self.replay_buffer = replay_buffer
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = collections.deque()

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) >= self.n_step:
            self._emit_transition(self.n_step)
            self.buffer.popleft()

    def flush_episode(self):
        """Flush remaining partial trajectories when an episode ends."""
        while len(self.buffer) > 0:
            self._emit_transition(len(self.buffer))
            self.buffer.popleft()

    def _emit_transition(self, horizon):
        state, action, _, _, _ = self.buffer[0]
        n_step_return = 0.0
        for i in range(horizon):
            _, _, reward, _, _ = self.buffer[i]
            n_step_return += (self.gamma ** i) * reward

        _, _, _, next_state, done = self.buffer[horizon - 1]
        self.replay_buffer.push(state, action, n_step_return, next_state, done)

    def __len__(self):
        return len(self.replay_buffer)
