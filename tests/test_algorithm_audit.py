"""Analytic replay/target checks, independent of learning-curve rankings.

Run with: python -m unittest discover -s tests -p 'test_algorithm_audit.py'
"""
import random
import unittest

import numpy as np

from rct.replay import NStepCollector, PrioritizedReplayBuffer, UniformReplayBuffer


def observation(value):
    return np.array([value], dtype=np.float32)


class ReplayAuditTests(unittest.TestCase):
    def test_new_per_transition_receives_maximum_priority_once(self):
        replay = PrioritizedReplayBuffer(8, alpha=.6)
        replay.push(observation(0), 0, 0, observation(1), False)
        replay.update_priorities([0], [-100.])
        replay.push(observation(1), 0, 0, observation(2), False)
        expected = (100. + replay.priority_eps) ** .6
        self.assertAlmostEqual(replay._leaf_priority(0), expected, places=5)
        self.assertAlmostEqual(replay._leaf_priority(1), expected, places=5)
        self.assertAlmostEqual(replay.max_priority, 100. + replay.priority_eps)

    def test_per_probabilities_and_global_importance_weights(self):
        # Non-power-of-two capacity and partial fill exercise the custom tree.
        random.seed(932)
        replay = PrioritizedReplayBuffer(11, alpha=.6, beta_start=.4)
        errors = np.array([.01, .5, 2., 10., 100.])
        for i in range(len(errors)):
            replay.push(observation(i), i, i, observation(i+1), False)
        replay.update_priorities(range(len(errors)), errors)
        mass = (errors + replay.priority_eps) ** replay.alpha
        expected_probs = mass / mass.sum()
        counts = np.zeros(len(errors), dtype=int)
        for _ in range(2000):
            indices, states, actions, rewards, _, _, weights = replay.sample(5)
            np.add.at(counts, indices, 1)
            np.testing.assert_array_equal(states[:, 0], indices)
            np.testing.assert_array_equal(actions, indices)
            np.testing.assert_array_equal(rewards, indices)
            np.testing.assert_allclose(weights, (mass.min()/mass[indices])**.4, rtol=1e-6)
        np.testing.assert_allclose(counts/counts.sum(), expected_probs, atol=.015, rtol=0)

    def test_per_alpha_zero_weights_and_beta_schedule(self):
        replay = PrioritizedReplayBuffer(7, alpha=0)
        for i in range(14):  # Includes ring overwrite.
            replay.push(observation(i), i, 0, observation(i+1), False)
        replay.update_priorities(range(7), [0, .01, .1, 1, 10, 100, 1000])
        self.assertEqual(replay.sum_tree.total(), 7)
        for frame, expected in [(0, .4), (50, .7), (100, 1.), (200, 1.)]:
            replay.set_beta(frame, 100)
            self.assertAlmostEqual(replay.beta, expected)
            np.testing.assert_array_equal(replay.sample(7)[-1], np.ones(7))

    def test_nstep_matches_independent_episode_reference(self):
        rng = np.random.default_rng(91)
        for n in [1, 3, 5]:
            for length in [1, 2, 3, 4, 7, 20]:
                with self.subTest(n=n, length=length):
                    replay = UniformReplayBuffer(100)
                    collector = NStepCollector(replay, n, .9)
                    rewards = rng.normal(size=length)
                    for i, reward in enumerate(rewards):
                        collector.push(observation(i), i % 2, reward,
                                       observation(i+1), i == length-1)
                    collector.flush_episode()
                    self.assertEqual(len(replay), length)
                    self.assertFalse(collector.buffer)
                    for i in range(length):
                        end = min(i+n, length)
                        expected = sum(.9**k * rewards[i+k] for k in range(end-i))
                        self.assertAlmostEqual(replay._rewards[i], expected, places=5)
                        self.assertEqual(replay._next_states[i, 0], end)
                        self.assertEqual(replay._dones[i], end == length)
                        self.assertEqual(replay._states[i, 0], i)
                        self.assertEqual(replay._actions[i], i % 2)
                    collector.push(observation(100), 1, 7., observation(101), True)
                    collector.flush_episode()
                    self.assertEqual(replay._rewards[length], 7.)
                    self.assertEqual(replay._states[length, 0], 100)

    def test_nstep_nonterminal_window_waits_for_full_horizon(self):
        replay = UniformReplayBuffer(10)
        collector = NStepCollector(replay, 3, .9)
        for i in range(2):
            collector.push(observation(i), 0, i+1, observation(i+1), False)
        self.assertEqual(len(replay), 0)
        collector.push(observation(2), 0, 3., observation(3), False)
        self.assertEqual(len(replay), 1)
        self.assertAlmostEqual(replay._rewards[0], 1+.9*2+.9**2*3, places=5)
        self.assertEqual(replay._dones[0], 0)


class TargetAuditTests(unittest.TestCase):
    def test_nstep_bootstrap_terminal_mask_and_one_step_equivalence(self):
        import torch
        from rct.targets import control_target, nstep_target

        class ConstantQ(torch.nn.Module):
            def forward(self, states):
                return torch.tensor([[2., 4.]]).repeat(len(states), 1)

        states = torch.zeros(3, 1)
        rewards = torch.tensor([[1.], [2.], [3.]])
        dones = torch.tensor([[0.], [1.], [1.]])
        network = ConstantQ()
        actual = nstep_target(states, rewards, dones, network, .9, 3)
        torch.testing.assert_close(actual, torch.tensor([[1+.9**3*4], [2.], [3.]]))
        torch.testing.assert_close(nstep_target(states, rewards, dones, network, .9, 1),
                                   control_target(states, rewards, dones, network, .9))
        self.assertFalse(actual.requires_grad)

    def test_n_one_matches_control_through_actual_minatar_training_loop(self):
        import contextlib
        import io
        from unittest.mock import patch
        import torch
        from rct import trainer

        torch.set_num_threads(1)
        individual = dict(seed=1337, config=dict(lr=.0001, buffer_capacity=100,
                          batch_size=8, num_layers=1, replay_start_size=12,
                          target_update_freq=20, epsilon_decay_steps=100))
        histories = []
        for variant in ["control", "nstep", "per"]:
            snapshots = []

            def evaluate(network, *args, **kwargs):
                snapshots.append(torch.cat([p.detach().flatten().clone() for p in network.parameters()]))
                return 0.

            with patch.dict(trainer.TREATMENT_DEFAULTS["nstep"], n_step=1), \
                 patch.object(trainer, "evaluate_agent", side_effect=evaluate), \
                 patch.object(trainer, "should_stop_for_time_limit", return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                trajectory, completed = trainer.train_individual(
                    variant, "MinAtar/Breakout-v1", individual, 4, 3,
                    total_frames=120, eval_frequency=40, num_eval_episodes=1, log_every=40)
            self.assertTrue(completed)
            self.assertEqual(len(trajectory), 3)
            self.assertTrue(all(torch.isfinite(s).all() for s in snapshots))
            self.assertIn("sampled_mse:", output.getvalue())
            if variant == "per":
                self.assertIn("is_weight_mean:", output.getvalue())
            histories.append(snapshots)
        for control, n_one in zip(histories[0], histories[1]):
            torch.testing.assert_close(control, n_one, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
