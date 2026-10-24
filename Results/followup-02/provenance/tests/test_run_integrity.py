"""Protect result provenance, episode evaluation and interrupted-run handling."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from rct import evaluation, results


class ResultIntegrityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = str(Path(directory.name) / 'results.json')
        self.metadata = results.build_metadata('per', 'MinAtar/Breakout-v1', 100, 50, 2)
        self.individual = dict(individual_id=0, seed=1337, config={'lr': .0001})
        self.run = dict(self.individual, eval_trajectory=[2., 4.], auc=150.)

    def test_matching_resume_and_atomic_round_trip(self):
        results.save_results(self.path, self.metadata, [self.run])
        loaded, ids = results.load_existing_results(self.path, self.metadata, [self.individual])
        self.assertEqual(loaded, [self.run])
        self.assertEqual(ids, {0})
        self.assertFalse(Path(self.path+'.tmp').exists())

    def test_legacy_read_allowed_but_training_resume_rejected(self):
        Path(self.path).write_text(json.dumps({'results': [self.run]}))
        self.assertEqual(results.load_existing_results(self.path)[1], {0})
        with self.assertRaisesRegex(ValueError, 'Refusing incompatible resume'):
            results.load_existing_results(self.path, self.metadata, [self.individual])

    def test_changed_protocol_source_budget_and_cohort_are_rejected(self):
        results.save_results(self.path, self.metadata, [self.run])
        for key in ['provenance', 'training_protocol', 'total_frames', 'eval_frequency', 'eval_seeds', 'game_id', 'variant']:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.metadata)
                changed[key] = 'different'
                with self.assertRaisesRegex(ValueError, 'Refusing incompatible resume'):
                    results.load_existing_results(self.path, changed, [self.individual])
        for key, value in [('seed', 99), ('config', {'lr': .01}), ('individual_id', 1)]:
            changed = dict(self.individual, **{key: value})
            with self.assertRaisesRegex(ValueError, 'cohort mismatch'):
                results.load_existing_results(self.path, self.metadata, [changed])

    def test_invalid_results_never_replace_existing_file(self):
        results.save_results(self.path, self.metadata, [self.run])
        before = Path(self.path).read_bytes()
        bad_runs = [[self.run, self.run], [dict(self.run, auc=1.)],
                    [dict(self.run, eval_trajectory=[2.])],
                    [dict(self.run, eval_trajectory=[2., float('nan')])],
                    [dict(self.run, auc=float('inf'))]]
        for runs in bad_runs:
            with self.assertRaises(ValueError):
                results.save_results(self.path, self.metadata, runs)
            self.assertEqual(Path(self.path).read_bytes(), before)

    def test_corrupt_completed_trajectory_rejected_on_resume(self):
        payload = dict(self.metadata, results=[dict(self.run, eval_trajectory=[2.])])
        Path(self.path).write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            results.load_existing_results(self.path, self.metadata, [self.individual])


class TinyEnv:
    def __init__(self, truncated=False, reward=1.):
        self.steps = 0
        self.seeds = []
        self.closed = False
        self.truncated = truncated
        self.reward = reward

    def reset(self, seed=None):
        self.seeds.append(seed)
        return np.zeros((10, 10, 1), dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros((10, 10, 1), dtype=np.float32), self.reward, not self.truncated, self.truncated, {}

    def close(self):
        self.closed = True


class TrainingIntegrityTests(unittest.TestCase):
    def train(self, environment, **kwargs):
        from rct import trainer
        individual = dict(seed=1337, config=dict(lr=.0001, buffer_capacity=100,
                          batch_size=2, num_layers=1, replay_start_size=20,
                          target_update_freq=2, epsilon_decay_steps=100))
        with patch.object(trainer.gym, 'make', return_value=environment), \
             patch.object(trainer, 'evaluate_agent', return_value=1.), \
             contextlib.redirect_stdout(io.StringIO()):
            return trainer.train_individual('control', 'test', individual, 1, 2,
                                            total_frames=3, eval_frequency=3,
                                            num_eval_episodes=1, log_every=3, **kwargs)

    def test_stop_before_final_step_is_not_completed(self):
        from rct import trainer
        env = TinyEnv()
        with patch.object(trainer, '_TIME_CHECK_INTERVAL', 1), \
             patch.object(trainer, 'should_stop_for_time_limit', side_effect=[False, False, True]):
            trajectory, completed = self.train(env)
        self.assertFalse(completed)
        self.assertEqual(env.steps, 2)
        self.assertEqual(trajectory, [])
        self.assertTrue(env.closed)

    def test_unexpected_external_truncation_fails_instead_of_zero_bootstrap(self):
        from rct import trainer
        env = TinyEnv(truncated=True)
        with patch.object(trainer, 'should_stop_for_time_limit', return_value=False):
            with self.assertRaisesRegex(ValueError, 'truncation is unsupported'):
                self.train(env)
        self.assertTrue(env.closed)

    def test_evaluation_is_greedy_fixed_seed_and_restores_training_mode(self):
        class Policy(torch.nn.Module):
            def forward(self, state):
                assert not self.training
                assert not torch.is_grad_enabled()
                return torch.tensor([[0., 10.]])

        env = TinyEnv()
        policy = Policy()
        original_step = env.step
        with patch.object(evaluation.gym, 'make', return_value=env), \
             patch.object(env, 'step', wraps=original_step) as step:
            self.assertEqual(evaluation.evaluate_agent(policy, 'test', 2, [10, 11]), 1.)
        self.assertEqual(env.seeds, [10, 11])
        self.assertTrue(all(call.args == (1,) for call in step.call_args_list))
        self.assertTrue(policy.training)
        self.assertTrue(env.closed)

    def test_invalid_policy_values_fail_and_environment_is_closed(self):
        class InvalidPolicy(torch.nn.Module):
            def forward(self, state):
                return torch.tensor([[float('nan'), 0.]])

        env, policy = TinyEnv(), InvalidPolicy()
        with patch.object(evaluation.gym, 'make', return_value=env):
            with self.assertRaises(FloatingPointError):
                evaluation.evaluate_agent(policy, 'test', 1)
        self.assertTrue(env.closed)
        self.assertTrue(policy.training)

    def test_trial_rejects_unevaluated_final_policy(self):
        from rct.trial import execute_rct_trial
        with self.assertRaisesRegex(ValueError, 'divisible'):
            execute_rct_trial('control', total_frames=100, eval_frequency=30)

    def test_nonterminating_evaluation_fails_without_saving_partial_return(self):
        class Policy(torch.nn.Module):
            def forward(self, state):
                return torch.zeros(1, 2)

        class ContinuingEnv(TinyEnv):
            def step(self, action):
                state, reward, _, _, info = super().step(action)
                return state, reward, False, False, info

        env, policy = ContinuingEnv(), Policy()
        with patch.object(evaluation.gym, 'make', return_value=env):
            with self.assertRaisesRegex(RuntimeError, 'No partial return was recorded'):
                evaluation.evaluate_agent(policy, 'test', 1, max_episode_steps=5)
        self.assertEqual(env.steps, 5)
        self.assertTrue(env.closed)
        self.assertTrue(policy.training)

    def test_actual_seaquest_surface_policy_hits_guard(self):
        from rct import torch_setup  # Registers MinAtar environments.

        class NoOp(torch.nn.Module):
            def forward(self, state):
                return torch.tensor([[1., 0., 0., 0., 0., 0.]])

        with self.assertRaisesRegex(RuntimeError, 'exceeded 300 steps'):
            evaluation.evaluate_agent(NoOp(), 'MinAtar/Seaquest-v1', 1, max_episode_steps=300)


if __name__ == '__main__':
    unittest.main()
