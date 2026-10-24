"""Follow-up options: capped evaluation, the Obando-Ceron & Castro recipe,
individual selection with config overrides, and the follow-up manifest."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rct import evaluation, results, trainer, trial
from rct.constants import RECIPES, REFERENCE_CONFIGS, TRAINING_SETTINGS
from rct.sampling import sample_individuals
from rct.trainer import train_individual
from rct.trial import execute_rct_trial

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import followup  # noqa: E402


class SurfaceNetwork(torch.nn.Module):
    """Always chooses no-op, so the Seaquest submarine stays on the surface."""

    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def forward(self, x):
        q = torch.zeros(x.shape[0], self.num_actions)
        q[:, 0] = 1.0
        return q


class CappedEvaluationTests(unittest.TestCase):
    def test_raise_is_default_and_truncate_keeps_partial_return(self):
        net = SurfaceNetwork(18)
        with self.assertRaisesRegex(RuntimeError, 'exceeded 50 steps'):
            evaluation.evaluate_agent(net, 'MinAtar/Seaquest-v1', num_episodes=2, max_episode_steps=50)
        log = []
        score = evaluation.evaluate_agent(net, 'MinAtar/Seaquest-v1', num_episodes=2, max_episode_steps=50,
                                          on_limit='truncate', truncation_log=log)
        self.assertEqual(score, 0.0)
        self.assertEqual(log, [2])

    def test_terminating_episodes_are_not_counted(self):
        log = []
        net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LazyLinear(3))
        evaluation.evaluate_agent(net, 'MinAtar/Breakout-v1', num_episodes=2, on_limit='truncate',
                                  truncation_log=log)
        self.assertEqual(log, [0])

    def test_unknown_behaviour_rejected(self):
        with self.assertRaises(ValueError):
            evaluation.evaluate_agent(SurfaceNetwork(18), 'MinAtar/Seaquest-v1', on_limit='skip')


def ignore_job_deadline(test_case):
    """Training tests must not obey the SLURM deadline of the job running the preflight."""
    for module in (trainer, trial):
        patcher = patch.object(module, 'should_stop_for_time_limit', return_value=False)
        patcher.start()
        test_case.addCleanup(patcher.stop)


class RecipeTests(unittest.TestCase):
    def setUp(self):
        ignore_job_deadline(self)

    def _count_steps(self, optimizer_cls, settings, frames=400):
        individual = sample_individuals(1)[0]
        individual['config'] = dict(individual['config'], replay_start_size=100, batch_size=32)
        with patch.object(optimizer_cls, 'step', autospec=True, side_effect=lambda self, *a, **k: None) as step, \
                contextlib.redirect_stdout(io.StringIO()):
            train_individual('control', 'MinAtar/Breakout-v1', individual, 4, 3, total_frames=frames,
                             eval_frequency=frames, num_eval_episodes=1, log_every=10**6, settings=settings)
        return step.call_count

    def test_shared_recipe_updates_every_frame_with_rmsprop(self):
        # Training starts once the buffer holds replay_start_size (100) transitions.
        self.assertEqual(self._count_steps(torch.optim.RMSprop, None), 400 - 99)

    def test_obando_recipe_uses_adam_every_fourth_frame(self):
        frames = [f for f in range(99, 400) if (f + 1) % 4 == 0]
        self.assertEqual(self._count_steps(torch.optim.Adam, RECIPES['obando2021']), len(frames))

    def test_obando_recipe_values(self):
        recipe = RECIPES['obando2021']
        self.assertEqual((recipe['optimizer'], recipe['optimizer_options'], recipe['update_period'],
                          recipe['epsilon_final'], recipe['loss'], recipe['gamma']),
                         ('Adam', {'eps': 3.125e-4}, 4, 0.01, 'mse', 0.99))
        self.assertIs(RECIPES['shared'], TRAINING_SETTINGS)
        self.assertNotIn('update_period', TRAINING_SETTINGS)
        self.assertEqual(REFERENCE_CONFIGS['obando2021']['lr'], 0.00025)


class SelectionAndMetadataTests(unittest.TestCase):
    def setUp(self):
        ignore_job_deadline(self)

    def test_default_metadata_unchanged(self):
        metadata = results.build_metadata('per', 'MinAtar/Breakout-v1', 100, 50, 2)
        protocol = metadata['training_protocol']
        self.assertEqual(protocol['shared_settings'], TRAINING_SETTINGS)
        self.assertNotIn('recipe', protocol)
        self.assertNotIn('config_overrides', protocol)

    def test_selected_individuals_overrides_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(variant='control', total_frames=300, eval_frequency=150, num_eval_episodes=1,
                          output_dir=directory, games=['MinAtar/Seaquest-v1'], log_every=10**6,
                          individual_ids=[13], config_overrides={'lr': 3e-5, 'replay_start_size': 100},
                          recipe='obando2021', eval_limit='truncate')
            with contextlib.redirect_stdout(io.StringIO()):
                execute_rct_trial(**kwargs)
            path = Path(results.result_path(directory, 'control', 'MinAtar/Seaquest-v1'))
            data = json.loads(path.read_text())
            (run,) = data['results']
            expected = dict(sample_individuals(14)[13]['config'], lr=3e-5, replay_start_size=100)
            self.assertEqual((run['individual_id'], run['seed'], run['config']), (13, 1350, expected))
            self.assertEqual(len(run['eval_truncated_episodes']), 2)
            protocol = data['training_protocol']
            self.assertEqual(protocol['recipe'], 'obando2021')
            self.assertEqual(protocol['config_overrides'], {'lr': 3e-5, 'replay_start_size': 100})
            self.assertEqual(protocol['shared_settings']['evaluation_limit_behavior'],
                             'truncate_and_record_partial_return')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                execute_rct_trial(**kwargs)
            self.assertIn('already done', out.getvalue())
            with self.assertRaisesRegex(ValueError, 'Refusing incompatible resume'), \
                    contextlib.redirect_stdout(io.StringIO()):
                execute_rct_trial(**dict(kwargs, recipe='shared'))

    def test_unknown_override_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown config override'):
            execute_rct_trial('control', total_frames=10, eval_frequency=10, num_eval_episodes=1,
                              individual_ids=[0], config_overrides={'learning_rate': 1e-5})


class ManifestTests(unittest.TestCase):
    def test_groups_sizes_and_unique_outputs(self):
        rows = followup.manifest()
        sizes = {}
        for row in rows:
            sizes[row['group']] = sizes.get(row['group'], 0) + 1
        self.assertEqual(sizes, {'recipe-sampled': 42, 'recipe-reference': 40, 'lr-low-dqn': 56,
                                 'seaquest-capped': 150, 'lr-low-arms': 224})
        self.assertEqual([r['row'] for r in rows], list(range(len(rows))))
        outputs = {followup.result_file('/x', r).as_posix() + f":{r['individual_id']}" for r in rows}
        self.assertEqual(len(outputs), len(rows))

    def test_lr_ids_are_the_lr_1e4_individuals(self):
        ids = [i['individual_id'] for i in sample_individuals(30) if i['config']['lr'] == 1e-4]
        self.assertEqual(ids, followup.LR_1E4_IDS)

    def test_prepare_creates_every_run_folder(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(followup, 'ROOT', Path(directory)), contextlib.redirect_stdout(io.StringIO()):
            root = str(Path(directory) / 'experiments' / 'x')
            followup.cmd_prepare(type('Args', (), {'root': root})())
            followup.cmd_prepare(type('Args', (), {'root': root})())  # idempotent
            for row in followup.manifest():
                for folder in followup.run_dirs(root, row):
                    self.assertTrue(folder.is_dir(), folder)
            self.assertTrue((Path(directory) / 'logs' / 'rct-followup' / 'rows').is_dir())

    def test_ensure_dir_retries_transient_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'a' / 'b'
            real_mkdir = Path.mkdir
            calls = []

            def flaky(path, *args, **kwargs):
                calls.append(path)
                if len(calls) == 1:
                    raise OSError('mkdir: Already exists')
                return real_mkdir(path, *args, **kwargs)

            with patch.object(Path, 'mkdir', flaky), patch.object(followup.time, 'sleep'), \
                    contextlib.redirect_stdout(io.StringIO()):
                followup.ensure_dir(target)
            self.assertTrue(target.is_dir())
            self.assertGreaterEqual(len(calls), 2)

    def test_task_chunks_cover_rows_once_without_mixing_groups(self):
        rows = followup.manifest()
        for per_task in (1, 3, 4, 7):
            chunks = followup.task_chunks(per_task)
            flat = [i for _, ids in chunks for i in ids]
            self.assertEqual(flat, list(range(len(rows))))
            for group, ids in chunks:
                self.assertLessEqual(len(ids), per_task)
                self.assertEqual({rows[i]['group'] for i in ids}, {group})


if __name__ == '__main__':
    unittest.main()
