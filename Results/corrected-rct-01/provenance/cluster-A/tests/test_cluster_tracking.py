"""Protect paired RNGs, attempt lifecycles and the cluster task mapping."""

import contextlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from rct import trainer, trial  # Initialize CPU threading before any tensor work.
from rct.constants import GAMES, VARIANTS
from rct.results import build_metadata, result_path
from rct.sampling import sample_individuals
from rct.tasks import task_assignment
from rct.tracking import AttemptTracker, TrackingSettings
from sync_wandb import pending_runs

ROOT = Path(__file__).resolve().parents[1]


def consume_rng(*args, **kwargs):
    random.random()
    np.random.rand()
    torch.rand(1)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = TrackingSettings(mode="offline", directory=self.directory.name)
        self.metadata = build_metadata("control", GAMES[0], 100, 50, 1)
        self.individual = sample_individuals(1)[0]
        self.run = Mock(summary={})
        self.run.log.side_effect = consume_rng
        self.run.finish.side_effect = consume_rng

        def init(**kwargs):
            consume_rng()
            return self.run

        self.sdk = types.SimpleNamespace(init=Mock(side_effect=init), Settings=Mock())
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, wandb=self.sdk).start()
        patch("rct.tracking.version", return_value="test").start()

    def records(self, tracker):
        path = Path(self.directory.name) / f"attempt-{tracker.attempt_id}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_logging_preserves_rng_and_records_final_endpoints(self):
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        expected = (random.random(), np.random.rand(), torch.rand(1).item())
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        tracker = AttemptTracker(self.settings, self.metadata, self.individual)
        tracker.log(100, {"eval/mean_return": 4.})
        tracker.finish("completed", {"auc": 150., "final_return": 4.})
        self.assertEqual(expected, (random.random(), np.random.rand(), torch.rand(1).item()))
        self.assertTrue(self.run.summary["completed"])
        self.assertEqual(self.records(tracker)[-1]["final_return"], 4.)
        self.assertEqual(self.sdk.init.call_args.kwargs["mode"], "offline")

    def test_restarted_attempts_are_distinct_and_incomplete_is_not_a_result(self):
        first = AttemptTracker(self.settings, self.metadata, self.individual)
        first.log(50, {"eval/mean_return": 2.})
        first.finish("interrupted")
        second = AttemptTracker(self.settings, self.metadata, self.individual)
        second.finish("failed")
        self.assertNotEqual(first.attempt_id, second.attempt_id)
        for item in [first, second]:
            final = self.records(item)[-1]
            self.assertFalse(final["completed"])
            self.assertNotIn("auc", final)

    def test_disabled_tracking_does_not_import_sdk_or_write_files(self):
        tracker = AttemptTracker(TrackingSettings(), self.metadata, self.individual)
        tracker.log(1, {"eval/mean_return": 2.})
        tracker.finish("completed")
        self.sdk.init.assert_not_called()
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_initialization_failure_is_recorded_and_raised(self):
        self.sdk.init.side_effect = RuntimeError("disk full")
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            AttemptTracker(self.settings, self.metadata, self.individual)
        journal = next(Path(self.directory.name).glob("*.jsonl"))
        self.assertEqual(json.loads(journal.read_text().splitlines()[-1])["status"], "initialization_failed")

    def test_trial_saves_before_marking_complete_and_resume_skips_tracking(self):
        from rct import trial
        settings = self.settings
        def check_saved(*args, **kwargs):
            path = result_path(self.directory.name, "control", GAMES[0])
            data = json.loads(Path(path).read_text())
            self.assertEqual(data["results"][0]["final_return"], 4.)
        self.run.finish.side_effect = check_saved
        with patch.object(trial, "train_individual", return_value=([2., 4.], True)) as train, \
             patch.object(trial, "should_stop_for_time_limit", return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                trial.execute_rct_trial("control", num_individuals=1, total_frames=100,
                                        eval_frequency=50, num_eval_episodes=1,
                                        output_dir=self.directory.name, games=[GAMES[0]], tracking=settings)
        train.assert_called_once()
        self.sdk.init.assert_called_once()
        self.assertTrue(self.run.summary["completed"])

    def test_trial_failure_and_interruption_do_not_save_completed_results(self):
        from rct import trial
        for outcome in [([], False), RuntimeError("evaluation failed")]:
            with self.subTest(outcome=outcome), \
                 patch.object(trial, "train_individual") as train, \
                 patch.object(trial, "should_stop_for_time_limit", return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                kwargs = dict(num_individuals=1, total_frames=100, eval_frequency=50,
                              num_eval_episodes=1, output_dir=self.directory.name,
                              games=[GAMES[0]], tracking=self.settings)
                if isinstance(outcome, Exception):
                    train.side_effect = outcome
                    with self.assertRaises(RuntimeError):
                        trial.execute_rct_trial("control", **kwargs)
                else:
                    train.return_value = outcome
                    self.assertTrue(trial.execute_rct_trial("control", **kwargs))
                self.assertFalse(Path(result_path(self.directory.name, "control", GAMES[0])).exists())
                self.assertFalse(self.run.summary["completed"])

    def test_tracked_and_untracked_training_have_identical_weights_all_arms(self):
        from rct import trainer
        torch.set_num_threads(1)
        individual = dict(self.individual, config=dict(lr=.0001, buffer_capacity=100,
            batch_size=8, num_layers=1, replay_start_size=12,
            target_update_freq=20, epsilon_decay_steps=100))
        factory = trainer.MinAtarQNetwork
        for variant in VARIANTS:
            outputs = []
            with self.subTest(variant=variant):
                for tracked in (False, True):
                    nets = []
                    def network(*args, **kwargs):
                        net = factory(*args, **kwargs)
                        nets.append(net)
                        return net
                    tracker = AttemptTracker(self.settings, self.metadata, individual) if tracked else None
                    with patch.object(trainer, "MinAtarQNetwork", side_effect=network), \
                         patch.object(trainer, "evaluate_agent", return_value=1.), \
                         patch.object(trainer, "should_stop_for_time_limit", return_value=False), \
                         contextlib.redirect_stdout(io.StringIO()):
                        result = trainer.train_individual(variant, GAMES[0], individual, 4, 3,
                            60, 30, 1, log_every=10, metrics_callback=tracker.log if tracked else None)
                    outputs.append((result, {k: v.clone() for k, v in nets[0].state_dict().items()}))
                    if tracker:
                        tracker.finish("completed")
                self.assertEqual(outputs[0][0], outputs[1][0])
                for key in outputs[0][1]:
                    torch.testing.assert_close(outputs[0][1][key], outputs[1][1][key], rtol=0, atol=0)

    def test_training_error_is_preserved_when_sdk_cleanup_also_fails(self):
        self.run.finish.side_effect = OSError("tracking disk unavailable")
        with patch.object(trial, "train_individual", side_effect=RuntimeError("evaluation failed")), \
             patch.object(trial, "should_stop_for_time_limit", return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                trial.execute_rct_trial("control", num_individuals=1, total_frames=100,
                    eval_frequency=50, num_eval_episodes=1, output_dir=self.directory.name,
                    games=[GAMES[0]], tracking=self.settings)
        journal = next(Path(self.directory.name).glob("*.jsonl"))
        final = json.loads(journal.read_text().splitlines()[-1])
        self.assertFalse(final["wandb_finalized"])
        self.assertFalse(final["completed"])


class ClusterTests(unittest.TestCase):
    def test_all_25_shell_assignments_match_python(self):
        env = dict(os.environ, PYTHON=sys.executable)
        script = 'source slurm/common.sh; for task in {0..24}; do export SLURM_ARRAY_TASK_ID="$task"; rct_map_task || exit; echo "$RCT_VARIANT $RCT_GAME $RCT_GAME_SLUG"; done'
        result = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        actual = [tuple(line.split()) for line in result.stdout.splitlines()]
        self.assertEqual(actual, [task_assignment(i) for i in range(25)])
        self.assertEqual({row[:2] for row in actual}, {(v, g) for v in VARIANTS for g in GAMES})
        for bad in [-1, 25]:
            with self.assertRaises(ValueError):
                task_assignment(bad)

    def test_offline_phase_paths_and_experiment_id_survive_resubmission(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PROJECT_ROOT=directory, RCT_VARIANT="per", RCT_GAME_SLUG="breakout",
                       SLURM_ARRAY_JOB_ID="123", WANDB_MODE="offline", WANDB_API_KEY="test-placeholder")
            for name in ("RCT_EXPERIMENT_ID", "RCT_EXPERIMENT_ROOT"):
                env.pop(name, None)
            script = 'source slurm/common.sh; rct_setup_experiment full; echo "$OUTPUT_DIR"; export SLURM_ARRAY_JOB_ID=999; rct_setup_experiment full; echo "$OUTPUT_DIR"; rct_setup_experiment pilot; echo "$OUTPUT_DIR"; test -z "${WANDB_API_KEY:-}"'
            result = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            self.assertEqual(lines[0], lines[1])
            self.assertNotEqual(lines[0], lines[2])
            self.assertIn("/123/full/results/per/breakout", lines[0])

    def test_slurm_unix_end_timestamp_stops_before_limit(self):
        from rct.slurm_utils import should_stop_for_time_limit
        from datetime import datetime, timedelta, timezone
        env = {"SLURM_JOB_END_TIME": str(int(datetime.now().timestamp()) + 100)}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(should_stop_for_time_limit(900))
            self.assertFalse(should_stop_for_time_limit(10))
        end = datetime.now(timezone.utc) + timedelta(seconds=100)
        with patch.dict(os.environ, {"SLURM_JOB_END_TIME": end.strftime("%Y-%m-%dT%H:%M:%S%z")}, clear=True):
            self.assertTrue(should_stop_for_time_limit(900))

    def test_shell_time_budget_replaces_or_clears_previous_allocation_deadline(self):
        env = dict(os.environ, RCT_JOB_END_EPOCH="1", SLURM_JOB_END_TIME="2000000000")
        script = 'source slurm/common.sh; rct_export_time_budget; test "$RCT_JOB_END_EPOCH" = 2000000000; unset SLURM_JOB_END_TIME SLURM_JOB_ID; rct_export_time_budget; test -z "${RCT_JOB_END_EPOCH:-}"'
        subprocess.run(["bash", "-ec", script], cwd=ROOT, env=env, capture_output=True, check=True)

    def test_sync_skips_active_and_already_synced_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for attempt, status in [("a", "completed"), ("b", "interrupted"), ("c", "running")]:
                run = root / "wandb" / f"offline-run-date-{attempt}" / f"run-{attempt}.wandb"
                run.parent.mkdir(parents=True)
                run.write_bytes(b"test")
                (root / f"attempt-{attempt}.jsonl").write_text(json.dumps({"event": "finish" if status != "running" else "start", "status": status, "wandb_finalized": status != "running"}) + "\n")
            self.assertEqual(len(list(pending_runs(root))), 2)
            next(pending_runs(root)).with_suffix(".wandb.synced").touch()
            self.assertEqual(len(list(pending_runs(root))), 1)


@unittest.skipUnless(os.environ.get("RCT_TEST_WANDB_SDK") == "1", "opt-in real offline SDK integration")
class RealOfflineSDKTest(unittest.TestCase):
    def test_offline_artifact_and_sync_preview(self):
        with tempfile.TemporaryDirectory(prefix="rct-wandb-offline-") as directory:
            tracker = AttemptTracker(TrackingSettings(mode="offline", directory=directory),
                build_metadata("control", GAMES[0], 100, 50, 1), sample_individuals(1)[0])
            tracker.log(50, {"train/loss": 1., "eval/mean_return": 2.})
            tracker.log(100, {"train/loss": .5, "eval/mean_return": 4.})
            tracker.finish("completed", {"auc": 150., "final_return": 4.})
            artifacts = list(pending_runs(directory))
            self.assertEqual(len(artifacts), 1)
            self.assertGreater(artifacts[0].stat().st_size, 0)

    def test_real_offline_sdk_preserves_learned_per_policy(self):
        individual = dict(sample_individuals(1)[0], config=dict(lr=.0001, buffer_capacity=100,
            batch_size=8, num_layers=1, replay_start_size=12,
            target_update_freq=20, epsilon_decay_steps=100))
        factory = trainer.MinAtarQNetwork
        outcomes = []
        with tempfile.TemporaryDirectory(prefix="rct-wandb-training-") as directory:
            for tracked in (False, True):
                nets = []
                def network(*args, **kwargs):
                    net = factory(*args, **kwargs)
                    nets.append(net)
                    return net
                tracker = AttemptTracker(TrackingSettings(mode="offline", directory=directory),
                    build_metadata("per", GAMES[0], 120, 60, 1), individual, pilot=True) if tracked else None
                with patch.object(trainer, "MinAtarQNetwork", side_effect=network), \
                     patch.object(trainer, "should_stop_for_time_limit", return_value=False), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = trainer.train_individual("per", GAMES[0], individual, 4, 3,
                        120, 60, 1, log_every=20, metrics_callback=tracker.log if tracked else None)
                outcomes.append((result, {k: v.clone() for k, v in nets[0].state_dict().items()}))
                if tracker:
                    tracker.finish("completed")
            self.assertEqual(outcomes[0][0], outcomes[1][0])
            for key in outcomes[0][1]:
                torch.testing.assert_close(outcomes[0][1][key], outcomes[1][1][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
