"""Regression checks for duplicated logs, restarted runs, and result matching."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis import review


class LogAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.audit = self.root / "audit"
        self.tables = self.root / "tables"
        self.audit.mkdir()
        self.tables.mkdir()
        self.run = dict(individual_id=0, seed=1337, config={"lr": .0001},
                        auc=4_950_000., eval_trajectory=[1.] * 100)
        self.data = {(g,v): [self.run.copy()] for g in review.GAMES for v in review.VARIANTS}
        for name, value in {"ROOT":self.root, "LOG_ROOTS":[self.logs],
                            "AUDIT":self.audit,"TABLES":self.tables}.items():
            patcher = patch.object(review, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def log(self, complete=True, seed=1337):
        text = ("Job ID: 123\nVariant: control\nGame: MinAtar/Breakout-v1\n"
                "--- Individual 1/30 ---\n"
                f"Seed: {seed} | Settings: {{'lr': 0.0001}}\n"
                "[control] frame 5000/5000000 | loss: 0.100000 | epsilon: 1.0000\n")
        if complete:
            text += "".join(f"[control] frame {frame}/5000000 | eval mean return: 1.000\n"
                            for frame in range(50000, 5000001, 50000))
            text += "[control] frame 5000000/5000000 | loss: 0.010000 | epsilon: 0.1000\n"
            text += "Completed control. AUC: 4950000.0000\n"
        return text

    def test_identical_copies_are_audited_but_not_counted_twice(self):
        for name in ("a", "b"):
            (self.logs / f"{name}.out").write_text(self.log())
        attempts, bins, jobs = review.parse_logs(self.data)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(jobs[1]["duplicate_stdout_of"], "logs/a.out")
        self.assertEqual(jobs[0]["job_id"], "123")  # Old header has no '#'.
        self.assertTrue(attempts[0]["selected"])
        self.assertEqual(int(bins.samples.sum()), 2)

    def test_restart_is_separate_from_completed_history(self):
        (self.logs / "a.out").write_text(self.log(complete=False))
        (self.logs / "b.out").write_text(self.log())
        attempts, bins, _ = review.parse_logs(self.data)
        self.assertEqual(len(attempts), 2)
        self.assertFalse(attempts[0]["selected"])
        self.assertTrue(attempts[1]["selected"])
        self.assertEqual(int(bins[bins.selected].samples.sum()), 2)

    def test_matching_auc_does_not_override_mismatched_seed(self):
        (self.logs / "a.out").write_text(self.log(seed=999))
        attempts, bins, _ = review.parse_logs(self.data)
        self.assertTrue(attempts[0]["completed"])
        self.assertFalse(attempts[0]["matches_json"])
        self.assertFalse(bins.selected.any())

    def test_same_individual_in_distinct_jobs_is_selected_once(self):
        (self.logs / "a.out").write_text(self.log())
        (self.logs / "b.out").write_text(self.log().replace("Job ID: 123", "# Job ID: 456"))
        attempts, _, jobs = review.parse_logs(self.data)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(a["matches_json"] for a in attempts))
        self.assertEqual(sum(a["selected"] for a in attempts), 1)
        self.assertEqual(jobs[1]["job_id"], "456")


if __name__ == "__main__":
    unittest.main()
