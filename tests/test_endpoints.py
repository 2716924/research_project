"""Verify endpoint definition and pairing independently of saved rankings."""
import unittest

import numpy as np
from scipy import stats

from analysis.endpoints import endpoint, paired_arrays, paired_summary


class EndpointTests(unittest.TestCase):
    def run_data(self, i, scores):
        return dict(individual_id=i, seed=100+i, config={"lr": .001},
                    auc=123., eval_trajectory=scores)

    def test_final_checkpoint_is_not_final_window(self):
        run = self.run_data(0, list(range(100)))
        self.assertEqual(endpoint(run, "final_return"), 99)
        self.assertEqual(endpoint(run, "last_10_return"), 94.5)
        self.assertEqual(endpoint(run, "auc"), 123.)

    def test_pairs_by_id_and_rejects_mismatched_seed_or_config(self):
        control = [self.run_data(0, [2]), self.run_data(1, [3])]
        treatment = [self.run_data(1, [5]), self.run_data(0, [7])]
        c, t = paired_arrays(control, treatment, "final_return")
        np.testing.assert_array_equal(c, [2, 3])
        np.testing.assert_array_equal(t, [7, 5])
        treatment[0]["seed"] += 1
        with self.assertRaises(ValueError):
            paired_arrays(control, treatment, "final_return")
        with self.assertRaises(ValueError):
            paired_arrays(control, control[:1], "final_return")

    def test_paired_uncertainty_uses_differences(self):
        c = np.array([1., 10., 100., 1000.])
        t = c + np.array([1., 2., 3., 4.])
        result = paired_summary(c, t, np.random.default_rng(1))
        expected_radius = stats.t.ppf(1-.0125/2, 3)*stats.sem(t-c)
        self.assertEqual(result["mean_difference"], 2.5)
        self.assertAlmostEqual(result["ci_98_75_high"], 2.5+expected_radius)
        self.assertAlmostEqual(result["p_value"], stats.ttest_1samp(t-c, 0).pvalue)
        self.assertEqual(result["matched_win_fraction"], 1)
        self.assertLessEqual(result["bootstrap_95_low"], 2.5)
        self.assertGreaterEqual(result["bootstrap_95_high"], 2.5)


if __name__ == '__main__':
    unittest.main()
