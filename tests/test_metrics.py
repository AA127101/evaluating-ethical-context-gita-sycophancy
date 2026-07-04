"""Unit tests for the statistics and lexical-metric functions.

Run:  python -m unittest discover -s tests        (from the project root)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_results as ar  # noqa: E402
import objective_metrics as om  # noqa: E402


class TestPairedStats(unittest.TestCase):
    def test_known_difference_higher_better(self):
        base = np.array([3.0, 3, 3, 3])
        nish = np.array([4.0, 4, 4, 4])
        s = ar.paired_stats(base, nish, higher_better=True)
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["baseline_mean"], 3.0)
        self.assertAlmostEqual(s["nishkama_mean"], 4.0)
        self.assertAlmostEqual(s["mean_diff"], 1.0)
        self.assertEqual(s["n_favorable"], 4)
        self.assertEqual(s["n_nonzero"], 4)

    def test_sycophancy_direction_lower_better(self):
        base = np.array([4.0, 4, 4, 4])
        nish = np.array([2.0, 2, 2, 2])
        s = ar.paired_stats(base, nish, higher_better=False)
        # lower sycophancy is better, so every drop counts as favorable
        self.assertEqual(s["n_favorable"], 4)
        self.assertAlmostEqual(s["mean_diff"], -2.0)

    def test_nan_pairs_dropped(self):
        base = np.array([3.0, np.nan, 3.0])
        nish = np.array([4.0, 4.0, np.nan])
        s = ar.paired_stats(base, nish, higher_better=True)
        self.assertEqual(s["n"], 1)

    def test_boot_ci_brackets_constant_diff(self):
        diff = np.array([1.0, 1, 1, 1, 1])
        lo, hi = ar.boot_ci(diff)
        self.assertLessEqual(lo, 1.0)
        self.assertGreaterEqual(hi, 1.0)


class TestObjectiveMetrics(unittest.TestCase):
    def test_word_count(self):
        self.assertEqual(om.word_count("hello world foo"), 3)
        self.assertEqual(om.word_count(""), 1)  # guarded to a minimum of 1

    def test_rate_per_100w(self):
        # 'effort' appears once in 4 words -> 100 * 1/4 = 25.0
        self.assertAlmostEqual(om.rate_per_100w("i value effort here", ["effort"]), 25.0)

    def test_rate_zero_when_absent(self):
        self.assertEqual(om.rate_per_100w("nothing matches here", ["xyzzy"]), 0.0)

    def test_lexicons_nonempty(self):
        for key, terms in om.LEXICONS.items():
            self.assertTrue(len(terms) > 0, f"empty lexicon: {key}")


if __name__ == "__main__":
    unittest.main()
