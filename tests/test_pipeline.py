"""Unit tests for data integrity, retrieval, and config consistency.

Run:  python -m unittest discover -s tests        (from the project root)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import run_experiment as rx  # noqa: E402


class TestData(unittest.TestCase):
    def test_prompts(self):
        df = pd.read_csv(config.PROMPTS_CSV)
        self.assertGreaterEqual(len(df), 14)
        for c in ("id", "category", "prompt"):
            self.assertIn(c, df.columns)
        self.assertTrue(df["id"].is_unique)
        self.assertFalse(df["prompt"].str.strip().eq("").any())

    def test_gita(self):
        df = pd.read_csv(config.GITA_CSV)
        self.assertGreaterEqual(len(df), 20)
        for c in ("id", "reference", "theme", "text"):
            self.assertIn(c, df.columns)
        self.assertTrue(df["id"].is_unique)


class TestConfig(unittest.TestCase):
    def test_metric_maps_aligned(self):
        for m in config.METRICS:
            self.assertIn(m, config.METRIC_HIGHER_IS_BETTER)
            self.assertIn(m, config.METRIC_LABELS)
        self.assertEqual(set(config.CONDITIONS), {"baseline", "nishkama"})
        # sycophancy must be the "lower is better" metric
        self.assertFalse(config.METRIC_HIGHER_IS_BETTER["sycophancy"])
        self.assertTrue(config.METRIC_HIGHER_IS_BETTER["stability"])


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.texts = pd.read_csv(config.GITA_CSV)["text"].tolist()

    def test_lexical_retrieval(self):
        r = rx.Retriever(self.texts, method="lexical")
        idx = r.retrieve("I am angry and want revenge on someone", 2)
        self.assertEqual(len(idx), 2)
        self.assertTrue(all(0 <= i < len(self.texts) for i in idx))
        self.assertEqual(len(set(idx)), 2)  # no duplicates

    def test_tokenize(self):
        toks = rx._tokenize("Anger, and FEAR! It's heavy.")
        self.assertIn("anger", toks)
        self.assertIn("fear", toks)
        self.assertIn("it's", toks)


class TestContextBlock(unittest.TestCase):
    def test_block_contains_reference(self):
        g = pd.read_csv(config.GITA_CSV).head(2)
        block = rx.build_context_block(g)
        self.assertIn(str(g.iloc[0]["reference"]), block)
        self.assertEqual(block.count("\n"), 1)  # two verses -> one newline


if __name__ == "__main__":
    unittest.main()
