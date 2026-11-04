"""Synthetic-only tests; do not materialize official or external competition rows."""
import unittest

import pandas as pd

from scripts.audit_lora24_sibling_holdout import family, summarize_predictions


class TestStage1SiblingHoldout(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            [
                (0, "abdullah", "a", "gemini-3.7-flash", "gemini-3.8-flash"),
                (1, "semenoff", "b", "gpt-5.5", "claude-opus-4-6"),
                (2, "zakir", "c", "gpt-5.6-sol", "gpt-5.6-sol"),
            ],
            columns=["row_index", "source", "prompt_id", "model", "prediction"],
        )
        self.labels = sorted(self.frame["model"].tolist())

    def test_known_families(self):
        self.assertEqual(family("gpt-6-astra"), "openai")
        self.assertEqual(family("qwen3.8-max"), "qwen")
        with self.assertRaises(ValueError):
            family("unknown-5")

    def test_exact_family_error_accounting(self):
        s = summarize_predictions(
            self.frame, eligible_labels=self.labels, require_frozen_counts=False
        )
        self.assertEqual(s["pooled"]["wrong"], 2)
        self.assertEqual(s["pooled"]["within_family_wrong"], 1)
        self.assertEqual(s["pooled"]["across_family_wrong"], 1)
        self.assertAlmostEqual(s["pooled"]["within_family_fraction_of_wrong"], 0.5)
        self.assertEqual(s["by_source"]["zakir"]["wrong"], 0)
        self.assertIsNone(s["by_source"]["zakir"]["within_family_fraction_of_wrong"])
        self.assertEqual(s["reference_official_stage2_oof"]["directly_comparable"], False)
        self.assertEqual(s["decision"], "descriptive_only_no_promotion")

    def test_missing_row_and_label_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "frozen eligible"):
            summarize_predictions(self.frame, eligible_labels=self.labels)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            f = self.frame.copy()
            f.loc[2, "row_index"] = 0
            summarize_predictions(f, eligible_labels=self.labels, require_frozen_counts=False)
        with self.assertRaisesRegex(ValueError, "non-eligible"):
            summarize_predictions(self.frame, eligible_labels=self.labels[:1], require_frozen_counts=False)

    def test_source_change_rejected(self):
        f = self.frame.copy()
        f.loc[2, "source"] = "another"
        with self.assertRaisesRegex(ValueError, "source set"):
            summarize_predictions(f, eligible_labels=self.labels, require_frozen_counts=False)


if __name__ == "__main__":
    unittest.main()
