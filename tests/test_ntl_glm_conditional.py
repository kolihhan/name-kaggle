"""CPU-only contract tests for frozen GLM-only prediction routing."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from ntl_glm_conditional_runner import eligible_inputs, routed_predictions, FULL, FLASH


class ConditionalGLMContract(unittest.TestCase):
    def setUp(self):
        self.p = [
            {"id": "a", "model": FULL},
            {"id": "b", "model": "gpt-5.5"},
            {"id": "c", "model": FLASH},
            {"id": "d", "model": "qwen3.8-max"},
        ]
        self.x = [{"id": row["id"], "text": f"independent text {row['id']}"} for row in self.p]

    def test_only_predicted_glm_texts_without_labels_or_metadata(self):
        selected = eligible_inputs(self.p, self.x)
        self.assertEqual([r["id"] for r in selected], ["a", "c"])
        self.assertEqual([set(r) for r in selected], [{"id", "text"}, {"id", "text"}])
        self.assertNotIn("model", str(selected))
        self.assertEqual([r["model"] for r in self.p], [FULL, "gpt-5.5", FLASH, "qwen3.8-max"])

    def test_frozen_zero_threshold_and_no_changes_outside_family(self):
        output = routed_predictions(self.p, {"a": -0.25, "c": 0.5})
        self.assertEqual([r["model"] for r in output], [FLASH, "gpt-5.5", FULL, "qwen3.8-max"])
        tied = routed_predictions(self.p, {"a": 0.0, "c": 0.0})
        self.assertEqual(tied, self.p)

    def test_fail_closed_on_extra_missing_stale_or_nonfinite_scores(self):
        for d in ({"a": 1.0}, {"a": 1.0, "c": 0.5, "b": 2.0},
                  {"a": math.nan, "c": 2.0},
                  {"a": 1.0, "c": math.inf}, {"a": True, "c": 1.0}):
            with self.assertRaises(ValueError):
                routed_predictions(self.p, d)

    def test_duplicate_misaligned_or_truth_bearing_inputs_fail(self):
        for preds, inputs in (
            (self.p, self.x[:-1]),
            (self.p, self.x[:3] + [self.x[0]]),
            (self.p, [dict(self.x[0], model="glm-5.3")] + self.x[1:]),
            (self.p, [dict(self.x[0], truth="glm-5.3")] + self.x[1:]),
            (self.p + [self.p[0]], self.x + [self.x[0]]),
        ):
            with self.assertRaises(ValueError):
                eligible_inputs(preds, inputs)

    def test_zero_eligible_no_gpu_scores_needed(self):
        preds = [{"id": "x", "model": "gpt-5.5"}]
        self.assertEqual(eligible_inputs(preds, [{"id": "x", "text": "text"}]), [])
        self.assertEqual(routed_predictions(preds, {}), preds)


if __name__ == "__main__":
    unittest.main()
