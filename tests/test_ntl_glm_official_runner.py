"""CPU integration controls for no-fit conditional GLM official output packaging."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from unittest import TestCase, main
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from ntl_glm_official_runner import finalize


class FrozenOutputContract(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root.joinpath("data").mkdir()
        self.out = self.root / "out"
        self.out.mkdir()
        self.work = self.root / "work"

    def seed(self, models):
        rows = [{"id": str(i), "model": model} for i, model in enumerate(models)]
        pd.DataFrame(rows).to_parquet(self.out / "submission.parquet", index=False)
        pd.DataFrame([{"id": str(i), "text": f"test response {i}"} for i in range(len(rows))]).to_csv(
            self.root / "data" / "test.csv", index=False)
        return rows

    def stub(self, score):
        fake = ModuleType("ntl_glm_runner")
        fake.WORK = self.work
        fake.scores = score
        return fake

    def test_only_glm_predictions_scored_and_parquet_order_preserved(self):
        source = self.seed(["glm-5.3", "gpt-5.5", "glm-5.3-flash", "qwen3.8-max"])
        called = []
        def score(payload):
            selected = pd.read_csv(self.work / "ratio-input.csv", dtype={"id": str})
            called.extend(selected.to_dict(orient="records"))
            return pd.Series([-0.5, 0.75], index=["0", "2"]), {"flash": 1.2, "glm53": 2.1}
        with patch.dict(sys.modules, {"ntl_glm_runner": self.stub(score)}):
            finalize(self.root / "data", "not-invoked-in-stub", output_root=self.out)
        self.assertEqual([r["id"] for r in called], ["0", "2"])
        self.assertEqual(set(called[0]), {"id", "text"})
        actual = pd.read_parquet(self.out / "submission.parquet")
        self.assertEqual(actual["id"].tolist(), ["0", "1", "2", "3"])
        self.assertEqual(actual["model"].tolist(),
                         ["glm-5.3-flash", "gpt-5.5", "glm-5.3", "qwen3.8-max"])
        metrics = json.loads((self.out / "glm-candidate-runtime.json").read_text())
        self.assertEqual((metrics["total"], metrics["eligible_glm"], metrics["changed"]), (4, 2, 2))
        self.assertFalse(metrics["new_training"])
        self.assertFalse(metrics["used_truth_labels"])

    def test_mismatched_scored_ids_leave_original_champion(self):
        source = self.seed(["glm-5.3", "gpt-5.5"])
        def bad_score(payload):
            return pd.Series([0.9], index=["NOT_PREDICTED"]), {"flash": 0, "glm53": 0}
        with patch.dict(sys.modules, {"ntl_glm_runner": self.stub(bad_score)}):
            with self.assertRaisesRegex(ValueError, "incomplete likelihood rows"):
                finalize(self.root / "data", "not-invoked", output_root=self.out)
        self.assertEqual(pd.read_parquet(self.out / "submission.parquet").to_dict("records"), source)
        self.assertFalse((self.out / "glm-candidate-runtime.json").exists())

    def test_no_eligible_predictions_skip_expensive_likelihood(self):
        source = self.seed(["gpt-5.5", "claude-opus-5"])
        def unexpected(payload):
            raise AssertionError("GLM inference must not be called")
        with patch.dict(sys.modules, {"ntl_glm_runner": self.stub(unexpected)}):
            finalize(self.root / "data", "not-invoked", output_root=self.out)
        self.assertEqual(pd.read_parquet(self.out / "submission.parquet").to_dict("records"), source)
        result = json.loads((self.out / "glm-candidate-runtime.json").read_text())
        self.assertEqual(result["eligible_glm"], 0)
        self.assertEqual(result["changed"], 0)


if __name__ == "__main__":
    main()
