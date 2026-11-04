"""Verify no-training package contract without fetching any model or competition data."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_lora24_sibling_holdout_kernel import (
    CODE_FILE,
    KERNEL_REF,
    SOURCES,
    write_kernel_bundle,
)


class TestPrivateSiblingInferenceBundle(unittest.TestCase):
    def test_private_no_submission_no_competition_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_kernel_bundle(root, source_texts={name: "# synthetic test fixture\n" for name in SOURCES})
            meta = json.loads((root / "kernel-metadata.json").read_text())
            code = (root / CODE_FILE).read_text()
            compile(code, "kernel_main.py", "exec")
            self.assertEqual(meta["id"], KERNEL_REF)
            self.assertEqual(meta["title"].lower().replace(" ", "-"), KERNEL_REF.split("/")[1])
            self.assertEqual(meta["code_file"], CODE_FILE)
            self.assertTrue(meta["is_private"])
            self.assertFalse(meta["enable_internet"])
            self.assertTrue(meta["enable_gpu"])
            self.assertEqual(meta["competition_sources"], [])
            self.assertEqual(meta["kernel_sources"], [])
            self.assertEqual(len(meta["dataset_sources"]), 3)
            self.assertEqual(len(meta["model_sources"]), 1)
            self.assertNotIn("kaggle competitions submit", code)
            self.assertNotIn("run_final_fit(", code)
            self.assertNotIn("train_two_stage2_epochs(", code)
            self.assertIn("INFERENCE_ONLY_SUMMARY_COMPLETE", code)

    def test_packager_runs_with_stdlib_only(self):
        """Regression: GitHub GPU job installs kaggle only; no NumPy/Pandas required."""
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "bundle"
            outcome = subprocess.run(
                [
                    sys.executable, "-S",
                    str(repo_root / "scripts/build_lora24_sibling_holdout_kernel.py"),
                    "--repo-root", str(repo_root),
                    "--output-dir", str(destination),
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(
                outcome.returncode, 0,
                "builder imported third-party ML modules in GitHub packaging runner:\n"
                + outcome.stdout + outcome.stderr,
            )
            self.assertTrue((destination / "kernel_main.py").exists())

    def test_incorrect_source_set_is_not_packaged(self):
        from scripts.build_lora24_sibling_holdout_kernel import kernel_source
        with self.assertRaisesRegex(RuntimeError, "source set drift"):
            kernel_source({})


if __name__ == "__main__":
    unittest.main()
