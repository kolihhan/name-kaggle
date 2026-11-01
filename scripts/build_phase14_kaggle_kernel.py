from __future__ import annotations

import argparse
import json
from pathlib import Path


KERNEL_REF = "kolihhan/name-that-llm-phase14-public-sparse"
KERNEL_TITLE = "Name That LLM Phase14 Public Sparse"
NOTEBOOK_NAME = "name-that-llm-phase14-public-sparse.ipynb"
PUBLIC_DATASET = "nishantdahal/ntl-handwriting-blend"
COMPETITION = "name-that-llm"
EXPECTED_RAW_SHA256 = "37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec"
EXPECTED_RAW_ROWS = 75_783
EXPECTED_EXTERNAL_ROWS = 43_781
EXPECTED_OFFICIAL_ROWS = 2_699
EXPECTED_LABELS = 18


def build_notebook_source(runner_source: str, submission_source: str) -> str:
    runner_literal = repr(runner_source)
    submission_literal = repr(submission_source)
    return f'''from pathlib import Path
import sys
import types
import pandas as pd

RUNNER_SOURCE = {runner_literal}
SUBMISSION_SOURCE = {submission_literal}
EXPECTED_RAW_SHA256 = "{EXPECTED_RAW_SHA256}"
EXPECTED_RAW_ROWS = {EXPECTED_RAW_ROWS}
EXPECTED_EXTERNAL_ROWS = {EXPECTED_EXTERNAL_ROWS}
EXPECTED_OFFICIAL_ROWS = {EXPECTED_OFFICIAL_ROWS}
EXPECTED_LABELS = {EXPECTED_LABELS}

# The frozen CV module imports helpers that are only used by its CV entry point.
# Register minimal modules so its reusable Phase 14 feature/subset functions can
# be loaded without copying or altering the validated recipe.
scripts_pkg = types.ModuleType("scripts")
scripts_pkg.__path__ = []
sys.modules["scripts"] = scripts_pkg
anchor = types.ModuleType("scripts.run_strong_sparse_anchor")
def _cv_only(*args, **kwargs):
    raise RuntimeError("CV-only helper is unavailable in the submission notebook")
anchor.load_train = _cv_only
anchor.summarize_oof = _cv_only
sys.modules["scripts.run_strong_sparse_anchor"] = anchor

phase14 = types.ModuleType("scripts.run_phase14_public_sparse")
exec(RUNNER_SOURCE, phase14.__dict__)
sys.modules["scripts.run_phase14_public_sparse"] = phase14
setattr(scripts_pkg, "run_phase14_public_sparse", phase14)

phase14_submission = types.ModuleType("scripts.phase14_submission")
exec(SUBMISSION_SOURCE, phase14_submission.__dict__)
sys.modules["scripts.phase14_submission"] = phase14_submission
setattr(scripts_pkg, "phase14_submission", phase14_submission)


def first_existing(paths):
    for candidate in paths:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("none of the expected Kaggle input paths exist: " + ", ".join(paths))


def only_glob(pattern):
    matches = sorted(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one match for {{pattern}}, found {{matches}}")
    return matches[0]


train_path = first_existing([
    "/kaggle/input/name-that-llm/train.csv",
    "/kaggle/input/competitions/name-that-llm/train.csv",
])
test_path = first_existing([
    "/kaggle/input/name-that-llm/test.csv",
    "/kaggle/input/competitions/name-that-llm/test.csv",
])
pool_path = only_glob("**/pool_v4.parquet")

observed_sha = phase14.sha256_file(pool_path)
if observed_sha != EXPECTED_RAW_SHA256:
    raise RuntimeError(f"public pool SHA mismatch: {{observed_sha}} != {{EXPECTED_RAW_SHA256}}")

pool = pd.read_parquet(pool_path)
if len(pool) != EXPECTED_RAW_ROWS:
    raise RuntimeError(f"public pool row count mismatch: {{len(pool)}} != {{EXPECTED_RAW_ROWS}}")
external = phase14.build_conservative_subset(pool, validate_full_contract=True)
if len(external) != EXPECTED_EXTERNAL_ROWS:
    raise RuntimeError(f"external subset row count mismatch: {{len(external)}} != {{EXPECTED_EXTERNAL_ROWS}}")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
if len(train) != EXPECTED_OFFICIAL_ROWS:
    raise RuntimeError(f"official train row count mismatch: {{len(train)}} != {{EXPECTED_OFFICIAL_ROWS}}")
if not {{"id", "text", "model"}} <= set(train.columns):
    raise RuntimeError(f"unexpected official train schema: {{sorted(train.columns)}}")
if not {{"id", "text"}} <= set(test.columns):
    raise RuntimeError(f"unexpected official test schema: {{sorted(test.columns)}}")

allowed_labels = set(train["model"].astype(str).unique())
if len(allowed_labels) != EXPECTED_LABELS:
    raise RuntimeError(f"official label count mismatch: {{len(allowed_labels)}} != {{EXPECTED_LABELS}}")
if set(external["model"].astype(str).unique()) != allowed_labels:
    raise RuntimeError("external label set does not match official label set")

print(f"train={{train.shape}} test={{test.shape}} raw_pool={{pool.shape}} external={{external.shape}}")
print(external["model"].astype(str).value_counts().sort_index().to_string())

predictions, meta = phase14_submission.train_full_and_predict(train, external, test)
submission = phase14_submission.build_submission_frame(
    test,
    predictions,
    allowed_labels=allowed_labels,
)

output_path = Path("/kaggle/working/submission.parquet")
submission.to_parquet(output_path, index=False)
print("feature_meta:", meta)
print(f"wrote {{len(submission)}} rows to {{output_path}}")
print(submission["model"].value_counts().sort_index().to_string())
'''


def write_kernel_bundle(
    output_dir: Path,
    *,
    runner_path: Path,
    submission_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_source = runner_path.read_text(encoding="utf-8")
    submission_source = submission_path.read_text(encoding="utf-8")
    source = build_notebook_source(runner_source, submission_source)

    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (output_dir / NOTEBOOK_NAME).write_text(
        json.dumps(notebook, ensure_ascii=False), encoding="utf-8"
    )

    metadata = {
        "id": KERNEL_REF,
        "title": KERNEL_TITLE,
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [PUBLIC_DATASET],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"built {KERNEL_REF} with pinned public pool SHA {EXPECTED_RAW_SHA256}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen Phase 14 Kaggle notebook bundle.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_phase14_public_sparse.py"),
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("scripts/phase14_submission.py"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_kernel_bundle(
        args.output_dir,
        runner_path=args.runner,
        submission_path=args.submission,
    )


if __name__ == "__main__":
    main()
