from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


KERNEL_REF = "kolihhan/name-that-llm-phase7-first-submission"
KERNEL_TITLE = "Name That LLM Phase7 First Submission"
NOTEBOOK_NAME = "name-that-llm-phase7-first-submission.ipynb"
PUBLIC_DATASET = "abdullah1afify/ntl-generations"
COMPETITION = "name-that-llm"
EXPECTED_FROZEN_ROWS = 2503


def row_hash(model: str, text: str) -> str:
    payload = f"{model}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_frozen_hashes(external_path: Path) -> list[str]:
    frame = pd.read_parquet(external_path)
    required = {"model", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"frozen external snapshot missing columns: {sorted(missing)}")
    hashes = sorted(
        {
            row_hash(str(model), str(text))
            for model, text in frame[["model", "text"]].itertuples(index=False, name=None)
        }
    )
    if len(frame) != EXPECTED_FROZEN_ROWS or len(hashes) != EXPECTED_FROZEN_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_FROZEN_ROWS} unique frozen rows, "
            f"found rows={len(frame)} unique_hashes={len(hashes)}"
        )
    return hashes


def build_notebook_source(module_source: str, frozen_hashes: list[str]) -> str:
    hashes_literal = json.dumps(frozen_hashes, separators=(",", ":"))
    module_literal = repr(module_source)
    return f'''from pathlib import Path
import hashlib
import types
import pandas as pd

MODULE_SOURCE = {module_literal}
phase7 = types.ModuleType("phase7_submission")
exec(MODULE_SOURCE, phase7.__dict__)

FROZEN_HASHES = set({hashes_literal})
EXPECTED_FROZEN_ROWS = {EXPECTED_FROZEN_ROWS}


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


def frozen_row_hash(model, text):
    return hashlib.sha256(f"{{model}}\\0{{text}}".encode("utf-8")).hexdigest()


train_path = first_existing([
    "/kaggle/input/name-that-llm/train.csv",
    "/kaggle/input/competitions/name-that-llm/train.csv",
])
test_path = first_existing([
    "/kaggle/input/name-that-llm/test.csv",
    "/kaggle/input/competitions/name-that-llm/test.csv",
])
external_path = only_glob("**/ntl_generations.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
public_external = pd.read_csv(external_path)
if "model" not in public_external.columns and "label" in public_external.columns:
    public_external = public_external.rename(columns={{"label": "model"}})
if not {{"model", "text"}} <= set(public_external.columns):
    raise RuntimeError(f"unexpected public external schema: {{sorted(public_external.columns)}}")

public_hashes = [
    frozen_row_hash(str(model), str(text))
    for model, text in public_external[["model", "text"]].itertuples(index=False, name=None)
]
mask = [value in FROZEN_HASHES for value in public_hashes]
external = public_external.loc[mask].copy().reset_index(drop=True)
selected_hashes = {{
    frozen_row_hash(str(model), str(text))
    for model, text in external[["model", "text"]].itertuples(index=False, name=None)
}}
if len(external) != EXPECTED_FROZEN_ROWS:
    raise RuntimeError(f"frozen external row count mismatch: {{len(external)}}")
if selected_hashes != FROZEN_HASHES:
    missing = len(FROZEN_HASHES - selected_hashes)
    extra = len(selected_hashes - FROZEN_HASHES)
    raise RuntimeError(f"frozen external hash mismatch: missing={{missing}} extra={{extra}}")

print(f"train={{train.shape}} test={{test.shape}} public_external={{public_external.shape}} frozen_external={{external.shape}}")
print(external["model"].astype(str).value_counts().sort_index().to_string())

official_labels = phase7.validate_training_frames(train, external)
predictions = phase7.predict_phase7(train, external, test, seed=phase7.DEFAULT_SEED)
submission = phase7.build_submission_frame(test, predictions, allowed_labels=official_labels)

output_path = Path("/kaggle/working/submission.parquet")
submission.to_parquet(output_path, index=False)
print(f"wrote {{len(submission)}} rows to {{output_path}}")
print(submission["model"].value_counts().sort_index().to_string())
'''


def write_kernel_bundle(
    output_dir: Path,
    *,
    module_path: Path,
    external_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    module_source = module_path.read_text(encoding="utf-8")
    frozen_hashes = compute_frozen_hashes(external_path)
    source = build_notebook_source(module_source, frozen_hashes)

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
    print(f"built {KERNEL_REF} with {len(frozen_hashes)} frozen external row hashes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen Phase 7 Kaggle notebook bundle.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--module", type=Path, default=Path("scripts/make_submission.py"))
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_kernel_bundle(
        args.output_dir,
        module_path=args.module,
        external_path=args.external,
    )


if __name__ == "__main__":
    main()
