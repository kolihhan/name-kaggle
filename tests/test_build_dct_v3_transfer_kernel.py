from __future__ import annotations

import json
from pathlib import Path

from scripts.build_dct_v3_transfer_kernel import (
    EXPECTED_ELIGIBLE_LABELS,
    EXPECTED_ELIGIBLE_ROWS,
    EXPECTED_HELDOUT_ROWS,
    EXPECTED_HOLDOUT_SOURCES,
    EXPECTED_TRAIN_ROWS,
    KERNEL_REF,
    write_kernel_bundle,
)


def test_frozen_transfer_constants() -> None:
    assert KERNEL_REF == "kolihhan/ntl-v3-dct-transfer"
    assert EXPECTED_HOLDOUT_SOURCES == ["alperen", "hailey", "willrice"]
    assert EXPECTED_TRAIN_ROWS == 14502
    assert EXPECTED_HELDOUT_ROWS == 14506
    assert EXPECTED_ELIGIBLE_ROWS == 14506
    assert EXPECTED_ELIGIBLE_LABELS == 17


def test_bundle_is_private_offline_t4_and_has_exact_sources(tmp_path: Path) -> None:
    write_kernel_bundle(tmp_path, repo_root=Path("."))
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == KERNEL_REF
    assert meta["is_private"] is True
    assert meta["enable_internet"] is False
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["competition_sources"] == []
    assert meta["dataset_sources"] == [
        "nishantdahal/ntl-handwriting-blend",
        "kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2",
    ]
    assert meta["model_sources"] == ["qwen-lm/qwen-3/transformers/8b-base/1"]


def test_kernel_has_no_submission_and_preserves_v3_checkpoint(tmp_path: Path) -> None:
    write_kernel_bundle(tmp_path, repo_root=Path("."))
    src = (tmp_path / "kernel_main.py").read_text()
    assert "competitions submit" not in src.lower()
    assert "v3_stage1_checkpoint" in src
    assert "paired_transfer_result.json" in src
    assert "paired_transfer_predictions.csv" in src
    assert "EXPECTED_HOLDOUT_SOURCES" in src
    assert "build_transfer_split" in src
    assert "evaluate_source_transfer" in src
    assert "run_dct_v3_gate_a_gpu" in src
