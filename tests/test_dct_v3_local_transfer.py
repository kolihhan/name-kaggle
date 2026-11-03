from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dct_v3_local_transfer import (
    EXPECTED_ELIGIBLE_ROWS,
    EXPECTED_HOLDOUT_SOURCES,
    EXPECTED_POOL_SHA256,
    validate_local_result_payload,
    verify_model_snapshot,
)


def test_frozen_local_transfer_constants() -> None:
    assert EXPECTED_HOLDOUT_SOURCES == ["alperen", "hailey", "willrice"]
    assert EXPECTED_ELIGIBLE_ROWS == 14506
    assert EXPECTED_POOL_SHA256 == "37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec"


def test_verify_model_snapshot_accepts_exact_qwen3_8b_shape(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36}),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"placeholder")
    summary = verify_model_snapshot(tmp_path)
    assert summary["model_type"] == "qwen3"
    assert summary["hidden_size"] == 4096
    assert summary["num_hidden_layers"] == 36


def test_verify_model_snapshot_rejects_wrong_shape(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "hidden_size": 2048, "num_hidden_layers": 36}),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"placeholder")
    with pytest.raises(RuntimeError, match="unexpected local model config"):
        verify_model_snapshot(tmp_path)


def test_validate_local_result_payload_accepts_clean_frozen_lineage() -> None:
    payload = {
        "candidate": "dct-temporal-fingerprint-v3",
        "scope": "paired_external_source_transfer_no_official_rows_no_public_lb",
        "pool_sha256": EXPECTED_POOL_SHA256,
        "holdout_sources": EXPECTED_HOLDOUT_SOURCES,
        "eligible_rows": EXPECTED_ELIGIBLE_ROWS,
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "v3_stage1_checkpoint_preserved": True,
        "decision": "pass",
    }
    validate_local_result_payload(payload)


def test_validate_local_result_payload_rejects_contamination() -> None:
    payload = {
        "candidate": "dct-temporal-fingerprint-v3",
        "scope": "paired_external_source_transfer_no_official_rows_no_public_lb",
        "pool_sha256": EXPECTED_POOL_SHA256,
        "holdout_sources": EXPECTED_HOLDOUT_SOURCES,
        "eligible_rows": EXPECTED_ELIGIBLE_ROWS,
        "official_rows_scored": 1,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "v3_stage1_checkpoint_preserved": True,
        "decision": "pass",
    }
    with pytest.raises(RuntimeError, match="official rows"):
        validate_local_result_payload(payload)
