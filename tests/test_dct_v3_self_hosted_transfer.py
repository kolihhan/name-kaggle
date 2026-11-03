import json
from pathlib import Path

import pytest

from scripts.dct_v3_self_hosted_transfer import (
    EXPECTED_ELIGIBLE_ROWS,
    EXPECTED_HOLDOUT_SOURCES,
    validate_result_payload,
)


def _valid_payload():
    return {
        "candidate": "dct-temporal-fingerprint-v3",
        "scope": "paired_external_source_transfer_no_official_rows_no_public_lb",
        "holdout_sources": list(EXPECTED_HOLDOUT_SOURCES),
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "v3_stage1_checkpoint_preserved": True,
        "prediction_rows": EXPECTED_ELIGIBLE_ROWS,
        "decision": "pass",
        "transfer_gate": {"decision": {"status": "pass"}},
    }


def test_validate_result_payload_accepts_exact_frozen_lineage():
    payload = _valid_payload()
    summary = validate_result_payload(payload)
    assert summary["decision"] == "pass"
    assert summary["prediction_rows"] == EXPECTED_ELIGIBLE_ROWS
    assert summary["submission_authorized"] is False


def test_validate_result_payload_rejects_official_row_contamination():
    payload = _valid_payload()
    payload["official_rows_scored"] = 1
    with pytest.raises(RuntimeError, match="official_rows_scored"):
        validate_result_payload(payload)


def test_validate_result_payload_rejects_holdout_drift():
    payload = _valid_payload()
    payload["holdout_sources"] = ["other"]
    with pytest.raises(RuntimeError, match="holdout_sources"):
        validate_result_payload(payload)


def test_validate_result_payload_rejects_public_lb_tuning():
    payload = _valid_payload()
    payload["public_lb_tuning"] = True
    with pytest.raises(RuntimeError, match="public_lb_tuning"):
        validate_result_payload(payload)


def test_result_contract_never_authorizes_submission():
    payload = _valid_payload()
    assert validate_result_payload(payload)["submission_authorized"] is False
