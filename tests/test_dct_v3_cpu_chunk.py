import pytest

from scripts.dct_v3_cpu_chunk import (
    ACCUMULATION_SIZE,
    CANDIDATE,
    SCIENTIFIC_BASE_SHA,
    SCOPE,
    chunk_contract,
    next_chunk_boundary,
    validate_chunk_manifest,
)


def _manifest(**overrides):
    payload = {
        "candidate": CANDIDATE,
        "scope": SCOPE,
        "scientific_base_sha": SCIENTIFIC_BASE_SHA,
        "accumulation_size": ACCUMULATION_SIZE,
        "cursor": 7,
        "accumulation_position": 1,
        "total_microbatches": 100,
        "generation": 2,
        "predecessor_state_sha256": "a" * 64,
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "firewall_rows_scored": 0,
        "public_lb_tuning": False,
        "submission_authorized": False,
    }
    payload.update(overrides)
    return payload


def test_chunk_contract_pins_frozen_scientific_and_zero_scoring_scope():
    contract = chunk_contract()
    assert contract["candidate"] == "dct-temporal-fingerprint-v3"
    assert contract["scientific_base_sha"] == "2d0a86fffa880709ad25c8ee9153f2b2838617a8"
    assert contract["accumulation_size"] == 6
    assert contract["official_rows_scored"] == 0
    assert contract["competition_test_rows_used"] == 0
    assert contract["firewall_rows_scored"] == 0
    assert contract["public_lb_tuning"] is False
    assert contract["submission_authorized"] is False


def test_next_chunk_boundary_is_monotonic_and_never_exceeds_total():
    assert next_chunk_boundary(cursor=7, total_microbatches=100, microbatch_budget=1) == 8
    assert next_chunk_boundary(cursor=98, total_microbatches=100, microbatch_budget=10) == 100
    assert next_chunk_boundary(cursor=100, total_microbatches=100, microbatch_budget=10) == 100
    with pytest.raises(ValueError):
        next_chunk_boundary(cursor=7, total_microbatches=100, microbatch_budget=0)


def test_manifest_accepts_mid_accumulation_cursor_and_reports_progress():
    summary = validate_chunk_manifest(
        _manifest(), expected_predecessor_sha256="a" * 64
    )
    assert summary["cursor"] == 7
    assert summary["accumulation_position"] == 1
    assert summary["generation"] == 2
    assert summary["complete"] is False


def test_manifest_accepts_completed_partial_final_group_with_zero_accumulation_position():
    summary = validate_chunk_manifest(
        _manifest(
            cursor=100,
            total_microbatches=100,
            accumulation_position=0,
        )
    )
    assert summary["complete"] is True


def test_manifest_rejects_stale_predecessor_lineage():
    with pytest.raises(RuntimeError, match="predecessor"):
        validate_chunk_manifest(
            _manifest(predecessor_state_sha256="b" * 64),
            expected_predecessor_sha256="a" * 64,
        )


def test_manifest_rejects_scoring_contamination_and_cursor_drift():
    with pytest.raises(RuntimeError, match="official_rows_scored"):
        validate_chunk_manifest(_manifest(official_rows_scored=1))
    with pytest.raises(RuntimeError, match="accumulation position"):
        validate_chunk_manifest(_manifest(accumulation_position=4))
    with pytest.raises(RuntimeError, match="scientific_base_sha"):
        validate_chunk_manifest(_manifest(scientific_base_sha="wrong"))
