import pandas as pd
import pytest

from scripts.phase12_audit_healthbench_anthropic import (
    OPUS_LABEL,
    SONNET_LABEL,
    audit_response_rows,
    extract_target_responses,
    normalize_text,
)


def _row(model, prompt_id, response_text="model response", **extra):
    return {
        "model": model,
        "prompt_id": prompt_id,
        "response_text": response_text,
        "stop_reason": extra.get("stop_reason"),
        "is_refusal": extra.get("is_refusal", False),
        "correction": extra.get("correction"),
    }


def test_extracts_response_only_for_targets_and_excludes_empty_refusals():
    rows, stats = extract_target_responses(
        [
            _row(OPUS_LABEL, "o1", "  Opus answer  "),
            _row(OPUS_LABEL, "o2", "", stop_reason="refusal", is_refusal=True),
            _row(SONNET_LABEL, "s1", "Sonnet answer"),
            _row("gpt-5.5", "g1", "not a target"),
        ],
        expected_raw_rows={OPUS_LABEL: 2, SONNET_LABEL: 1},
    )

    assert [row["text"] for row in rows] == ["Opus answer", "Sonnet answer"]
    assert [row["model"] for row in rows] == [OPUS_LABEL, SONNET_LABEL]
    assert stats["raw_rows_by_model"] == {OPUS_LABEL: 2, SONNET_LABEL: 1}
    assert stats["empty_rows_by_model"] == {OPUS_LABEL: 1, SONNET_LABEL: 0}
    assert stats["retained_rows_by_model"] == {OPUS_LABEL: 1, SONNET_LABEL: 1}


def test_rejects_duplicate_prompt_ids_within_target_model():
    records = [
        _row(OPUS_LABEL, "dup", "a"),
        _row(OPUS_LABEL, "dup", "b"),
        _row(SONNET_LABEL, "s1", "c"),
    ]

    with pytest.raises(ValueError, match="prompt_id"):
        extract_target_responses(
            records,
            expected_raw_rows={OPUS_LABEL: 2, SONNET_LABEL: 1},
        )


def test_rejects_raw_count_drift():
    with pytest.raises(ValueError, match="raw row count"):
        extract_target_responses(
            [_row(OPUS_LABEL, "o1", "a"), _row(SONNET_LABEL, "s1", "b")],
            expected_raw_rows={OPUS_LABEL: 2, SONNET_LABEL: 1},
        )


def test_audit_blocks_official_existing_external_and_candidate_cross_label_overlap():
    responses = pd.DataFrame(
        {
            "text": [" Alpha ", "Beta", "BETA"],
            "model": [OPUS_LABEL, OPUS_LABEL, SONNET_LABEL],
            "prompt_id": ["o1", "o2", "s1"],
        }
    )
    official = pd.DataFrame({"text": ["alpha"]})
    admitted = pd.DataFrame({"text": [" beta "]})

    report = audit_response_rows(responses, official, admitted)

    assert normalize_text("  ＡLPHA\n") == "alpha"
    assert report["official_overlap"]["normalized"] == 1
    assert report["existing_external_overlap"]["normalized"] == 1
    assert report["candidate_cross_label_overlap"]["normalized"] == 1
    assert report["decision"]["training_authorized"] is False


def test_audit_authorizes_clean_unique_two_label_responses():
    responses = pd.DataFrame(
        {
            "text": ["unique opus answer", "unique sonnet answer"],
            "model": [OPUS_LABEL, SONNET_LABEL],
            "prompt_id": ["o1", "s1"],
        }
    )
    official = pd.DataFrame({"text": ["official answer"]})
    admitted = pd.DataFrame({"text": ["existing external"]})

    report = audit_response_rows(responses, official, admitted)

    assert report["response_duplicates"]["within_label_normalized_duplicate_groups"] == 0
    assert report["candidate_cross_label_overlap"]["normalized"] == 0
    assert report["official_overlap"]["normalized"] == 0
    assert report["existing_external_overlap"]["normalized"] == 0
    assert report["decision"]["training_authorized"] is True
