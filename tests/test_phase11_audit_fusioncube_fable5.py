import pandas as pd
import pytest

from scripts.phase11_audit_fusioncube_fable5 import (
    audit_response_rows,
    extract_fable5_responses,
    normalize_text,
)


def _row(*, category="coding", model="claude-fable-5", prompt="SECRET PROMPT", response="model response"):
    return {
        "category": category,
        "model": model,
        "prompt": prompt,
        "response": response,
    }


def test_extracts_response_only_and_excludes_source_defined_decoys():
    rows, stats = extract_fable5_responses(
        [
            _row(response="  Real response  "),
            _row(category="decoy", response="decoy response"),
        ],
        expected_model="claude-fable-5",
    )

    assert [row["text"] for row in rows] == ["Real response"]
    assert "SECRET PROMPT" not in rows[0]["text"]
    assert stats["rows_seen"] == 2
    assert stats["decoy_rows_excluded"] == 1
    assert stats["responses_extracted"] == 1


def test_accepts_singleton_usage_dict_for_expected_model():
    rows, stats = extract_fable5_responses(
        [
            _row(
                model={
                    "claude-fable-5": {
                        "inputTokens": 10,
                        "outputTokens": 20,
                    }
                },
                response="singleton dict response",
            )
        ],
        expected_model="claude-fable-5",
    )

    assert [row["text"] for row in rows] == ["singleton dict response"]
    assert stats["singleton_model_dict_rows"] == 1
    assert stats["ambiguous_model_rows_excluded"] == 0
    assert stats["responses_extracted"] == 1


def test_excludes_mixed_model_usage_dict_instead_of_guessing_identity():
    rows, stats = extract_fable5_responses(
        [
            _row(
                model={
                    "claude-fable-5": {"outputTokens": 10},
                    "claude-opus-4-8": {"outputTokens": 5},
                },
                response="ambiguous response",
            )
        ],
        expected_model="claude-fable-5",
    )

    assert rows == []
    assert stats["ambiguous_model_rows_excluded"] == 1
    assert stats["responses_extracted"] == 0


def test_rejects_wrong_model_identity_on_retained_row():
    with pytest.raises(ValueError, match="model identity"):
        extract_fable5_responses(
            [_row(model="claude-opus-5")],
            expected_model="claude-fable-5",
        )


def test_rejects_singleton_usage_dict_for_wrong_model():
    with pytest.raises(ValueError, match="model identity"):
        extract_fable5_responses(
            [_row(model={"claude-opus-4-8": {"outputTokens": 1}})],
            expected_model="claude-fable-5",
        )


def test_empty_response_is_counted_and_not_used():
    rows, stats = extract_fable5_responses(
        [_row(response="   ")],
        expected_model="claude-fable-5",
    )

    assert rows == []
    assert stats["empty_response_rows"] == 1


def test_audit_blocks_official_or_cross_label_overlap():
    responses = pd.DataFrame({"text": [" Alpha ", "Beta"]})
    official = pd.DataFrame({"text": ["alpha"]})
    admitted = pd.DataFrame({"text": ["BETA"], "model": ["gpt-5.6-sol"]})

    report = audit_response_rows(responses, official, admitted)

    assert normalize_text("  ＡLPHA\n") == "alpha"
    assert report["official_overlap"]["normalized"] == 1
    assert report["cross_label_overlap"]["normalized"] == 1
    assert report["decision"]["training_authorized"] is False


def test_audit_authorizes_clean_unique_response_only_rows():
    responses = pd.DataFrame({"text": ["one unique fable response", "another fable response"]})
    official = pd.DataFrame({"text": ["official answer"]})
    admitted = pd.DataFrame({"text": ["other model answer"], "model": ["gpt-5.6-sol"]})

    report = audit_response_rows(responses, official, admitted)

    assert report["responses"]["unique_normalized"] == 2
    assert report["official_overlap"]["normalized"] == 0
    assert report["cross_label_overlap"]["normalized"] == 0
    assert report["decision"]["training_authorized"] is True
