import pandas as pd
import pytest

from scripts.phase10_audit_sonnet5 import audit_frames, normalize_text


def test_normalize_text_is_case_whitespace_and_unicode_stable():
    assert normalize_text("  Ｈello\nWORLD  ") == "hello world"


def test_audit_keeps_missing_row_level_model_identity_unknown():
    external = pd.DataFrame(
        {
            "question": ["q1", "q2", "q3"],
            "answer": ["Alpha", " beta  ", "BETA"],
        }
    )
    official = pd.DataFrame({"text": ["alpha", "something else"]})

    report = audit_frames(external, official, expected_model="claude-sonnet-5")

    assert report["external"]["rows"] == 3
    assert report["external"]["null_or_empty_answers"] == 0
    assert report["external"]["normalized_duplicate_groups"] == 1
    assert report["official_answer_overlap"]["exact_overlap_count"] == 0
    assert report["official_answer_overlap"]["normalized_overlap_count"] == 1
    assert report["model_identity"]["gate"] == "UNKNOWN"
    assert report["decision"]["training_authorized"] is False


def test_audit_passes_row_level_identity_only_when_every_row_matches():
    external = pd.DataFrame(
        {
            "question": ["q1", "q2"],
            "answer": ["one", "two"],
            "model": ["claude-sonnet-5", "claude-sonnet-5"],
        }
    )
    official = pd.DataFrame({"text": ["other"]})

    report = audit_frames(external, official, expected_model="claude-sonnet-5")
    assert report["model_identity"]["gate"] == "PASS"


def test_audit_rejects_wrong_schema():
    external = pd.DataFrame({"prompt": ["q"], "response": ["a"]})
    official = pd.DataFrame({"text": ["x"]})

    with pytest.raises(ValueError, match="question.*answer"):
        audit_frames(external, official, expected_model="claude-sonnet-5")
