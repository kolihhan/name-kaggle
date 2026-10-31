import pandas as pd
import pytest

from scripts.phase10_audit_arc_opus48 import (
    audit_response_rows,
    extract_assistant_responses,
    normalize_text,
)


def _attempt(model="claude-opus-4-8", assistant="model output", user="secret prompt", answer=None):
    return {
        "answer": answer if answer is not None else [[1, 2], [3, 4]],
        "metadata": {
            "model": model,
            "choices": [
                {"message": {"role": "user", "content": user}},
                {"message": {"role": "assistant", "content": assistant}},
            ],
        },
    }


def test_extracts_only_assistant_content_and_never_prompt_or_answer():
    records = [{"attempt_1": _attempt(assistant="  Final reasoning  ", user="DO NOT LEAK") }]

    rows, stats = extract_assistant_responses(
        records,
        setting="high",
        source_file="task.json",
        expected_model="claude-opus-4-8",
    )

    assert [row["text"] for row in rows] == ["Final reasoning"]
    assert rows[0]["setting"] == "high"
    assert rows[0]["source_file"] == "task.json"
    assert "DO NOT LEAK" not in rows[0]["text"]
    assert "1" not in rows[0]["text"]
    assert stats["attempts_seen"] == 1
    assert stats["responses_extracted"] == 1
    assert stats["missing_assistant_response"] == 0


def test_rejects_any_attempt_with_wrong_model_identity():
    records = [{"attempt_1": _attempt(model="claude-opus-5")}]

    with pytest.raises(ValueError, match="model identity"):
        extract_assistant_responses(
            records,
            setting="high",
            source_file="task.json",
            expected_model="claude-opus-4-8",
        )


def test_missing_assistant_message_is_counted_but_not_used_as_training_text():
    attempt = _attempt()
    attempt["metadata"]["choices"] = [
        {"message": {"role": "user", "content": "prompt only"}},
    ]
    records = [{"attempt_1": attempt}]

    rows, stats = extract_assistant_responses(
        records,
        setting="medium",
        source_file="task.json",
        expected_model="claude-opus-4-8",
    )

    assert rows == []
    assert stats["attempts_seen"] == 1
    assert stats["missing_assistant_response"] == 1


def test_audit_reports_normalized_overlap_and_cross_label_duplicates():
    responses = pd.DataFrame(
        {
            "text": ["  Alpha  ", "Beta", "BETA"],
            "setting": ["high", "medium", "max"],
            "source_file": ["a.json", "b.json", "c.json"],
        }
    )
    official = pd.DataFrame({"text": ["alpha", "unrelated"]})
    admitted = pd.DataFrame(
        {
            "text": [" beta ", "other"],
            "model": ["gpt-5.6-sol", "gemini-3.8-flash"],
        }
    )

    report = audit_response_rows(responses, official, admitted)

    assert normalize_text("  ＡLPHA\n") == "alpha"
    assert report["official_overlap"]["exact"] == 0
    assert report["official_overlap"]["normalized"] == 1
    assert report["cross_label_overlap"]["exact"] == 0
    assert report["cross_label_overlap"]["normalized"] == 1
    assert report["response_duplicates"]["normalized_duplicate_groups"] == 1
    assert report["decision"]["training_authorized"] is False


def test_audit_authorizes_clean_response_only_rows():
    responses = pd.DataFrame(
        {
            "text": ["one unique response", "another unique response"],
            "setting": ["high", "max"],
            "source_file": ["a.json", "b.json"],
        }
    )
    official = pd.DataFrame({"text": ["official text"]})
    admitted = pd.DataFrame({"text": ["existing external"], "model": ["gpt-5.6-sol"]})

    report = audit_response_rows(responses, official, admitted)

    assert report["official_overlap"]["normalized"] == 0
    assert report["cross_label_overlap"]["normalized"] == 0
    assert report["decision"]["training_authorized"] is True
