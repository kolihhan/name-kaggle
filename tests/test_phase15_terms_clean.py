import pandas as pd

from scripts.phase15_terms_clean import (
    BLOCKED_EXTERNAL_LABELS,
    EXPECTED_PHASE14_EXTERNAL_ROWS,
    EXPECTED_TERMS_CLEAN_ROWS,
    build_terms_clean_subset,
)


def test_phase15_blocked_labels_are_frozen_before_cv():
    assert BLOCKED_EXTERNAL_LABELS == {
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "glm-5.3",
        "glm-5.3-flash",
        "kimi-k3",
    }
    assert EXPECTED_PHASE14_EXTERNAL_ROWS == 43_781
    assert EXPECTED_TERMS_CLEAN_ROWS == 26_025


def test_phase15_filter_removes_only_blocked_external_labels():
    frame = pd.DataFrame(
        {
            "prompt_id": ["p1", "p1", "p2", "p2", "p3"],
            "text": ["a", "b", "c", "d", "e"],
            "model": [
                "gpt-5.5",
                "gemini-3.7-flash",
                "claude-opus-5",
                "glm-5.3",
                "qwen3.8-max",
            ],
            "source": ["s"] * 5,
        }
    )
    clean = build_terms_clean_subset(frame, validate_full_contract=False)
    assert clean["model"].tolist() == ["gpt-5.5", "claude-opus-5", "qwen3.8-max"]
    assert clean["text"].tolist() == ["a", "c", "e"]
    assert not set(clean["model"]) & BLOCKED_EXTERNAL_LABELS
