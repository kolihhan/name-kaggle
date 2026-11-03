import pandas as pd
import pytest

from scripts.analyze_v1_mechanism import analyze_frames, family_of_model, render_markdown


def _frames():
    labels = [
        "gpt-5.5", "gpt-5.5", "gpt-5.6-sol", "gpt-5.6-sol",
        "claude-opus-5", "claude-opus-5", "gemini-3.7-flash", "gemini-3.7-flash",
    ]
    lengths = [2, 20, 3, 21, 4, 22, 5, 23]
    train = pd.DataFrame({
        "id": [f"r{i}" for i in range(8)],
        "text": ["x " * n for n in lengths],
        "model": labels,
    })
    gate = pd.DataFrame({
        "id": train["id"],
        "model": labels,
        "plain_prediction": [
            "gpt-5.6-sol", "gpt-5.5", "claude-opus-5", "gpt-5.6-sol",
            "claude-opus-5", "claude-opus-5", "gemini-3.6-flash", "gemini-3.7-flash",
        ],
        "centered_prediction": [
            "gpt-5.5", "gpt-5.5", "gpt-5.6-sol", "gpt-5.6-sol",
            "gpt-5.5", "claude-opus-5", "gemini-3.6-flash", "gemini-3.7-flash",
        ],
    })
    sparse = pd.DataFrame({
        "id": train["id"],
        "model": labels,
        "prediction": [
            "gpt-5.6-sol", "gpt-5.5", "gpt-5.6-sol", "gpt-5.5",
            "claude-opus-5", "claude-opus-5", "gemini-3.7-flash", "claude-opus-5",
        ],
    })
    return train, gate, sparse


def test_family_of_model_groups_sibling_versions():
    assert family_of_model("gpt-5.6-sol") == "gpt"
    assert family_of_model("claude-opus-5") == "claude"
    assert family_of_model("gemini-3.7-flash") == "gemini"
    assert family_of_model("glm-5.3") == "glm"
    assert family_of_model("kimi-k3") == "kimi"
    assert family_of_model("qwen3.8-max") == "qwen"


def test_analyze_frames_rejects_misaligned_ids():
    train, gate, sparse = _frames()
    sparse = sparse.iloc[:-1].copy()
    with pytest.raises(ValueError, match="id sets differ"):
        analyze_frames(train, gate, sparse)


def test_analyze_frames_counts_mechanism_signals():
    train, gate, sparse = _frames()
    report = analyze_frames(train, gate, sparse)

    assert report["rows"] == 8
    assert report["plain_errors"]["count"] == 3
    assert report["plain_errors"]["same_family_count"] == 2
    assert report["plain_errors"]["cross_family_count"] == 1
    assert report["plain_errors"]["same_family_rate"] == pytest.approx(2 / 3)

    centered = report["complementarity"]["centered_vs_plain"]
    assert centered["rescues"] == 2
    assert centered["breaks"] == 1
    assert centered["rescued_same_family_plain_errors"] == 1
    assert centered["rescued_cross_family_plain_errors"] == 1

    sparse_diag = report["complementarity"]["sparse_vs_plain"]
    assert sparse_diag["rescues"] == 2
    assert sparse_diag["breaks"] == 2
    assert sparse_diag["rescued_same_family_plain_errors"] == 1
    assert sparse_diag["rescued_cross_family_plain_errors"] == 1

    patterns = report["agreement_patterns"]
    assert sum(item["rows"] for item in patterns.values()) == 8
    assert patterns["all_agree"]["rows"] == 2
    assert patterns["plain_centered_agree"]["rows"] == 3
    assert patterns["plain_sparse_agree"]["rows"] == 2
    assert patterns["centered_sparse_agree"]["rows"] == 1
    assert patterns["all_distinct"]["rows"] == 0

    within = report["length"]["within_model_median"]
    assert within["models_evaluable"] == 4
    assert within["models_with_positive_long_minus_short"] == 3
    assert within["macro_long_minus_short_accuracy"] == pytest.approx(0.75)


def test_top_confusions_are_deterministic_and_markdown_is_labeled_posthoc():
    train, gate, sparse = _frames()
    report = analyze_frames(train, gate, sparse)
    assert report["top_plain_confusions"][0] == {
        "true": "gemini-3.7-flash",
        "predicted": "gemini-3.6-flash",
        "count": 1,
    }
    markdown = render_markdown(report)
    assert "POST-HOC DIAGNOSTIC ONLY" in markdown
    assert "Same-family" in markdown
    assert "Within-model length" in markdown
