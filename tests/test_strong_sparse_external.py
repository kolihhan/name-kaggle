import pandas as pd
import pytest

from scripts.run_strong_sparse_external import (
    EXPECTED_EXTERNAL_LABELS,
    combine_fit_data,
    grouped_recall_means,
    validate_external,
)


def test_combine_fit_data_adds_external_only_to_training_rows():
    official = pd.DataFrame({"text": ["o1", "o2"], "model": ["a", "b"]})
    external = pd.DataFrame({"text": ["e1", "e2", "e3"], "model": ["a", "a", "b"]})

    texts, labels = combine_fit_data(official, external)

    assert texts == ["o1", "o2", "e1", "e2", "e3"]
    assert labels == ["a", "b", "a", "a", "b"]


def test_validate_external_enforces_identity_contract():
    external = pd.DataFrame(
        {
            "text": ["x", "y"],
            "model": ["a", "b"],
            "prompt_id": ["p1", "p2"],
        }
    )
    validate_external(
        external,
        official_labels={"a", "b", "c"},
        expected_rows=2,
        expected_prompts=2,
        expected_labels={"a", "b"},
    )

    bad = external.copy()
    bad.loc[1, "model"] = "unknown"
    with pytest.raises(RuntimeError, match="labels"):
        validate_external(
            bad,
            official_labels={"a", "b", "c"},
            expected_rows=2,
            expected_prompts=2,
            expected_labels={"a", "b"},
        )


def test_grouped_recall_means_separates_covered_and_uncovered_labels():
    recalls = {"a": 0.8, "b": 0.4, "c": 0.2}
    grouped = grouped_recall_means(recalls, covered_labels={"a", "b"})
    assert grouped["covered"] == pytest.approx(0.6)
    assert grouped["uncovered"] == pytest.approx(0.2)


def test_production_external_label_contract_is_the_audited_eight_labels():
    assert EXPECTED_EXTERNAL_LABELS == {
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-6-astra",
    }
