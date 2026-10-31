import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import recall_score

from scripts.run_gated_specialist import (
    EXPECTED_COVERED_LABELS,
    assert_uncovered_recall_invariant,
    gate_predictions,
    specialist_training_rows,
    validate_anchor_oof_alignment,
    validate_external,
)


def test_gate_only_replaces_anchor_predictions_inside_covered_set():
    anchor = np.array(
        ["claude-opus-5", "gpt-5.5", "qwen3.8-max", "gemini-3.6-flash"],
        dtype=object,
    )
    specialist = np.array(
        ["gpt-6-astra", "gpt-6-astra", "gpt-5.5", "gpt-5.6-luna"],
        dtype=object,
    )

    result = gate_predictions(anchor, specialist, EXPECTED_COVERED_LABELS)

    assert result.tolist() == [
        "claude-opus-5",
        "gpt-6-astra",
        "qwen3.8-max",
        "gpt-5.6-luna",
    ]


def test_gate_rejects_specialist_predictions_outside_covered_set():
    anchor = np.array(["gpt-5.5", "claude-opus-5"], dtype=object)
    specialist = np.array(["claude-opus-5", "gpt-6-astra"], dtype=object)

    with pytest.raises(RuntimeError, match="outside covered label set"):
        gate_predictions(anchor, specialist, EXPECTED_COVERED_LABELS)


def test_gate_preserves_true_uncovered_recall_by_construction():
    y_true = np.array(
        ["claude-opus-5", "claude-opus-5", "gpt-5.5", "gpt-5.5"], dtype=object
    )
    anchor = np.array(
        ["claude-opus-5", "gpt-5.5", "claude-opus-5", "gpt-5.5"], dtype=object
    )
    specialist = np.array(
        ["gpt-6-astra", "gpt-6-astra", "gpt-6-astra", "gpt-6-astra"],
        dtype=object,
    )
    covered = {"gpt-5.5", "gpt-6-astra"}

    ensemble = gate_predictions(anchor, specialist, covered)
    anchor_recall = recall_score(
        y_true, anchor, labels=["claude-opus-5"], average=None, zero_division=0
    )[0]
    ensemble_recall = recall_score(
        y_true, ensemble, labels=["claude-opus-5"], average=None, zero_division=0
    )[0]

    assert ensemble_recall == anchor_recall


def test_specialist_training_rows_excludes_uncovered_official_labels():
    official = pd.DataFrame(
        {
            "text": ["covered official", "uncovered official", "covered official 2"],
            "model": ["gpt-5.5", "claude-opus-5", "gpt-6-astra"],
        }
    )
    external = pd.DataFrame(
        {
            "text": ["external 1", "external 2"],
            "model": ["gpt-5.5", "gpt-6-astra"],
        }
    )

    texts, labels = specialist_training_rows(
        official, external, covered_labels={"gpt-5.5", "gpt-6-astra"}
    )

    assert texts == ["covered official", "covered official 2", "external 1", "external 2"]
    assert labels == ["gpt-5.5", "gpt-6-astra", "gpt-5.5", "gpt-6-astra"]


def test_validate_anchor_oof_alignment_requires_exact_id_and_fold_order():
    train = pd.DataFrame(
        {
            "id": ["a", "b"],
            "prompt_id": ["p1", "p2"],
            "model": ["gpt-5.5", "claude-opus-5"],
            "fold_id": [0, 1],
        }
    )
    anchor_oof = train.copy()
    anchor_oof["prediction"] = ["gpt-5.5", "claude-opus-5"]

    validate_anchor_oof_alignment(train, anchor_oof)

    misaligned = anchor_oof.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="anchor OOF row alignment"):
        validate_anchor_oof_alignment(train, misaligned)


def test_production_covered_label_contract_is_exactly_the_audited_eight():
    assert EXPECTED_COVERED_LABELS == {
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-6-astra",
    }


def test_validate_external_still_rejects_blank_required_fields():
    external = pd.DataFrame(
        {
            "text": ["x", "   "],
            "model": ["gpt-5.5", "gpt-6-astra"],
            "prompt_id": ["p1", "p2"],
        }
    )
    with pytest.raises(RuntimeError, match="text contains blank"):
        validate_external(
            external,
            official_labels={"gpt-5.5", "gpt-6-astra", "claude-opus-5"},
            expected_rows=2,
            expected_prompts=2,
            expected_labels={"gpt-5.5", "gpt-6-astra"},
        )


def test_uncovered_recall_invariance_checker_rejects_any_drift():
    anchor = {"gpt-5.5": 0.5, "claude-opus-5": 0.4, "qwen3.8-max": 0.2}
    ensemble = {"gpt-5.5": 0.6, "claude-opus-5": 0.4, "qwen3.8-max": 0.1}

    with pytest.raises(RuntimeError, match="uncovered recall drift"):
        assert_uncovered_recall_invariant(
            anchor,
            ensemble,
            covered_labels={"gpt-5.5"},
        )
