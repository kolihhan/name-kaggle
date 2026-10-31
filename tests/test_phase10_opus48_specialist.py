import numpy as np
import pandas as pd

from scripts.run_phase10_opus48_specialist import (
    PHASE10_COVERED_LABELS,
    PHASE7_OPUS48_RECALL,
    PHASE7_SCORE,
    gate_predictions,
    promotion_decision,
    specialist_training_rows,
)


def test_phase10_expands_phase7_to_exactly_nine_classes():
    assert len(PHASE10_COVERED_LABELS) == 9
    assert "claude-opus-4-8" in PHASE10_COVERED_LABELS
    assert "claude-fable-5" not in PHASE10_COVERED_LABELS
    assert PHASE7_SCORE == 0.3700521998508576
    assert PHASE7_OPUS48_RECALL == 0.20666666666666667


def test_gate_changes_only_anchor_predictions_inside_nine_class_set():
    anchor = np.array(
        ["claude-opus-4-8", "claude-fable-5", "gpt-5.6-sol"],
        dtype=object,
    )
    specialist = np.array(
        ["gpt-5.5", "gemini-3.8-flash", "claude-opus-4-8"],
        dtype=object,
    )

    result = gate_predictions(anchor, specialist, PHASE10_COVERED_LABELS)

    assert result.tolist() == ["gpt-5.5", "claude-fable-5", "claude-opus-4-8"]


def test_training_rows_include_only_covered_official_plus_both_external_sets():
    official = pd.DataFrame(
        {
            "text": ["official opus", "official gpt", "official fable"],
            "model": ["claude-opus-4-8", "gpt-5.6-sol", "claude-fable-5"],
        }
    )
    phase7_external = pd.DataFrame(
        {"text": ["external gpt"], "model": ["gpt-5.6-sol"]}
    )
    opus_external = pd.DataFrame(
        {"text": ["external opus"], "model": ["claude-opus-4-8"]}
    )

    texts, labels = specialist_training_rows(official, phase7_external, opus_external)

    assert texts == ["official opus", "official gpt", "external gpt", "external opus"]
    assert labels == [
        "claude-opus-4-8",
        "gpt-5.6-sol",
        "gpt-5.6-sol",
        "claude-opus-4-8",
    ]


def test_promotion_requires_both_overall_and_opus48_improvement_and_invariant():
    assert promotion_decision(0.371, 0.22, uncovered_invariant=True)["promote_phase10"] is True
    assert promotion_decision(PHASE7_SCORE, 0.22, uncovered_invariant=True)["promote_phase10"] is False
    assert promotion_decision(0.371, PHASE7_OPUS48_RECALL, uncovered_invariant=True)["promote_phase10"] is False
    assert promotion_decision(0.371, 0.22, uncovered_invariant=False)["promote_phase10"] is False
