import numpy as np
import pandas as pd
import pytest

from scripts.run_phase9_precompetition_specialist import (
    PHASE9_SPECIALIST_LABELS,
    assert_outside_recall_invariant,
    gate_predictions,
    validate_precompetition_external,
)

EXPECTED = {
    "claude-opus-4-6",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
    "qwen3.8-max",
}


def test_specialist_label_set_is_frozen():
    assert PHASE9_SPECIALIST_LABELS == EXPECTED


def test_gate_only_changes_rows_baseline_predicted_inside_specialist_set():
    baseline = np.array(["glm-5.3", "claude-fable-5", "qwen3.8-max"], dtype=object)
    specialist = np.array(["qwen3.8-max", "glm-5.3", "kimi-k3"], dtype=object)
    result = gate_predictions(baseline, specialist, PHASE9_SPECIALIST_LABELS)
    assert result.tolist() == ["qwen3.8-max", "claude-fable-5", "kimi-k3"]


def test_gate_rejects_specialist_prediction_outside_frozen_set():
    baseline = np.array(["glm-5.3"], dtype=object)
    specialist = np.array(["claude-fable-5"], dtype=object)
    with pytest.raises(RuntimeError, match="outside specialist label set"):
        gate_predictions(baseline, specialist, PHASE9_SPECIALIST_LABELS)


def test_outside_recall_invariant_rejects_any_drift():
    before = {"glm-5.3": 0.2, "claude-fable-5": 0.4}
    after = {"glm-5.3": 0.3, "claude-fable-5": 0.39}
    with pytest.raises(RuntimeError, match="outside-specialist recall drift"):
        assert_outside_recall_invariant(before, after, specialist_labels={"glm-5.3"})


def test_external_contract_requires_exact_2921_rows_and_five_labels():
    counts = {
        "claude-opus-4-6": 1934,
        "glm-5.3": 200,
        "glm-5.3-flash": 284,
        "kimi-k3": 317,
        "qwen3.8-max": 186,
    }
    rows = []
    i = 0
    for label, n in counts.items():
        for _ in range(n):
            rows.append({"response_id": f"r{i}", "text": f"text-{i}", "model": label})
            i += 1
    external = pd.DataFrame(rows)
    validate_precompetition_external(external)

    with pytest.raises(RuntimeError, match="2921"):
        validate_precompetition_external(external.iloc[:-1])
