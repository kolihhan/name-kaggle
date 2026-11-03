from __future__ import annotations

import pandas as pd

from scripts.dct_v3_source_eval import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    evaluate_source_transfer,
)


def _frame(v3_better: bool = True) -> pd.DataFrame:
    rows = []
    labels = ["a", "b", "c"]
    sources = ["s1", "s2", "s3"]
    for s_idx, source in enumerate(sources):
        for p in range(20):
            for l_idx, label in enumerate(labels):
                truth = label
                v1 = label if (p + l_idx + s_idx) % 4 else labels[(l_idx + 1) % 3]
                if v3_better:
                    v3 = label if (p + l_idx + s_idx) % 7 else labels[(l_idx + 1) % 3]
                else:
                    v3 = v1
                rows.append(
                    {
                        "source": source,
                        "prompt_id": f"{source}-p{p}",
                        "model": truth,
                        "v1_prediction": v1,
                        "v3_prediction": v3,
                    }
                )
    return pd.DataFrame(rows)


def test_gate_constants_are_frozen() -> None:
    assert BOOTSTRAP_RESAMPLES == 5000
    assert BOOTSTRAP_SEED == 20261103


def test_clear_paired_improvement_passes() -> None:
    decision, details = evaluate_source_transfer(_frame(), resamples=400, seed=20261103)
    assert decision.status == "pass"
    assert decision.v3_balanced_accuracy > decision.v1_balanced_accuracy
    assert decision.bootstrap_ci_lower > 0
    assert decision.mean_source_delta > 0
    assert decision.negative_source_count <= 1
    assert decision.safety_veto_labels == ()
    assert details["gate"]["bootstrap_seed"] == 20261103


def test_no_improvement_fails() -> None:
    decision, _ = evaluate_source_transfer(_frame(v3_better=False), resamples=200, seed=20261103)
    assert decision.status == "fail"
    assert "v3_pooled_ba_not_above_v1" in decision.reasons
    assert "prompt_bootstrap_lower_bound_not_positive" in decision.reasons


def test_large_class_recall_drop_triggers_safety_veto() -> None:
    frame = _frame()
    # Make class a substantially worse while leaving enough rows for the veto.
    mask = frame["model"].eq("a")
    frame.loc[mask, "v3_prediction"] = "b"
    decision, _ = evaluate_source_transfer(frame, resamples=100, seed=20261103)
    assert decision.status == "fail"
    assert "a" in decision.safety_veto_labels
    assert "per_label_safety_veto" in decision.reasons
