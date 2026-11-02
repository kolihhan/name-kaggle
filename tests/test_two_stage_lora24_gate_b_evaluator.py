from __future__ import annotations

import inspect

import pandas as pd
import pytest

from scripts.run_two_stage_lora24_gate_b import (
    PHASE7_METRIC_ATOL,
    PHASE7_SEED,
    evaluate_gate_b,
    regenerate_phase7_oof,
    validate_phase7_reference,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_rows = []
    phase7_rows = []
    candidate_rows = []
    row_id = 0
    for fold in range(5):
        for offset, label in enumerate(("a", "b")):
            row = {
                "id": f"id{row_id}",
                "prompt_id": f"p{fold}",
                "model": label,
                "fold_id": fold,
            }
            train_rows.append(row)
            phase7_rows.append(
                {
                    **row,
                    "prediction": "a",
                }
            )
            candidate_rows.append(
                {
                    **row,
                    "plain_prediction": label,
                    "centered_prediction": "a",
                }
            )
            row_id += 1

    train = pd.DataFrame(train_rows)
    phase7_oof = pd.DataFrame(phase7_rows)
    candidate_oof = pd.DataFrame(candidate_rows)
    phase7_report = {
        "ensemble": {
            "result": {
                "oof_rows": 10,
                "balanced_accuracy": 0.5,
                "folds": [0.5, 0.5, 0.5, 0.5, 0.5],
                "per_model_recall": {"a": 1.0, "b": 0.0},
            }
        }
    }
    return train, phase7_oof, candidate_oof, phase7_report


def test_phase7_preflight_constants_are_frozen():
    assert PHASE7_SEED == 20261031
    assert PHASE7_METRIC_ATOL == 1e-12


def test_regenerate_phase7_oof_uses_frozen_pipeline_not_aggregate_inference():
    source = inspect.getsource(regenerate_phase7_oof)
    assert "run_anchor_cv" in source
    assert "run_specialist_cv" in source
    assert "build_phase7_report" in source
    assert "validate_phase7_reference" in source
    assert "external" in inspect.signature(regenerate_phase7_oof).parameters


def test_validate_phase7_reference_requires_exact_row_alignment_and_frozen_metrics():
    train, phase7_oof, _, phase7_report = _frames()
    validated = validate_phase7_reference(train, phase7_oof, phase7_report)
    assert validated["balanced_accuracy"] == pytest.approx(0.5, abs=1e-12)
    assert validated["folds"] == pytest.approx([0.5] * 5, abs=1e-12)
    assert validated["per_model_recall"] == {"a": 1.0, "b": 0.0}

    reordered = phase7_oof.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="row alignment"):
        validate_phase7_reference(train, reordered, phase7_report)

    drifted = {
        "ensemble": {
            "result": {
                **phase7_report["ensemble"]["result"],
                "balanced_accuracy": 0.500000000002,
            }
        }
    }
    with pytest.raises(RuntimeError, match="frozen Phase7"):
        validate_phase7_reference(train, phase7_oof, drifted)


def test_evaluate_gate_b_reports_frozen_diagnostics_bootstrap_and_decision():
    train, phase7_oof, candidate_oof, phase7_report = _frames()
    report = evaluate_gate_b(
        candidate_oof,
        train,
        phase7_oof,
        phase7_report,
        n_resamples=5000,
        seed=20261101,
    )

    assert report["candidate"]["plain"]["balanced_accuracy"] == pytest.approx(1.0)
    assert report["candidate"]["plain"]["folds"] == pytest.approx([1.0] * 5)
    assert report["candidate"]["centered"]["balanced_accuracy"] == pytest.approx(0.5)
    assert report["phase7"]["balanced_accuracy"] == pytest.approx(0.5)
    assert report["delta"]["balanced_accuracy"] == pytest.approx(0.5)
    assert report["delta"]["folds"] == pytest.approx([0.5] * 5)
    assert report["delta"]["per_model_recall"] == {"a": 0.0, "b": 1.0}
    assert report["paired_rows"]["disagreement_rate"] == pytest.approx(0.5)
    assert report["paired_rows"]["candidate_only_correct"] == 5
    assert report["paired_rows"]["phase7_only_correct"] == 0

    bootstrap = report["bootstrap"]
    assert bootstrap["n_resamples"] == 5000
    assert bootstrap["seed"] == 20261101
    assert bootstrap["point_delta"] == pytest.approx(0.5)
    assert bootstrap["ci_lower"] > 0.0
    assert bootstrap["ci_upper"] >= bootstrap["ci_lower"]

    decision = report["decision"]
    assert decision["status"] == "pass"
    assert decision["candidate_beats_phase7"] is True
    assert decision["bootstrap_ci_lower_positive"] is True
    assert decision["beats_phase7_on_at_least_3_of_5_folds"] is True
    assert decision["class_recall_safety_veto"] is False


def test_evaluate_gate_b_rejects_candidate_row_misalignment():
    train, phase7_oof, candidate_oof, phase7_report = _frames()
    broken = candidate_oof.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="candidate OOF row alignment"):
        evaluate_gate_b(broken, train, phase7_oof, phase7_report)
