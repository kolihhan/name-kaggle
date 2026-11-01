import numpy as np
import pandas as pd

from scripts.run_phase19_shadow_diagnostic import (
    ORIGINAL_PHASE7_SCORE,
    ORIGINAL_PHASE15_SCORE,
    prompt_bootstrap_delta,
    summarize_shadow_result,
)


def test_original_reference_delta_is_frozen():
    assert ORIGINAL_PHASE7_SCORE == 0.3700521998508576
    assert ORIGINAL_PHASE15_SCORE == 0.4053218990802883
    assert np.isclose(
        ORIGINAL_PHASE15_SCORE - ORIGINAL_PHASE7_SCORE,
        0.03526969922943074,
        rtol=0.0,
        atol=1e-15,
    )


def test_prompt_bootstrap_delta_is_deterministic_and_prompt_grouped():
    oof = pd.DataFrame(
        {
            "prompt_id": ["p1", "p1", "p2", "p2"],
            "model": ["a", "b", "a", "b"],
            "phase7_prediction": ["a", "a", "b", "b"],
            "phase15_prediction": ["a", "b", "a", "b"],
        }
    )
    first = prompt_bootstrap_delta(oof, seed=123, reps=200)
    second = prompt_bootstrap_delta(oof, seed=123, reps=200)
    assert first == second
    assert first["reps"] == 200
    assert first["prompt_groups"] == 2
    assert 0.0 <= first["probability_delta_gt_zero"] <= 1.0
    assert first["p025"] <= first["median"] <= first["p975"]


def test_summarize_shadow_result_reports_paired_delta_and_correctness():
    oof = pd.DataFrame(
        {
            "prompt_id": ["p1", "p1", "p2", "p2"],
            "model": ["a", "b", "a", "b"],
            "phase7_prediction": ["a", "a", "a", "b"],
            "phase15_prediction": ["a", "b", "b", "b"],
        }
    )
    summary = summarize_shadow_result(
        "shadow_a",
        oof,
        phase7_score=0.75,
        phase15_score=0.75,
        phase7_recalls={"a": 1.0, "b": 0.5},
        phase15_recalls={"a": 0.5, "b": 1.0},
        bootstrap_seed=7,
        bootstrap_reps=100,
    )
    assert summary["shadow"] == "shadow_a"
    assert summary["phase15_minus_phase7"] == 0.0
    assert summary["paired_correctness"] == {
        "both_correct": 2,
        "phase7_only_correct": 1,
        "phase15_only_correct": 1,
        "both_wrong": 0,
    }
    assert summary["per_model_recall_delta"] == {"a": -0.5, "b": 0.5}
