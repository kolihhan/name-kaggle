from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_two_stage_lora24_gate_b import (
    EXPECTED_FOLD_FINGERPRINT,
    EXPECTED_TRAIN_SHA256,
    gate_b_decision,
    gate_b_training_spec,
    paired_prompt_bootstrap_delta,
    split_official_fold,
    validate_gate_a_checkpoint_meta,
)


MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"


def _checkpoint_meta() -> dict:
    return {
        "candidate": "two-stage-lora24-v1",
        "condition": "A1",
        "model_id": "Qwen/Qwen3-8B-Base",
        "model_revision": MODEL_REVISION,
        "labels": [f"m{i:02d}" for i in range(18)],
        "training_spec": {"seed": 20261101},
        "implementation_invariants": {
            "stock_qwen_final_norm_retained": True,
            "triplet_padding": False,
            "exact_one_epoch": True,
        },
    }


def test_gate_b_training_spec_matches_frozen_contract():
    spec = gate_b_training_spec()
    assert spec["backbone"] == "Qwen3-8B-Base"
    assert spec["keep_layers"] == 24
    assert spec["pooled_layers"] == [2, 4, 6, 8, 12, 16, 20, 24]
    assert spec["max_length_before_eos"] == 447
    assert spec["epochs"] == 2
    assert spec["adapter_lr"] == 5e-5
    assert spec["head_lr"] == 5e-4
    assert spec["weight_decay"] == 0.01
    assert spec["warmup_fraction"] == 0.10
    assert spec["gradient_accumulation_triplets"] == 6
    assert spec["gradient_clip"] == 1.0
    assert spec["seed"] == 20261101
    assert spec["replay_same_triplet_order_each_epoch"] is True
    assert spec["primary_head"] == "plain"
    assert spec["centered_head_role"] == "diagnostic_only"


def test_gate_b_pins_official_train_and_fold_fingerprints():
    assert EXPECTED_TRAIN_SHA256 == "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
    assert EXPECTED_FOLD_FINGERPRINT == "bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d"


def test_checkpoint_meta_accepts_exact_gate_a_a1_lineage():
    validate_gate_a_checkpoint_meta(_checkpoint_meta())


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("candidate", "other-candidate"),
        ("condition", "A0"),
        ("model_id", "Qwen/Qwen3-14B-Base"),
        ("model_revision", "wrong-revision"),
    ],
)
def test_checkpoint_meta_rejects_substitution(field: str, bad_value: str):
    meta = _checkpoint_meta()
    meta[field] = bad_value
    with pytest.raises(ValueError):
        validate_gate_a_checkpoint_meta(meta)


def test_checkpoint_meta_rejects_missing_gate_a_runtime_invariant():
    meta = _checkpoint_meta()
    meta["implementation_invariants"]["triplet_padding"] = True
    with pytest.raises(ValueError):
        validate_gate_a_checkpoint_meta(meta)


def test_split_official_fold_is_prompt_isolated():
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "prompt_id": ["p0", "p0", "p1", "p1", "p2", "p2"],
            "model": ["a", "b", "a", "b", "a", "b"],
            "fold_id": [0, 0, 1, 1, 2, 2],
        }
    )
    fit_idx, valid_idx = split_official_fold(frame, 1)
    assert valid_idx.tolist() == [2, 3]
    assert fit_idx.tolist() == [0, 1, 4, 5]
    assert set(frame.iloc[fit_idx]["prompt_id"]).isdisjoint(
        set(frame.iloc[valid_idx]["prompt_id"])
    )


def test_split_official_fold_rejects_prompt_leakage():
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "prompt_id": ["shared", "shared", "other"],
            "model": ["a", "b", "a"],
            "fold_id": [0, 1, 1],
        }
    )
    with pytest.raises(RuntimeError, match="prompt leakage"):
        split_official_fold(frame, 0)


def test_paired_prompt_bootstrap_is_deterministic_and_prompt_grouped():
    frame = pd.DataFrame(
        {
            "prompt_id": ["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"],
            "model": ["a", "b"] * 4,
            "candidate_prediction": ["a", "b", "a", "b", "a", "a", "a", "b"],
            "phase7_prediction": ["a", "a", "a", "b", "a", "b", "b", "b"],
        }
    )
    first = paired_prompt_bootstrap_delta(frame, n_resamples=5000, seed=20261101)
    second = paired_prompt_bootstrap_delta(frame, n_resamples=5000, seed=20261101)
    assert first == second
    assert first["n_resamples"] == 5000
    assert first["seed"] == 20261101
    assert first["sampling_unit"] == "prompt_id"
    assert first["prompt_groups"] == 4
    assert first["ci_lower"] <= first["point_delta"] <= first["ci_upper"]


def test_gate_b_decision_passes_only_when_all_frozen_conditions_pass():
    decision = gate_b_decision(
        candidate_ba=0.41,
        phase7_ba=0.37,
        bootstrap_ci_lower=0.01,
        candidate_fold_ba=[0.40, 0.42, 0.39, 0.41, 0.43],
        phase7_fold_ba=[0.38, 0.34, 0.37, 0.39, 0.37],
        candidate_recall={"a": 0.50, "b": 0.40},
        phase7_recall={"a": 0.45, "b": 0.35},
    )
    assert decision["status"] == "pass"
    assert decision["candidate_beats_phase7"] is True
    assert decision["bootstrap_ci_lower_positive"] is True
    assert decision["fold_wins"] == 4
    assert decision["class_recall_safety_veto"] is False


def test_gate_b_decision_fails_class_recall_safety_veto():
    decision = gate_b_decision(
        candidate_ba=0.41,
        phase7_ba=0.37,
        bootstrap_ci_lower=0.01,
        candidate_fold_ba=[0.40, 0.42, 0.39, 0.41, 0.43],
        phase7_fold_ba=[0.38, 0.34, 0.37, 0.39, 0.37],
        candidate_recall={"stable": 0.30, "damaged": 0.20},
        phase7_recall={"stable": 0.40, "damaged": 0.40},
    )
    assert decision["status"] == "fail"
    assert decision["class_recall_safety_veto"] is True
    assert decision["veto_labels"] == ["damaged"]
