from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from scripts.run_style_baseline import (
    EXPECTED_FOLD_FINGERPRINT,
    EXPECTED_TRAIN_SHA256,
)
from scripts.run_two_stage_lora24_gate_a import gate_a_training_spec


CANDIDATE = "two-stage-lora24-v1"
MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
GATE_B_SEED = 20261101
BOOTSTRAP_RESAMPLES = 5000


def gate_b_training_spec() -> dict[str, object]:
    """Return the frozen Stage-2 training specification from the candidate contract."""
    spec = dict(gate_a_training_spec())
    spec.update(
        {
            "epochs": 2,
            "adapter_lr": 5e-5,
            "head_lr": 5e-4,
            "replay_same_triplet_order_each_epoch": True,
            "primary_head": "plain",
            "centered_head_role": "diagnostic_only",
        }
    )
    return spec


def validate_gate_a_checkpoint_meta(meta: dict) -> None:
    """Fail closed unless metadata identifies the exact Gate-A A1 checkpoint contract."""
    exact = {
        "candidate": CANDIDATE,
        "condition": "A1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    for field, expected in exact.items():
        if meta.get(field) != expected:
            raise ValueError(
                f"Gate-A checkpoint {field} mismatch: {meta.get(field)!r} != {expected!r}"
            )

    labels = [str(value) for value in meta.get("labels", [])]
    if len(labels) != 18 or len(set(labels)) != 18:
        raise ValueError("Gate-A checkpoint must contain exactly 18 unique labels")

    training_spec = meta.get("training_spec")
    if not isinstance(training_spec, dict) or training_spec.get("seed") != GATE_B_SEED:
        raise ValueError("Gate-A checkpoint seed mismatch")

    invariants = meta.get("implementation_invariants")
    expected_invariants = {
        "stock_qwen_final_norm_retained": True,
        "triplet_padding": False,
        "exact_one_epoch": True,
    }
    if not isinstance(invariants, dict):
        raise ValueError("Gate-A checkpoint implementation invariants missing")
    for field, expected in expected_invariants.items():
        if invariants.get(field) is not expected:
            raise ValueError(
                f"Gate-A checkpoint invariant mismatch for {field}: "
                f"{invariants.get(field)!r} != {expected!r}"
            )


def split_official_fold(
    train: pd.DataFrame,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leakage-safe fit/validation row indices for one permanent official fold."""
    if fold not in range(5):
        raise ValueError(f"fold must be in 0..4, got {fold}")
    required = {"prompt_id", "fold_id"}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"official frame missing columns: {missing}")

    fold_values = train["fold_id"].to_numpy()
    fit_idx = np.flatnonzero(fold_values != fold)
    valid_idx = np.flatnonzero(fold_values == fold)
    if len(fit_idx) == 0 or len(valid_idx) == 0:
        raise RuntimeError(f"empty official fit/validation partition for fold {fold}")

    fit_prompts = set(train.iloc[fit_idx]["prompt_id"].astype(str))
    valid_prompts = set(train.iloc[valid_idx]["prompt_id"].astype(str))
    overlap = fit_prompts & valid_prompts
    if overlap:
        raise RuntimeError(f"prompt leakage in fold {fold}: {sorted(overlap)[:5]}")
    return fit_idx, valid_idx


def score_oof(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, object]:
    """Score OOF predictions using balanced accuracy and per-class recall."""
    true_values = np.asarray(y_true, dtype=object).astype(str)
    pred_values = np.asarray(y_pred, dtype=object).astype(str)
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")
    if labels is None:
        labels = sorted(set(true_values.tolist()))
    labels = [str(label) for label in labels]
    recalls = recall_score(
        true_values,
        pred_values,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "balanced_accuracy": float(balanced_accuracy_score(true_values, pred_values)),
        "per_model_recall": {
            label: float(recall) for label, recall in zip(labels, recalls)
        },
    }


def paired_prompt_bootstrap_delta(
    frame: pd.DataFrame,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = GATE_B_SEED,
) -> dict[str, object]:
    """Paired bootstrap of candidate-minus-Phase7 BA by resampling prompt groups."""
    required = {
        "prompt_id",
        "model",
        "candidate_prediction",
        "phase7_prediction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"bootstrap frame missing columns: {missing}")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if len(frame) == 0:
        raise ValueError("bootstrap frame must not be empty")

    prompts = sorted(frame["prompt_id"].astype(str).unique().tolist())
    if not prompts:
        raise ValueError("bootstrap requires at least one prompt group")
    prompt_values = frame["prompt_id"].astype(str).to_numpy()
    grouped_indices = {
        prompt: np.flatnonzero(prompt_values == prompt) for prompt in prompts
    }

    true_values = frame["model"].astype(str).to_numpy()
    candidate = frame["candidate_prediction"].astype(str).to_numpy()
    phase7 = frame["phase7_prediction"].astype(str).to_numpy()
    point_delta = float(
        balanced_accuracy_score(true_values, candidate)
        - balanced_accuracy_score(true_values, phase7)
    )

    rng = np.random.RandomState(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled_prompts = rng.choice(prompts, size=len(prompts), replace=True)
        sampled_rows = np.concatenate(
            [grouped_indices[str(prompt)] for prompt in sampled_prompts]
        )
        sampled_true = true_values[sampled_rows]
        deltas[index] = float(
            balanced_accuracy_score(sampled_true, candidate[sampled_rows])
            - balanced_accuracy_score(sampled_true, phase7[sampled_rows])
        )

    return {
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "sampling_unit": "prompt_id",
        "prompt_groups": int(len(prompts)),
        "point_delta": point_delta,
        "ci_lower": float(np.quantile(deltas, 0.025)),
        "ci_upper": float(np.quantile(deltas, 0.975)),
    }


def gate_b_decision(
    *,
    candidate_ba: float,
    phase7_ba: float,
    bootstrap_ci_lower: float,
    candidate_fold_ba: Sequence[float],
    phase7_fold_ba: Sequence[float],
    candidate_recall: dict[str, float],
    phase7_recall: dict[str, float],
) -> dict[str, object]:
    """Apply the four frozen Gate-B PASS conditions mechanically."""
    if len(candidate_fold_ba) != 5 or len(phase7_fold_ba) != 5:
        raise ValueError("Gate B requires exactly five fold scores per system")
    if set(candidate_recall) != set(phase7_recall):
        raise ValueError("candidate and Phase7 recall labels differ")

    candidate_beats_phase7 = float(candidate_ba) > float(phase7_ba)
    bootstrap_ci_lower_positive = float(bootstrap_ci_lower) > 0.0
    fold_wins = int(
        sum(
            float(candidate) > float(reference)
            for candidate, reference in zip(candidate_fold_ba, phase7_fold_ba)
        )
    )
    veto_labels = sorted(
        label
        for label, reference_recall in phase7_recall.items()
        if float(reference_recall) >= 0.25
        and float(reference_recall) - float(candidate_recall[label]) > 0.15
    )
    class_recall_safety_veto = bool(veto_labels)

    passes = (
        candidate_beats_phase7
        and bootstrap_ci_lower_positive
        and fold_wins >= 3
        and not class_recall_safety_veto
    )
    return {
        "status": "pass" if passes else "fail",
        "candidate_beats_phase7": candidate_beats_phase7,
        "bootstrap_ci_lower_positive": bootstrap_ci_lower_positive,
        "fold_wins": fold_wins,
        "beats_phase7_on_at_least_3_of_5_folds": fold_wins >= 3,
        "class_recall_safety_veto": class_recall_safety_veto,
        "veto_labels": veto_labels,
    }
