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
PHASE7_SEED = 20261031
BOOTSTRAP_RESAMPLES = 5000
PHASE7_METRIC_ATOL = 1e-12


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
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": GATE_B_SEED,
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
        "prompt_id_overlap": 0,
        "prompt_key_overlap": 0,
    }
    for field, expected in exact.items():
        if meta.get(field) != expected:
            raise ValueError(
                f"Gate-A checkpoint {field} mismatch: {meta.get(field)!r} != {expected!r}"
            )

    labels = [str(value) for value in meta.get("labels", [])]
    if len(labels) != 18 or len(set(labels)) != 18 or labels != sorted(labels):
        raise ValueError(
            "Gate-A checkpoint label order must contain exactly 18 unique sorted labels"
        )

    training_spec = meta.get("training_spec")
    expected_training_spec = gate_a_training_spec()
    if training_spec != expected_training_spec:
        raise ValueError("Gate-A checkpoint training spec mismatch")

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

    observed_folds = set(int(value) for value in train["fold_id"].unique().tolist())
    if observed_folds != set(range(5)):
        raise RuntimeError(
            f"official fold ids mismatch: {sorted(observed_folds)} != [0, 1, 2, 3, 4]"
        )

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


def _normalized_alignment_keys(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["id", "prompt_id", "model", "fold_id"]
    missing = [column for column in keys if column not in frame.columns]
    if missing:
        raise RuntimeError(f"OOF alignment columns missing: {missing}")
    return frame[keys].reset_index(drop=True).astype(
        {"id": str, "prompt_id": str, "model": str, "fold_id": int}
    )


def _score_with_folds(
    frame: pd.DataFrame,
    prediction_column: str,
    *,
    labels: Sequence[str],
) -> dict[str, object]:
    if prediction_column not in frame.columns:
        raise RuntimeError(f"prediction column missing: {prediction_column}")
    overall = score_oof(
        frame["model"].astype(str).tolist(),
        frame[prediction_column].astype(str).tolist(),
        labels=labels,
    )
    fold_scores: list[float] = []
    for fold in range(5):
        fold_frame = frame.loc[frame["fold_id"].astype(int) == fold]
        if fold_frame.empty:
            raise RuntimeError(f"missing permanent fold {fold}")
        fold_scores.append(
            float(
                balanced_accuracy_score(
                    fold_frame["model"].astype(str),
                    fold_frame[prediction_column].astype(str),
                )
            )
        )
    return {
        "oof_rows": int(len(frame)),
        "balanced_accuracy": float(overall["balanced_accuracy"]),
        "folds": fold_scores,
        "per_model_recall": dict(overall["per_model_recall"]),
    }


def _assert_phase7_result_matches_frozen(
    observed: dict[str, object],
    frozen: dict[str, object],
) -> None:
    if int(observed.get("oof_rows", -1)) != int(frozen.get("oof_rows", -2)):
        raise RuntimeError("regenerated frozen Phase7 OOF row count drift")

    observed_ba = float(observed["balanced_accuracy"])
    frozen_ba = float(frozen["balanced_accuracy"])
    if abs(observed_ba - frozen_ba) > PHASE7_METRIC_ATOL:
        raise RuntimeError(
            f"regenerated frozen Phase7 balanced accuracy drift: {observed_ba} != {frozen_ba}"
        )

    observed_folds = [float(value) for value in observed["folds"]]
    frozen_folds = [float(value) for value in frozen["folds"]]
    if len(observed_folds) != 5 or len(frozen_folds) != 5:
        raise RuntimeError("frozen Phase7 requires exactly five fold scores")
    for fold, (value, reference) in enumerate(zip(observed_folds, frozen_folds)):
        if abs(value - reference) > PHASE7_METRIC_ATOL:
            raise RuntimeError(
                f"regenerated frozen Phase7 fold {fold} drift: {value} != {reference}"
            )

    observed_recall = {
        str(label): float(value)
        for label, value in dict(observed["per_model_recall"]).items()
    }
    frozen_recall = {
        str(label): float(value)
        for label, value in dict(frozen["per_model_recall"]).items()
    }
    if set(observed_recall) != set(frozen_recall):
        raise RuntimeError("regenerated frozen Phase7 recall label set drift")
    for label in sorted(frozen_recall):
        if abs(observed_recall[label] - frozen_recall[label]) > PHASE7_METRIC_ATOL:
            raise RuntimeError(
                "regenerated frozen Phase7 per-model recall drift for "
                f"{label}: {observed_recall[label]} != {frozen_recall[label]}"
            )


def validate_phase7_reference(
    train: pd.DataFrame,
    phase7_oof: pd.DataFrame,
    phase7_report: dict,
) -> dict[str, object]:
    """Require row-exact Phase7 OOF and reproduce its frozen aggregate metrics."""
    required = {"id", "prompt_id", "model", "fold_id", "prediction"}
    missing = sorted(required - set(phase7_oof.columns))
    if missing:
        raise RuntimeError(f"Phase7 OOF columns missing: {missing}")
    if len(phase7_oof) != len(train):
        raise RuntimeError("Phase7 row alignment mismatch: row count differs")
    if not _normalized_alignment_keys(train).equals(
        _normalized_alignment_keys(phase7_oof)
    ):
        raise RuntimeError("Phase7 row alignment mismatch")
    if phase7_oof["prediction"].isna().any():
        raise RuntimeError("Phase7 OOF contains null predictions")

    labels = sorted(train["model"].astype(str).unique().tolist())
    outside = sorted(set(phase7_oof["prediction"].astype(str)) - set(labels))
    if outside:
        raise RuntimeError(f"Phase7 OOF predictions outside official labels: {outside}")

    observed = _score_with_folds(
        phase7_oof,
        "prediction",
        labels=labels,
    )
    try:
        frozen = phase7_report["ensemble"]["result"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("frozen Phase7 report is missing ensemble.result") from exc
    _assert_phase7_result_matches_frozen(observed, frozen)
    return observed


def regenerate_phase7_oof(
    train: pd.DataFrame,
    external: pd.DataFrame,
    frozen_phase7_report: dict,
    *,
    seed: int = PHASE7_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Regenerate row-level Phase7 predictions from the frozen pipeline."""
    from scripts.run_gated_specialist import (
        build_report as build_phase7_report,
        run_specialist_cv,
    )
    from scripts.run_strong_sparse_anchor import run_cv as run_anchor_cv

    anchor_report, anchor_oof = run_anchor_cv(train, seed=seed)
    specialist_bundle, phase7_oof = run_specialist_cv(
        train,
        external,
        anchor_oof,
        seed=seed,
    )
    regenerated_report = build_phase7_report(
        train,
        external,
        anchor_report,
        specialist_bundle,
        phase7_oof,
    )
    validate_phase7_reference(train, phase7_oof, frozen_phase7_report)
    _assert_phase7_result_matches_frozen(
        regenerated_report["ensemble"]["result"],
        frozen_phase7_report["ensemble"]["result"],
    )
    return phase7_oof, regenerated_report


def evaluate_gate_b(
    candidate_oof: pd.DataFrame,
    train: pd.DataFrame,
    phase7_oof: pd.DataFrame,
    phase7_report: dict,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = GATE_B_SEED,
) -> dict[str, object]:
    """Evaluate Gate B mechanically against regenerated frozen Phase7 OOF."""
    required = {
        "id",
        "prompt_id",
        "model",
        "fold_id",
        "plain_prediction",
        "centered_prediction",
    }
    missing = sorted(required - set(candidate_oof.columns))
    if missing:
        raise RuntimeError(f"candidate OOF columns missing: {missing}")
    if len(candidate_oof) != len(train):
        raise RuntimeError("candidate OOF row alignment mismatch: row count differs")
    if not _normalized_alignment_keys(train).equals(
        _normalized_alignment_keys(candidate_oof)
    ):
        raise RuntimeError("candidate OOF row alignment mismatch")

    labels = sorted(train["model"].astype(str).unique().tolist())
    allowed = set(labels)
    for column in ("plain_prediction", "centered_prediction"):
        if candidate_oof[column].isna().any():
            raise RuntimeError(f"candidate OOF {column} contains null predictions")
        outside = sorted(set(candidate_oof[column].astype(str)) - allowed)
        if outside:
            raise RuntimeError(
                f"candidate OOF {column} contains predictions outside label set: {outside}"
            )

    phase7 = validate_phase7_reference(train, phase7_oof, phase7_report)
    candidate_plain = _score_with_folds(
        candidate_oof,
        "plain_prediction",
        labels=labels,
    )
    candidate_centered = _score_with_folds(
        candidate_oof,
        "centered_prediction",
        labels=labels,
    )
    candidate_centered["role"] = "diagnostic_only"

    candidate_recall = {
        str(label): float(value)
        for label, value in dict(candidate_plain["per_model_recall"]).items()
    }
    phase7_recall = {
        str(label): float(value)
        for label, value in dict(phase7["per_model_recall"]).items()
    }
    candidate_folds = [float(value) for value in candidate_plain["folds"]]
    phase7_folds = [float(value) for value in phase7["folds"]]

    candidate_prediction = candidate_oof["plain_prediction"].astype(str).to_numpy()
    phase7_prediction = phase7_oof["prediction"].astype(str).to_numpy()
    truth = train["model"].astype(str).to_numpy()
    disagreements = candidate_prediction != phase7_prediction
    candidate_correct = candidate_prediction == truth
    phase7_correct = phase7_prediction == truth

    bootstrap_frame = train[["prompt_id", "model"]].copy()
    bootstrap_frame["candidate_prediction"] = candidate_prediction
    bootstrap_frame["phase7_prediction"] = phase7_prediction
    bootstrap = paired_prompt_bootstrap_delta(
        bootstrap_frame,
        n_resamples=n_resamples,
        seed=seed,
    )

    decision = gate_b_decision(
        candidate_ba=float(candidate_plain["balanced_accuracy"]),
        phase7_ba=float(phase7["balanced_accuracy"]),
        bootstrap_ci_lower=float(bootstrap["ci_lower"]),
        candidate_fold_ba=candidate_folds,
        phase7_fold_ba=phase7_folds,
        candidate_recall=candidate_recall,
        phase7_recall=phase7_recall,
    )

    return {
        "candidate": {
            "plain": candidate_plain,
            "centered": candidate_centered,
        },
        "phase7": phase7,
        "delta": {
            "balanced_accuracy": float(candidate_plain["balanced_accuracy"])
            - float(phase7["balanced_accuracy"]),
            "folds": [
                float(candidate) - float(reference)
                for candidate, reference in zip(candidate_folds, phase7_folds)
            ],
            "per_model_recall": {
                label: candidate_recall[label] - phase7_recall[label]
                for label in labels
            },
        },
        "paired_rows": {
            "rows": int(len(train)),
            "disagreement_rate": float(np.mean(disagreements)),
            "candidate_only_correct": int(np.sum(candidate_correct & ~phase7_correct)),
            "phase7_only_correct": int(np.sum(~candidate_correct & phase7_correct)),
            "both_correct": int(np.sum(candidate_correct & phase7_correct)),
            "both_wrong": int(np.sum(~candidate_correct & ~phase7_correct)),
        },
        "bootstrap": bootstrap,
        "decision": decision,
    }
