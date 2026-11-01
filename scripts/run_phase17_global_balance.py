from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from scripts.run_strong_sparse_anchor import (
    DEFAULT_SEED,
    build_fold_features,
    load_train,
    make_svc,
)


PHASE5B_SCORE = 0.36708923688789463
PHASE7_SCORE = 0.3700521998508576
NUMERIC_ATOL = 1e-12
SINKHORN_ITERATIONS = 100
EXPECTED_ROWS = 2_699
EXPECTED_LABELS = 18


def row_softmax(margins: np.ndarray) -> np.ndarray:
    values = np.asarray(margins, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise RuntimeError(f"expected a 2D multiclass margin matrix, found {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError("margin matrix contains non-finite values")

    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sums).all() or (row_sums <= 0).any():
        raise RuntimeError("softmax row normalization is invalid")
    probabilities /= row_sums
    if not np.isfinite(probabilities).all():
        raise RuntimeError("softmax produced non-finite probabilities")
    return probabilities


def sinkhorn_uniform(
    probabilities: np.ndarray,
    *,
    iterations: int = SINKHORN_ITERATIONS,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise RuntimeError(f"expected a 2D probability matrix, found {values.shape}")
    if iterations < 1:
        raise RuntimeError("Sinkhorn iterations must be positive")
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("probability matrix is invalid")

    balanced = values.copy()
    initial_rows = balanced.sum(axis=1)
    initial_cols = balanced.sum(axis=0)
    if (initial_rows <= 0).any() or (initial_cols <= 0).any():
        raise RuntimeError("probability matrix contains an empty row or column")

    n_rows, n_classes = balanced.shape
    target = np.full(n_classes, n_rows / n_classes, dtype=np.float64)
    for _ in range(iterations):
        column_sums = balanced.sum(axis=0)
        if not np.isfinite(column_sums).all() or (column_sums <= 0).any():
            raise RuntimeError("Sinkhorn encountered an invalid column sum")
        balanced *= target / column_sums

        row_sums = balanced.sum(axis=1, keepdims=True)
        if not np.isfinite(row_sums).all() or (row_sums <= 0).any():
            raise RuntimeError("Sinkhorn encountered an invalid row sum")
        balanced /= row_sums

    if not np.isfinite(balanced).all():
        raise RuntimeError("Sinkhorn produced non-finite values")
    if not np.allclose(balanced.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("Sinkhorn final rows are not normalized")
    return balanced


def promotion_decision(score: float) -> dict[str, float | bool]:
    score = float(score)
    delta = score - PHASE7_SCORE
    promoted = score > PHASE7_SCORE + NUMERIC_ATOL
    return {
        "phase7_reference": PHASE7_SCORE,
        "balanced_accuracy": score,
        "delta_vs_phase7": delta,
        "beats_phase7_beyond_numeric_equality": promoted,
        "promote_phase17": promoted,
    }


def _recalls(y_true: pd.Series, predictions: np.ndarray, labels: list[str]) -> dict[str, float]:
    values = recall_score(
        y_true.astype(str),
        np.asarray(predictions, dtype=object),
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {label: float(value) for label, value in zip(labels, values)}


def _counts(predictions: np.ndarray, labels: list[str]) -> dict[str, int]:
    series = pd.Series(np.asarray(predictions, dtype=object), dtype="object")
    observed = series.value_counts()
    return {label: int(observed.get(label, 0)) for label in labels}


def run_cv(train: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    labels = sorted(train["model"].astype(str).unique())
    if len(train) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} official rows, found {len(train)}")
    if len(labels) != EXPECTED_LABELS:
        raise RuntimeError(f"expected {EXPECTED_LABELS} labels, found {len(labels)}")

    fold_ids = train["fold_id"].to_numpy()
    margins = np.full((len(train), len(labels)), np.nan, dtype=np.float64)
    raw_predictions = np.full(len(train), None, dtype=object)
    raw_fold_scores: list[float] = []
    fold_feature_meta: list[dict] = []

    for fold in range(5):
        fit_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]

        x_train, x_valid, meta = build_fold_features(
            fit["text"].astype(str).tolist(),
            valid["text"].astype(str).tolist(),
        )
        model = make_svc(DEFAULT_SEED + fold)
        model.fit(x_train, fit["model"].astype(str))

        fold_margins = np.asarray(model.decision_function(x_valid), dtype=np.float64)
        model_classes = list(map(str, model.classes_.tolist()))
        if set(model_classes) != set(labels):
            raise RuntimeError(
                f"fold {fold} class set mismatch: {model_classes} != {labels}"
            )
        reorder = [model_classes.index(label) for label in labels]
        fold_margins = fold_margins[:, reorder]
        if fold_margins.shape != (len(valid_idx), len(labels)):
            raise RuntimeError(
                f"fold {fold} margin shape mismatch: {fold_margins.shape}"
            )
        if not np.isfinite(fold_margins).all():
            raise RuntimeError(f"fold {fold} margins contain non-finite values")

        margins[valid_idx] = fold_margins
        fold_pred = np.asarray(labels, dtype=object)[np.argmax(fold_margins, axis=1)]
        raw_predictions[valid_idx] = fold_pred
        raw_score = float(balanced_accuracy_score(valid["model"], fold_pred))
        raw_fold_scores.append(raw_score)

        meta.update(
            {
                "fold": fold,
                "official_training_rows": int(len(fit)),
                "official_validation_rows": int(len(valid)),
                "external_training_rows": 0,
            }
        )
        fold_feature_meta.append(meta)
        print(
            f"phase17 fold={fold} raw_score={raw_score:.6f} "
            f"train={len(fit)} valid={len(valid)} features={meta['total_features']}",
            flush=True,
        )

        del x_train, x_valid, model

    if np.isnan(margins).any() or not np.isfinite(margins).all():
        raise RuntimeError("OOF margin matrix is incomplete or invalid")
    if any(value is None for value in raw_predictions.tolist()):
        raise RuntimeError("raw OOF predictions are incomplete")

    raw_ba = float(balanced_accuracy_score(train["model"], raw_predictions))
    if not np.isclose(raw_ba, PHASE5B_SCORE, atol=NUMERIC_ATOL, rtol=0):
        raise RuntimeError(
            f"raw OOF score failed Phase 5b reproduction: {raw_ba} != {PHASE5B_SCORE}"
        )

    probabilities = row_softmax(margins)
    balanced = sinkhorn_uniform(probabilities, iterations=SINKHORN_ITERATIONS)
    predictions = np.asarray(labels, dtype=object)[np.argmax(balanced, axis=1)]

    final_ba = float(balanced_accuracy_score(train["model"], predictions))
    final_fold_scores = [
        float(
            balanced_accuracy_score(
                train.loc[fold_ids == fold, "model"],
                predictions[fold_ids == fold],
            )
        )
        for fold in range(5)
    ]

    row_sums = balanced.sum(axis=1)
    column_masses = balanced.sum(axis=0)
    target_mass = len(train) / len(labels)
    if not np.allclose(row_sums, 1.0, atol=1e-12, rtol=0):
        raise RuntimeError("final balanced probabilities are not row normalized")

    report = {
        "phase": 17,
        "experiment": "official_only_phase5b_uniform_global_balance",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": "bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d",
        },
        "data": {
            "official_rows": int(len(train)),
            "labels": labels,
            "external_rows": 0,
        },
        "configuration": {
            "base_model": "exact Phase 5b official-only strong sparse LinearSVC",
            "margin_source": "fold-local untouched validation decision_function",
            "softmax_temperature": 1.0,
            "sinkhorn_iterations": SINKHORN_ITERATIONS,
            "target_prior": "uniform",
            "target_column_mass": float(target_mass),
            "new_tuned_parameters": 0,
        },
        "fold_feature_meta": fold_feature_meta,
        "raw_phase5b": {
            "balanced_accuracy": raw_ba,
            "folds": raw_fold_scores,
            "per_model_recall": _recalls(train["model"], raw_predictions, labels),
            "prediction_counts": _counts(raw_predictions, labels),
        },
        "phase17": {
            "balanced_accuracy": final_ba,
            "folds": final_fold_scores,
            "per_model_recall": _recalls(train["model"], predictions, labels),
            "prediction_counts": _counts(predictions, labels),
            "final_column_masses": {
                label: float(value) for label, value in zip(labels, column_masses)
            },
            "max_abs_column_mass_error": float(np.max(np.abs(column_masses - target_mass))),
        },
        "invariants": {
            "margin_shape": list(margins.shape),
            "all_margins_finite": bool(np.isfinite(margins).all()),
            "all_balanced_values_finite": bool(np.isfinite(balanced).all()),
            "rows_normalized": bool(np.allclose(row_sums, 1.0, atol=1e-12, rtol=0)),
            "raw_argmax_reproduces_phase5b": True,
            "external_training_rows": 0,
        },
        "decision": {
            **promotion_decision(final_ba),
            "phase5b_reference": PHASE5B_SCORE,
            "delta_vs_phase5b": final_ba - PHASE5B_SCORE,
        },
    }

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["raw_prediction"] = raw_predictions
    oof["prediction"] = predictions
    if len(oof) != EXPECTED_ROWS or oof["id"].nunique() != EXPECTED_ROWS:
        raise RuntimeError("Phase 17 OOF row/ID invariant failed")
    if oof[["raw_prediction", "prediction"]].isna().any().any():
        raise RuntimeError("Phase 17 OOF contains null predictions")
    return report, oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen Phase 17 official-only global prior balancing experiment."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase17/global_balance.json"),
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=Path("artifacts/phase17/global_balance_oof.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train(args.train, args.folds)
    report, oof = run_cv(train)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
