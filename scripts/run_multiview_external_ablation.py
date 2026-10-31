from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from scripts.run_honest_style_baseline import (
    EXPECTED_FOLD_FINGERPRINT,
    build_multiview_model,
    load_official_train,
)


PHASE5_REFERENCE = 0.36003977131493914
PHASE5_TOLERANCE = 1e-10
EXPECTED_EXTERNAL_ROWS = 2503
EXPECTED_EXTERNAL_PROMPTS = 430
EXPECTED_EXTERNAL_LABELS = (
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
)
EXPECTED_BALANCED_CAP = 140
EXPECTED_BALANCED_ROWS = EXPECTED_BALANCED_CAP * len(EXPECTED_EXTERNAL_LABELS)


def load_phase5_reference(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("phase") != 5:
        raise ValueError("Phase 5 reference must have phase=5")
    observed_fold = report["validation"]["fold_fingerprint"]
    if observed_fold != EXPECTED_FOLD_FINGERPRINT:
        raise ValueError(
            f"Phase 5 fold fingerprint mismatch: {observed_fold}"
        )
    observed = float(report["candidates"]["multiview_style_svm"]["balanced_accuracy"])
    if abs(observed - PHASE5_REFERENCE) > PHASE5_TOLERANCE:
        raise ValueError(
            f"Phase 5 reference score drift: {observed} != {PHASE5_REFERENCE}"
        )
    if int(report["data"]["official_rows"]) != 2699:
        raise ValueError("Phase 5 reference must score 2,699 official rows")
    if int(report["data"]["external_rows"]) != 0:
        raise ValueError("Phase 5 reference must be official-only")
    return report


def validate_external_snapshot(
    external: pd.DataFrame,
    *,
    official_labels: set[str],
    enforce_canonical_shape: bool = True,
) -> None:
    required = {"text", "model", "prompt_id"}
    missing = sorted(required - set(external.columns))
    if missing:
        raise ValueError(f"external snapshot missing columns: {missing}")
    if external[list(required)].isna().any().any():
        raise ValueError("external snapshot has null text/model/prompt_id rows")

    external_labels = set(external["model"].astype(str).unique())
    unknown = sorted(external_labels - official_labels)
    if unknown:
        raise ValueError(f"external labels outside competition label set: {unknown}")

    if enforce_canonical_shape:
        if len(external) != EXPECTED_EXTERNAL_ROWS:
            raise ValueError(
                f"expected {EXPECTED_EXTERNAL_ROWS} external rows, found {len(external)}"
            )
        prompts = int(external["prompt_id"].nunique())
        if prompts != EXPECTED_EXTERNAL_PROMPTS:
            raise ValueError(
                f"expected {EXPECTED_EXTERNAL_PROMPTS} external prompts, found {prompts}"
            )
        observed_labels = tuple(sorted(external_labels))
        if observed_labels != tuple(sorted(EXPECTED_EXTERNAL_LABELS)):
            raise ValueError(
                f"external label set mismatch: {observed_labels}"
            )
        if int(external["text"].astype(str).duplicated().sum()) != 0:
            raise ValueError("external snapshot contains duplicate exact text rows")
        min_count = int(external["model"].value_counts().min())
        if min_count != EXPECTED_BALANCED_CAP:
            raise ValueError(
                f"expected minimum external class count {EXPECTED_BALANCED_CAP}, found {min_count}"
            )


def balanced_external_slice(
    external: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    cap = int(external["model"].value_counts().min())
    pieces = []
    for label in sorted(external["model"].astype(str).unique()):
        group = external.loc[external["model"].astype(str) == label]
        pieces.append(group.sample(n=cap, random_state=seed))
    return pd.concat(pieces, ignore_index=True).reset_index(drop=True)


def prepare_augmented_fold(
    train: pd.DataFrame,
    external: pd.DataFrame,
    *,
    fold: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, np.ndarray]:
    train_idx = np.flatnonzero(train["fold_id"].to_numpy() != fold)
    valid_idx = np.flatnonzero(train["fold_id"].to_numpy() == fold)

    official_train = train.iloc[train_idx]
    valid = train.iloc[valid_idx]
    fit_text = pd.concat(
        [official_train["text"], external["text"]], ignore_index=True
    ).astype(str)
    fit_y = pd.concat(
        [official_train["model"], external["model"]], ignore_index=True
    ).astype(str)
    valid_text = valid["text"].astype(str).reset_index(drop=True)
    valid_y = valid["model"].astype(str).reset_index(drop=True)
    return fit_text, fit_y, valid_text, valid_y, valid_idx


def run_augmented_cv(
    train: pd.DataFrame,
    external: pd.DataFrame,
    *,
    name: str,
) -> tuple[dict, np.ndarray]:
    predictions = np.empty(len(train), dtype=object)
    predictions[:] = None
    fold_scores: list[float] = []

    for fold in range(5):
        fit_text, fit_y, valid_text, valid_y, valid_idx = prepare_augmented_fold(
            train, external, fold=fold
        )
        model = build_multiview_model(seed=20261031 + fold)
        model.fit(fit_text, fit_y)
        fold_pred = model.predict(valid_text)
        predictions[valid_idx] = fold_pred
        score = float(balanced_accuracy_score(valid_y, fold_pred))
        fold_scores.append(score)
        print(
            f"condition={name} fold={fold} balanced_accuracy={score:.6f}",
            flush=True,
        )

    if any(value is None for value in predictions.tolist()):
        raise RuntimeError("OOF predictions contain unfilled official rows")

    labels = sorted(train["model"].astype(str).unique())
    recalls = recall_score(
        train["model"],
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return (
        {
            "name": name,
            "external_rows": int(len(external)),
            "oof_rows": int(len(train)),
            "balanced_accuracy": float(
                balanced_accuracy_score(train["model"], predictions)
            ),
            "folds": [float(x) for x in fold_scores],
            "per_model_recall": {
                label: float(recall) for label, recall in zip(labels, recalls)
            },
        },
        predictions,
    )


def add_deltas(
    condition: dict,
    *,
    baseline_score: float,
    baseline_recall: dict[str, float],
    augmented_labels: list[str],
    untouched_labels: list[str],
) -> dict:
    per_model_delta = {
        label: float(condition["per_model_recall"][label] - baseline_recall[label])
        for label in baseline_recall
    }
    condition = dict(condition)
    condition["delta_vs_phase5"] = float(
        condition["balanced_accuracy"] - baseline_score
    )
    condition["per_model_delta_vs_phase5"] = per_model_delta
    condition["mean_delta_augmented_labels"] = float(
        np.mean([per_model_delta[label] for label in augmented_labels])
    )
    condition["mean_delta_untouched_labels"] = float(
        np.mean([per_model_delta[label] for label in untouched_labels])
    )
    return condition


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: external-data ablation on the Phase 5 multiview anchor."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument(
        "--phase5-reference",
        type=Path,
        default=Path("artifacts/phase5/honest_style_baseline.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase6/multiview_external_ablation.json"),
    )
    parser.add_argument(
        "--all-oof-output",
        type=Path,
        default=Path("artifacts/phase6/multiview_all_external_oof.csv"),
    )
    parser.add_argument(
        "--balanced-oof-output",
        type=Path,
        default=Path("artifacts/phase6/multiview_balanced_external_oof.csv"),
    )
    args = parser.parse_args()

    phase5 = load_phase5_reference(args.phase5_reference)
    train = load_official_train(args.train)
    external = pd.read_parquet(args.external)
    validate_external_snapshot(
        external,
        official_labels=set(train["model"].astype(str).unique()),
        enforce_canonical_shape=True,
    )

    # Independent prompt bank: keep an explicit guard against accidental namespace overlap.
    prompt_overlap = set(train["prompt_id"].astype(str)) & set(
        external["prompt_id"].astype(str)
    )
    if prompt_overlap:
        raise RuntimeError(
            f"external prompt_id overlaps official prompt_id namespace: {sorted(prompt_overlap)[:5]}"
        )

    balanced = balanced_external_slice(external, seed=42)
    if len(balanced) != EXPECTED_BALANCED_ROWS:
        raise RuntimeError(
            f"balanced external slice must have {EXPECTED_BALANCED_ROWS} rows"
        )

    all_result, all_oof = run_augmented_cv(
        train,
        external[["text", "model", "prompt_id"]].copy(),
        name="official_plus_all_external",
    )
    balanced_result, balanced_oof = run_augmented_cv(
        train,
        balanced[["text", "model", "prompt_id"]].copy(),
        name="official_plus_balanced_external",
    )

    baseline = phase5["candidates"]["multiview_style_svm"]
    baseline_score = float(baseline["balanced_accuracy"])
    baseline_recall = {
        str(k): float(v) for k, v in baseline["per_model_recall"].items()
    }
    augmented_labels = sorted(EXPECTED_EXTERNAL_LABELS)
    untouched_labels = sorted(set(baseline_recall) - set(augmented_labels))
    all_result = add_deltas(
        all_result,
        baseline_score=baseline_score,
        baseline_recall=baseline_recall,
        augmented_labels=augmented_labels,
        untouched_labels=untouched_labels,
    )
    balanced_result = add_deltas(
        balanced_result,
        baseline_score=baseline_score,
        baseline_recall=baseline_recall,
        augmented_labels=augmented_labels,
        untouched_labels=untouched_labels,
    )

    candidates = [
        {
            "name": "official_only_phase5_reference",
            "external_rows": 0,
            "oof_rows": 2699,
            "balanced_accuracy": baseline_score,
            "folds": baseline["folds"],
            "per_model_recall": baseline_recall,
            "delta_vs_phase5": 0.0,
        },
        all_result,
        balanced_result,
    ]
    best = max(candidates, key=lambda item: item["balanced_accuracy"])
    report = {
        "phase": 6,
        "experiment": "phase5_multiview_external_ablation",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": EXPECTED_FOLD_FINGERPRINT,
            "phase5_reference": PHASE5_REFERENCE,
            "official_reference_reused_not_refit": True,
        },
        "data": {
            "official_rows": int(len(train)),
            "external_rows": int(len(external)),
            "external_prompts": int(external["prompt_id"].nunique()),
            "external_labels": augmented_labels,
            "balanced_cap_per_label": EXPECTED_BALANCED_CAP,
            "balanced_external_rows": int(len(balanced)),
            "prompt_id_overlap_with_official": 0,
        },
        "model": {
            "representation": "Phase 5 fixed multiview style SVM",
            "config_source": "scripts.run_honest_style_baseline.build_multiview_model",
            "hyperparameters_changed": False,
        },
        "conditions": candidates,
        "decision": {
            "best_condition": best["name"],
            "best_balanced_accuracy": float(best["balanced_accuracy"]),
            "best_delta_vs_phase5": float(best.get("delta_vs_phase5", 0.0)),
            "external_improves_new_anchor": best["external_rows"] > 0
            and best.get("delta_vs_phase5", 0.0) > 0,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    base_oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    all_frame = base_oof.copy()
    all_frame["prediction"] = all_oof
    all_frame.to_csv(args.all_oof_output, index=False)
    balanced_frame = base_oof.copy()
    balanced_frame["prediction"] = balanced_oof
    balanced_frame.to_csv(args.balanced_oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
