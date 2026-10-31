from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from scripts.run_strong_sparse_anchor import (
    EXPECTED_FOLD_FINGERPRINT,
    build_fold_features,
    load_train,
    make_svc,
    summarize_oof,
)


EXPECTED_EXTERNAL_COUNTS = {
    "gemini-3.1-pro-preview": 140,
    "gemini-3.6-flash": 314,
    "gemini-3.8-flash": 320,
    "gpt-5.5": 313,
    "gpt-5.6-luna": 370,
    "gpt-5.6-sol": 321,
    "gpt-5.6-terra": 370,
    "gpt-6-astra": 355,
}
EXPECTED_EXTERNAL_ROWS = 2503
EXPECTED_BASELINE = 0.36708923688789463
BASELINE_TOLERANCE = 0.001
DEFAULT_SEED = 20261031


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def validate_external(
    external: pd.DataFrame,
    *,
    official_labels: Sequence[str],
    official_texts: Sequence[str] | None = None,
) -> pd.DataFrame:
    required = {"text", "model"}
    missing = sorted(required - set(external.columns))
    if missing:
        raise ValueError(f"external data missing columns: {missing}")

    out = external[["text", "model"]].copy()
    out["text"] = out["text"].astype(str)
    out["model"] = out["model"].astype(str)
    counts = out["model"].value_counts().to_dict()
    if len(out) != EXPECTED_EXTERNAL_ROWS or counts != EXPECTED_EXTERNAL_COUNTS:
        raise ValueError(
            f"audited external counts mismatch: rows={len(out)} counts={counts}"
        )

    unknown = sorted(set(out["model"]) - set(str(x) for x in official_labels))
    if unknown:
        raise ValueError(f"external labels outside official label set: {unknown}")
    if out["text"].str.strip().eq("").any():
        raise ValueError("external data contains blank text")
    normalized = out["text"].map(normalize_text)
    if normalized.duplicated().any():
        raise ValueError("external data contains duplicate normalized text")

    if official_texts is not None:
        official_normalized = {normalize_text(x) for x in official_texts}
        overlap = normalized.isin(official_normalized)
        if overlap.any():
            raise ValueError(
                f"external data has {int(overlap.sum())} exact normalized official-text overlaps"
            )
    return out.reset_index(drop=True)


def prepare_condition_rows(
    train: pd.DataFrame,
    external: pd.DataFrame,
    *,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    official_fit = train.loc[train["fold_id"] != fold].copy()
    valid = train.loc[train["fold_id"] == fold].copy()
    official_fit["is_external"] = False
    valid["is_external"] = False

    external_fit = external[["text", "model"]].copy()
    external_fit["is_external"] = True
    # Keep the schema needed by the feature builder/model; provenance remains in the source snapshot.
    augmented_fit = pd.concat(
        [official_fit[["text", "model", "is_external"]], external_fit],
        ignore_index=True,
    )
    return official_fit, valid, augmented_fit


def summarize_ablation(
    *,
    train: pd.DataFrame,
    official_predictions: np.ndarray,
    augmented_predictions: np.ndarray,
    official_fold_scores: list[float],
    augmented_fold_scores: list[float],
    covered_labels: set[str],
) -> dict:
    official = summarize_oof(train, official_predictions, official_fold_scores)
    augmented = summarize_oof(train, augmented_predictions, augmented_fold_scores)
    labels = sorted(official["per_model_recall"])
    per_model_delta = {
        label: augmented["per_model_recall"][label] - official["per_model_recall"][label]
        for label in labels
    }
    covered = [label for label in labels if label in covered_labels]
    uncovered = [label for label in labels if label not in covered_labels]
    covered_mean = float(np.mean([per_model_delta[x] for x in covered])) if covered else 0.0
    uncovered_mean = float(np.mean([per_model_delta[x] for x in uncovered])) if uncovered else 0.0
    return {
        "official_only": official,
        "official_plus_external": augmented,
        "delta": augmented["balanced_accuracy"] - official["balanced_accuracy"],
        "covered_labels": covered,
        "uncovered_labels": uncovered,
        "per_model_recall_delta": per_model_delta,
        "covered_mean_recall_delta": covered_mean,
        "uncovered_mean_recall_delta": uncovered_mean,
    }


def run_cv(
    train: pd.DataFrame,
    external: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[dict, pd.DataFrame]:
    labels = sorted(train["model"].astype(str).unique())
    if len(labels) != 18:
        raise RuntimeError(f"expected 18 official labels, found {len(labels)}")

    official_predictions = np.full(len(train), None, dtype=object)
    augmented_predictions = np.full(len(train), None, dtype=object)
    official_fold_scores: list[float] = []
    augmented_fold_scores: list[float] = []
    fold_meta: list[dict] = []

    for fold in range(5):
        official_fit, valid, augmented_fit = prepare_condition_rows(
            train, external, fold=fold
        )
        valid_idx = np.flatnonzero(train["fold_id"].to_numpy() == fold)
        valid_text = valid["text"].astype(str).tolist()

        condition_results = {}
        for condition, fit in [
            ("official_only", official_fit),
            ("official_plus_external", augmented_fit),
        ]:
            x_fit, x_valid, feature_meta = build_fold_features(
                fit["text"].astype(str).tolist(),
                valid_text,
            )
            model = make_svc(seed + fold)
            model.fit(x_fit, fit["model"])
            pred = model.predict(x_valid)
            score = float(balanced_accuracy_score(valid["model"], pred))
            condition_results[condition] = {
                "pred": pred,
                "score": score,
                "feature_meta": feature_meta,
                "train_rows": int(len(fit)),
            }

        official_predictions[valid_idx] = condition_results["official_only"]["pred"]
        augmented_predictions[valid_idx] = condition_results["official_plus_external"]["pred"]
        official_fold_scores.append(condition_results["official_only"]["score"])
        augmented_fold_scores.append(condition_results["official_plus_external"]["score"])
        fold_meta.append({
            "fold": fold,
            "official_only": {
                "score": condition_results["official_only"]["score"],
                "train_rows": condition_results["official_only"]["train_rows"],
                "features": condition_results["official_only"]["feature_meta"],
            },
            "official_plus_external": {
                "score": condition_results["official_plus_external"]["score"],
                "train_rows": condition_results["official_plus_external"]["train_rows"],
                "features": condition_results["official_plus_external"]["feature_meta"],
            },
        })
        print(
            "fold={} official={:.6f} augmented={:.6f} delta={:+.6f}".format(
                fold,
                official_fold_scores[-1],
                augmented_fold_scores[-1],
                augmented_fold_scores[-1] - official_fold_scores[-1],
            ),
            flush=True,
        )

    summary = summarize_ablation(
        train=train,
        official_predictions=official_predictions,
        augmented_predictions=augmented_predictions,
        official_fold_scores=official_fold_scores,
        augmented_fold_scores=augmented_fold_scores,
        covered_labels=set(EXPECTED_EXTERNAL_COUNTS),
    )
    reproduction_error = abs(
        summary["official_only"]["balanced_accuracy"] - EXPECTED_BASELINE
    )
    if reproduction_error > BASELINE_TOLERANCE:
        raise RuntimeError(
            "Phase 5b baseline did not reproduce: "
            f"observed={summary['official_only']['balanced_accuracy']:.9f} "
            f"expected={EXPECTED_BASELINE:.9f} error={reproduction_error:.9f}"
        )

    report = {
        "phase": 6,
        "experiment": "strong_sparse_external_ablation",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": EXPECTED_FOLD_FINGERPRINT,
        },
        "data": {
            "official_rows": int(len(train)),
            "external_rows": int(len(external)),
            "external_labels": sorted(EXPECTED_EXTERNAL_COUNTS),
            "external_label_counts": EXPECTED_EXTERNAL_COUNTS,
        },
        "configuration": {
            "model": "Phase 5b strong sparse anchor unchanged",
            "classifier": "LinearSVC(C=1.5,class_weight=balanced,dual=True,max_iter=15000)",
            "condition_feature_fit": "independent fold-local vocabularies/scaler per A/B condition",
        },
        "baseline_reproduction": {
            "expected": EXPECTED_BASELINE,
            "observed": summary["official_only"]["balanced_accuracy"],
            "absolute_error": reproduction_error,
            "tolerance": BASELINE_TOLERANCE,
        },
        "folds": fold_meta,
        "ablation": summary,
        "decision": {
            "external_improves_strong_sparse_oof": summary["delta"] > 0,
            "delta": summary["delta"],
        },
    }

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["prediction_official_only"] = official_predictions
    oof["prediction_official_plus_external"] = augmented_predictions
    return report, oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 strong sparse external A/B.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase6/strong_sparse_external_ablation.json"),
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=Path("artifacts/phase6/strong_sparse_external_oof.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train(args.train, args.folds)
    external_raw = pd.read_parquet(args.external)
    external = validate_external(
        external_raw,
        official_labels=sorted(train["model"].astype(str).unique()),
        official_texts=train["text"].astype(str).tolist(),
    )
    report, oof = run_cv(train, external, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
