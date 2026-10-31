from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from scripts.run_strong_sparse_anchor import (
    DEFAULT_SEED,
    build_fold_features,
    load_train,
    make_svc,
    run_cv as run_anchor_cv,
    summarize_oof,
)
from scripts.run_strong_sparse_external import (
    grouped_recall_means,
    validate_external,
)


EXPECTED_COVERED_LABELS = {
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
}


def gate_predictions(
    anchor_predictions: np.ndarray,
    specialist_predictions: np.ndarray,
    covered_labels: Iterable[str],
) -> np.ndarray:
    if len(anchor_predictions) != len(specialist_predictions):
        raise RuntimeError("anchor and specialist prediction lengths differ")

    covered = set(covered_labels)
    observed_specialist = set(str(x) for x in specialist_predictions.tolist())
    outside = sorted(observed_specialist - covered)
    if outside:
        raise RuntimeError(f"specialist predictions outside covered label set: {outside}")

    result = np.asarray(anchor_predictions, dtype=object).copy()
    anchor = np.asarray(anchor_predictions, dtype=object)
    specialist = np.asarray(specialist_predictions, dtype=object)
    mask = np.fromiter(
        (str(label) in covered for label in anchor), dtype=bool, count=len(anchor)
    )
    result[mask] = specialist[mask]
    return result


def specialist_training_rows(
    official_fit: pd.DataFrame,
    external: pd.DataFrame,
    *,
    covered_labels: Iterable[str],
) -> tuple[list[str], list[str]]:
    covered = set(covered_labels)
    official_covered = official_fit.loc[
        official_fit["model"].astype(str).isin(covered), ["text", "model"]
    ]
    texts = pd.concat(
        [official_covered["text"], external["text"]], ignore_index=True
    ).astype(str).tolist()
    labels = pd.concat(
        [official_covered["model"], external["model"]], ignore_index=True
    ).astype(str).tolist()
    return texts, labels


def validate_anchor_oof_alignment(train: pd.DataFrame, anchor_oof: pd.DataFrame) -> None:
    required = ["id", "prompt_id", "model", "fold_id", "prediction"]
    missing = [column for column in required if column not in anchor_oof.columns]
    if missing:
        raise RuntimeError(f"anchor OOF missing columns: {missing}")
    if len(anchor_oof) != len(train):
        raise RuntimeError(
            f"anchor OOF row alignment mismatch: {len(anchor_oof)} != {len(train)}"
        )

    keys = ["id", "prompt_id", "model", "fold_id"]
    expected = train[keys].reset_index(drop=True)
    observed = anchor_oof[keys].reset_index(drop=True)
    if not expected.equals(observed):
        raise RuntimeError("anchor OOF row alignment mismatch")
    if anchor_oof["prediction"].isna().any():
        raise RuntimeError("anchor OOF predictions contain nulls")


def assert_uncovered_recall_invariant(
    anchor_recalls: dict[str, float],
    ensemble_recalls: dict[str, float],
    *,
    covered_labels: Iterable[str],
) -> None:
    covered = set(covered_labels)
    if set(anchor_recalls) != set(ensemble_recalls):
        raise RuntimeError("anchor and ensemble recall label sets differ")
    for label, anchor_recall in anchor_recalls.items():
        if label in covered:
            continue
        ensemble_recall = ensemble_recalls[label]
        if not np.isclose(anchor_recall, ensemble_recall, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"uncovered recall drift for {label}: {anchor_recall} -> {ensemble_recall}"
            )


def run_specialist_cv(
    train: pd.DataFrame,
    external: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[dict, pd.DataFrame]:
    validate_anchor_oof_alignment(train, anchor_oof)
    official_labels = set(train["model"].astype(str).unique())
    if len(official_labels) != 18:
        raise RuntimeError(f"expected 18 official labels, found {len(official_labels)}")
    if not EXPECTED_COVERED_LABELS < official_labels:
        raise RuntimeError("covered label set must be a strict subset of official labels")

    validate_external(external, official_labels=official_labels)

    specialist_predictions = np.full(len(train), None, dtype=object)
    specialist_covered_fold_scores: list[float] = []
    fold_feature_meta: list[dict] = []

    fold_ids = train["fold_id"].to_numpy()
    for fold in range(5):
        fit_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        official_fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]

        fit_text, fit_y = specialist_training_rows(
            official_fit,
            external,
            covered_labels=EXPECTED_COVERED_LABELS,
        )
        if set(fit_y) != EXPECTED_COVERED_LABELS:
            raise RuntimeError(
                f"specialist training labels mismatch on fold {fold}: {sorted(set(fit_y))}"
            )

        x_train, x_valid, meta = build_fold_features(
            fit_text,
            valid["text"].astype(str).tolist(),
        )
        model = make_svc(seed + fold)
        model.fit(x_train, fit_y)
        pred = model.predict(x_valid)
        observed_pred = set(str(x) for x in pred.tolist())
        if not observed_pred <= EXPECTED_COVERED_LABELS:
            raise RuntimeError(
                f"specialist emitted non-covered labels on fold {fold}: "
                f"{sorted(observed_pred - EXPECTED_COVERED_LABELS)}"
            )
        specialist_predictions[valid_idx] = pred

        covered_valid = valid["model"].astype(str).isin(EXPECTED_COVERED_LABELS).to_numpy()
        covered_score = float(
            balanced_accuracy_score(valid.loc[covered_valid, "model"], pred[covered_valid])
        )
        specialist_covered_fold_scores.append(covered_score)
        meta.update(
            {
                "fold": fold,
                "official_covered_training_rows": int(
                    official_fit["model"].astype(str).isin(EXPECTED_COVERED_LABELS).sum()
                ),
                "external_training_rows": int(len(external)),
                "validation_rows": int(len(valid)),
                "covered_validation_rows": int(covered_valid.sum()),
            }
        )
        fold_feature_meta.append(meta)
        print(
            f"specialist fold={fold} covered_score={covered_score:.6f} "
            f"official_covered_fit={meta['official_covered_training_rows']} "
            f"external={len(external)} total_features={meta['total_features']}",
            flush=True,
        )

    if any(value is None for value in specialist_predictions.tolist()):
        raise RuntimeError("specialist OOF predictions contain unfilled rows")

    anchor_predictions = anchor_oof["prediction"].astype(str).to_numpy(dtype=object)
    ensemble_predictions = gate_predictions(
        anchor_predictions,
        specialist_predictions,
        EXPECTED_COVERED_LABELS,
    )

    ensemble_fold_scores = []
    for fold in range(5):
        valid_idx = np.flatnonzero(fold_ids == fold)
        ensemble_fold_scores.append(
            float(
                balanced_accuracy_score(
                    train.iloc[valid_idx]["model"], ensemble_predictions[valid_idx]
                )
            )
        )
    ensemble_result = summarize_oof(train, ensemble_predictions, ensemble_fold_scores)

    covered_mask = train["model"].astype(str).isin(EXPECTED_COVERED_LABELS).to_numpy()
    covered_labels = sorted(EXPECTED_COVERED_LABELS)
    specialist_recalls = recall_score(
        train.loc[covered_mask, "model"],
        specialist_predictions[covered_mask],
        labels=covered_labels,
        average=None,
        zero_division=0,
    )
    specialist_result = {
        "scored_rows": int(covered_mask.sum()),
        "balanced_accuracy_on_covered_true_labels": float(
            balanced_accuracy_score(
                train.loc[covered_mask, "model"], specialist_predictions[covered_mask]
            )
        ),
        "folds_on_covered_true_labels": specialist_covered_fold_scores,
        "per_model_recall": {
            label: float(recall)
            for label, recall in zip(covered_labels, specialist_recalls)
        },
    }

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["anchor_prediction"] = anchor_predictions
    oof["specialist_prediction"] = specialist_predictions
    oof["prediction"] = ensemble_predictions
    oof["anchor_was_covered"] = oof["anchor_prediction"].isin(EXPECTED_COVERED_LABELS)
    oof["changed_by_specialist"] = oof["anchor_prediction"] != oof["prediction"]

    return {
        "ensemble_result": ensemble_result,
        "specialist_result": specialist_result,
        "fold_feature_meta": fold_feature_meta,
    }, oof


def build_report(
    train: pd.DataFrame,
    external: pd.DataFrame,
    anchor_report: dict,
    specialist_bundle: dict,
    oof: pd.DataFrame,
) -> dict:
    anchor_result = anchor_report["result"]
    ensemble_result = specialist_bundle["ensemble_result"]
    assert_uncovered_recall_invariant(
        anchor_result["per_model_recall"],
        ensemble_result["per_model_recall"],
        covered_labels=EXPECTED_COVERED_LABELS,
    )

    anchor_groups = grouped_recall_means(
        anchor_result["per_model_recall"], covered_labels=EXPECTED_COVERED_LABELS
    )
    ensemble_groups = grouped_recall_means(
        ensemble_result["per_model_recall"], covered_labels=EXPECTED_COVERED_LABELS
    )
    anchor_score = float(anchor_result["balanced_accuracy"])
    ensemble_score = float(ensemble_result["balanced_accuracy"])
    anchor_covered_mask = oof["anchor_was_covered"].to_numpy(dtype=bool)

    return {
        "phase": 7,
        "experiment": "gated_external_specialist",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": anchor_report["validation"]["fold_fingerprint"],
        },
        "data": {
            "official_rows": int(len(train)),
            "official_prompts": int(train["prompt_id"].nunique()),
            "external_rows": int(len(external)),
            "external_prompts": int(external["prompt_id"].astype(str).nunique()),
            "covered_labels": sorted(EXPECTED_COVERED_LABELS),
            "uncovered_labels": sorted(
                set(train["model"].astype(str).unique()) - EXPECTED_COVERED_LABELS
            ),
        },
        "configuration": {
            "anchor": anchor_report["configuration"],
            "specialist": anchor_report["configuration"],
            "specialist_classes": 8,
            "gate": (
                "replace anchor label with specialist label iff anchor prediction "
                "is in the 8 covered labels"
            ),
            "tuned_gate_parameters": 0,
        },
        "anchor": {
            "result": anchor_result,
            "mean_recall": anchor_groups,
        },
        "specialist": specialist_bundle["specialist_result"],
        "ensemble": {
            "result": ensemble_result,
            "mean_recall": ensemble_groups,
        },
        "gate_stats": {
            "anchor_predicted_covered_rows": int(anchor_covered_mask.sum()),
            "anchor_predicted_uncovered_rows": int((~anchor_covered_mask).sum()),
            "changed_rows": int(oof["changed_by_specialist"].sum()),
        },
        "deltas": {
            "balanced_accuracy": ensemble_score - anchor_score,
            "covered_mean_recall": ensemble_groups["covered"] - anchor_groups["covered"],
            "uncovered_mean_recall": ensemble_groups["uncovered"] - anchor_groups["uncovered"],
            "per_model_recall": {
                label: float(ensemble_result["per_model_recall"][label])
                - float(anchor_result["per_model_recall"][label])
                for label in anchor_result["per_model_recall"]
            },
        },
        "invariants": {
            "uncovered_recall_exactly_preserved": True,
            "anchor_predictions_outside_covered_set_never_changed": bool(
                (
                    oof.loc[~oof["anchor_was_covered"], "anchor_prediction"]
                    == oof.loc[~oof["anchor_was_covered"], "prediction"]
                ).all()
            ),
        },
        "fold_feature_meta": specialist_bundle["fold_feature_meta"],
        "decision": {
            "beats_phase5b_anchor": ensemble_score > anchor_score,
            "promote_gated_specialist": ensemble_score > anchor_score,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7: fixed gated 8-class external specialist."
    )
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
        default=Path("artifacts/phase7/gated_specialist.json"),
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=Path("artifacts/phase7/gated_specialist_oof.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train(args.train, args.folds)
    external = pd.read_parquet(args.external)

    anchor_report, anchor_oof = run_anchor_cv(train, seed=args.seed)
    specialist_bundle, oof = run_specialist_cv(
        train,
        external,
        anchor_oof,
        seed=args.seed,
    )
    report = build_report(train, external, anchor_report, specialist_bundle, oof)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
