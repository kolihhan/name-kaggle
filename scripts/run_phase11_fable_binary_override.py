from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from scripts.phase11_audit_fusioncube_fable5 import normalize_text
from scripts.run_gated_specialist import run_specialist_cv as run_phase7_specialist_cv
from scripts.run_strong_sparse_anchor import (
    DEFAULT_SEED,
    build_fold_features,
    load_train,
    make_svc,
    run_cv as run_anchor_cv,
    summarize_oof,
)


FABLE_LABEL = "claude-fable-5"
OTHER_LABEL = "__other__"
EXPECTED_FABLE_EXTERNAL_ROWS = 344
PHASE7_REFERENCE_BA = 0.3700521998508576
PHASE7_REFERENCE_FABLE_RECALL = 0.14093959731543623
NUMERICAL_ATOL = 1e-12


def validate_fable_external(external: pd.DataFrame) -> None:
    required = {"text", "model"}
    missing = sorted(required - set(external.columns))
    if missing:
        raise RuntimeError(f"Fable external missing columns: {missing}")
    if len(external) != EXPECTED_FABLE_EXTERNAL_ROWS:
        raise RuntimeError(
            f"expected exactly 344 Fable external rows, found {len(external)}"
        )
    if external["text"].isna().any() or external["model"].isna().any():
        raise RuntimeError("Fable external contains null text/model")
    if external["text"].astype(str).str.strip().eq("").any():
        raise RuntimeError("Fable external contains empty text")

    observed_models = set(external["model"].astype(str))
    if observed_models != {FABLE_LABEL}:
        raise RuntimeError(
            f"Fable external model identity drifted: {sorted(observed_models)}"
        )

    normalized = external["text"].astype(str).map(normalize_text)
    if normalized.nunique() != EXPECTED_FABLE_EXTERNAL_ROWS:
        raise RuntimeError(
            "Fable external normalized text must remain exactly 344 unique rows"
        )


def binary_training_rows(
    official_fit: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    validate_fable_external(external)
    required = {"text", "model"}
    missing = sorted(required - set(official_fit.columns))
    if missing:
        raise RuntimeError(f"official fit missing columns: {missing}")

    official_text = official_fit["text"].astype(str).tolist()
    official_binary = [
        FABLE_LABEL if str(model) == FABLE_LABEL else OTHER_LABEL
        for model in official_fit["model"].tolist()
    ]
    external_text = external["text"].astype(str).tolist()

    texts = official_text + external_text
    labels = official_binary + [FABLE_LABEL] * len(external_text)
    if set(labels) != {FABLE_LABEL, OTHER_LABEL}:
        raise RuntimeError("binary training labels must contain Fable and __other__")
    return texts, labels


def apply_fable_override(
    phase7_predictions: np.ndarray,
    detector_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phase7 = np.asarray(phase7_predictions, dtype=object)
    detector = np.asarray(detector_predictions, dtype=object)
    if len(phase7) != len(detector):
        raise RuntimeError("phase7 and detector prediction length mismatch")

    observed = set(str(value) for value in detector.tolist())
    allowed = {FABLE_LABEL, OTHER_LABEL}
    outside = sorted(observed - allowed)
    if outside:
        raise RuntimeError(f"detector predictions outside binary label set: {outside}")

    final = phase7.copy()
    positive = detector == FABLE_LABEL
    final[positive] = FABLE_LABEL
    changed = final != phase7

    phase7_fable = phase7 == FABLE_LABEL
    if not np.all(final[phase7_fable] == FABLE_LABEL):
        raise RuntimeError("Fable override removed an existing Phase 7 Fable prediction")
    if np.any(changed & (final != FABLE_LABEL)):
        raise RuntimeError("Fable override changed a prediction to a non-Fable label")

    return final, changed


def phase11_promotion_decision(
    *,
    phase7_ba: float,
    final_ba: float,
    phase7_fable_recall: float,
    final_fable_recall: float,
    no_fable_removed: bool,
    only_changes_to_fable: bool,
) -> dict:
    material_ba_gain = bool(
        final_ba > phase7_ba
        and not np.isclose(final_ba, phase7_ba, rtol=0.0, atol=NUMERICAL_ATOL)
    )
    improves_fable_recall = bool(final_fable_recall > phase7_fable_recall)
    promote = bool(
        material_ba_gain
        and improves_fable_recall
        and no_fable_removed
        and only_changes_to_fable
    )
    return {
        "material_ba_gain": material_ba_gain,
        "improves_fable_recall": improves_fable_recall,
        "no_fable_removed": bool(no_fable_removed),
        "only_changes_to_fable": bool(only_changes_to_fable),
        "promote_phase11": promote,
    }


def validate_phase7_alignment(train: pd.DataFrame, phase7_oof: pd.DataFrame) -> None:
    required = ["id", "prompt_id", "model", "fold_id", "prediction"]
    missing = [column for column in required if column not in phase7_oof.columns]
    if missing:
        raise RuntimeError(f"Phase 7 OOF missing columns: {missing}")
    if len(train) != len(phase7_oof):
        raise RuntimeError("Phase 7 OOF row count mismatch")
    keys = ["id", "prompt_id", "model", "fold_id"]
    if not train[keys].reset_index(drop=True).equals(
        phase7_oof[keys].reset_index(drop=True)
    ):
        raise RuntimeError("Phase 7 OOF row alignment mismatch")
    if phase7_oof["prediction"].isna().any():
        raise RuntimeError("Phase 7 OOF contains null predictions")


def run_binary_detector_cv(
    train: pd.DataFrame,
    fable_external: pd.DataFrame,
    phase7_oof: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[dict, pd.DataFrame]:
    validate_fable_external(fable_external)
    validate_phase7_alignment(train, phase7_oof)

    fold_ids = train["fold_id"].to_numpy()
    detector_predictions = np.full(len(train), None, dtype=object)
    detector_fold_scores: list[float] = []
    feature_meta: list[dict] = []

    for fold in range(5):
        fit_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        official_fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]

        fit_text, fit_y = binary_training_rows(official_fit, fable_external)
        x_train, x_valid, meta = build_fold_features(
            fit_text,
            valid["text"].astype(str).tolist(),
        )
        model = make_svc(seed + fold)
        model.fit(x_train, fit_y)
        pred = model.predict(x_valid)
        observed = set(str(value) for value in pred.tolist())
        if not observed <= {FABLE_LABEL, OTHER_LABEL}:
            raise RuntimeError(
                f"binary detector emitted outside labels on fold {fold}: {sorted(observed)}"
            )
        detector_predictions[valid_idx] = pred

        valid_binary = np.where(
            valid["model"].astype(str).to_numpy() == FABLE_LABEL,
            FABLE_LABEL,
            OTHER_LABEL,
        )
        detector_score = float(balanced_accuracy_score(valid_binary, pred))
        detector_fold_scores.append(detector_score)
        meta.update(
            {
                "fold": fold,
                "official_training_rows": int(len(official_fit)),
                "official_fable_training_rows": int(
                    official_fit["model"].astype(str).eq(FABLE_LABEL).sum()
                ),
                "official_other_training_rows": int(
                    official_fit["model"].astype(str).ne(FABLE_LABEL).sum()
                ),
                "external_fable_training_rows": int(len(fable_external)),
                "validation_rows": int(len(valid)),
                "detector_positive_predictions": int(
                    np.sum(np.asarray(pred, dtype=object) == FABLE_LABEL)
                ),
            }
        )
        feature_meta.append(meta)
        print(
            f"phase11 fold={fold} binary_ba={detector_score:.6f} "
            f"external_fable={len(fable_external)} positives={meta['detector_positive_predictions']} "
            f"features={meta['total_features']}",
            flush=True,
        )

    if any(value is None for value in detector_predictions.tolist()):
        raise RuntimeError("binary detector OOF contains unfilled rows")

    phase7_predictions = phase7_oof["prediction"].astype(str).to_numpy(dtype=object)
    final_predictions, changed = apply_fable_override(
        phase7_predictions,
        detector_predictions,
    )

    final_fold_scores: list[float] = []
    for fold in range(5):
        valid_idx = np.flatnonzero(fold_ids == fold)
        final_fold_scores.append(
            float(
                balanced_accuracy_score(
                    train.iloc[valid_idx]["model"],
                    final_predictions[valid_idx],
                )
            )
        )
    final_result = summarize_oof(train, final_predictions, final_fold_scores)

    binary_truth = np.where(
        train["model"].astype(str).to_numpy() == FABLE_LABEL,
        FABLE_LABEL,
        OTHER_LABEL,
    )
    binary_labels = [FABLE_LABEL, OTHER_LABEL]
    binary_recalls = recall_score(
        binary_truth,
        detector_predictions,
        labels=binary_labels,
        average=None,
        zero_division=0,
    )
    detector_result = {
        "oof_rows": int(len(train)),
        "balanced_accuracy": float(
            balanced_accuracy_score(binary_truth, detector_predictions)
        ),
        "folds": detector_fold_scores,
        "per_binary_label_recall": {
            label: float(value)
            for label, value in zip(binary_labels, binary_recalls)
        },
        "predicted_fable_rows": int(
            np.sum(detector_predictions == FABLE_LABEL)
        ),
    }

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["phase7_prediction"] = phase7_predictions
    oof["fable_detector_prediction"] = detector_predictions
    oof["prediction"] = final_predictions
    oof["changed_by_phase11"] = changed

    return {
        "detector_result": detector_result,
        "final_result": final_result,
        "feature_meta": feature_meta,
    }, oof


def build_report(
    train: pd.DataFrame,
    fable_external: pd.DataFrame,
    phase7_result: dict,
    phase11_bundle: dict,
    oof: pd.DataFrame,
    *,
    fold_fingerprint: str,
) -> dict:
    validate_fable_external(fable_external)

    phase7_ba = float(phase7_result["balanced_accuracy"])
    phase7_fable_recall = float(phase7_result["per_model_recall"][FABLE_LABEL])
    if not np.isclose(
        phase7_ba,
        PHASE7_REFERENCE_BA,
        rtol=0.0,
        atol=NUMERICAL_ATOL,
    ):
        raise RuntimeError(
            f"Phase 7 BA drifted: {phase7_ba} != {PHASE7_REFERENCE_BA}"
        )
    if not np.isclose(
        phase7_fable_recall,
        PHASE7_REFERENCE_FABLE_RECALL,
        rtol=0.0,
        atol=NUMERICAL_ATOL,
    ):
        raise RuntimeError(
            "Phase 7 Fable recall drifted: "
            f"{phase7_fable_recall} != {PHASE7_REFERENCE_FABLE_RECALL}"
        )

    final_result = phase11_bundle["final_result"]
    final_ba = float(final_result["balanced_accuracy"])
    final_fable_recall = float(final_result["per_model_recall"][FABLE_LABEL])

    phase7_prediction = oof["phase7_prediction"].astype(str)
    final_prediction = oof["prediction"].astype(str)
    changed = oof["changed_by_phase11"].to_numpy(dtype=bool)
    phase7_fable_mask = phase7_prediction.eq(FABLE_LABEL).to_numpy()

    no_fable_removed = bool(
        final_prediction.to_numpy(dtype=object)[phase7_fable_mask].tolist()
        == [FABLE_LABEL] * int(phase7_fable_mask.sum())
    )
    only_changes_to_fable = bool(
        (final_prediction[changed] == FABLE_LABEL).all()
    )
    if not no_fable_removed:
        raise RuntimeError("Phase 11 removed an existing Phase 7 Fable prediction")
    if not only_changes_to_fable:
        raise RuntimeError("Phase 11 performed a non-Fable label rewrite")

    decision = phase11_promotion_decision(
        phase7_ba=phase7_ba,
        final_ba=final_ba,
        phase7_fable_recall=phase7_fable_recall,
        final_fable_recall=final_fable_recall,
        no_fable_removed=no_fable_removed,
        only_changes_to_fable=only_changes_to_fable,
    )

    true_fable = oof["model"].astype(str).eq(FABLE_LABEL).to_numpy()
    changed_true_fable = int(np.sum(changed & true_fable))
    changed_non_fable = int(np.sum(changed & ~true_fable))

    return {
        "phase": 11,
        "experiment": "global_fable_binary_override",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": fold_fingerprint,
        },
        "data": {
            "official_rows": int(len(train)),
            "official_prompts": int(train["prompt_id"].nunique()),
            "fable_external_rows": int(len(fable_external)),
            "fable_external_source": "TheFusionCube/Fable-5-CoT-Traces",
            "fable_external_revision": "3624e3a8aab36a89dc3b89aebef0e034a64ea04e",
        },
        "configuration": {
            "baseline": "reproduce canonical Phase 7 in same run",
            "detector": {
                "task": "claude-fable-5 vs __other__",
                "official_training_rows": "all 18 labels from four training folds",
                "external_positive_rows": 344,
                "features": "exact Phase 5b strong-sparse feature construction",
                "classifier": "LinearSVC(C=1.5,class_weight=balanced,dual=True,max_iter=15000)",
                "prediction_rule": "native classifier predict; no tuned threshold",
            },
            "override": (
                "if detector predicts claude-fable-5, output claude-fable-5; "
                "otherwise preserve Phase 7 prediction"
            ),
            "new_tuned_parameters": 0,
        },
        "phase7_reference": {
            "balanced_accuracy": phase7_ba,
            "fable_recall": phase7_fable_recall,
        },
        "detector": phase11_bundle["detector_result"],
        "phase11_final": final_result,
        "override_stats": {
            "changed_rows": int(changed.sum()),
            "changed_true_fable_rows": changed_true_fable,
            "changed_non_fable_rows": changed_non_fable,
            "phase7_predicted_fable_rows": int(phase7_prediction.eq(FABLE_LABEL).sum()),
            "final_predicted_fable_rows": int(final_prediction.eq(FABLE_LABEL).sum()),
        },
        "deltas_vs_phase7": {
            "balanced_accuracy": final_ba - phase7_ba,
            "fable_recall": final_fable_recall - phase7_fable_recall,
            "per_model_recall": {
                label: float(final_result["per_model_recall"][label])
                - float(phase7_result["per_model_recall"][label])
                for label in phase7_result["per_model_recall"]
            },
        },
        "invariants": {
            "no_phase7_fable_prediction_removed": no_fable_removed,
            "every_changed_prediction_changes_to_fable": only_changes_to_fable,
        },
        "feature_meta": phase11_bundle["feature_meta"],
        "decision": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 11 frozen global Claude Fable 5 binary override."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument(
        "--phase7-external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument("--fable-external", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase11/fable_binary_override.json"),
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=Path("artifacts/phase11/fable_binary_override_oof.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train(args.train, args.folds)
    phase7_external = pd.read_parquet(args.phase7_external)
    fable_external = pd.read_parquet(args.fable_external)
    validate_fable_external(fable_external)

    anchor_report, anchor_oof = run_anchor_cv(train, seed=args.seed)
    phase7_bundle, phase7_oof = run_phase7_specialist_cv(
        train,
        phase7_external,
        anchor_oof,
        seed=args.seed,
    )
    phase7_result = phase7_bundle["ensemble_result"]

    phase11_bundle, oof = run_binary_detector_cv(
        train,
        fable_external,
        phase7_oof,
        seed=args.seed,
    )
    report = build_report(
        train,
        fable_external,
        phase7_result,
        phase11_bundle,
        oof,
        fold_fingerprint=anchor_report["validation"]["fold_fingerprint"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.oof_output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
