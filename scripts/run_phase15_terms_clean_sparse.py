from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from scripts.phase15_terms_clean import (
    BLOCKED_EXTERNAL_LABELS,
    EXPECTED_RETAINED_LABEL_COUNTS,
    EXPECTED_TERMS_CLEAN_ROWS,
)
from scripts.run_phase14_public_sparse import (
    build_fold_features,
    load_train,
    make_svc,
    summarize_oof,
)


PHASE7_SCORE = 0.3700521998508576
PHASE14_DIAGNOSTIC_SCORE = 0.4178100919711658
NUMERIC_ATOL = 1e-12
FOLD_FINGERPRINT = "bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d"
EXPECTED_EXTERNAL_PROMPTS = 2_801


def promotion_decision(score: float) -> dict[str, float | bool]:
    score = float(score)
    beats_phase7 = score > PHASE7_SCORE + NUMERIC_ATOL
    beats_phase14 = score > PHASE14_DIAGNOSTIC_SCORE + NUMERIC_ATOL
    return {
        "phase7_reference": PHASE7_SCORE,
        "phase14_diagnostic_reference": PHASE14_DIAGNOSTIC_SCORE,
        "balanced_accuracy": score,
        "delta_vs_phase7": score - PHASE7_SCORE,
        "delta_vs_phase14_diagnostic": score - PHASE14_DIAGNOSTIC_SCORE,
        "beats_phase7_beyond_numeric_equality": beats_phase7,
        "beats_phase14_diagnostic": beats_phase14,
        "promote_phase15": beats_phase7,
    }


def validate_terms_clean_external(external: pd.DataFrame) -> None:
    required = {"prompt_id", "text", "model"}
    missing = sorted(required - set(external.columns))
    if missing:
        raise RuntimeError(f"Phase 15 external frame missing required columns: {missing}")
    if len(external) != EXPECTED_TERMS_CLEAN_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_TERMS_CLEAN_ROWS} terms-clean external rows, found {len(external)}"
        )
    observed_prompts = int(external["prompt_id"].astype(str).nunique())
    if observed_prompts != EXPECTED_EXTERNAL_PROMPTS:
        raise RuntimeError(
            f"expected {EXPECTED_EXTERNAL_PROMPTS} external prompt IDs, found {observed_prompts}"
        )
    observed_counts = {
        str(label): int(count)
        for label, count in external["model"].astype(str).value_counts().sort_index().items()
    }
    if observed_counts != EXPECTED_RETAINED_LABEL_COUNTS:
        raise RuntimeError(
            f"Phase 15 retained label counts drifted: {observed_counts} != {EXPECTED_RETAINED_LABEL_COUNTS}"
        )
    blocked = set(external["model"].astype(str).unique()) & BLOCKED_EXTERNAL_LABELS
    if blocked:
        raise RuntimeError(f"provider-terms-blocked external labels remain: {sorted(blocked)}")
    if external["text"].isna().any() or external["model"].isna().any():
        raise RuntimeError("Phase 15 external frame contains null text/model")
    if (external["text"].astype(str).str.strip() == "").any():
        raise RuntimeError("Phase 15 external frame contains empty text")


def run_cv(train: pd.DataFrame, external: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    labels = sorted(train["model"].astype(str).unique())
    if len(train) != 2_699:
        raise RuntimeError(f"expected 2699 official rows, found {len(train)}")
    if len(labels) != 18:
        raise RuntimeError(f"expected 18 official labels, found {len(labels)}")
    validate_terms_clean_external(external)

    fold_ids = train["fold_id"].to_numpy()
    predictions = np.full(len(train), None, dtype=object)
    fold_scores: list[float] = []
    feature_meta: list[dict] = []
    ext_text = external["text"].astype(str).tolist()
    ext_y = external["model"].astype(str).tolist()

    for fold in range(5):
        fit_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]

        fit_text = fit["text"].astype(str).tolist() + ext_text
        fit_y = fit["model"].astype(str).tolist() + ext_y
        valid_text = valid["text"].astype(str).tolist()

        x_train, x_valid, meta = build_fold_features(fit_text, valid_text)
        model = make_svc()
        model.fit(x_train, fit_y)
        pred = model.predict(x_valid)
        predictions[valid_idx] = pred
        score = float(balanced_accuracy_score(valid["model"], pred))
        fold_scores.append(score)
        meta.update(
            {
                "fold": fold,
                "official_training_rows": int(len(fit)),
                "external_training_rows": int(len(external)),
                "total_training_rows": int(len(fit_text)),
                "official_validation_rows": int(len(valid)),
                "external_validation_rows": 0,
            }
        )
        feature_meta.append(meta)
        print(
            f"phase15 fold={fold} score={score:.6f} "
            f"train={len(fit_text)} valid={len(valid)} features={meta['total_features']}",
            flush=True,
        )

        del x_train, x_valid, model, fit_text, fit_y, valid_text

    result = summarize_oof(train, predictions, fold_scores)
    decision = promotion_decision(result["balanced_accuracy"])

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["prediction"] = predictions
    if len(oof) != 2_699 or oof["id"].nunique() != 2_699:
        raise RuntimeError("Phase 15 OOF row/ID invariant failed")
    if oof["prediction"].isna().any():
        raise RuntimeError("Phase 15 OOF contains null predictions")

    report = {
        "phase": 15,
        "experiment": "provider_terms_clean_public_four_view_svc",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": FOLD_FINGERPRINT,
        },
        "data": {
            "official_rows": int(len(train)),
            "external_rows": int(len(external)),
            "external_prompt_ids": int(external["prompt_id"].astype(str).nunique()),
            "external_labels": {
                str(label): int(count)
                for label, count in external["model"].astype(str).value_counts().sort_index().items()
            },
            "blocked_provider_labels": sorted(BLOCKED_EXTERNAL_LABELS),
        },
        "configuration": {
            "recipe": "exact_phase14_four_view_sparse_recipe_unchanged",
            "classifier": "LinearSVC(C=0.5,random_state=42,max_iter=5000,class_weight=balanced)",
            "new_tuned_parameters": 0,
        },
        "fold_feature_meta": feature_meta,
        "result": result,
        "decision": decision,
        "invariants": {
            "no_blocked_external_labels": True,
            "external_validation_rows_all_zero": all(
                meta["external_validation_rows"] == 0 for meta in feature_meta
            ),
        },
    }
    return report, oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen Phase 15 terms-clean public sparse CV.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train(args.train, args.folds)
    external = pd.read_parquet(args.external)
    report, oof = run_cv(train, external)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.oof_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
