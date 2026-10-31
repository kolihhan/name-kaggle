from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from scripts.phase8_audit_response_corpus import (
    PRECOMPETITION_SOURCES,
    build_clean_mask,
    candidate_from_sources,
)
from scripts.run_gated_specialist import (
    build_report as build_phase7_report,
    run_specialist_cv as run_phase7_specialist_cv,
)
from scripts.run_strong_sparse_anchor import (
    DEFAULT_SEED,
    build_fold_features,
    load_train,
    make_svc,
    run_cv as run_anchor_cv,
    summarize_oof,
)

PHASE9_SPECIALIST_LABELS = {
    "claude-opus-4-6",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
    "qwen3.8-max",
}

EXPECTED_EXTERNAL_COUNTS = {
    "claude-opus-4-6": 1934,
    "glm-5.3": 200,
    "glm-5.3-flash": 284,
    "kimi-k3": 317,
    "qwen3.8-max": 186,
}
EXPECTED_EXTERNAL_ROWS = 2921


def gate_predictions(
    baseline_predictions: np.ndarray,
    specialist_predictions: np.ndarray,
    specialist_labels: Iterable[str],
) -> np.ndarray:
    if len(baseline_predictions) != len(specialist_predictions):
        raise RuntimeError("baseline and specialist prediction lengths differ")
    labels = set(specialist_labels)
    observed = set(str(x) for x in specialist_predictions.tolist())
    outside = sorted(observed - labels)
    if outside:
        raise RuntimeError(f"specialist predictions outside specialist label set: {outside}")

    baseline = np.asarray(baseline_predictions, dtype=object)
    specialist = np.asarray(specialist_predictions, dtype=object)
    result = baseline.copy()
    mask = np.fromiter((str(x) in labels for x in baseline), dtype=bool, count=len(baseline))
    result[mask] = specialist[mask]
    return result


def assert_outside_recall_invariant(
    before: dict[str, float],
    after: dict[str, float],
    *,
    specialist_labels: Iterable[str],
) -> None:
    labels = set(specialist_labels)
    if set(before) != set(after):
        raise RuntimeError("recall label sets differ")
    for label, old in before.items():
        if label in labels:
            continue
        new = after[label]
        if not np.isclose(old, new, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"outside-specialist recall drift for {label}: {old} -> {new}")


def validate_precompetition_external(external: pd.DataFrame) -> None:
    required = {"response_id", "text", "model"}
    missing = sorted(required - set(external.columns))
    if missing:
        raise RuntimeError(f"precompetition external missing columns: {missing}")
    if len(external) != EXPECTED_EXTERNAL_ROWS:
        raise RuntimeError(f"expected exactly 2921 precompetition rows, found {len(external)}")
    if external["response_id"].astype(str).nunique() != EXPECTED_EXTERNAL_ROWS:
        raise RuntimeError("precompetition response_id values must be unique")
    if external["text"].isna().any() or external["model"].isna().any():
        raise RuntimeError("precompetition external contains null text/model")
    observed = external["model"].astype(str).value_counts().to_dict()
    if observed != EXPECTED_EXTERNAL_COUNTS:
        raise RuntimeError(f"precompetition label counts drifted: {observed}")


def load_precompetition_external(corpus_root: Path) -> pd.DataFrame:
    wide = pd.read_parquet(corpus_root / "external_short_wide.parquet")
    cells = pd.read_parquet(corpus_root / "external_cell_metadata.parquet")
    prov = pd.read_parquet(corpus_root / "external_cell_provenance.parquet")

    model_columns = [c for c in wide.columns if c not in {"id", "prompt"}]
    long = wide.melt(
        id_vars=["id", "prompt"],
        value_vars=model_columns,
        var_name="model",
        value_name="text",
    ).dropna(subset=["text"])
    short_cells = cells[cells["view"].eq("short")].copy()
    merged = long.merge(
        short_cells,
        on=["id", "model"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_meta"),
    )
    if merged["response_id"].isna().any():
        raise RuntimeError("short response cells failed metadata join")
    clean = merged[build_clean_mask(merged)].copy()
    clean_ids = set(clean["response_id"].astype(str))
    clean_prov = prov[
        prov["cell_id_short"].notna()
        & prov["response_id"].astype(str).isin(clean_ids)
    ].copy()
    candidate = candidate_from_sources(clean, clean_prov, sorted(PRECOMPETITION_SOURCES))
    candidate = candidate[["response_id", "text", "model"]].reset_index(drop=True)
    validate_precompetition_external(candidate)
    return candidate


def specialist_training_rows(
    official_fit: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    official = official_fit.loc[
        official_fit["model"].astype(str).isin(PHASE9_SPECIALIST_LABELS),
        ["text", "model"],
    ]
    text = pd.concat([official["text"], external["text"]], ignore_index=True).astype(str).tolist()
    labels = pd.concat([official["model"], external["model"]], ignore_index=True).astype(str).tolist()
    return text, labels


def run_phase9_cv(
    train: pd.DataFrame,
    external: pd.DataFrame,
    phase7_oof: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[dict, pd.DataFrame]:
    validate_precompetition_external(external)
    required = ["id", "prompt_id", "model", "fold_id", "prediction"]
    if any(c not in phase7_oof.columns for c in required):
        raise RuntimeError("Phase 7 OOF missing required columns")
    if not train[["id", "prompt_id", "model", "fold_id"]].reset_index(drop=True).equals(
        phase7_oof[["id", "prompt_id", "model", "fold_id"]].reset_index(drop=True)
    ):
        raise RuntimeError("Phase 7 OOF alignment mismatch")

    fold_ids = train["fold_id"].to_numpy()
    specialist_predictions = np.full(len(train), None, dtype=object)
    specialist_fold_scores: list[float] = []
    feature_meta: list[dict] = []

    for fold in range(5):
        fit_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        official_fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]
        fit_text, fit_y = specialist_training_rows(official_fit, external)
        if set(fit_y) != PHASE9_SPECIALIST_LABELS:
            raise RuntimeError(f"Phase 9 training label set drifted on fold {fold}")

        x_train, x_valid, meta = build_fold_features(
            fit_text,
            valid["text"].astype(str).tolist(),
        )
        model = make_svc(seed + fold)
        model.fit(x_train, fit_y)
        pred = model.predict(x_valid)
        if not set(str(x) for x in pred.tolist()) <= PHASE9_SPECIALIST_LABELS:
            raise RuntimeError(f"Phase 9 specialist emitted outside label on fold {fold}")
        specialist_predictions[valid_idx] = pred

        true_mask = valid["model"].astype(str).isin(PHASE9_SPECIALIST_LABELS).to_numpy()
        specialist_fold_scores.append(
            float(balanced_accuracy_score(valid.loc[true_mask, "model"], pred[true_mask]))
        )
        meta.update(
            {
                "fold": fold,
                "official_specialist_training_rows": int(
                    official_fit["model"].astype(str).isin(PHASE9_SPECIALIST_LABELS).sum()
                ),
                "external_training_rows": int(len(external)),
                "validation_rows": int(len(valid)),
                "specialist_true_validation_rows": int(true_mask.sum()),
            }
        )
        feature_meta.append(meta)
        print(
            f"phase9 fold={fold} specialist_true_ba={specialist_fold_scores[-1]:.6f} "
            f"external={len(external)} features={meta['total_features']}",
            flush=True,
        )

    if any(x is None for x in specialist_predictions.tolist()):
        raise RuntimeError("Phase 9 specialist OOF contains unfilled rows")

    baseline = phase7_oof["prediction"].astype(str).to_numpy(dtype=object)
    final = gate_predictions(baseline, specialist_predictions, PHASE9_SPECIALIST_LABELS)
    fold_scores = []
    for fold in range(5):
        idx = np.flatnonzero(fold_ids == fold)
        fold_scores.append(float(balanced_accuracy_score(train.iloc[idx]["model"], final[idx])))
    final_result = summarize_oof(train, final, fold_scores)

    true_mask = train["model"].astype(str).isin(PHASE9_SPECIALIST_LABELS).to_numpy()
    labels = sorted(PHASE9_SPECIALIST_LABELS)
    recalls = recall_score(
        train.loc[true_mask, "model"],
        specialist_predictions[true_mask],
        labels=labels,
        average=None,
        zero_division=0,
    )
    specialist_result = {
        "scored_rows": int(true_mask.sum()),
        "balanced_accuracy_on_specialist_true_labels": float(
            balanced_accuracy_score(train.loc[true_mask, "model"], specialist_predictions[true_mask])
        ),
        "folds_on_specialist_true_labels": specialist_fold_scores,
        "per_model_recall": {label: float(value) for label, value in zip(labels, recalls)},
    }

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["phase7_prediction"] = baseline
    oof["phase9_specialist_prediction"] = specialist_predictions
    oof["prediction"] = final
    oof["phase7_predicted_specialist_label"] = oof["phase7_prediction"].isin(PHASE9_SPECIALIST_LABELS)
    oof["changed_by_phase9"] = oof["phase7_prediction"] != oof["prediction"]

    return {
        "final_result": final_result,
        "specialist_result": specialist_result,
        "feature_meta": feature_meta,
    }, oof


def build_report(
    train: pd.DataFrame,
    external: pd.DataFrame,
    phase7_report: dict,
    phase9_bundle: dict,
    oof: pd.DataFrame,
) -> dict:
    baseline_result = phase7_report["ensemble"]["result"]
    final_result = phase9_bundle["final_result"]
    assert_outside_recall_invariant(
        baseline_result["per_model_recall"],
        final_result["per_model_recall"],
        specialist_labels=PHASE9_SPECIALIST_LABELS,
    )
    baseline_score = float(baseline_result["balanced_accuracy"])
    final_score = float(final_result["balanced_accuracy"])
    gate_mask = oof["phase7_predicted_specialist_label"].to_numpy(dtype=bool)

    return {
        "phase": 9,
        "experiment": "precompetition_five_class_gated_specialist",
        "validation_statement": (
            "Official validation remains the frozen five-fold GroupKFold(prompt_id) OOF. "
            "External sources/prompt banks demonstrably predate competition launch and have zero "
            "exact/normalized official answer-text overlap. Official prompt text is unavailable, "
            "so residual accidental prompt identity is unobservable; this run is not claimed to be "
            "provably prompt-leakage-free. External rows are training-only."
        ),
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": phase7_report["validation"]["fold_fingerprint"],
        },
        "data": {
            "official_rows": int(len(train)),
            "external_rows": int(len(external)),
            "external_unique_response_ids": int(external["response_id"].nunique()),
            "external_counts": {k: int(v) for k, v in external["model"].value_counts().sort_index().items()},
            "external_sources": sorted(PRECOMPETITION_SOURCES),
            "specialist_labels": sorted(PHASE9_SPECIALIST_LABELS),
        },
        "configuration": {
            "phase7_baseline": "reproduced in same run from canonical anchor + Phase 7 8-class specialist",
            "phase9_specialist": phase7_report["configuration"]["anchor"],
            "specialist_classes": 5,
            "gate": "replace Phase 7 prediction iff it is one of the five Phase 9 specialist labels",
            "new_tuned_parameters": 0,
        },
        "phase7_baseline": baseline_result,
        "phase9_specialist": phase9_bundle["specialist_result"],
        "phase9_final": final_result,
        "gate_stats": {
            "phase7_predicted_specialist_rows": int(gate_mask.sum()),
            "phase7_predicted_other_rows": int((~gate_mask).sum()),
            "changed_rows": int(oof["changed_by_phase9"].sum()),
        },
        "deltas": {
            "balanced_accuracy": final_score - baseline_score,
            "per_model_recall": {
                label: float(final_result["per_model_recall"][label]) - float(baseline_result["per_model_recall"][label])
                for label in baseline_result["per_model_recall"]
            },
        },
        "invariants": {
            "outside_13_label_recall_exactly_preserved": True,
            "phase7_predictions_outside_five_class_gate_never_changed": bool(
                (
                    oof.loc[~oof["phase7_predicted_specialist_label"], "phase7_prediction"]
                    == oof.loc[~oof["phase7_predicted_specialist_label"], "prediction"]
                ).all()
            ),
        },
        "feature_meta": phase9_bundle["feature_meta"],
        "decision": {
            "beats_phase7": final_score > baseline_score,
            "promote_phase9": final_score > baseline_score,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 frozen precompetition five-class specialist")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument(
        "--phase7-external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase9/precompetition_specialist.json"))
    parser.add_argument("--oof-output", type=Path, default=Path("artifacts/phase9/precompetition_specialist_oof.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train(args.train, args.folds)
    phase7_external = pd.read_parquet(args.phase7_external)
    phase9_external = load_precompetition_external(args.corpus_root)

    anchor_report, anchor_oof = run_anchor_cv(train, seed=args.seed)
    phase7_bundle, phase7_oof = run_phase7_specialist_cv(
        train, phase7_external, anchor_oof, seed=args.seed
    )
    phase7_report = build_phase7_report(
        train, phase7_external, anchor_report, phase7_bundle, phase7_oof
    )
    phase9_bundle, oof = run_phase9_cv(
        train, phase9_external, phase7_oof, seed=args.seed
    )
    report = build_report(train, phase9_external, phase7_report, phase9_bundle, oof)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
