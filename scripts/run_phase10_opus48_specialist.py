from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from scripts.phase10_audit_arc_opus48 import normalize_text
from scripts.run_gated_specialist import (
    EXPECTED_COVERED_LABELS as PHASE7_COVERED_LABELS,
    assert_uncovered_recall_invariant,
    gate_predictions,
    validate_anchor_oof_alignment,
)
from scripts.run_strong_sparse_anchor import (
    DEFAULT_SEED,
    build_fold_features,
    load_train,
    make_svc,
    run_cv as run_anchor_cv,
    summarize_oof,
)
from scripts.run_strong_sparse_external import validate_external


OPUS48_LABEL = "claude-opus-4-8"
PHASE10_COVERED_LABELS = set(PHASE7_COVERED_LABELS) | {OPUS48_LABEL}
PHASE7_SCORE = 0.3700521998508576
PHASE7_OPUS48_RECALL = 0.20666666666666667
EXPECTED_PHASE7_EXTERNAL_ROWS = 2503
EXPECTED_OPUS48_EXTERNAL_ROWS = 868


def specialist_training_rows(
    official_fit: pd.DataFrame,
    phase7_external: pd.DataFrame,
    opus_external: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    official_covered = official_fit.loc[
        official_fit["model"].astype(str).isin(PHASE10_COVERED_LABELS),
        ["text", "model"],
    ]
    texts = pd.concat(
        [official_covered["text"], phase7_external["text"], opus_external["text"]],
        ignore_index=True,
    ).astype(str).tolist()
    labels = pd.concat(
        [official_covered["model"], phase7_external["model"], opus_external["model"]],
        ignore_index=True,
    ).astype(str).tolist()
    return texts, labels


def promotion_decision(
    balanced_accuracy: float,
    opus48_recall: float,
    *,
    uncovered_invariant: bool,
) -> dict[str, bool]:
    beats_phase7 = float(balanced_accuracy) > PHASE7_SCORE
    improves_opus48 = float(opus48_recall) > PHASE7_OPUS48_RECALL
    promote = beats_phase7 and improves_opus48 and bool(uncovered_invariant)
    return {
        "beats_phase7": beats_phase7,
        "improves_opus48_recall": improves_opus48,
        "uncovered_recall_invariant": bool(uncovered_invariant),
        "promote_phase10": promote,
    }


def validate_phase10_external(
    phase7_external: pd.DataFrame,
    opus_external: pd.DataFrame,
    *,
    official_labels: set[str],
) -> None:
    validate_external(phase7_external, official_labels=official_labels)
    if len(phase7_external) != EXPECTED_PHASE7_EXTERNAL_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_PHASE7_EXTERNAL_ROWS} frozen Phase 7 external rows, "
            f"found {len(phase7_external)}"
        )
    observed_phase7 = set(phase7_external["model"].astype(str))
    if observed_phase7 != PHASE7_COVERED_LABELS:
        raise RuntimeError(
            f"Phase 7 external label set drifted: {sorted(observed_phase7)}"
        )

    required = {"text", "model"}
    missing = sorted(required - set(opus_external.columns))
    if missing:
        raise RuntimeError(f"Opus 4.8 external missing columns: {missing}")
    if len(opus_external) != EXPECTED_OPUS48_EXTERNAL_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_OPUS48_EXTERNAL_ROWS} Opus 4.8 external rows, "
            f"found {len(opus_external)}"
        )
    if opus_external["text"].isna().any() or opus_external["model"].isna().any():
        raise RuntimeError("Opus 4.8 external contains null text/model")
    if (opus_external["text"].astype(str).str.strip() == "").any():
        raise RuntimeError("Opus 4.8 external contains empty text")
    observed_opus = set(opus_external["model"].astype(str))
    if observed_opus != {OPUS48_LABEL}:
        raise RuntimeError(f"Opus 4.8 model identity drifted: {sorted(observed_opus)}")
    normalized = opus_external["text"].astype(str).map(normalize_text)
    if normalized.nunique() != EXPECTED_OPUS48_EXTERNAL_ROWS:
        raise RuntimeError("Opus 4.8 normalized response uniqueness drifted")


def run_phase10_cv(
    train: pd.DataFrame,
    phase7_external: pd.DataFrame,
    opus_external: pd.DataFrame,
    anchor_oof: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[dict, pd.DataFrame]:
    validate_anchor_oof_alignment(train, anchor_oof)
    official_labels = set(train["model"].astype(str).unique())
    if len(official_labels) != 18:
        raise RuntimeError(f"expected 18 official labels, found {len(official_labels)}")
    if not PHASE10_COVERED_LABELS < official_labels:
        raise RuntimeError("Phase 10 covered labels must be a strict subset of official labels")
    validate_phase10_external(
        phase7_external,
        opus_external,
        official_labels=official_labels,
    )

    fold_ids = train["fold_id"].to_numpy()
    specialist_predictions = np.full(len(train), None, dtype=object)
    specialist_fold_scores: list[float] = []
    feature_meta: list[dict] = []

    for fold in range(5):
        fit_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        official_fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]

        fit_text, fit_y = specialist_training_rows(
            official_fit,
            phase7_external,
            opus_external,
        )
        if set(fit_y) != PHASE10_COVERED_LABELS:
            raise RuntimeError(
                f"Phase 10 specialist labels mismatch on fold {fold}: {sorted(set(fit_y))}"
            )

        x_train, x_valid, meta = build_fold_features(
            fit_text,
            valid["text"].astype(str).tolist(),
        )
        model = make_svc(seed + fold)
        model.fit(x_train, fit_y)
        pred = model.predict(x_valid)
        observed = set(str(x) for x in pred.tolist())
        if not observed <= PHASE10_COVERED_LABELS:
            raise RuntimeError(
                f"Phase 10 specialist emitted outside labels on fold {fold}: "
                f"{sorted(observed - PHASE10_COVERED_LABELS)}"
            )
        specialist_predictions[valid_idx] = pred

        true_covered = valid["model"].astype(str).isin(PHASE10_COVERED_LABELS).to_numpy()
        covered_score = float(
            balanced_accuracy_score(valid.loc[true_covered, "model"], pred[true_covered])
        )
        specialist_fold_scores.append(covered_score)
        meta.update(
            {
                "fold": fold,
                "official_covered_training_rows": int(
                    official_fit["model"].astype(str).isin(PHASE10_COVERED_LABELS).sum()
                ),
                "phase7_external_training_rows": int(len(phase7_external)),
                "opus48_external_training_rows": int(len(opus_external)),
                "validation_rows": int(len(valid)),
                "covered_validation_rows": int(true_covered.sum()),
            }
        )
        feature_meta.append(meta)
        print(
            f"phase10 fold={fold} covered_score={covered_score:.6f} "
            f"phase7_external={len(phase7_external)} opus48_external={len(opus_external)} "
            f"features={meta['total_features']}",
            flush=True,
        )

    if any(value is None for value in specialist_predictions.tolist()):
        raise RuntimeError("Phase 10 specialist OOF contains unfilled rows")

    anchor_predictions = anchor_oof["prediction"].astype(str).to_numpy(dtype=object)
    final_predictions = gate_predictions(
        anchor_predictions,
        specialist_predictions,
        PHASE10_COVERED_LABELS,
    )

    fold_scores = []
    for fold in range(5):
        valid_idx = np.flatnonzero(fold_ids == fold)
        fold_scores.append(
            float(
                balanced_accuracy_score(
                    train.iloc[valid_idx]["model"], final_predictions[valid_idx]
                )
            )
        )
    final_result = summarize_oof(train, final_predictions, fold_scores)

    covered_mask = train["model"].astype(str).isin(PHASE10_COVERED_LABELS).to_numpy()
    covered_labels = sorted(PHASE10_COVERED_LABELS)
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
        "folds_on_covered_true_labels": specialist_fold_scores,
        "per_model_recall": {
            label: float(value)
            for label, value in zip(covered_labels, specialist_recalls)
        },
    }

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["anchor_prediction"] = anchor_predictions
    oof["specialist_prediction"] = specialist_predictions
    oof["prediction"] = final_predictions
    oof["anchor_was_covered"] = oof["anchor_prediction"].isin(PHASE10_COVERED_LABELS)
    oof["changed_by_phase10"] = oof["anchor_prediction"] != oof["prediction"]

    return {
        "final_result": final_result,
        "specialist_result": specialist_result,
        "feature_meta": feature_meta,
    }, oof


def build_report(
    train: pd.DataFrame,
    phase7_external: pd.DataFrame,
    opus_external: pd.DataFrame,
    anchor_report: dict,
    phase10_bundle: dict,
    oof: pd.DataFrame,
) -> dict:
    anchor_result = anchor_report["result"]
    final_result = phase10_bundle["final_result"]

    assert_uncovered_recall_invariant(
        anchor_result["per_model_recall"],
        final_result["per_model_recall"],
        covered_labels=PHASE10_COVERED_LABELS,
    )
    outside_mask = ~oof["anchor_was_covered"]
    unchanged_outside = bool(
        (
            oof.loc[outside_mask, "anchor_prediction"]
            == oof.loc[outside_mask, "prediction"]
        ).all()
    )
    if not unchanged_outside:
        raise RuntimeError("Phase 10 changed an anchor prediction outside the nine-class gate")

    final_score = float(final_result["balanced_accuracy"])
    opus48_recall = float(final_result["per_model_recall"][OPUS48_LABEL])
    decision = promotion_decision(
        final_score,
        opus48_recall,
        uncovered_invariant=True,
    )

    return {
        "phase": 10,
        "experiment": "opus48_nine_class_gated_specialist",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": anchor_report["validation"]["fold_fingerprint"],
        },
        "data": {
            "official_rows": int(len(train)),
            "phase7_external_rows": int(len(phase7_external)),
            "opus48_external_rows": int(len(opus_external)),
            "covered_labels": sorted(PHASE10_COVERED_LABELS),
            "uncovered_labels": sorted(
                set(train["model"].astype(str).unique()) - PHASE10_COVERED_LABELS
            ),
            "opus48_source_revision": "026789c1c12a4c34580a32e84dcaf5630d7e8f31",
        },
        "configuration": {
            "anchor": anchor_report["configuration"],
            "specialist": anchor_report["configuration"],
            "specialist_classes": 9,
            "gate": (
                "replace Phase 5b anchor label with specialist label iff anchor prediction "
                "is in the nine covered labels"
            ),
            "new_tuned_parameters": 0,
        },
        "phase7_reference": {
            "balanced_accuracy": PHASE7_SCORE,
            "opus48_recall": PHASE7_OPUS48_RECALL,
        },
        "anchor": anchor_result,
        "phase10_specialist": phase10_bundle["specialist_result"],
        "phase10_final": final_result,
        "gate_stats": {
            "anchor_predicted_covered_rows": int(oof["anchor_was_covered"].sum()),
            "anchor_predicted_uncovered_rows": int((~oof["anchor_was_covered"]).sum()),
            "changed_rows": int(oof["changed_by_phase10"].sum()),
        },
        "deltas_vs_phase7": {
            "balanced_accuracy": final_score - PHASE7_SCORE,
            "opus48_recall": opus48_recall - PHASE7_OPUS48_RECALL,
        },
        "invariants": {
            "remaining_nine_uncovered_recalls_exactly_preserved": True,
            "anchor_predictions_outside_nine_class_gate_never_changed": unchanged_outside,
        },
        "feature_meta": phase10_bundle["feature_meta"],
        "decision": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 10 frozen nine-class specialist with admitted ARC Opus 4.8 responses."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument(
        "--phase7-external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument("--opus48-external", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase10/opus48_specialist.json"),
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=Path("artifacts/phase10/opus48_specialist_oof.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train(args.train, args.folds)
    phase7_external = pd.read_parquet(args.phase7_external)
    opus_external = pd.read_parquet(args.opus48_external)

    anchor_report, anchor_oof = run_anchor_cv(train, seed=args.seed)
    phase10_bundle, oof = run_phase10_cv(
        train,
        phase7_external,
        opus_external,
        anchor_oof,
        seed=args.seed,
    )
    report = build_report(
        train,
        phase7_external,
        opus_external,
        anchor_report,
        phase10_bundle,
        oof,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
