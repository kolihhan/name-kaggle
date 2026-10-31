from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from scripts.phase12_audit_healthbench_anthropic import normalize_text
from scripts.run_gated_specialist import (
    assert_uncovered_recall_invariant,
    run_specialist_cv as run_phase7_specialist_cv,
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


OPUS_LABEL = "claude-opus-5"
SONNET_LABEL = "claude-sonnet-5"
PAIR_LABELS = {OPUS_LABEL, SONNET_LABEL}
PHASE7_SCORE = 0.3700521998508576
PHASE7_OPUS_RECALL = 0.6133333333333333
PHASE7_SONNET_RECALL = 0.38
PHASE7_PAIR_MEAN_RECALL = (PHASE7_OPUS_RECALL + PHASE7_SONNET_RECALL) / 2.0
EXPECTED_PHASE7_EXTERNAL_ROWS = 2503
EXPECTED_OPUS_EXTERNAL_ROWS = 600
EXPECTED_SONNET_EXTERNAL_ROWS = 610
EXPECTED_HEALTHBENCH_ROWS = EXPECTED_OPUS_EXTERNAL_ROWS + EXPECTED_SONNET_EXTERNAL_ROWS
GAIN_EPSILON = 1e-12


def gate_pair_predictions(
    phase7_predictions: np.ndarray,
    specialist_predictions: np.ndarray,
) -> np.ndarray:
    if len(phase7_predictions) != len(specialist_predictions):
        raise RuntimeError("Phase 7 and specialist prediction lengths differ")

    baseline = np.asarray(phase7_predictions, dtype=object)
    specialist = np.asarray(specialist_predictions, dtype=object)
    observed_specialist = {str(value) for value in specialist.tolist()}
    outside = sorted(observed_specialist - PAIR_LABELS)
    if outside:
        raise RuntimeError(f"specialist predictions outside pair: {outside}")

    result = baseline.copy()
    mask = np.fromiter(
        (str(label) in PAIR_LABELS for label in baseline),
        dtype=bool,
        count=len(baseline),
    )
    result[mask] = specialist[mask]
    return result


def specialist_training_rows(
    official_fit: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    official_pair = official_fit.loc[
        official_fit["model"].astype(str).isin(PAIR_LABELS), ["text", "model"]
    ]
    texts = pd.concat(
        [official_pair["text"], external["text"]], ignore_index=True
    ).astype(str).tolist()
    labels = pd.concat(
        [official_pair["model"], external["model"]], ignore_index=True
    ).astype(str).tolist()
    return texts, labels


def validate_healthbench_external(external: pd.DataFrame) -> None:
    required = {"text", "model", "prompt_id"}
    missing = sorted(required - set(external.columns))
    if missing:
        raise RuntimeError(f"HealthBench external missing columns: {missing}")
    if len(external) != EXPECTED_HEALTHBENCH_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_HEALTHBENCH_ROWS} HealthBench rows, found {len(external)}"
        )
    if external[["text", "model", "prompt_id"]].isna().any().any():
        raise RuntimeError("HealthBench external contains null text/model/prompt_id")
    if (external["text"].astype(str).str.strip() == "").any():
        raise RuntimeError("HealthBench external contains empty text")

    observed_labels = set(external["model"].astype(str))
    if observed_labels != PAIR_LABELS:
        raise RuntimeError(
            f"HealthBench external label set drifted: {sorted(observed_labels)}"
        )

    counts = external["model"].astype(str).value_counts().to_dict()
    expected_counts = {
        OPUS_LABEL: EXPECTED_OPUS_EXTERNAL_ROWS,
        SONNET_LABEL: EXPECTED_SONNET_EXTERNAL_ROWS,
    }
    if counts != expected_counts:
        raise RuntimeError(
            f"HealthBench external row counts drifted: expected={expected_counts} observed={counts}"
        )

    normalized = external["text"].astype(str).map(normalize_text)
    if normalized.nunique() != EXPECTED_HEALTHBENCH_ROWS:
        raise RuntimeError("HealthBench external response text must be globally normalized-unique")

    for label, expected_count in expected_counts.items():
        subset = external.loc[external["model"].astype(str).eq(label)]
        if subset["prompt_id"].astype(str).nunique() != expected_count:
            raise RuntimeError(f"HealthBench prompt_id values must be unique within {label}")


def promotion_decision(
    balanced_accuracy: float,
    pair_mean_recall: float,
    *,
    other_16_recalls_preserved: bool,
    pair_only_changes: bool,
) -> dict[str, bool]:
    material_ba_gain = float(balanced_accuracy) - PHASE7_SCORE > GAIN_EPSILON
    material_pair_gain = (
        float(pair_mean_recall) - PHASE7_PAIR_MEAN_RECALL > GAIN_EPSILON
    )
    promote = (
        material_ba_gain
        and material_pair_gain
        and bool(other_16_recalls_preserved)
        and bool(pair_only_changes)
    )
    return {
        "material_ba_gain": material_ba_gain,
        "material_pair_mean_recall_gain": material_pair_gain,
        "other_16_recalls_preserved": bool(other_16_recalls_preserved),
        "pair_only_changes": bool(pair_only_changes),
        "promote_phase12": promote,
    }


def _validate_phase7_reference(phase7_result: dict) -> None:
    score = float(phase7_result["balanced_accuracy"])
    if not np.isclose(score, PHASE7_SCORE, rtol=0.0, atol=GAIN_EPSILON):
        raise RuntimeError(f"Phase 7 score drifted: {score} != {PHASE7_SCORE}")

    recalls = phase7_result["per_model_recall"]
    expected = {
        OPUS_LABEL: PHASE7_OPUS_RECALL,
        SONNET_LABEL: PHASE7_SONNET_RECALL,
    }
    for label, expected_recall in expected.items():
        observed = float(recalls[label])
        if not np.isclose(observed, expected_recall, rtol=0.0, atol=GAIN_EPSILON):
            raise RuntimeError(
                f"Phase 7 recall drifted for {label}: {observed} != {expected_recall}"
            )


def run_pair_cv(
    train: pd.DataFrame,
    healthbench_external: pd.DataFrame,
    phase7_oof: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[dict, pd.DataFrame]:
    validate_healthbench_external(healthbench_external)
    validate_anchor_oof_alignment(train, phase7_oof)

    official_labels = set(train["model"].astype(str).unique())
    if len(official_labels) != 18:
        raise RuntimeError(f"expected 18 official labels, found {len(official_labels)}")
    if not PAIR_LABELS < official_labels:
        raise RuntimeError("pair labels must be a strict subset of official labels")

    fold_ids = train["fold_id"].to_numpy()
    specialist_predictions = np.full(len(train), None, dtype=object)
    pair_fold_scores: list[float] = []
    feature_meta: list[dict] = []

    for fold in range(5):
        fit_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        official_fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]

        fit_text, fit_y = specialist_training_rows(official_fit, healthbench_external)
        if set(fit_y) != PAIR_LABELS:
            raise RuntimeError(
                f"pair specialist training labels mismatch on fold {fold}: {sorted(set(fit_y))}"
            )

        x_train, x_valid, meta = build_fold_features(
            fit_text,
            valid["text"].astype(str).tolist(),
        )
        model = make_svc(seed + fold)
        model.fit(x_train, fit_y)
        pred = model.predict(x_valid)
        observed = {str(value) for value in pred.tolist()}
        outside = sorted(observed - PAIR_LABELS)
        if outside:
            raise RuntimeError(
                f"pair specialist emitted labels outside pair on fold {fold}: {outside}"
            )
        specialist_predictions[valid_idx] = pred

        true_pair = valid["model"].astype(str).isin(PAIR_LABELS).to_numpy()
        pair_score = float(
            balanced_accuracy_score(valid.loc[true_pair, "model"], pred[true_pair])
        )
        pair_fold_scores.append(pair_score)
        meta.update(
            {
                "fold": fold,
                "official_pair_training_rows": int(
                    official_fit["model"].astype(str).isin(PAIR_LABELS).sum()
                ),
                "healthbench_external_rows": int(len(healthbench_external)),
                "validation_rows": int(len(valid)),
                "pair_validation_rows": int(true_pair.sum()),
            }
        )
        feature_meta.append(meta)
        print(
            f"phase12 fold={fold} pair_score={pair_score:.6f} "
            f"official_pair_fit={meta['official_pair_training_rows']} "
            f"external={len(healthbench_external)} features={meta['total_features']}",
            flush=True,
        )

    if any(value is None for value in specialist_predictions.tolist()):
        raise RuntimeError("pair specialist OOF predictions contain unfilled rows")

    baseline_predictions = phase7_oof["prediction"].astype(str).to_numpy(dtype=object)
    final_predictions = gate_pair_predictions(
        baseline_predictions,
        specialist_predictions,
    )

    final_fold_scores = []
    for fold in range(5):
        valid_idx = np.flatnonzero(fold_ids == fold)
        final_fold_scores.append(
            float(
                balanced_accuracy_score(
                    train.iloc[valid_idx]["model"], final_predictions[valid_idx]
                )
            )
        )
    final_result = summarize_oof(train, final_predictions, final_fold_scores)

    pair_mask = train["model"].astype(str).isin(PAIR_LABELS).to_numpy()
    pair_labels = [OPUS_LABEL, SONNET_LABEL]
    pair_recalls = recall_score(
        train.loc[pair_mask, "model"],
        specialist_predictions[pair_mask],
        labels=pair_labels,
        average=None,
        zero_division=0,
    )
    specialist_result = {
        "scored_rows": int(pair_mask.sum()),
        "balanced_accuracy_on_pair_true_labels": float(
            balanced_accuracy_score(
                train.loc[pair_mask, "model"], specialist_predictions[pair_mask]
            )
        ),
        "folds_on_pair_true_labels": pair_fold_scores,
        "per_model_recall": {
            label: float(value) for label, value in zip(pair_labels, pair_recalls)
        },
    }

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["phase7_prediction"] = baseline_predictions
    oof["pair_specialist_prediction"] = specialist_predictions
    oof["prediction"] = final_predictions
    oof["phase7_predicted_pair"] = oof["phase7_prediction"].isin(PAIR_LABELS)
    oof["changed_by_phase12"] = oof["phase7_prediction"] != oof["prediction"]

    return {
        "final_result": final_result,
        "specialist_result": specialist_result,
        "feature_meta": feature_meta,
    }, oof


def build_report(
    train: pd.DataFrame,
    phase7_external: pd.DataFrame,
    healthbench_external: pd.DataFrame,
    anchor_report: dict,
    phase7_bundle: dict,
    phase7_oof: pd.DataFrame,
    phase12_bundle: dict,
    oof: pd.DataFrame,
) -> dict:
    phase7_result = phase7_bundle["ensemble_result"]
    final_result = phase12_bundle["final_result"]
    _validate_phase7_reference(phase7_result)

    assert_uncovered_recall_invariant(
        phase7_result["per_model_recall"],
        final_result["per_model_recall"],
        covered_labels=PAIR_LABELS,
    )
    other_16_preserved = all(
        np.isclose(
            float(phase7_result["per_model_recall"][label]),
            float(final_result["per_model_recall"][label]),
            rtol=0.0,
            atol=1e-12,
        )
        for label in phase7_result["per_model_recall"]
        if label not in PAIR_LABELS
    )

    changed = oof["changed_by_phase12"]
    pair_only_changes = bool(
        (
            oof.loc[changed, "phase7_prediction"].isin(PAIR_LABELS)
            & oof.loc[changed, "prediction"].isin(PAIR_LABELS)
        ).all()
    )

    final_pair_mean = float(
        np.mean([float(final_result["per_model_recall"][label]) for label in PAIR_LABELS])
    )
    final_score = float(final_result["balanced_accuracy"])
    decision = promotion_decision(
        final_score,
        final_pair_mean,
        other_16_recalls_preserved=other_16_preserved,
        pair_only_changes=pair_only_changes,
    )

    return {
        "phase": 12,
        "experiment": "healthbench_opus5_sonnet5_pair_specialist",
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
            "phase7_external_rows": int(len(phase7_external)),
            "healthbench_external_rows": int(len(healthbench_external)),
            "healthbench_rows_by_model": {
                label: int(
                    healthbench_external["model"].astype(str).eq(label).sum()
                )
                for label in sorted(PAIR_LABELS)
            },
            "healthbench_source_revision": "7a116ea7b4fefe0c72eabb8398ec390c6375f823",
            "pair_labels": sorted(PAIR_LABELS),
        },
        "configuration": {
            "anchor": anchor_report["configuration"],
            "phase7_specialist": anchor_report["configuration"],
            "pair_specialist": anchor_report["configuration"],
            "pair_specialist_classes": 2,
            "gate": (
                "replace Phase 7 prediction with pair-specialist prediction iff Phase 7 "
                "prediction is claude-opus-5 or claude-sonnet-5"
            ),
            "new_tuned_parameters": 0,
        },
        "phase7_reference": {
            "balanced_accuracy": PHASE7_SCORE,
            "opus5_recall": PHASE7_OPUS_RECALL,
            "sonnet5_recall": PHASE7_SONNET_RECALL,
            "pair_mean_recall": PHASE7_PAIR_MEAN_RECALL,
            "reproduced_result": phase7_result,
        },
        "phase12_specialist": phase12_bundle["specialist_result"],
        "phase12_final": final_result,
        "pair_mean_recall": final_pair_mean,
        "gate_stats": {
            "phase7_predicted_pair_rows": int(oof["phase7_predicted_pair"].sum()),
            "phase7_predicted_nonpair_rows": int((~oof["phase7_predicted_pair"]).sum()),
            "changed_rows": int(changed.sum()),
            "changed_opus_to_sonnet": int(
                (
                    changed
                    & oof["phase7_prediction"].eq(OPUS_LABEL)
                    & oof["prediction"].eq(SONNET_LABEL)
                ).sum()
            ),
            "changed_sonnet_to_opus": int(
                (
                    changed
                    & oof["phase7_prediction"].eq(SONNET_LABEL)
                    & oof["prediction"].eq(OPUS_LABEL)
                ).sum()
            ),
        },
        "deltas_vs_phase7": {
            "balanced_accuracy": final_score - PHASE7_SCORE,
            "pair_mean_recall": final_pair_mean - PHASE7_PAIR_MEAN_RECALL,
            "opus5_recall": float(final_result["per_model_recall"][OPUS_LABEL])
            - PHASE7_OPUS_RECALL,
            "sonnet5_recall": float(final_result["per_model_recall"][SONNET_LABEL])
            - PHASE7_SONNET_RECALL,
        },
        "invariants": {
            "other_16_recalls_exactly_preserved": other_16_preserved,
            "every_changed_prediction_stays_inside_pair": pair_only_changes,
        },
        "feature_meta": phase12_bundle["feature_meta"],
        "decision": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 12 frozen Opus 5 / Sonnet 5 paired HealthBench specialist."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument(
        "--phase7-external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument("--healthbench-external", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase12/anthropic_pair_specialist.json"),
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=Path("artifacts/phase12/anthropic_pair_specialist_oof.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train(args.train, args.folds)
    phase7_external = pd.read_parquet(args.phase7_external)
    if len(phase7_external) != EXPECTED_PHASE7_EXTERNAL_ROWS:
        raise RuntimeError(
            f"expected frozen Phase 7 external rows={EXPECTED_PHASE7_EXTERNAL_ROWS}, "
            f"found {len(phase7_external)}"
        )
    healthbench_external = pd.read_parquet(args.healthbench_external)
    validate_healthbench_external(healthbench_external)

    anchor_report, anchor_oof = run_anchor_cv(train, seed=args.seed)
    phase7_bundle, phase7_oof = run_phase7_specialist_cv(
        train,
        phase7_external,
        anchor_oof,
        seed=args.seed,
    )
    _validate_phase7_reference(phase7_bundle["ensemble_result"])

    phase12_bundle, oof = run_pair_cv(
        train,
        healthbench_external,
        phase7_oof,
        seed=args.seed,
    )
    report = build_report(
        train,
        phase7_external,
        healthbench_external,
        anchor_report,
        phase7_bundle,
        phase7_oof,
        phase12_bundle,
        oof,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
