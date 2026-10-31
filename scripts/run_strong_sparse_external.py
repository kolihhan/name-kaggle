from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from scripts.run_strong_sparse_anchor import (
    DEFAULT_SEED,
    build_fold_features,
    load_train,
    make_svc,
    summarize_oof,
)


EXPECTED_EXTERNAL_ROWS = 2503
EXPECTED_EXTERNAL_PROMPTS = 430
EXPECTED_EXTERNAL_LABELS = {
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
}


def validate_external(
    external: pd.DataFrame,
    *,
    official_labels: set[str],
    expected_rows: int = EXPECTED_EXTERNAL_ROWS,
    expected_prompts: int = EXPECTED_EXTERNAL_PROMPTS,
    expected_labels: set[str] = EXPECTED_EXTERNAL_LABELS,
) -> None:
    required = {"text", "model", "prompt_id"}
    missing = sorted(required - set(external.columns))
    if missing:
        raise RuntimeError(f"external data missing columns: {missing}")
    if len(external) != expected_rows:
        raise RuntimeError(
            f"external row count mismatch: {len(external)} != {expected_rows}"
        )
    if external["prompt_id"].astype(str).nunique() != expected_prompts:
        raise RuntimeError(
            "external prompt count mismatch: "
            f"{external['prompt_id'].astype(str).nunique()} != {expected_prompts}"
        )
    observed_labels = set(external["model"].astype(str).unique())
    if observed_labels != set(expected_labels):
        raise RuntimeError(
            f"external labels mismatch: observed={sorted(observed_labels)} "
            f"expected={sorted(expected_labels)}"
        )
    if not observed_labels <= official_labels:
        raise RuntimeError(
            f"external labels outside official label set: {sorted(observed_labels - official_labels)}"
        )
    for column in ["text", "model", "prompt_id"]:
        values = external[column].astype(str)
        if values.map(lambda value: bool(value.strip())).all() is False:
            raise RuntimeError(f"external {column} contains blank values")


def combine_fit_data(
    official_train: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    texts = pd.concat(
        [official_train["text"], external["text"]], ignore_index=True
    ).astype(str).tolist()
    labels = pd.concat(
        [official_train["model"], external["model"]], ignore_index=True
    ).astype(str).tolist()
    return texts, labels


def grouped_recall_means(
    recalls: dict[str, float],
    *,
    covered_labels: Iterable[str],
) -> dict[str, float]:
    covered = set(covered_labels)
    present = set(recalls)
    covered_present = sorted(present & covered)
    uncovered_present = sorted(present - covered)
    if not covered_present or not uncovered_present:
        raise RuntimeError("covered and uncovered recall groups must both be non-empty")
    return {
        "covered": float(np.mean([recalls[label] for label in covered_present])),
        "uncovered": float(np.mean([recalls[label] for label in uncovered_present])),
    }


def run_cv(
    train: pd.DataFrame,
    external: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[dict, pd.DataFrame]:
    labels = sorted(train["model"].astype(str).unique())
    if len(labels) != 18:
        raise RuntimeError(f"expected 18 labels, found {len(labels)}")
    validate_external(external, official_labels=set(labels))

    predictions = np.full(len(train), None, dtype=object)
    fold_scores: list[float] = []
    fold_feature_meta: list[dict] = []

    external_text = external["text"].astype(str).tolist()
    external_model = external["model"].astype(str).tolist()

    for fold in range(5):
        fit_idx = np.flatnonzero(train["fold_id"].to_numpy() != fold)
        valid_idx = np.flatnonzero(train["fold_id"].to_numpy() == fold)
        official_fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]

        fit_text, fit_y = combine_fit_data(official_fit, external)
        if len(fit_text) != len(official_fit) + len(external):
            raise RuntimeError("combined training row count mismatch")
        if fit_text[-len(external_text) :] != external_text:
            raise RuntimeError("external training rows were not appended exactly once")
        if fit_y[-len(external_model) :] != external_model:
            raise RuntimeError("external training labels were not appended exactly once")

        x_train, x_valid, meta = build_fold_features(
            fit_text,
            valid["text"].astype(str).tolist(),
        )
        model = make_svc(seed + fold)
        model.fit(x_train, fit_y)
        pred = model.predict(x_valid)
        predictions[valid_idx] = pred
        score = float(balanced_accuracy_score(valid["model"], pred))
        fold_scores.append(score)
        meta.update(
            {
                "fold": fold,
                "official_training_rows": int(len(official_fit)),
                "external_training_rows": int(len(external)),
                "validation_rows": int(len(valid)),
            }
        )
        fold_feature_meta.append(meta)
        print(
            f"fold={fold} score={score:.6f} official_fit={len(official_fit)} "
            f"external={len(external)} total_features={meta['total_features']}",
            flush=True,
        )

    result = summarize_oof(train, predictions, fold_scores)
    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["prediction"] = predictions
    return {
        "result": result,
        "fold_feature_meta": fold_feature_meta,
    }, oof


def build_report(
    train: pd.DataFrame,
    external: pd.DataFrame,
    transfer: dict,
    anchor_report: dict,
) -> dict:
    anchor_result = anchor_report["result"]
    result = transfer["result"]
    anchor_score = float(anchor_result["balanced_accuracy"])
    score = float(result["balanced_accuracy"])
    anchor_groups = grouped_recall_means(
        anchor_result["per_model_recall"], covered_labels=EXPECTED_EXTERNAL_LABELS
    )
    transfer_groups = grouped_recall_means(
        result["per_model_recall"], covered_labels=EXPECTED_EXTERNAL_LABELS
    )
    labels = sorted(train["model"].astype(str).unique())
    return {
        "phase": 6,
        "experiment": "strong_sparse_plus_audited_external",
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
            "external_labels": sorted(EXPECTED_EXTERNAL_LABELS),
            "external_source": "data/external/gpt_gemini_generations.parquet",
        },
        "configuration": anchor_report["configuration"],
        "anchor": {
            "name": "phase5b_official_only_strong_sparse",
            "balanced_accuracy": anchor_score,
            "mean_recall": anchor_groups,
        },
        "result": result,
        "mean_recall": transfer_groups,
        "deltas": {
            "balanced_accuracy": score - anchor_score,
            "covered_mean_recall": transfer_groups["covered"] - anchor_groups["covered"],
            "uncovered_mean_recall": transfer_groups["uncovered"] - anchor_groups["uncovered"],
            "per_model_recall": {
                label: float(result["per_model_recall"][label])
                - float(anchor_result["per_model_recall"][label])
                for label in labels
            },
        },
        "fold_feature_meta": transfer["fold_feature_meta"],
        "decision": {
            "beats_official_only_anchor": score > anchor_score,
            "promote_external_for_strong_sparse": score > anchor_score,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6: strong sparse anchor plus audited external generations."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument(
        "--anchor",
        type=Path,
        default=Path("artifacts/phase5b/strong_sparse_anchor.json"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase6/strong_sparse_external.json"),
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
    external = pd.read_parquet(args.external)
    anchor_report = json.loads(args.anchor.read_text())
    transfer, oof = run_cv(train, external, seed=args.seed)
    report = build_report(train, external, transfer, anchor_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
