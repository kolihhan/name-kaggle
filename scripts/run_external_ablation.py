from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.svm import LinearSVC


BASELINE_CONFIG = {
    "name": "char_default",
    "analyzer": "char",
    "sublinear_tf": False,
    "min_df": 1,
}
BASELINE_TOLERANCE = 0.002


def run_cv(train: pd.DataFrame, external: pd.DataFrame | None, config: dict) -> dict:
    labels = sorted(train["model"].unique())
    predictions = np.empty(len(train), dtype=object)
    fold_scores: list[float] = []

    for fold in range(5):
        train_fold = train[train.fold_id != fold]
        valid_fold = train[train.fold_id == fold]

        if external is None:
            fit_text = train_fold["text"].astype(str).tolist()
            fit_y = train_fold["model"].astype(str).tolist()
        else:
            fit_text = pd.concat(
                [train_fold["text"], external["text"]], ignore_index=True
            ).astype(str).tolist()
            fit_y = pd.concat(
                [train_fold["model"], external["model"]], ignore_index=True
            ).astype(str).tolist()

        vectorizer = TfidfVectorizer(
            analyzer=config["analyzer"],
            ngram_range=(1, 5),
            max_features=80000,
            min_df=config["min_df"],
            sublinear_tf=config["sublinear_tf"],
            dtype=np.float32,
        )
        x_train = vectorizer.fit_transform(fit_text)
        x_valid = vectorizer.transform(valid_fold["text"].astype(str))
        classifier = LinearSVC(class_weight="balanced")
        classifier.fit(x_train, fit_y)
        fold_pred = classifier.predict(x_valid)
        predictions[valid_fold.index.to_numpy()] = fold_pred
        fold_scores.append(
            float(balanced_accuracy_score(valid_fold["model"], fold_pred))
        )

    balanced_accuracy = float(balanced_accuracy_score(train["model"], predictions))
    recalls = recall_score(
        train["model"],
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "balanced_accuracy": balanced_accuracy,
        "folds": fold_scores,
        "per_model_recall": {
            label: float(recall) for label, recall in zip(labels, recalls)
        },
    }


def run_ablation(
    train_path: Path,
    folds_path: Path,
    external_path: Path,
    phase1_baseline_path: Path,
    output_path: Path,
) -> dict:
    train = pd.read_csv(train_path)
    folds = pd.read_csv(folds_path)[["id", "fold_id"]]
    train = train.merge(folds, on="id", how="left", validate="one_to_one")
    if train["fold_id"].isna().any():
        raise RuntimeError("fold mapping is incomplete")

    external = pd.read_parquet(external_path)
    required_external = {"text", "model", "prompt_id"}
    missing = sorted(required_external - set(external.columns))
    if missing:
        raise RuntimeError(f"external data missing columns: {missing}")

    labels = sorted(train["model"].unique())
    target_baseline = json.loads(phase1_baseline_path.read_text())[
        "oof_balanced_accuracy"
    ]

    baseline = run_cv(train, None, BASELINE_CONFIG)
    reproduction_delta = abs(baseline["balanced_accuracy"] - target_baseline)
    if reproduction_delta > BASELINE_TOLERANCE:
        raise RuntimeError(
            "Frozen Phase-1 baseline no longer reproduces closely: "
            f"target={target_baseline:.8f}, "
            f"actual={baseline['balanced_accuracy']:.8f}, "
            f"delta={reproduction_delta:.8f}"
        )

    all_external = external[["text", "model"]].copy()
    augmented_all = run_cv(train, all_external, BASELINE_CONFIG)

    cap = int(external["model"].value_counts().min())
    balanced_external = (
        external.groupby("model", group_keys=False)
        .apply(lambda frame: frame.sample(n=min(cap, len(frame)), random_state=42))
        .reset_index(drop=True)[["text", "model"]]
    )
    augmented_balanced = run_cv(train, balanced_external, BASELINE_CONFIG)

    augmented_labels = sorted(external["model"].unique())
    untouched_labels = sorted(set(labels) - set(augmented_labels))

    def summarize(name: str, result: dict, n_external: int) -> dict:
        per_model = result["per_model_recall"]
        return {
            "name": name,
            "external_rows": int(n_external),
            "balanced_accuracy": result["balanced_accuracy"],
            "delta_vs_baseline": result["balanced_accuracy"]
            - baseline["balanced_accuracy"],
            "folds": result["folds"],
            "mean_recall_augmented_labels": float(
                np.mean([per_model[label] for label in augmented_labels])
            ),
            "mean_recall_untouched_labels": float(
                np.mean([per_model[label] for label in untouched_labels])
            ),
            "per_model_recall": per_model,
            "per_model_delta_vs_baseline": {
                label: per_model[label] - baseline["per_model_recall"][label]
                for label in labels
            },
        }

    report = {
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "phase1_target": target_baseline,
            "frozen_reproduction_config": BASELINE_CONFIG,
            "reproduction_delta": reproduction_delta,
        },
        "external_source": {
            "file": str(external_path),
            "rows": int(len(external)),
            "prompts": int(external["prompt_id"].nunique()),
            "represented_labels": augmented_labels,
            "balanced_cap_per_label": cap,
        },
        "experiments": [
            summarize("official_only", baseline, 0),
            summarize("official_plus_all_external", augmented_all, len(all_external)),
            summarize(
                "official_plus_balanced_external",
                augmented_balanced,
                len(balanced_external),
            ),
        ],
    }

    best = max(report["experiments"], key=lambda item: item["balanced_accuracy"])
    report["decision"] = {
        "best_experiment": best["name"],
        "best_balanced_accuracy": best["balanced_accuracy"],
        "best_delta_vs_baseline": best["delta_vs_baseline"],
        "promote_external_to_encoder_stage": best["name"] != "official_only"
        and best["delta_vs_baseline"] > 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--folds", type=Path, default=Path("artifacts/phase1/folds.csv")
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument(
        "--phase1-baseline",
        type=Path,
        default=Path("artifacts/phase1/baseline_char_1_5.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2/gpt_gemini_char_ablation.json"),
    )
    args = parser.parse_args()

    report = run_ablation(
        train_path=args.train,
        folds_path=args.folds,
        external_path=args.external,
        phase1_baseline_path=args.phase1_baseline,
        output_path=args.output,
    )
    experiments = {item["name"]: item for item in report["experiments"]}
    print(
        json.dumps(
            {
                "baseline": experiments["official_only"]["balanced_accuracy"],
                "all_external": experiments["official_plus_all_external"][
                    "balanced_accuracy"
                ],
                "balanced_external": experiments[
                    "official_plus_balanced_external"
                ]["balanced_accuracy"],
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
