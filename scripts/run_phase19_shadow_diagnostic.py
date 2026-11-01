from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.make_shadow_folds import SHADOW_SALTS
from scripts.run_gated_specialist import run_specialist_cv
from scripts.run_phase15_terms_clean_sparse import run_cv as run_phase15_cv
from scripts.run_strong_sparse_anchor import DEFAULT_SEED, run_cv as run_anchor_cv
from scripts.run_style_baseline import EXPECTED_TRAIN_SHA256, sha256_file


ORIGINAL_PHASE7_SCORE = 0.3700521998508576
ORIGINAL_PHASE15_SCORE = 0.4053218990802883
BOOTSTRAP_SEED = 20261101
BOOTSTRAP_REPS = 5000


def load_shadow_train(
    train_path: Path,
    shadow_folds_path: Path,
    shadow: str,
) -> pd.DataFrame:
    if shadow not in SHADOW_SALTS:
        raise ValueError(f"unknown shadow split: {shadow}")
    observed_sha = sha256_file(train_path)
    if observed_sha != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(
            f"official train SHA mismatch: {observed_sha} != {EXPECTED_TRAIN_SHA256}"
        )

    train = pd.read_csv(train_path)
    shadow_folds = pd.read_csv(shadow_folds_path)
    required = {"id", "prompt_id", shadow}
    missing = sorted(required - set(shadow_folds.columns))
    if missing:
        raise RuntimeError(f"shadow fold file missing columns: {missing}")

    mapping = shadow_folds[["id", "prompt_id", shadow]].rename(
        columns={"prompt_id": "shadow_prompt_id", shadow: "fold_id"}
    )
    merged = train.merge(mapping, on="id", how="left", validate="one_to_one")
    if merged["fold_id"].isna().any():
        raise RuntimeError("missing shadow fold assignments")
    if not merged["prompt_id"].astype(str).equals(
        merged["shadow_prompt_id"].astype(str)
    ):
        raise RuntimeError("shadow fold prompt_id alignment mismatch")
    merged = merged.drop(columns=["shadow_prompt_id"])
    merged["fold_id"] = merged["fold_id"].astype(int)

    if len(merged) != 2699 or merged["id"].nunique() != 2699:
        raise RuntimeError("official row/id invariant failed")
    if merged["prompt_id"].astype(str).nunique() != 150:
        raise RuntimeError("expected 150 official prompt groups")
    if merged["model"].astype(str).nunique() != 18:
        raise RuntimeError("expected 18 official labels")
    if set(merged["fold_id"]) != {0, 1, 2, 3, 4}:
        raise RuntimeError("shadow fold IDs must be exactly 0..4")
    if merged.groupby(merged["prompt_id"].astype(str))["fold_id"].nunique().max() != 1:
        raise RuntimeError("shadow split leaks a prompt across folds")
    prompt_counts = (
        merged[["prompt_id", "fold_id"]]
        .assign(prompt_id=lambda frame: frame["prompt_id"].astype(str))
        .drop_duplicates()["fold_id"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    if prompt_counts != {0: 30, 1: 30, 2: 30, 3: 30, 4: 30}:
        raise RuntimeError(f"shadow prompt fold counts drifted: {prompt_counts}")
    return merged


def prompt_bootstrap_delta(
    oof: pd.DataFrame,
    *,
    seed: int = BOOTSTRAP_SEED,
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, float | int]:
    required = {"prompt_id", "model", "phase7_prediction", "phase15_prediction"}
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"OOF frame missing bootstrap columns: {missing}")
    if reps <= 0:
        raise ValueError("bootstrap reps must be positive")

    prompts = sorted(oof["prompt_id"].astype(str).unique())
    labels = sorted(oof["model"].astype(str).unique())
    prompt_index = {prompt: i for i, prompt in enumerate(prompts)}
    label_index = {label: i for i, label in enumerate(labels)}

    totals = np.zeros((len(prompts), len(labels)), dtype=np.int32)
    correct7 = np.zeros_like(totals)
    correct15 = np.zeros_like(totals)

    for row in oof.itertuples(index=False):
        p = prompt_index[str(row.prompt_id)]
        label = str(row.model)
        c = label_index[label]
        totals[p, c] += 1
        correct7[p, c] += int(str(row.phase7_prediction) == label)
        correct15[p, c] += int(str(row.phase15_prediction) == label)

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(prompts),
        np.full(len(prompts), 1.0 / len(prompts), dtype=np.float64),
        size=reps,
    )
    denominators = weights @ totals
    if (denominators == 0).any():
        raise RuntimeError("prompt bootstrap produced a replicate with a missing class")

    recall7 = (weights @ correct7) / denominators
    recall15 = (weights @ correct15) / denominators
    delta = recall15.mean(axis=1) - recall7.mean(axis=1)

    return {
        "seed": int(seed),
        "reps": int(reps),
        "prompt_groups": int(len(prompts)),
        "mean": float(delta.mean()),
        "std": float(delta.std(ddof=1)) if reps > 1 else 0.0,
        "p025": float(np.quantile(delta, 0.025)),
        "median": float(np.quantile(delta, 0.5)),
        "p975": float(np.quantile(delta, 0.975)),
        "probability_delta_gt_zero": float(np.mean(delta > 0.0)),
    }


def summarize_shadow_result(
    shadow: str,
    oof: pd.DataFrame,
    *,
    phase7_score: float,
    phase15_score: float,
    phase7_recalls: dict[str, float],
    phase15_recalls: dict[str, float],
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> dict:
    if set(phase7_recalls) != set(phase15_recalls):
        raise RuntimeError("Phase 7 and Phase 15 recall label sets differ")

    truth = oof["model"].astype(str)
    p7_correct = oof["phase7_prediction"].astype(str).eq(truth)
    p15_correct = oof["phase15_prediction"].astype(str).eq(truth)

    return {
        "shadow": shadow,
        "phase7_balanced_accuracy": float(phase7_score),
        "phase15_balanced_accuracy": float(phase15_score),
        "phase15_minus_phase7": float(phase15_score - phase7_score),
        "per_model_recall_phase7": {
            label: float(phase7_recalls[label]) for label in sorted(phase7_recalls)
        },
        "per_model_recall_phase15": {
            label: float(phase15_recalls[label]) for label in sorted(phase15_recalls)
        },
        "per_model_recall_delta": {
            label: float(phase15_recalls[label] - phase7_recalls[label])
            for label in sorted(phase7_recalls)
        },
        "paired_correctness": {
            "both_correct": int((p7_correct & p15_correct).sum()),
            "phase7_only_correct": int((p7_correct & ~p15_correct).sum()),
            "phase15_only_correct": int((~p7_correct & p15_correct).sum()),
            "both_wrong": int((~p7_correct & ~p15_correct).sum()),
        },
        "prompt_bootstrap_phase15_minus_phase7": prompt_bootstrap_delta(
            oof,
            seed=bootstrap_seed,
            reps=bootstrap_reps,
        ),
    }


def run_shadow_diagnostic(
    train: pd.DataFrame,
    phase7_external: pd.DataFrame,
    phase15_external: pd.DataFrame,
    *,
    shadow: str,
) -> tuple[dict, pd.DataFrame]:
    anchor_report, anchor_oof = run_anchor_cv(train, seed=DEFAULT_SEED)
    phase7_bundle, phase7_oof = run_specialist_cv(
        train,
        phase7_external,
        anchor_oof,
        seed=DEFAULT_SEED,
    )
    phase7_result = phase7_bundle["ensemble_result"]

    del anchor_report, anchor_oof
    gc.collect()

    phase15_report, phase15_oof = run_phase15_cv(train, phase15_external)
    phase15_result = phase15_report["result"]

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["phase7_prediction"] = phase7_oof["prediction"].astype(str).to_numpy()
    oof["phase15_prediction"] = phase15_oof["prediction"].astype(str).to_numpy()
    if oof[["phase7_prediction", "phase15_prediction"]].isna().any().any():
        raise RuntimeError("diagnostic OOF contains null predictions")

    summary = summarize_shadow_result(
        shadow,
        oof,
        phase7_score=float(phase7_result["balanced_accuracy"]),
        phase15_score=float(phase15_result["balanced_accuracy"]),
        phase7_recalls=phase7_result["per_model_recall"],
        phase15_recalls=phase15_result["per_model_recall"],
    )
    report = {
        "phase": 19,
        "experiment": "historical_phase7_vs_phase15_shadow_split_diagnostic",
        "purpose": "diagnostic_only_not_model_selection",
        "shadow": shadow,
        "salt": SHADOW_SALTS[shadow],
        "original_permanent_split_reference": {
            "phase7_balanced_accuracy": ORIGINAL_PHASE7_SCORE,
            "phase15_balanced_accuracy": ORIGINAL_PHASE15_SCORE,
            "phase15_minus_phase7": ORIGINAL_PHASE15_SCORE - ORIGINAL_PHASE7_SCORE,
        },
        "frozen_recipes": {
            "phase7": "exact gated 8-class specialist recipe and frozen external snapshot",
            "phase15": "exact provider-terms-clean four-view sparse recipe and frozen 26,025-row external subset",
            "retuned_parameters": 0,
        },
        "result": summary,
        "invariants": {
            "official_rows": int(len(train)),
            "official_prompt_groups": int(train["prompt_id"].astype(str).nunique()),
            "phase7_external_rows": int(len(phase7_external)),
            "phase15_external_rows": int(len(phase15_external)),
            "shadow_prompts_per_fold": 30,
            "kaggle_submission_authorized": False,
            "shadow_is_spent_after_this_diagnostic": True,
        },
    }
    return report, oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen Phase 7 and Phase 15 systems on one frozen shadow split."
    )
    parser.add_argument("--shadow", choices=sorted(SHADOW_SALTS), required=True)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--shadow-folds", type=Path, required=True)
    parser.add_argument(
        "--phase7-external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument("--phase15-external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_shadow_train(args.train, args.shadow_folds, args.shadow)
    phase7_external = pd.read_parquet(args.phase7_external)
    phase15_external = pd.read_parquet(args.phase15_external)

    report, oof = run_shadow_diagnostic(
        train,
        phase7_external,
        phase15_external,
        shadow=args.shadow,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.oof_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
