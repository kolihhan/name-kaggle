from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scripts.run_style_baseline import (
    EXPECTED_FOLD_FINGERPRINT,
    EXPECTED_TRAIN_SHA256,
    extract_style_features,
    fold_fingerprint,
    sha256_file,
)


STYLE_WEIGHT = 0.2
DEFAULT_C = 1.5
DEFAULT_SEED = 20261031


def build_fold_features(
    train_text: Sequence[str],
    valid_text: Sequence[str],
    *,
    min_df: int = 2,
    max_features: int | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    limits = {
        "char": max_features or 180_000,
        "word": max_features or 150_000,
        "boundary": max_features or 100_000,
        "punct": max_features or 100_000,
    }

    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 5),
        lowercase=False,
        sublinear_tf=True,
        min_df=min_df,
        max_features=limits["char"],
        dtype=np.float32,
    )
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=False,
        token_pattern=r"\S+",
        sublinear_tf=True,
        min_df=min_df,
        max_features=limits["word"],
        dtype=np.float32,
    )
    boundary = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=min_df,
        max_features=limits["boundary"],
        dtype=np.float32,
    )
    punct = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        ngram_range=(1, 2),
        lowercase=False,
        sublinear_tf=True,
        min_df=min_df,
        max_features=limits["punct"],
        dtype=np.float32,
    )

    train_parts = []
    valid_parts = []
    widths = {}
    for name, vectorizer in [
        ("char", char),
        ("word", word),
        ("boundary", boundary),
        ("punct", punct),
    ]:
        x_train = vectorizer.fit_transform(train_text).tocsr()
        x_valid = vectorizer.transform(valid_text).tocsr()
        train_parts.append(x_train)
        valid_parts.append(x_valid)
        widths[name] = int(x_train.shape[1])

    style_train_raw = extract_style_features(train_text)
    style_valid_raw = extract_style_features(valid_text)
    scaler = StandardScaler()
    style_train = scaler.fit_transform(style_train_raw).astype(np.float32) * STYLE_WEIGHT
    style_valid = scaler.transform(style_valid_raw).astype(np.float32) * STYLE_WEIGHT
    train_parts.append(sparse.csr_matrix(style_train))
    valid_parts.append(sparse.csr_matrix(style_valid))
    widths["style"] = int(style_train.shape[1])

    x_train = sparse.hstack(train_parts, format="csr", dtype=np.float32)
    x_valid = sparse.hstack(valid_parts, format="csr", dtype=np.float32)
    meta = {
        "widths": widths,
        "total_features": int(x_train.shape[1]),
        "style_features": int(style_train.shape[1]),
        "style_weight": STYLE_WEIGHT,
    }
    return x_train, x_valid, meta


def make_svc(seed: int = DEFAULT_SEED) -> LinearSVC:
    return LinearSVC(
        C=DEFAULT_C,
        class_weight="balanced",
        dual=True,
        max_iter=15_000,
        random_state=seed,
    )


def summarize_oof(
    train: pd.DataFrame,
    predictions: np.ndarray,
    fold_scores: list[float],
) -> dict:
    if len(predictions) != len(train):
        raise RuntimeError("OOF prediction count mismatch")
    if any(x is None for x in predictions.tolist()):
        raise RuntimeError("OOF predictions contain unfilled rows")
    labels = sorted(train["model"].astype(str).unique())
    recalls = recall_score(
        train["model"], predictions, labels=labels, average=None, zero_division=0
    )
    return {
        "oof_rows": int(len(train)),
        "balanced_accuracy": float(balanced_accuracy_score(train["model"], predictions)),
        "folds": [float(x) for x in fold_scores],
        "per_model_recall": {
            label: float(recall) for label, recall in zip(labels, recalls)
        },
    }


def load_train(train_path: Path, folds_path: Path) -> pd.DataFrame:
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("official train SHA mismatch")
    train = pd.read_csv(train_path)
    folds = pd.read_csv(folds_path)[["id", "fold_id"]]
    train = train.merge(folds, on="id", how="left", validate="one_to_one")
    if train["fold_id"].isna().any():
        raise RuntimeError("missing fold assignments")
    observed = fold_fingerprint(train)
    if observed != EXPECTED_FOLD_FINGERPRINT:
        raise RuntimeError(
            f"fold mapping mismatch: {observed} != {EXPECTED_FOLD_FINGERPRINT}"
        )
    return train


def run_cv(train: pd.DataFrame, *, seed: int = DEFAULT_SEED) -> tuple[dict, pd.DataFrame]:
    labels = sorted(train["model"].astype(str).unique())
    if len(labels) != 18:
        raise RuntimeError(f"expected 18 labels, found {len(labels)}")

    predictions = np.full(len(train), None, dtype=object)
    fold_scores: list[float] = []
    fold_feature_meta: list[dict] = []

    for fold in range(5):
        fit_idx = np.flatnonzero(train["fold_id"].to_numpy() != fold)
        valid_idx = np.flatnonzero(train["fold_id"].to_numpy() == fold)
        fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]

        x_train, x_valid, meta = build_fold_features(
            fit["text"].astype(str).tolist(),
            valid["text"].astype(str).tolist(),
        )
        model = make_svc(seed + fold)
        model.fit(x_train, fit["model"])
        pred = model.predict(x_valid)
        predictions[valid_idx] = pred
        score = float(balanced_accuracy_score(valid["model"], pred))
        fold_scores.append(score)
        meta["fold"] = fold
        fold_feature_meta.append(meta)
        print(
            f"fold={fold} score={score:.6f} total_features={meta['total_features']}",
            flush=True,
        )

    result = summarize_oof(train, predictions, fold_scores)
    report = {
        "phase": "5b",
        "experiment": "official_only_strong_sparse_anchor",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": EXPECTED_FOLD_FINGERPRINT,
        },
        "data": {
            "official_rows": int(len(train)),
            "prompt_count": int(train["prompt_id"].nunique()),
            "external_rows": 0,
            "labels": labels,
        },
        "configuration": {
            "char": "char TF-IDF 1-5, lowercase=False, min_df=2, max_features=180000",
            "word": "word TF-IDF 1-2, whitespace token, lowercase=False, min_df=2, max_features=150000",
            "boundary": "char_wb TF-IDF 3-5, min_df=2, max_features=100000",
            "punct": "punctuation-aware word TF-IDF 1-2, lowercase=False, min_df=2, max_features=100000",
            "style_weight": STYLE_WEIGHT,
            "classifier": "LinearSVC(C=1.5,class_weight=balanced,dual=True,max_iter=15000)",
        },
        "fold_feature_meta": fold_feature_meta,
        "result": result,
        "decision": {
            "beats_phase5_blend": result["balanced_accuracy"] > 0.34820780512055677,
            "delta_vs_phase5_blend": result["balanced_accuracy"] - 0.34820780512055677,
            "reaches_public_reference_floor": result["balanced_accuracy"] >= 0.374,
        },
    }

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["prediction"] = predictions
    return report, oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-only strong sparse anchor.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase5b/strong_sparse_anchor.json"))
    parser.add_argument("--oof-output", type=Path, default=Path("artifacts/phase5b/strong_sparse_anchor_oof.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train(args.train, args.folds)
    report, oof = run_cv(train, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
