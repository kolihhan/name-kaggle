from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from scripts.run_strong_sparse_anchor import (
    EXPECTED_FOLD_FINGERPRINT,
    STYLE_WEIGHT,
    load_train,
    make_svc,
    summarize_oof,
)


REFERENCE_PUNCTUATION = [
    "—", "–", "-", ",", ".", ";", ":", "!", "?", "…", "...",
    '"', "'", "’", "“", "”", "(", ")", "[", "]", "*", "#", "\n", "\t", "/", "%",
]
REFERENCE_FUNCTION_WORDS = (
    "the a an and or but if then so because although however therefore thus moreover "
    "furthermore indeed perhaps might could would should may must can will we you i it "
    "they this that these those not no very more most such"
).split()


def extract_reference_style_features(texts: Sequence[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for raw in texts:
        text = str(raw)
        words = re.findall(r"\b\w+(?:['’]\w+)?\b", text)
        lower = [word.lower() for word in words]
        lengths = np.asarray([len(word) for word in words] or [0], dtype=float)
        sentences = [s for s in re.split(r"[.!?]+\s*", text) if s.strip()]
        sentence_lengths = np.asarray([len(s.split()) for s in sentences] or [0], dtype=float)
        nw = max(len(words), 1)
        nc = max(len(text), 1)
        counts = np.asarray([text.count(p) for p in REFERENCE_PUNCTUATION], dtype=float)
        rows.append([
            np.log1p(nc),
            np.log1p(nw),
            np.log1p(len(sentences)),
            float(lengths.mean()),
            float(lengths.std()),
            float(np.median(lengths)),
            float(sentence_lengths.mean()),
            float(sentence_lengths.std()),
            len(set(lower)) / nw,
            sum(len(w) <= 3 for w in words) / nw,
            sum(len(w) >= 8 for w in words) / nw,
            sum(c.isupper() for c in text) / nc,
            sum(c.isdigit() for c in text) / nc,
            sum(ord(c) > 127 for c in text) / nc,
            len(re.findall(r"\b\w+['’]\w+\b", text)) / nw,
            *np.log1p(counts).tolist(),
            *(counts / nw).tolist(),
            *[lower.count(word) / nw for word in REFERENCE_FUNCTION_WORDS],
        ])
    return np.asarray(rows, dtype=np.float64)


def build_reference_fold_features(
    train_text: Sequence[str],
    valid_text: Sequence[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    vectorizers = [
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(1, 5), lowercase=False,
            sublinear_tf=True, min_df=2, max_features=180_000, dtype=np.float32,
        )),
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), lowercase=False, token_pattern=r"\S+",
            sublinear_tf=True, min_df=2, max_features=150_000, dtype=np.float32,
        )),
        ("boundary", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            sublinear_tf=True, min_df=2, max_features=100_000, dtype=np.float32,
        )),
        ("punct", TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b|[^\w\s]", ngram_range=(1, 2),
            lowercase=False, sublinear_tf=True, min_df=2, max_features=100_000,
            dtype=np.float32,
        )),
    ]
    train_parts = []
    valid_parts = []
    widths = {}
    for name, vectorizer in vectorizers:
        x_train = vectorizer.fit_transform(train_text).tocsr()
        x_valid = vectorizer.transform(valid_text).tocsr()
        train_parts.append(x_train)
        valid_parts.append(x_valid)
        widths[name] = int(x_train.shape[1])

    raw_train = extract_reference_style_features(train_text)
    raw_valid = extract_reference_style_features(valid_text)
    scaler = StandardScaler()
    style_train = scaler.fit_transform(raw_train).astype(np.float32) * STYLE_WEIGHT
    style_valid = scaler.transform(raw_valid).astype(np.float32) * STYLE_WEIGHT
    train_parts.append(sparse.csr_matrix(style_train))
    valid_parts.append(sparse.csr_matrix(style_valid))
    widths["style"] = int(style_train.shape[1])

    x_train = sparse.hstack(train_parts, format="csr", dtype=np.float32)
    x_valid = sparse.hstack(valid_parts, format="csr", dtype=np.float32)
    return x_train, x_valid, {
        "widths": widths,
        "total_features": int(x_train.shape[1]),
        "style_features": int(style_train.shape[1]),
        "style_weight": STYLE_WEIGHT,
    }


def run_cv(train: pd.DataFrame, *, seed: int = 20261031) -> tuple[dict, pd.DataFrame]:
    labels = sorted(train["model"].astype(str).unique())
    if len(labels) != 18:
        raise RuntimeError(f"expected 18 labels, found {len(labels)}")
    predictions = np.full(len(train), None, dtype=object)
    fold_scores: list[float] = []
    feature_meta = []

    for fold in range(5):
        fit_idx = np.flatnonzero(train["fold_id"].to_numpy() != fold)
        valid_idx = np.flatnonzero(train["fold_id"].to_numpy() == fold)
        fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]
        x_train, x_valid, meta = build_reference_fold_features(
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
        feature_meta.append(meta)
        print(f"fold={fold} score={score:.6f}", flush=True)

    result = summarize_oof(train, predictions, fold_scores)
    report = {
        "phase": "5c",
        "experiment": "reference_rich_style_reproduction",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": EXPECTED_FOLD_FINGERPRINT,
        },
        "data": {
            "official_rows": int(len(train)),
            "external_rows": 0,
            "labels": labels,
        },
        "configuration": {
            "sparse_views": "identical to Phase 5b",
            "style_features": 108,
            "style_weight": STYLE_WEIGHT,
            "classifier": "LinearSVC(C=1.5,class_weight=balanced,dual=True,max_iter=15000)",
        },
        "fold_feature_meta": feature_meta,
        "result": result,
        "decision": {
            "beats_phase5b": result["balanced_accuracy"] > 0.36708923688789463,
            "delta_vs_phase5b": result["balanced_accuracy"] - 0.36708923688789463,
            "reaches_public_reference_floor": result["balanced_accuracy"] >= 0.374,
        },
    }
    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["prediction"] = predictions
    return report, oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference RichStyle reproduction.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase5c/reference_style_repro.json"))
    parser.add_argument("--oof-output", type=Path, default=Path("artifacts/phase5c/reference_style_repro_oof.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train(args.train, args.folds)
    report, oof = run_cv(train)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
