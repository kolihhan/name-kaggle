from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scripts.train_modernbert_cv import (
    EXPECTED_FOLD_FINGERPRINT,
    EXPECTED_TRAIN_SHA256,
    build_folds,
    fold_fingerprint,
    sha256_file,
)


PUNCTUATION = [
    ("em_dash", "—"),
    ("en_dash", "–"),
    ("hyphen", "-"),
    ("comma", ","),
    ("period", "."),
    ("semicolon", ";"),
    ("colon", ":"),
    ("exclamation", "!"),
    ("question", "?"),
    ("ellipsis_char", "…"),
    ("ellipsis_ascii", "..."),
    ("ascii_double_quote", '"'),
    ("ascii_apostrophe", "'"),
    ("curly_apostrophe", "’"),
    ("left_curly_quote", "“"),
    ("right_curly_quote", "”"),
    ("left_paren", "("),
    ("right_paren", ")"),
    ("left_bracket", "["),
    ("right_bracket", "]"),
    ("asterisk", "*"),
    ("hash", "#"),
    ("newline", "\n"),
    ("tab", "\t"),
    ("slash", "/"),
    ("percent", "%"),
    ("backtick", "`"),
    ("underscore", "_"),
    ("plus", "+"),
    ("equals", "="),
]

FUNCTION_WORDS = (
    "the a an and or but if then so because although however therefore thus moreover "
    "furthermore indeed perhaps might could would should may must can will we you i it "
    "they this that these those not no very more most such with without from into through "
    "about between as of to in on for by is are was were be been being"
).split()

STRUCTURAL_FEATURES = [
    "log_chars",
    "log_words",
    "log_sentences",
    "log_lines",
    "log_paragraphs",
    "mean_word_length",
    "std_word_length",
    "median_word_length",
    "mean_sentence_words",
    "std_sentence_words",
    "type_token_ratio",
    "short_word_ratio",
    "long_word_ratio",
    "uppercase_char_ratio",
    "digit_char_ratio",
    "nonascii_char_ratio",
    "contraction_ratio",
    "bullet_line_ratio",
    "heading_line_ratio",
    "code_fence_rate",
]

STYLE_FEATURE_NAMES = (
    STRUCTURAL_FEATURES
    + [f"count_{name}" for name, _ in PUNCTUATION]
    + [f"rate_{name}" for name, _ in PUNCTUATION]
    + [f"fw_{word}" for word in FUNCTION_WORDS]
)


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+(?:['’]\w+)?\b", text, flags=re.UNICODE)


def extract_style_features(texts: Iterable[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for raw_text in texts:
        text = str(raw_text)
        words = _words(text)
        lower = [word.lower() for word in words]
        word_lengths = np.asarray([len(word) for word in words] or [0], dtype=float)
        sentences = [s for s in re.split(r"[.!?]+\s*", text) if s.strip()]
        sentence_lengths = np.asarray(
            [len(_words(sentence)) for sentence in sentences] or [0], dtype=float
        )
        lines = text.splitlines() or [""]
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

        n_chars = max(len(text), 1)
        n_words = max(len(words), 1)
        n_lines = max(len(lines), 1)

        punct_counts = np.asarray(
            [text.count(symbol) for _, symbol in PUNCTUATION], dtype=float
        )
        bullets = sum(
            bool(re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line)) for line in lines
        )
        headings = sum(bool(re.match(r"^\s*#{1,6}\s+", line)) for line in lines)
        contractions = len(re.findall(r"\b\w+['’]\w+\b", text))

        structural = [
            float(np.log1p(len(text))),
            float(np.log1p(len(words))),
            float(np.log1p(len(sentences))),
            float(np.log1p(len(lines))),
            float(np.log1p(len(paragraphs))),
            float(word_lengths.mean()),
            float(word_lengths.std()),
            float(np.median(word_lengths)),
            float(sentence_lengths.mean()),
            float(sentence_lengths.std()),
            len(set(lower)) / n_words,
            sum(len(word) <= 3 for word in words) / n_words,
            sum(len(word) >= 8 for word in words) / n_words,
            sum(char.isupper() for char in text) / n_chars,
            sum(char.isdigit() for char in text) / n_chars,
            sum(ord(char) > 127 for char in text) / n_chars,
            contractions / n_words,
            bullets / n_lines,
            headings / n_lines,
            text.count("```") / n_words,
        ]
        count_features = np.log1p(punct_counts).tolist()
        rate_features = (punct_counts / n_words).tolist()
        function_features = [lower.count(word) / n_words for word in FUNCTION_WORDS]
        rows.append(structural + count_features + rate_features + function_features)

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(STYLE_FEATURE_NAMES):
        raise RuntimeError(
            f"style feature shape mismatch: {matrix.shape}; "
            f"expected width {len(STYLE_FEATURE_NAMES)}"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("style features contain non-finite values")
    return matrix


class RichStyleTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return extract_style_features(X)


def build_char_sanity_model(seed: int = 20261031) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 5),
                    lowercase=False,
                    sublinear_tf=True,
                    min_df=2,
                    max_features=80_000,
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=15_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_multiview_model(seed: int = 20261031) -> Pipeline:
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 5),
        lowercase=False,
        sublinear_tf=True,
        min_df=2,
        max_features=180_000,
        dtype=np.float32,
    )
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=False,
        strip_accents=None,
        token_pattern=r"\S+",
        sublinear_tf=True,
        min_df=2,
        max_features=150_000,
        dtype=np.float32,
    )
    boundary = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=False,
        sublinear_tf=True,
        min_df=2,
        max_features=100_000,
        dtype=np.float32,
    )
    punct = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        ngram_range=(1, 2),
        lowercase=False,
        sublinear_tf=True,
        min_df=2,
        max_features=100_000,
        dtype=np.float32,
    )
    style = Pipeline(
        [
            ("extract", RichStyleTransformer()),
            ("scale", StandardScaler()),
        ]
    )
    features = FeatureUnion(
        [
            ("char", char),
            ("word", word),
            ("boundary", boundary),
            ("punct", punct),
            ("style", style),
        ],
        transformer_weights={"style": 0.2},
        n_jobs=1,
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LinearSVC(
                    C=1.5,
                    class_weight="balanced",
                    dual=True,
                    max_iter=15_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def summarize_oof(
    train: pd.DataFrame,
    predictions: np.ndarray,
    *,
    fold_scores: list[float],
) -> dict:
    if len(predictions) != len(train):
        raise ValueError("OOF prediction count must equal official train row count")
    if any(value is None for value in predictions.tolist()):
        raise ValueError("OOF predictions contain unfilled rows")

    labels = sorted(train["model"].astype(str).unique())
    recalls = recall_score(
        train["model"], predictions, labels=labels, average=None, zero_division=0
    )
    return {
        "oof_rows": int(len(train)),
        "balanced_accuracy": float(balanced_accuracy_score(train["model"], predictions)),
        "folds": [float(score) for score in fold_scores],
        "per_model_recall": {
            label: float(recall) for label, recall in zip(labels, recalls)
        },
    }


def run_candidate_cv(
    train: pd.DataFrame,
    *,
    name: str,
    builder,
) -> tuple[dict, np.ndarray]:
    fold_ids = train["fold_id"].to_numpy()
    oof = np.empty(len(train), dtype=object)
    oof[:] = None
    fold_scores: list[float] = []

    for fold in range(5):
        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        model = builder(seed=20261031 + fold)
        model.fit(train.iloc[train_idx]["text"], train.iloc[train_idx]["model"])
        pred = model.predict(train.iloc[valid_idx]["text"])
        oof[valid_idx] = pred
        score = float(balanced_accuracy_score(train.iloc[valid_idx]["model"], pred))
        fold_scores.append(score)
        print(f"candidate={name} fold={fold} balanced_accuracy={score:.6f}", flush=True)

    return summarize_oof(train, oof, fold_scores=fold_scores), oof


def load_official_train(train_path: Path) -> pd.DataFrame:
    observed_sha = sha256_file(train_path)
    if observed_sha != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(
            f"official train SHA mismatch: {observed_sha} != {EXPECTED_TRAIN_SHA256}"
        )
    raw = pd.read_csv(train_path)
    train = build_folds(raw, n_splits=5)
    observed_fingerprint = fold_fingerprint(train)
    if observed_fingerprint != EXPECTED_FOLD_FINGERPRINT:
        raise RuntimeError(
            f"fold fingerprint mismatch: {observed_fingerprint} != "
            f"{EXPECTED_FOLD_FINGERPRINT}"
        )
    return train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Honest official-only multiview style baseline on permanent folds."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase5/honest_style_baseline.json"),
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=Path("artifacts/phase5/honest_style_oof.csv"),
    )
    args = parser.parse_args()

    train = load_official_train(args.train)
    sanity, _ = run_candidate_cv(
        train, name="char_sanity", builder=build_char_sanity_model
    )
    multiview, multiview_oof = run_candidate_cv(
        train, name="multiview_style_svm", builder=build_multiview_model
    )

    report = {
        "phase": 5,
        "experiment": "official_only_honest_style_baseline",
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
            "label_count": int(train["model"].nunique()),
            "external_rows": 0,
            "train_sha256": EXPECTED_TRAIN_SHA256,
        },
        "candidates": {
            "char_sanity": sanity,
            "multiview_style_svm": multiview,
        },
        "config": {
            "selection": "fixed_before_oof",
            "multiview": {
                "char": "char 1-5, max_features=180000, lowercase=False",
                "word": "word 1-2, max_features=150000, lowercase=False",
                "boundary": "char_wb 3-5, max_features=100000",
                "punct": "word-or-punctuation 1-2, max_features=100000",
                "style_feature_count": len(STYLE_FEATURE_NAMES),
                "style_weight": 0.2,
                "classifier": "LinearSVC(C=1.5, class_weight=balanced)",
            },
        },
        "delta_vs_char_sanity": (
            multiview["balanced_accuracy"] - sanity["balanced_accuracy"]
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    oof_frame = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof_frame["prediction"] = multiview_oof
    oof_frame.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
