from __future__ import annotations

import argparse
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


STYLE_WEIGHT = 0.2
DEFAULT_C = 1.5
DEFAULT_SEED = 20261031
EXPECTED_COVERED_LABELS = {
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
}
PUNCTUATION = [
    ".", ",", ";", ":", "!", "?", "—", "–", "-", "…", "...",
    '"', "'", "’", "“", "”", "(", ")", "[", "]", "{", "}", "*", "#",
    "\n", "\t", "/", "%",
]
FUNCTION_WORDS = (
    "the a an and or but if then so because although however therefore thus moreover "
    "furthermore indeed perhaps might could would should may must can will we you i it "
    "they this that these those not no very more most such into from with without about"
).split()


def validate_test_frame(test: pd.DataFrame) -> None:
    required = {"id", "text"}
    missing = sorted(required - set(test.columns))
    if missing:
        raise RuntimeError(f"test frame missing columns: {missing}")
    if test["id"].isna().any() or test["text"].isna().any():
        raise RuntimeError("test frame contains null ids or text")
    if test["id"].duplicated().any():
        raise RuntimeError("test frame contains duplicate ids")


def validate_training_frames(train: pd.DataFrame, external: pd.DataFrame) -> set[str]:
    for name, frame in [("train", train), ("external", external)]:
        missing = sorted({"text", "model"} - set(frame.columns))
        if missing:
            raise RuntimeError(f"{name} frame missing columns: {missing}")
        if frame[["text", "model"]].isna().any().any():
            raise RuntimeError(f"{name} frame contains null text/model values")

    official_labels = set(train["model"].astype(str).unique())
    if len(official_labels) != 18:
        raise RuntimeError(f"expected 18 official labels, found {len(official_labels)}")
    if not EXPECTED_COVERED_LABELS < official_labels:
        raise RuntimeError("covered label set must be a strict subset of official labels")

    external_labels = set(external["model"].astype(str).unique())
    if external_labels != EXPECTED_COVERED_LABELS:
        raise RuntimeError(
            "external label set mismatch: "
            f"{sorted(external_labels)} != {sorted(EXPECTED_COVERED_LABELS)}"
        )
    return official_labels


def extract_style_features(texts: Sequence[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for raw in texts:
        text = str(raw)
        words = re.findall(r"\b\w+(?:['’]\w+)?\b", text, flags=re.UNICODE)
        lower = [word.lower() for word in words]
        word_lengths = np.asarray([len(word) for word in words] or [0], dtype=float)
        sentences = [sentence for sentence in re.split(r"[.!?]+\s*", text) if sentence.strip()]
        sentence_lengths = np.asarray(
            [len(re.findall(r"\b\w+\b", sentence)) for sentence in sentences] or [0],
            dtype=float,
        )
        n_chars = max(len(text), 1)
        n_words = max(len(words), 1)
        n_sentences = max(len(sentences), 1)
        punct_counts = np.asarray([text.count(punct) for punct in PUNCTUATION], dtype=float)

        rows.append(
            [
                math.log1p(len(text)),
                math.log1p(len(words)),
                math.log1p(len(sentences)),
                float(word_lengths.mean()),
                float(word_lengths.std()),
                float(np.median(word_lengths)),
                float(word_lengths.max()),
                float(sentence_lengths.mean()),
                float(sentence_lengths.std()),
                float(sentence_lengths.max()),
                len(set(lower)) / n_words,
                sum(len(word) <= 3 for word in words) / n_words,
                sum(len(word) >= 8 for word in words) / n_words,
                sum(char.isupper() for char in text) / n_chars,
                sum(char.islower() for char in text) / n_chars,
                sum(char.isdigit() for char in text) / n_chars,
                sum(char.isspace() for char in text) / n_chars,
                sum(ord(char) > 127 for char in text) / n_chars,
                len(re.findall(r"\b\w+['’]\w+\b", text)) / n_words,
                text.count("\n") / n_chars,
                text.count("\n\n") / n_chars,
                len(re.findall(r"(?m)^\s*[-*+]\s+", text)) / n_sentences,
                len(re.findall(r"(?m)^\s*\d+[.)]\s+", text)) / n_sentences,
                text.count("```") / n_chars,
                text.count("**") / n_chars,
                *np.log1p(punct_counts).tolist(),
                *(punct_counts / n_chars).tolist(),
                *(punct_counts / n_words).tolist(),
                *[lower.count(word) / n_words for word in FUNCTION_WORDS],
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def build_features(
    train_text: Sequence[str],
    test_text: Sequence[str],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    vectorizers = [
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 5),
            lowercase=False,
            sublinear_tf=True,
            min_df=2,
            max_features=180_000,
            dtype=np.float32,
        ),
        TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=False,
            token_pattern=r"\S+",
            sublinear_tf=True,
            min_df=2,
            max_features=150_000,
            dtype=np.float32,
        ),
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=2,
            max_features=100_000,
            dtype=np.float32,
        ),
        TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
            ngram_range=(1, 2),
            lowercase=False,
            sublinear_tf=True,
            min_df=2,
            max_features=100_000,
            dtype=np.float32,
        ),
    ]

    train_parts: list[sparse.spmatrix] = []
    test_parts: list[sparse.spmatrix] = []
    for vectorizer in vectorizers:
        train_parts.append(vectorizer.fit_transform(train_text).tocsr())
        test_parts.append(vectorizer.transform(test_text).tocsr())

    scaler = StandardScaler()
    style_train = scaler.fit_transform(extract_style_features(train_text)).astype(np.float32)
    style_test = scaler.transform(extract_style_features(test_text)).astype(np.float32)
    train_parts.append(sparse.csr_matrix(style_train * STYLE_WEIGHT))
    test_parts.append(sparse.csr_matrix(style_test * STYLE_WEIGHT))

    return (
        sparse.hstack(train_parts, format="csr", dtype=np.float32),
        sparse.hstack(test_parts, format="csr", dtype=np.float32),
    )


def make_svc(seed: int = DEFAULT_SEED) -> LinearSVC:
    return LinearSVC(
        C=DEFAULT_C,
        class_weight="balanced",
        dual=True,
        max_iter=15_000,
        random_state=seed,
    )


def specialist_training_rows(
    train: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    official_covered = train.loc[
        train["model"].astype(str).isin(EXPECTED_COVERED_LABELS), ["text", "model"]
    ]
    texts = pd.concat(
        [official_covered["text"], external["text"]], ignore_index=True
    ).astype(str).tolist()
    labels = pd.concat(
        [official_covered["model"], external["model"]], ignore_index=True
    ).astype(str).tolist()
    if set(labels) != EXPECTED_COVERED_LABELS:
        raise RuntimeError("specialist training labels mismatch")
    return texts, labels


def gate_predictions(
    anchor_predictions: np.ndarray,
    specialist_predictions: np.ndarray,
) -> np.ndarray:
    anchor = np.asarray(anchor_predictions, dtype=object)
    specialist = np.asarray(specialist_predictions, dtype=object)
    if len(anchor) != len(specialist):
        raise RuntimeError("anchor and specialist prediction lengths differ")
    outside = sorted(set(str(value) for value in specialist.tolist()) - EXPECTED_COVERED_LABELS)
    if outside:
        raise RuntimeError(f"specialist predictions outside covered label set: {outside}")
    result = anchor.copy()
    mask = np.fromiter(
        (str(label) in EXPECTED_COVERED_LABELS for label in anchor),
        dtype=bool,
        count=len(anchor),
    )
    result[mask] = specialist[mask]
    return result


def predict_phase7(
    train: pd.DataFrame,
    external: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    validate_test_frame(test)
    validate_training_frames(train, external)

    train_text = train["text"].astype(str).tolist()
    test_text = test["text"].astype(str).tolist()
    x_anchor, x_test = build_features(train_text, test_text)
    anchor = make_svc(seed)
    anchor.fit(x_anchor, train["model"].astype(str))
    anchor_predictions = anchor.predict(x_test)

    specialist_text, specialist_labels = specialist_training_rows(train, external)
    x_specialist, x_specialist_test = build_features(specialist_text, test_text)
    specialist = make_svc(seed)
    specialist.fit(x_specialist, specialist_labels)
    specialist_predictions = specialist.predict(x_specialist_test)

    return gate_predictions(anchor_predictions, specialist_predictions)


def build_submission_frame(
    test: pd.DataFrame,
    predictions: np.ndarray,
    *,
    allowed_labels: Iterable[str],
) -> pd.DataFrame:
    validate_test_frame(test)
    predictions = np.asarray(predictions, dtype=object)
    if len(predictions) != len(test):
        raise RuntimeError(
            f"prediction count mismatch: {len(predictions)} != {len(test)}"
        )
    if any(value is None for value in predictions.tolist()):
        raise RuntimeError("predictions contain null values")

    allowed = {str(label) for label in allowed_labels}
    observed = {str(label) for label in predictions.tolist()}
    outside = sorted(observed - allowed)
    if outside:
        raise RuntimeError(f"predictions outside official label set: {outside}")

    submission = pd.DataFrame(
        {
            "id": test["id"].astype(str).to_numpy(),
            "model": predictions.astype(str),
        }
    )
    if submission["model"].isna().any():
        raise RuntimeError("submission contains null model values")
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen Phase 7 Kaggle submission.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("submission.parquet"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    external = pd.read_parquet(args.external)
    official_labels = validate_training_frames(train, external)
    predictions = predict_phase7(train, external, test, seed=args.seed)
    submission = build_submission_frame(test, predictions, allowed_labels=official_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(args.output, index=False)
    print(f"wrote {len(submission)} rows to {args.output}")
    print(submission["model"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
