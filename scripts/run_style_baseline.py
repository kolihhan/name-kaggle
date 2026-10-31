from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler


EXPECTED_TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_FOLD_FINGERPRINT = "bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d"
DEFAULT_SEED = 20261031

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_fingerprint(frame: pd.DataFrame) -> str:
    required = {"id", "prompt_id", "model", "fold_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fold frame missing columns: {missing}")
    ordered = frame.sort_values("id")
    payload = "\n".join(
        f"{row.id},{row.prompt_id},{row.model},{int(row.fold_id)}"
        for row in ordered.itertuples(index=False)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_style_features(texts: Sequence[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for raw in texts:
        text = str(raw)
        words = re.findall(r"\b\w+(?:['’]\w+)?\b", text, flags=re.UNICODE)
        lower = [w.lower() for w in words]
        word_lengths = np.asarray([len(w) for w in words] or [0], dtype=float)
        sentences = [s for s in re.split(r"[.!?]+\s*", text) if s.strip()]
        sentence_lengths = np.asarray(
            [len(re.findall(r"\b\w+\b", s)) for s in sentences] or [0],
            dtype=float,
        )
        n_chars = max(len(text), 1)
        n_words = max(len(words), 1)
        n_sentences = max(len(sentences), 1)
        punct_counts = np.asarray([text.count(p) for p in PUNCTUATION], dtype=float)

        row = [
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
            sum(len(w) <= 3 for w in words) / n_words,
            sum(len(w) >= 8 for w in words) / n_words,
            sum(c.isupper() for c in text) / n_chars,
            sum(c.islower() for c in text) / n_chars,
            sum(c.isdigit() for c in text) / n_chars,
            sum(c.isspace() for c in text) / n_chars,
            sum(ord(c) > 127 for c in text) / n_chars,
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
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def blend_probabilities(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    if not probabilities:
        raise ValueError("at least one probability matrix is required")
    shape = probabilities[0].shape
    if len(shape) != 2 or any(p.shape != shape for p in probabilities):
        raise ValueError("all probability matrices must have the same shape")
    eps = 1e-12
    log_mean = np.mean([np.log(np.clip(p, eps, 1.0)) for p in probabilities], axis=0)
    log_mean -= log_mean.max(axis=1, keepdims=True)
    blended = np.exp(log_mean)
    blended /= blended.sum(axis=1, keepdims=True)
    return blended


def align_probabilities(
    probabilities: np.ndarray,
    model_classes: Iterable[str],
    labels: Sequence[str],
) -> np.ndarray:
    classes = [str(x) for x in model_classes]
    lookup = {label: idx for idx, label in enumerate(classes)}
    missing = [label for label in labels if label not in lookup]
    if missing:
        raise RuntimeError(f"classifier missing labels: {missing}")
    return probabilities[:, [lookup[label] for label in labels]]


def summarize_oof(
    train: pd.DataFrame,
    predictions: np.ndarray,
    *,
    fold_scores: list[float],
) -> dict:
    if len(predictions) != len(train):
        raise ValueError("OOF prediction count must equal official train row count")
    if any(x is None for x in predictions.tolist()):
        raise ValueError("OOF predictions contain unfilled rows")
    labels = sorted(train["model"].astype(str).unique())
    score = float(balanced_accuracy_score(train["model"], predictions))
    recalls = recall_score(
        train["model"], predictions, labels=labels, average=None, zero_division=0
    )
    return {
        "oof_rows": int(len(train)),
        "balanced_accuracy": score,
        "folds": [float(x) for x in fold_scores],
        "per_model_recall": {
            label: float(recall) for label, recall in zip(labels, recalls)
        },
    }


def make_lr(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=4.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2500,
        random_state=seed,
    )


def run_cv(train: pd.DataFrame, *, seed: int = DEFAULT_SEED) -> tuple[dict, pd.DataFrame]:
    labels = sorted(train["model"].astype(str).unique())
    if len(labels) != 18:
        raise RuntimeError(f"expected 18 labels, found {len(labels)}")
    observed_fingerprint = fold_fingerprint(train)
    if observed_fingerprint != EXPECTED_FOLD_FINGERPRINT:
        raise RuntimeError(
            f"fold mapping mismatch: {observed_fingerprint} != {EXPECTED_FOLD_FINGERPRINT}"
        )

    names = ["char", "word", "style", "blend"]
    oof_predictions = {name: np.full(len(train), None, dtype=object) for name in names}
    fold_scores = {name: [] for name in names}

    for fold in range(5):
        fit_idx = np.flatnonzero(train["fold_id"].to_numpy() != fold)
        valid_idx = np.flatnonzero(train["fold_id"].to_numpy() == fold)
        fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]
        fit_text = fit["text"].astype(str).tolist()
        valid_text = valid["text"].astype(str).tolist()

        char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 5),
            lowercase=False,
            sublinear_tf=True,
            min_df=2,
            max_features=200_000,
            dtype=np.float32,
        )
        word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=False,
            token_pattern=r"\S+",
            sublinear_tf=True,
            min_df=2,
            max_features=150_000,
            dtype=np.float32,
        )
        char_x = char_vec.fit_transform(fit_text)
        char_v = char_vec.transform(valid_text)
        word_x = word_vec.fit_transform(fit_text)
        word_v = word_vec.transform(valid_text)

        style_x_raw = extract_style_features(fit_text)
        style_v_raw = extract_style_features(valid_text)
        scaler = StandardScaler()
        style_x = scaler.fit_transform(style_x_raw)
        style_v = scaler.transform(style_v_raw)

        fold_probs: dict[str, np.ndarray] = {}
        for view, x_fit, x_valid in [
            ("char", char_x, char_v),
            ("word", word_x, word_v),
            ("style", style_x, style_v),
        ]:
            model = make_lr(seed + fold)
            model.fit(x_fit, fit["model"])
            probs = align_probabilities(model.predict_proba(x_valid), model.classes_, labels)
            fold_probs[view] = probs
            pred = np.asarray(labels, dtype=object)[probs.argmax(axis=1)]
            oof_predictions[view][valid_idx] = pred
            fold_scores[view].append(
                float(balanced_accuracy_score(valid["model"], pred))
            )

        blend = blend_probabilities(
            [fold_probs["char"], fold_probs["word"], fold_probs["style"]]
        )
        blend_pred = np.asarray(labels, dtype=object)[blend.argmax(axis=1)]
        oof_predictions["blend"][valid_idx] = blend_pred
        fold_scores["blend"].append(
            float(balanced_accuracy_score(valid["model"], blend_pred))
        )
        print(
            "fold={} char={:.6f} word={:.6f} style={:.6f} blend={:.6f}".format(
                fold,
                fold_scores["char"][-1],
                fold_scores["word"][-1],
                fold_scores["style"][-1],
                fold_scores["blend"][-1],
            ),
            flush=True,
        )

    experiments = {
        name: summarize_oof(train, oof_predictions[name], fold_scores=fold_scores[name])
        for name in names
    }
    report = {
        "phase": 5,
        "experiment": "official_only_style_multiview",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": observed_fingerprint,
        },
        "data": {
            "official_rows": int(len(train)),
            "prompt_count": int(train["prompt_id"].nunique()),
            "labels": labels,
            "external_rows": 0,
        },
        "views": {
            "char": "TF-IDF char 1-5, case/punctuation preserved, max_features=200000",
            "word": "TF-IDF whitespace-token 1-2, case/punctuation preserved, max_features=150000",
            "style_feature_count": int(extract_style_features([""]).shape[1]),
            "classifier": "LogisticRegression(C=4,class_weight=balanced,solver=lbfgs)",
            "blend": "equal-weight geometric mean of three fold-local probability matrices",
        },
        "experiments": experiments,
        "decision": {
            "best_view": max(experiments, key=lambda name: experiments[name]["balanced_accuracy"]),
            "best_balanced_accuracy": max(
                exp["balanced_accuracy"] for exp in experiments.values()
            ),
            "blend_beats_phase1_char": experiments["blend"]["balanced_accuracy"] > 0.31930,
        },
    }

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    for name in names:
        oof[f"prediction_{name}"] = oof_predictions[name]
    return report, oof


def load_train(train_path: Path, folds_path: Path) -> pd.DataFrame:
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("official train SHA mismatch")
    train = pd.read_csv(train_path)
    folds = pd.read_csv(folds_path)[["id", "fold_id"]]
    merged = train.merge(folds, on="id", how="left", validate="one_to_one")
    if merged["fold_id"].isna().any():
        raise RuntimeError("missing fold assignments")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-only style multiview baseline.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase5/style_baseline.json"))
    parser.add_argument("--oof-output", type=Path, default=Path("artifacts/phase5/style_baseline_oof.csv"))
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
