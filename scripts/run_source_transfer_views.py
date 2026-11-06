#!/usr/bin/env python3
"""One-shot external-only source-transfer view ablation; prints aggregates only.

Nothing in this program loads official Kaggle Competition Data or submits.
See artifacts/research/source-transfer-views-20261106/CONTRACT.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from scripts.run_phase14_public_sparse import (
    EXPECTED_RAW_SHA256,
    VIEWS,
    WORD_TOKEN_PATTERN,
    _view_texts,
    build_conservative_subset,
    sha256_file,
)
from scripts.run_two_stage_lora24_gate_a import build_stage1_split

SHA_ALESS = "e98b1924269c30116f50623052f7fc4ea6148f7575d2fe483c9bd899b4ea2e21"
SHA_ELLIOT = "e8aeafcd75b9832c297324da360051a48fec3b58d1d8cfe251e03c2558f40185"
ARMS = {
    "full": ("char", "word", "shape", "small"),
    "lexical": ("char", "word"),
    "surface": ("shape", "small"),
}
FAMILIES = ("claude-", "gemini-", "glm-", "gpt-", "kimi", "qwen")


def check_file(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"file drift {path.name}: {actual} != {expected}")


def family(label: str) -> str:
    return next((p for p in FAMILIES if str(label).startswith(p)), "other")


def summaries(labels: np.ndarray, predicted: np.ndarray) -> dict:
    counts = pd.Series(labels).value_counts()
    eligible = sorted(label for label, count in counts.items() if count >= 20)
    recalls = {
        label: float(np.mean(predicted[labels == label] == label))
        for label in eligible
    }
    fam_recs = {
        fam: float(np.mean([v for k, v in recalls.items() if family(k) == fam]))
        for fam in FAMILIES if any(family(k) == fam for k in recalls)
    }
    within, cross, good = 0, 0, 0
    for correct, guess in zip(labels, predicted):
        if correct == guess:
            good += 1
        elif family(correct) == family(guess):
            within += 1
        else:
            cross += 1
    return {
        "eligible_label_count": len(eligible),
        "eligible_macro_recall": float(np.mean(list(recalls.values()))),
        "eligible_per_label_recall": recalls,
        "family_macro_recall": fam_recs,
        "correct_rows": good,
        "same_family_wrong_rows": within,
        "cross_family_wrong_rows": cross,
    }


def run(stage1: Path, aless: Path, elliot: Path) -> dict:
    for p, digest in (
        (stage1, EXPECTED_RAW_SHA256),
        (aless, SHA_ALESS),
        (elliot, SHA_ELLIOT),
    ):
        check_file(p, digest)
    pool = pd.read_parquet(stage1)
    safe = build_conservative_subset(pool)
    split = build_stage1_split(safe)
    train = safe.iloc[split["train_idx"]]
    if len(train) != 30133 or set(split["holdout_sources"]) != {"abdullah", "semenoff", "zakir"}:
        raise ValueError("original A1 training partition drifted")
    if train["model"].nunique() != 18:
        raise ValueError("Stage-1 label coverage changed")

    external = {
        "alessandrolair": pd.read_parquet(aless),
        "elliotthankinson": pd.read_parquet(elliot),
    }
    train_text = train["text"].astype(str).tolist()
    test_text = []
    slices = {}
    for source, frame in external.items():
        if not {"model", "text"}.issubset(frame.columns):
            raise ValueError(f"missing columns: {source}")
        start = len(test_text)
        test_text += frame["text"].astype(str).tolist()
        slices[source] = (start, len(test_text))
    tr_parts, te_parts = {}, {}
    widths = {}
    for name, lo, hi, cap, weight in VIEWS:
        analyzer = "char" if name in {"char", "shape"} else "word"
        opts = {
            "analyzer": analyzer,
            "ngram_range": (lo, hi),
            "max_features": cap,
            "min_df": 2,
            "lowercase": False,
            "sublinear_tf": True,
            "dtype": np.float32,
        }
        if analyzer == "word":
            opts["token_pattern"] = WORD_TOKEN_PATTERN
        vectorizer = TfidfVectorizer(**opts)
        x = vectorizer.fit_transform(_view_texts(name, train_text)).tocsr()
        v = vectorizer.transform(_view_texts(name, test_text)).tocsr()
        tr_parts[name] = x.multiply(np.float32(weight)).tocsr()
        te_parts[name] = v.multiply(np.float32(weight)).tocsr()
        widths[name] = x.shape[1]
        print(f"Fitted frozen {name} view: {x.shape[1]} features", flush=True)

    report = {
        "scope": "precommitted_exploratory_mechanism_not_fresh_holdout_or_official_18way",
        "train_rows": len(train),
        "source_holdout_train": split["holdout_sources"],
        "sha256": {k: sha256_file(p) for k, p in
                   {"stage1": stage1, "aless": aless, "elliot": elliot}.items()},
        "feature_widths": widths,
        "source_results": {
            k: {"rows": len(v), "labels": int(v["model"].nunique()), "arms": {}}
            for k, v in external.items()
        },
    }
    train_labels = train["model"].astype(str).to_numpy()
    for arm, views in ARMS.items():
        x = sparse.hstack([tr_parts[k] for k in views], format="csr", dtype=np.float32)
        v = sparse.hstack([te_parts[k] for k in views], format="csr", dtype=np.float32)
        model = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
        model.fit(x, train_labels)
        prediction = model.predict(v)
        for source, (lo, hi) in slices.items():
            actual = external[source]["model"].astype(str).to_numpy()
            report["source_results"][source]["arms"][arm] = summaries(actual, prediction[lo:hi])
            print(f"{arm} / {source}: {report['source_results'][source]['arms'][arm]['eligible_macro_recall']:.5f}", flush=True)
    for source, details in report["source_results"].items():
        arms = details["arms"]
        scores = {k: d["eligible_macro_recall"] for k, d in arms.items()}
        details["score_deltas"] = {
            "surface_minus_lexical": scores["surface"] - scores["lexical"],
            "full_minus_surface": scores["full"] - scores["surface"],
            "full_minus_lexical": scores["full"] - scores["lexical"],
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    for key in ("stage1", "aless", "elliot", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    report = run(args.stage1, args.aless, args.elliot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("AGGREGATE RESULTS " + json.dumps({
        name: info["score_deltas"] for name, info in report["source_results"].items()
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
