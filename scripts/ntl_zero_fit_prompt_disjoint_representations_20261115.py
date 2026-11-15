#!/usr/bin/env python3
"""Zero-fit, prompt-disjoint representation-class diagnostic for Name That LLM.

Three frozen representations: raw character n-grams, formatting-stripped
character n-grams, and lexical word n-grams. No supervised fit, calibration,
Kaggle test access, new training, or competition submission. The official
training prompts have already been inspected adaptively: diagnostic ONLY.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import linear_kernel

TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FEATURES = ("raw_char_3to5", "strip_format_char_3to5", "word_1to2")


def strip_format(text: str) -> str:
    # Fixed before observing results; intentionally removes punctuation signals.
    text = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def predict(texts: list[str], prompt_ids: list[str], family: str) -> np.ndarray:
    if family not in FEATURES:
        raise ValueError("unsupported representation")
    source = [strip_format(t) for t in texts] if family == "strip_format_char_3to5" else texts
    cfg = ({"analyzer": "word", "ngram_range": (1, 2)}
           if family == "word_1to2" else
           {"analyzer": "char", "ngram_range": (3, 5)})
    # HashingVectorizer is stateless: transform only, no fit or IDF estimation.
    vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False,
                                  norm="l2", lowercase=True, **cfg)
    vectors = vectorizer.transform(source)
    sims = linear_kernel(vectors, vectors)
    groups = np.asarray(prompt_ids)
    mask = groups[:, None] == groups[None, :]
    sims[mask] = -np.inf
    if not np.isfinite(sims).any(axis=1).all():
        raise ValueError("no prompt-disjoint reference for at least one row")
    return np.argmax(sims, axis=1)


def analyze(rows: list[dict], expected_classes: int | None = 18) -> dict:
    required = ("prompt_id", "model", "text")
    if not rows or any(any(k not in r or r[k] is None for k in required) for r in rows):
        raise ValueError("missing prompt/model/text")
    labels = np.asarray([str(r["model"]) for r in rows])
    prompts = [str(r["prompt_id"]) for r in rows]
    texts = [str(r["text"]) for r in rows]
    classes = sorted(set(labels))
    if expected_classes is not None and len(classes) != expected_classes:
        raise ValueError("wrong model coverage")
    if len(set(prompts)) < 2:
        raise ValueError("need independent prompt groups")
    results: dict = {}
    correct: dict = {}
    for family in FEATURES:
        indices = predict(texts, prompts, family)
        guesses = labels[indices]
        flags = guesses == labels
        correct[family] = flags
        recalls = {c: round(float(flags[labels == c].mean()), 5) for c in classes}
        results[family] = {
            "correct": int(flags.sum()),
            "total": len(rows),
            "prompt_excluded_1nn_balanced_accuracy": round(float(np.mean(list(recalls.values()))), 6),
            "per_model_recall": recalls,
        }
    baseline = correct[FEATURES[0]]
    comparisons = {}
    for family in FEATURES[1:]:
        candidate = correct[family]
        comparisons[family] = {
            "rescues": int((~baseline & candidate).sum()),
            "regressions": int((baseline & ~candidate).sum()),
            "net": int(candidate.sum() - baseline.sum()),
        }
    return {
        "evidence": "SPENT_OFFICIAL_TRAIN_PROMPT_DISJOINT_NOT_EXTERNAL_TRANSFER",
        "source_rows": len(rows),
        "prompt_groups": len(set(prompts)),
        "true_classes": len(classes),
        "training": False,
        "trained_prototypes": False,
        "new_kaggle_submission": False,
        "primary_question": "Which fixed representation retains generator clues across unseen prompts in the SAME already-studied source?",
        "arms": results,
        "paired_vs_raw_char": comparisons,
        "qualification": "No model promotion from these data; this is not a same-row comparison with the frozen 0.683 champion.",
    }


def smoke() -> None:
    rows = [
        {"prompt_id": p, "model": model, "text": text}
        for p, a, b in [
            ("p1", "apple useful apple", "banana useful banana"),
            ("p2", "apple normal apple", "banana normal banana"),
            ("p3", "apple bright apple", "banana bright banana"),
        ]
        for model, text in [("A", a), ("B", b)]
    ]
    out = analyze(rows, expected_classes=2)
    assert out["source_rows"] == 6 and out["prompt_groups"] == 3
    assert all(v["total"] == 6 for v in out["arms"].values())
    assert not out["training"] and not out["new_kaggle_submission"]
    assert all(v["rescues"] - v["regressions"] == v["net"]
               for v in out["paired_vs_raw_char"].values())
    try:
        predict(["a", "b"], ["same", "same"], FEATURES[0])
    except ValueError:
        pass
    else:
        raise AssertionError("same-prompt leakage should be rejected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke()
    if args.smoke:
        print("NTL_ZERO_FIT_SMOKE=PASS")
        return
    if args.input is None:
        parser.error("--input required")
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != TRAIN_SHA256:
        raise SystemExit("STOP_TRAIN_DATA_SHA_MISMATCH")
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = analyze(rows)
    print("NTL_ZERO_FIT_REPRESENTATION=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
