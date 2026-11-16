#!/usr/bin/env python3
"""One fixed, CPU-only falsifier for whether hidden prompt grouping survives sparsity.

No model-label fitting, no LLM inference, no Kaggle submission, no tuning or
raw text/prediction export. Exact registered TF-IDF+agglomerative recipe and
three deterministic row selections. The official 2699 training responses are
used only for source-agnostic structure feasibility, NEVER a transfer score.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances

EXPECTED_ROWS = 2699
EXPECTED_PROMPTS = 150
EXPECTED_MODELS = 18
SPARSITY_LEVELS = (3, 5, 10, 0)
DRAWS = (0, 1, 2)
DISTANCE_CUTOFF = 0.90   # fixed from a publicly described recipe; no sweep
STOPWORD_POLICY = "english"
MAX_FEATURES = 100000
CONTRACT = "ntl-pro-blind-cluster-sparsity-v1-20261116"


def choose(frame: pd.DataFrame, k: int, draw: int) -> np.ndarray:
    choices: list[int] = []
    for _, g in frame.groupby("prompt_id", sort=True):
        if k and len(g) < k:
            continue
        ordered = sorted(g.index.tolist(), key=lambda i: (
            hashlib.sha256((str(draw) + "|" + str(frame.at[i,"id"])).encode()).digest(),
            str(frame.at[i,"id"]),
        ))
        choices.extend(ordered if k == 0 else ordered[:k])
    return np.asarray(sorted(choices), dtype=np.int64)


def count_pairs(values: np.ndarray) -> int:
    return sum(v * (v - 1) // 2 for v in Counter(values.tolist()).values())


def pair_stats(actual: np.ndarray, pred: np.ndarray) -> dict:
    truth_pair = count_pairs(actual)
    prediction_pair = count_pairs(pred)
    both = Counter(zip(actual.tolist(), pred.tolist()))
    intersection = sum(v * (v - 1) // 2 for v in both.values())
    return {
        "true_pair_count": truth_pair,
        "predicted_pair_count": prediction_pair,
        "same_prompt_pair_precision": round(intersection / prediction_pair, 6) if prediction_pair else None,
        "same_prompt_pair_recall": round(intersection / truth_pair, 6) if truth_pair else None,
        "same_prompt_pair_f1": round(2 * intersection / (prediction_pair + truth_pair), 6)
        if prediction_pair + truth_pair else None,
    }


def eval_partition(frame: pd.DataFrame, indices: np.ndarray, k: int, draw: int) -> dict:
    data = frame.iloc[indices]
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORD_POLICY, ngram_range=(1, 2), min_df=1,
        sublinear_tf=True, norm="l2", max_features=MAX_FEATURES
    )
    x = vectorizer.fit_transform(data.text.astype(str).tolist())
    distance = cosine_distances(x)
    # Cosine is nonnegative. Matrix computed from ONLY unlabeled response text.
    np.clip(distance, 0, 2, out=distance)
    cluster = AgglomerativeClustering(
        n_clusters=None, distance_threshold=DISTANCE_CUTOFF,
        linkage="average", metric="precomputed",
    ).fit_predict(distance)
    truth = data.prompt_id.astype(str).to_numpy()
    counts = Counter(cluster.tolist())
    result = {
        "k": "all" if k == 0 else k,
        "draw": draw,
        "rows": len(data),
        "truth_groups": len(set(truth)),
        "blind_groups": len(counts),
        "blind_singleton_clusters": sum(v == 1 for v in counts.values()),
        "adjusted_rand_index": round(float(adjusted_rand_score(truth, cluster)), 6),
        **pair_stats(truth, cluster),
    }
    return result


def run() -> dict:
    frame = pd.read_csv(Path("data/train.csv"))
    if (len(frame) != EXPECTED_ROWS or frame.prompt_id.nunique() != EXPECTED_PROMPTS
            or frame.model.nunique() != EXPECTED_MODELS
            or frame[["id", "prompt_id", "text", "model"]].isna().any().any()
            or frame.id.duplicated().any()):
        raise ValueError("official existing training-set contract drift")
    runs = [
        eval_partition(frame, choose(frame, k, draw), k, draw)
        for k in SPARSITY_LEVELS
        for draw in (DRAWS[:1] if k == 0 else DRAWS)
    ]
    k5 = [r for r in runs if r["k"] == 5]
    viable = sum(
        r["adjusted_rand_index"] >= 0.90
        and r["same_prompt_pair_recall"] is not None
        and r["same_prompt_pair_recall"] >= 0.85
        for r in k5
    ) >= 2
    return {
        "protocol": CONTRACT,
        "source": "official_train_spent_not_hidden",
        "hypotheses": {
            "H_group": "blind reconstruction remains usable at <=5 answers per prompt",
            "H_sparse": "content-only grouping loses prompt-pair recall under sparse prompts",
        },
        "frozen_gate": "k=5, 2/3 draws ARI>=0.90 and pair recall>=0.85",
        "frozen_gate_pass": viable,
        "conclusion": "GROUPING_FEASIBLE_AS_PROXY_NOT_BA_GAIN" if viable else "GROUPING_AT_K5_NOT_ESTABLISHED",
        "runs": runs,
        "new_supervised_model_training": False,
        "llm_inference": False,
        "kaggle_submission": False,
        "independent_transfer": False,
        "balanced_accuracy": None,
        "note": "Unsupervised features are fit independently on each sampled evaluation pool; this is transductive clustering, not an unseen-distribution BA evaluation. No label-dependent tuning.",
    }


if __name__ == "__main__":
    report = run()
    print("NTL_PRO_BLIND_GROUP_SPARSITY=" + json.dumps(report, sort_keys=True), flush=True)
