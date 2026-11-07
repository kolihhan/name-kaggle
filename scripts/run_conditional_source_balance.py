#!/usr/bin/env python3
"""Frozen external-only CPU objective test; publish aggregates, never passages."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning

from scripts.run_phase14_public_sparse import (
    EXPECTED_RAW_SHA256,
    build_conservative_subset,
    build_fold_features,
    make_svc,
    sha256_file,
)
from scripts.run_two_stage_lora24_gate_a import build_stage1_split

SEED = 20261108
ARMS = ("B", "S", "P")


def conditional_weights(frame: pd.DataFrame) -> np.ndarray:
    """Keep each class's mass fixed while equalizing its observed sources."""
    if frame.empty or frame[["model", "source"]].isna().any().any():
        raise ValueError("nonempty model/source assignments required")
    counts = frame.groupby(["model", "source"])["model"].transform("size")
    label_n = frame.groupby("model")["model"].transform("size")
    source_n = frame.groupby("model")["source"].transform("nunique")
    return (label_n / (source_n * counts)).to_numpy(dtype=np.float64)


def shuffle_within_class(weights: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.asarray(weights, dtype=np.float64).copy()
    rng = np.random.default_rng(SEED)
    for label in sorted(set(labels)):
        idx = np.flatnonzero(labels == label)
        result[idx] = rng.permutation(result[idx])
    return result


def normalized_answer(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).lower().split())


def purge_overlap(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keep = np.ones(len(train), dtype=bool)
    for column in ("prompt_id", "prompt_key"):
        keep &= ~train[column].astype(str).isin(set(valid[column].astype(str))).to_numpy()
    valid_answers = set(valid["text"].map(normalized_answer))
    keep &= ~train["text"].map(normalized_answer).isin(valid_answers).to_numpy()
    kept = train.loc[keep].copy().reset_index(drop=True)
    report = {"removed_train_rows": int((~keep).sum())}
    for column in ("prompt_id", "prompt_key"):
        report[f"remaining_{column}_overlap"] = len(
            set(kept[column].astype(str)) & set(valid[column].astype(str))
        )
    report["remaining_answer_overlap"] = len(
        set(kept["text"].map(normalized_answer)) & valid_answers
    )
    if any(v for k, v in report.items() if k.startswith("remaining_")):
        raise ValueError("training/evaluation overlap survived exclusion")
    return kept, report


def family(label: str) -> str:
    return str(label).split("-", 1)[0]


def summarize_predictions(y, predicted, baseline, eligible) -> dict:
    rows = np.isin(y, eligible)
    y, predicted, baseline = y[rows], predicted[rows], baseline[rows]
    recalls, sibling_errors, provider_errors = {}, [], []
    for label in eligible:
        mask = y == label
        if not mask.any():
            raise ValueError(f"missing eligible label: {label}")
        guesses = predicted[mask]
        wrong = guesses != label
        same_family = np.array([family(v) == family(label) for v in guesses])
        recalls[label] = float((~wrong).mean())
        sibling_errors.append(float((wrong & same_family).mean()))
        provider_errors.append(float((wrong & ~same_family).mean()))
    return {
        "rows": int(len(y)),
        "eligible_label_count": len(eligible),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "per_label_recall": recalls,
        "macro_sibling_error": float(np.mean(sibling_errors)),
        "macro_provider_error": float(np.mean(provider_errors)),
        "rescues_vs_B": int(((predicted == y) & (baseline != y)).sum()),
        "regressions_vs_B": int(((predicted != y) & (baseline == y)).sum()),
        "disagreements_vs_B": int((predicted != baseline).sum()),
    }


def prompt_clusters(frame: pd.DataFrame) -> np.ndarray:
    keys = frame["prompt_key"].fillna("").astype(str)
    keys = keys.where(~keys.str.strip().isin(["", "nan", "None"]),
                      frame["prompt_id"].astype(str))
    if keys.str.strip().isin(["", "nan", "None"]).any():
        raise ValueError("missing prompt cluster identity")
    return keys.to_numpy()


def paired_bootstrap(frame, candidate, baseline, eligible, *, replicates=2000) -> dict:
    """Draw prompt multiplicities once per replicate, shared across sources/arms."""
    clusters = prompt_clusters(frame)
    unique, group = np.unique(clusters, return_inverse=True)
    y, source = frame["model"].to_numpy(), frame["source"].to_numpy()
    cells = [(s, label) for s in sorted(eligible) for label in eligible[s]]
    numerator = np.zeros((len(unique), len(cells)), dtype=float)
    denominator = np.zeros_like(numerator)
    delta = (candidate == y).astype(float) - (baseline == y).astype(float)
    for j, (s, label) in enumerate(cells):
        mask = (source == s) & (y == label)
        np.add.at(denominator[:, j], group[mask], 1)
        np.add.at(numerator[:, j], group[mask], delta[mask])
    cell_weights = np.array([1 / (len(eligible) * len(eligible[s])) for s, _ in cells])
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(replicates):
        multiplicity = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
        den = multiplicity @ denominator
        if np.any(den == 0):
            continue
        draws.append(float(((multiplicity @ numerator) / den) @ cell_weights))
    return {"requested_replicates": replicates, "retained_replicates": len(draws),
            "prompt_clusters": len(unique),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist() if draws else None}


def decision(delta_B, delta_P, source_deltas, ci95, retained) -> str:
    if retained < 1900 or ci95 is None:
        return "INCONCLUSIVE"
    passed = (delta_B >= 0.010 and delta_P >= 0.005
              and min(source_deltas) >= -0.010 and ci95[0] > 0)
    return "PASS_NEXT_STAGE_ONLY" if passed else "REJECT"


def run(pool_path: Path, output: Path, prediction_output: Path) -> dict:
    start = time.monotonic()
    if sha256_file(pool_path) != EXPECTED_RAW_SHA256:
        raise ValueError("public external pool SHA256 drift")
    if sklearn.__version__ != "1.6.1":
        raise ValueError("frozen scikit-learn version is 1.6.1")
    safe = build_conservative_subset(pd.read_parquet(pool_path))
    split = build_stage1_split(safe)
    train = safe.iloc[split["train_idx"]].copy()
    valid = safe.iloc[split["heldout_idx"]].copy().reset_index(drop=True)
    if len(train) != 30133 or set(split["holdout_sources"]) != {"abdullah", "semenoff", "zakir"}:
        raise ValueError("original source split drift")
    del safe
    train, overlap = purge_overlap(train, valid)
    classes = sorted(train["model"].astype(str).unique())
    if len(classes) != 18:
        raise ValueError("all 18 training labels are required")
    eligible = {}
    for source, group in valid.groupby("source", sort=True):
        counts = group["model"].value_counts()
        eligible[source] = sorted(k for k, v in counts.items() if v >= 20 and k in classes)
        eligible_rows = int(group["model"].isin(eligible[source]).sum())
        if len(eligible[source]) < 8 or eligible_rows < 300:
            raise ValueError(f"INCONCLUSIVE: frozen source eligibility failed: {source}")
    report = {
        "scope": "reused_external_source_disjoint_mechanism_diagnostic_not_promotion",
        "public_pool_sha256": EXPECTED_RAW_SHA256,
        "contract_sha256": sha256_file(Path(__file__).resolve().parents[1] /
                            "artifacts/research/conditional-source-balance-20261108/CONTRACT.md"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "train_rows": len(train), "heldout_rows": len(valid), "isolation": overlap,
        "eligible_labels": eligible, "arms": {}, "source_results": {},
        "training_cell_counts": {str(label): {str(s): int(n) for s, n in g["source"].value_counts().items()}
                                 for label, g in train.groupby("model")},
        "official_data_read": False, "gpu_used": False, "kaggle_submitted": False,
    }
    print(json.dumps({k: report[k] for k in ["train_rows", "heldout_rows", "isolation"]}), flush=True)
    print("Fitting the four frozen training-only TF-IDF views", flush=True)
    x, v, feature_meta = build_fold_features(train["text"].tolist(), valid["text"].tolist())
    report["features"] = feature_meta
    print(f"Features ready: train={x.shape}, evaluation={v.shape}", flush=True)
    y = train["model"].to_numpy()
    source_weights = conditional_weights(train)
    weights = {"B": np.ones(len(train)), "S": source_weights,
               "P": shuffle_within_class(source_weights, y)}
    predictions, margins = {}, {}
    for arm in ARMS:
        arm_start = time.monotonic()
        model = make_svc()
        print(f"Fitting frozen arm {arm}", flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y, sample_weight=weights[arm])
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError(f"INVALID: {arm} did not converge; do not interpret its scores")
        if list(model.classes_) != classes:
            raise ValueError("class order drift")
        raw_margins = model.decision_function(v)
        if raw_margins.shape != (len(valid), 18) or not np.isfinite(raw_margins).all():
            raise ValueError("invalid 18-label margins")
        predictions[arm] = np.asarray(classes)[raw_margins.argmax(axis=1)]
        margins[arm] = raw_margins.astype(np.float32)
        w = weights[arm]
        report["arms"][arm] = {"fit_seconds": time.monotonic() - arm_start,
                               "iterations": int(model.n_iter_),
                               "sample_weight_min": float(w.min()),
                               "sample_weight_max": float(w.max()),
                               "effective_sample_size": float(w.sum() ** 2 / (w @ w))}
        print(f"Completed {arm} in {report['arms'][arm]['fit_seconds']:.1f}s", flush=True)
    for source in sorted(eligible):
        rows = valid["source"].to_numpy() == source
        report["source_results"][source] = {
            arm: summarize_predictions(valid["model"].to_numpy()[rows], predictions[arm][rows],
                                       predictions["B"][rows], eligible[source]) for arm in ARMS
        }
    mean_ba = {a: float(np.mean([s[a]["balanced_accuracy"] for s in report["source_results"].values()]))
               for a in ARMS}
    source_deltas = [s["S"]["balanced_accuracy"] - s["B"]["balanced_accuracy"]
                     for s in report["source_results"].values()]
    boot = paired_bootstrap(valid, predictions["S"], predictions["B"], eligible)
    report["primary"] = {"mean_source_ba": mean_ba,
                         "S_minus_B": mean_ba["S"] - mean_ba["B"],
                         "S_minus_P": mean_ba["S"] - mean_ba["P"],
                         "per_source_S_minus_B": dict(zip(sorted(eligible), source_deltas)),
                         "bootstrap_S_minus_B": boot}
    report["decision"] = decision(report["primary"]["S_minus_B"], report["primary"]["S_minus_P"],
                                   source_deltas, boot["ci95"], boot["retained_replicates"])
    hashes = np.array([hashlib.sha256(t.encode()).hexdigest() for t in prompt_clusters(valid)])
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_output, y=valid["model"].to_numpy(dtype=str),
                        source=valid["source"].to_numpy(dtype=str), prompt_sha256=hashes,
                        labels=np.asarray(classes), **{f"pred_{a}": predictions[a] for a in ARMS},
                        **{f"margin_{a}": margins[a] for a in ARMS})
    report["aligned_prediction_archive_sha256"] = sha256_file(prediction_output)
    report["elapsed_seconds"] = time.monotonic() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"primary": report["primary"], "decision": report["decision"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    run(args.pool, args.output, args.predictions)


if __name__ == "__main__":
    main()
