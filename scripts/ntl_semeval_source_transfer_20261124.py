"""CPU-only source-transfer falsifier inspired by SemEval-2024 model attribution.
Read true competition TRAIN and pinned C1/C2 publisher panels in /tmp.
Output aggregate statistics only. No fitting, weights, row IDs, text or Kaggle submit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PANELS = {
    "C1": ("8746c4b77c2b09b9abafaeecc381a39321b9919214c005a3657042051c51b177", 360),
    "C2": ("bf25ef1e953d5c37b177f27b841a04098ae9b35f45fc43fa11036d91e8831e02", 470),
}
FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "do", "for", "from", "had", "has", "have", "he", "her", "his", "i", "if",
    "in", "into", "is", "it", "its", "may", "my", "not", "of", "on", "or",
    "our", "she", "should", "so", "than", "that", "the", "their", "them",
    "there", "these", "they", "this", "those", "to", "was", "we", "were",
    "what", "when", "which", "who", "will", "with", "would", "you", "your",
}
METRICS = ("word_count", "mean_word_length", "punct_per_100", "digits_per_100", "function_word_pct")
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[0-9]+")
PUNCT = re.compile(r"[,.;:!?]")
DIGITS = re.compile(r"[0-9]")
MIN_GROUP_MODELS = 6
MIN_RHO = 0.35
REQUIRED_METRICS = 3


def features(text: str) -> tuple[float, ...]:
    words = WORD.findall(text if isinstance(text, str) else "")
    n = len(words)
    if n == 0:
        raise ValueError("blank or unparseable text")
    return (float(n), float(sum(len(w) for w in words) / n),
            100.0 * len(PUNCT.findall(text)) / n,
            100.0 * len(DIGITS.findall(text)) / n,
            100.0 * sum(w.lower() in FUNCTION_WORDS for w in words) / n)


def load(path: Path, *, expected_sha: str | None = None, expected_rows: int | None = None) -> pd.DataFrame:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and sha != expected_sha:
        raise RuntimeError(f"source hash drift for {path.name}; reject and re-audit publisher")
    df = pd.read_csv(path, low_memory=False)
    if expected_rows is not None and len(df) != expected_rows:
        raise RuntimeError("source row count drift; reject")
    required = {"prompt_id", "model", "text"}
    if not required.issubset(df):
        raise ValueError("required columns missing: " + str(sorted(required - set(df))))
    if df[list(required)].isna().any().any():
        raise ValueError("null prompt/model/text")
    if df.duplicated(subset=["prompt_id", "model"]).any():
        raise ValueError("duplicate model in same prompt block")
    if df["text"].astype(str).str.strip().eq("").any():
        raise ValueError("blank text")
    return df


def center(df: pd.DataFrame, labels: set[str]) -> tuple[pd.DataFrame, int]:
    part = df.loc[df["model"].isin(labels), ["prompt_id", "model", "text"]].copy()
    part["prompt_id"] = part["prompt_id"].astype(str)
    arr = np.asarray([features(v) for v in part["text"]], dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("nonfinite feature")
    for k, name in enumerate(METRICS):
        part[name] = arr[:, k]
    size = part.groupby("prompt_id")["model"].transform("nunique")
    part = part[size >= MIN_GROUP_MODELS].copy()
    if part.empty:
        raise ValueError("no prompt blocks with enough genuine declared model classes")
    # Leave-one-out context subtraction: no model labels/weights fitted; prompt is nuisance.
    sizes = part.groupby("prompt_id")["model"].transform("size").to_numpy()
    for name in METRICS:
        total = part.groupby("prompt_id")[name].transform("sum").to_numpy()
        x = part[name].to_numpy()
        part[name] = x - (total - x) / (sizes - 1)
    return part, int(part.prompt_id.nunique())


def signatures(part: pd.DataFrame, ordered: list[str]) -> dict[str, list[float]]:
    by_model = part.groupby("model")[list(METRICS)].mean()
    if not set(ordered).issubset(by_model.index):
        raise ValueError("missing a declared shared label after group filtering")
    return {name: [float(v) for v in by_model.loc[ordered, name].to_numpy()] for name in METRICS}


def rho(a: list[float], b: list[float]) -> float | None:
    if len(a) < 6 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    score = float(spearmanr(a, b).statistic)
    return round(score, 6) if np.isfinite(score) else None


def analyze(official: pd.DataFrame, c1: pd.DataFrame, c2: pd.DataFrame) -> dict:
    counts = official["model"].value_counts()
    if len(official) != 2699 or len(counts) != 18:
        raise ValueError("not pinned original 2699-row 18-model competition TRAIN")
    labels = set(counts.index)
    a, b = set(c1.model), set(c2.model)
    shared = sorted(labels & a & b)
    if len(a) != 12 or len(b) != 12 or len(shared) != 12 or a != b:
        raise ValueError("source class support changed; refuse unsanctioned comparisons")
    frames = {}
    prompt_counts = {}
    covered = {}
    for tag, df in (("official", official), ("C1", c1), ("C2", c2)):
        f, n = center(df, set(shared))
        frames[tag] = signatures(f, shared)
        prompt_counts[tag] = n
        covered[tag] = int(len(f))
    comparisons = {}
    for name in METRICS:
        o, x, y = [frames[tag][name] for tag in ("official", "C1", "C2")]
        comparisons[name] = {
            "official_vs_c1_rho": rho(o, x),
            "official_vs_c2_rho": rho(o, y),
            "c1_vs_c2_rho": rho(x, y),
        }
    stable = sum(
        v["official_vs_c1_rho"] is not None
        and v["official_vs_c2_rho"] is not None
        and v["official_vs_c1_rho"] >= MIN_RHO
        and v["official_vs_c2_rho"] >= MIN_RHO
        for v in comparisons.values()
    )
    return {
        "status": "REAL_UNTRAINED_12CLASS_SOURCE_FINGERPRINT_DIAGNOSTIC",
        "official_rows": len(official),
        "official_labels": len(counts),
        "official_class_count_min": int(counts.min()),
        "official_class_count_max": int(counts.max()),
        "official_inverse_frequency_weight_ratio_max_min": round(float(counts.max()/counts.min()), 7),
        "shared_labels": len(shared),
        "source_rows": {"C1": len(c1), "C2": len(c2)},
        "qualified_prompt_groups": prompt_counts,
        "qualified_rows": covered,
        "fingerprint_rhos": comparisons,
        "stable_metric_count_fixed_threshold": stable,
        "stable_metric_gate": f"{stable} >= {REQUIRED_METRICS} of {len(METRICS)}",
        "passes_descriptive_stability_gate": stable >= REQUIRED_METRICS,
        "no_fitted_model_or_training": True,
        "no_0_715_same_row_prediction": True,
        "no_18class_holdout": True,
        "no_public_score_claim": True,
        "c1_c2_same_publisher_not_independent": True,
        "served_provider_identity_unverified": True,
    }


def selftest() -> None:
    assert features("The quick, brown fox jumps 123.") == (6.0, 4.0, 200/6, 50.0, 100/6)
    rows = [
        {"prompt_id": g, "model": f"m{i}", "text": f"Alpha beta gamma {i}."}
        for g in range(3) for i in range(6)
    ]
    s, n = center(pd.DataFrame(rows), {f"m{i}" for i in range(6)})
    assert n == 3 and len(s) == 18
    for _, group in s.groupby("prompt_id"):
        for k in METRICS:
            assert abs(float(group[k].sum())) < 1e-8
    assert rho([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]) == -1
    print("NTL_SEMEVAL_SOURCE_TRANSFER_SELFTEST=PASS", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--official", type=Path)
    p.add_argument("--c1", type=Path)
    p.add_argument("--c2", type=Path)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if not all((args.official, args.c1, args.c2)):
        p.error("official, c1, c2 are required")
    official = load(args.official, expected_rows=2699)
    c1 = load(args.c1, expected_sha=PANELS["C1"][0], expected_rows=PANELS["C1"][1])
    c2 = load(args.c2, expected_sha=PANELS["C2"][0], expected_rows=PANELS["C2"][1])
    print("NTL_SEMEVAL_SOURCE_TRANSFER_REAL_AGGREGATES="
          + json.dumps(analyze(official,c1,c2),sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
