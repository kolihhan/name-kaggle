#!/usr/bin/env python3
"""Descriptive paired sibling feature audit. No model fit, no test labels, no submission.

This is NOT a transfer benchmark: official train has been examined adaptively.
Only aggregate statistics are emitted; raw passages and prompt IDs are never output.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

PAIRS = (
    ("gpt-5.6-luna", "gpt-5.6-sol"),
    ("gpt-5.6-sol", "gpt-5.6-terra"),
    ("gemini-3.7-flash", "gemini-3.8-flash"),
    ("gemini-3.6-flash", "gemini-3.8-flash"),
    ("claude-opus-4-6", "claude-opus-4-8"),
    ("glm-5.3", "glm-5.3-flash"),
)
TERMS = ("however", "moreover", "indeed", "therefore")
FEATURES = (
    "words", "mean_word_len", "long_words_per100", "periods_per100",
    "commas_per100", "colons_per100", "semicolon_per100",
    "em_dash", "curly_apostrophe", "curly_double_quote", "question_mark",
    *TERMS, "in_conclusion", "newline", "markdown", "digit",
)

def featurize(text: str) -> dict[str, float]:
    ws = re.findall(r"\S+", text)
    alpha = re.findall(r"\b[A-Za-z]+\b", text)
    n = max(len(ws), 1)
    a = max(len(alpha), 1)
    out = {
        "words": float(len(ws)),
        "mean_word_len": sum(map(len, ws)) / n,
        "long_words_per100": 100 * sum(len(w) >= 9 for w in alpha) / a,
        "periods_per100": 100 * text.count(".") / n,
        "commas_per100": 100 * text.count(",") / n,
        "colons_per100": 100 * text.count(":") / n,
        "semicolon_per100": 100 * text.count(";") / n,
        "em_dash": float("—" in text),
        "curly_apostrophe": float("’" in text),
        "curly_double_quote": float("“" in text or "”" in text),
        "question_mark": float("?" in text),
        "in_conclusion": float(bool(re.search(r"\bin conclusion\b", text, re.I))),
        "newline": float("\n" in text),
        "markdown": float("**" in text or bool(re.search(r"(?m)^[-*] ", text))),
        "digit": float(bool(re.search(r"\d", text))),
    }
    for term in TERMS:
        out[term] = float(bool(re.search(r"\b" + term + r"\b", text, re.I)))
    return out

def paired_sign_p(a_more: int, b_more: int) -> float:
    n = a_more + b_more
    if not n:
        return 1.0
    k = min(a_more, b_more)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)

def bh_correct(rows: list[dict]) -> None:
    rows.sort(key=lambda r: r["p"])
    minimum = 1.0
    for idx in reversed(range(len(rows))):
        minimum = min(minimum, rows[idx]["p"] * len(rows) / (idx + 1))
        rows[idx]["q"] = minimum

def audit(records: list[dict[str, str]]) -> dict:
    required = {"prompt_id", "model", "text"}
    if not records or any(not required.issubset(r) for r in records):
        raise ValueError("Nonempty input with prompt_id/model/text is required")
    panels: dict[str, dict[str, dict]] = {}
    seen = set()
    for r in records:
        pair_id = (str(r["prompt_id"]), str(r["model"]))
        if pair_id in seen:
            raise ValueError("Duplicate model/prompt cell; cannot treat as independent")
        seen.add(pair_id)
        panels.setdefault(pair_id[0], {})[pair_id[1]] = featurize(str(r["text"]))
    tests: list[dict] = []
    for a, b in PAIRS:
        matched = [(panel[a], panel[b]) for panel in panels.values() if a in panel and b in panel]
        if not matched:
            raise ValueError(f"Missing matched prompts for {a} and {b}")
        for feature in FEATURES:
            differences = sorted(aa[feature] - bb[feature] for aa, bb in matched)
            plus = sum(d > 1e-12 for d in differences)
            minus = sum(d < -1e-12 for d in differences)
            m = len(differences)
            median = (differences[(m - 1) // 2] + differences[m // 2]) / 2.0
            tests.append({
                "pair_a": a, "pair_b": b, "feature": feature,
                "matched_prompts": m, "a_more": plus, "b_more": minus,
                "ties": m - plus - minus, "median_a_minus_b": round(median, 6),
                "p": paired_sign_p(plus, minus),
            })
    bh_correct(tests)
    notable = [t for t in tests if t["q"] < 0.05 and t["a_more"] + t["b_more"] >= 35]
    return {
        "input_rows": len(records),
        "distinct_prompt_ids": len(panels),
        "feature_count": len(FEATURES),
        "tested_pairs": len(PAIRS),
        "tests_total": len(tests),
        "bh_significant_q_below_005": len(notable),
        "significant_aggregate_only": notable,
        "status": "DESCRIPTIVE_SPENT_TRAIN_ONLY_NO_INDEPENDENT_TRANSFER",
        "limitations": [
            "Signs compare features of two responses to the SAME prompt; no classifier trained.",
            "Already-inspected official train is adaptively reused; q values are descriptive only.",
            "No generation request/settings, true served revisions or external validation provided.",
            "No effect is promoted to an inference rule, Kaggle submission or new score.",
        ],
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.csv_file.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = audit(rows)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")

if __name__ == "__main__":
    main()
