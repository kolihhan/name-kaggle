#!/usr/bin/env python3
"""Read-only public external dataset feasibility audit; never prints passages or requests.

Frozen question: Does a crossed 2-model × 2-temperature design exist for
the *same prompt, request text, and requested output length*?
This is an identifiability/support check, not an attribution benchmark.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
from pathlib import Path

SETTINGS = ("rlow_t1", "rlow_t0")
NEEDED = {"run", "bank", "prompt_id", "model", "model_slug", "setting",
          "target_words", "n_words", "created_at", "request", "text"}


def audit(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(NEEDED - set(reader.fieldnames or []))
        if missing:
            raise ValueError("Required metadata columns missing: " + str(missing))
        rows = list(reader)
    if not rows:
        raise ValueError("Empty dataset")

    def s(row, name):
        return str(row[name]).strip()

    # Three levels of pair matching: the last two cannot invent prompt identity.
    levels = {
        "prompt_only": ("bank", "prompt_id"),
        "prompt_and_target_words": ("bank", "prompt_id", "target_words"),
        "exact_request": ("bank", "prompt_id", "target_words", "request"),
    }
    crossed = {}
    for level, keys in levels.items():
        groups = collections.defaultdict(lambda: collections.defaultdict(set))
        for row in rows:
            if s(row, "setting") not in SETTINGS:
                continue
            if not all(s(row, k) for k in keys + ("model_slug",)):
                continue
            key = tuple(s(row, k) for k in keys)
            model = s(row, "model_slug")
            groups[key][model].add(s(row, "setting"))
        pairs = collections.Counter()
        within = collections.Counter()
        for key, members in groups.items():
            complete = sorted(m for m, settings in members.items()
                              if all(x in settings for x in SETTINGS))
            for m in complete:
                within[m] += 1
            for a, b in itertools.combinations(complete, 2):
                pairs[(a, b)] += 1
        crossed[level] = {
            "fully_crossed_2models_2temps": sum(pairs.values()),
            "unique_matched_prompt_keys": len(groups),
            "model_pairs": [
                {"model_a": a, "model_b": b, "matched_prompt_keys": n}
                for (a, b), n in pairs.most_common()
            ],
            "same_model_2temps_pairs": dict(sorted(within.items())),
        }
    # Additional source-coverage audit; no text, raw requests, or per-prompt IDs.
    by_model = collections.Counter(s(r, "model") for r in rows)
    by_setting = collections.Counter(s(r, "setting") for r in rows)
    by_run = collections.Counter(s(r, "run") for r in rows)
    by_bank = collections.Counter(s(r, "bank") for r in rows)
    model_request_counts = collections.defaultdict(set)
    model_target_counts = collections.defaultdict(set)
    for r in rows:
        key = (s(r, "bank"), s(r, "prompt_id"))
        model_request_counts[key].add(s(r, "model_slug"))
        model_target_counts[key].add(s(r, "target_words"))
    # Check if target length varies between model outputs of the same prompt.
    multiple_model_keys = [k for k, models in model_request_counts.items() if len(models) >= 2]
    length_varies = sum(len(model_target_counts[k]) > 1 for k in multiple_model_keys)

    result = {
        "source": "zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations",
        "sha256": digest,
        "rows": len(rows),
        "models": dict(sorted(by_model.items())),
        "settings": dict(sorted(by_setting.items())),
        "runs": dict(sorted(by_run.items())),
        "banks": dict(sorted(by_bank.items())),
        "unique_bank_prompt_keys": len(model_request_counts),
        "prompt_keys_with_2plus_models": len(multiple_model_keys),
        "those_with_different_requested_lengths": length_varies,
        "crossed_design": crossed,
        "pass_frozen_20_full_quadruples": crossed["exact_request"]["fully_crossed_2models_2temps"] >= 20,
        "limitations": [
            "Same proxy model slug is not proof of same immutable backend snapshot.",
            "Temperature comparison alone does not control output length or other protocol parameters.",
            "This dataset is a public external A/B source, not an independent 18-label Kaggle holdout.",
            "No fit, classifier inference, hidden label evaluation, or score measured.",
        ],
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    args = parser.parse_args()
    print("CROSS_DESIGN_RESULT", json.dumps(audit(args.csv), sort_keys=True))
