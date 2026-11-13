"""Measure a fixed GLM likelihood contrast on previously used train prompts."""
import argparse
from collections import defaultdict
import csv
import json
from math import comb, isfinite
from pathlib import Path


FULL = "glm-5.3"
FLASH = "glm-5.3-flash"


def read_index(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            key = row["id"]
            if not key or key in rows:
                raise ValueError(f"empty or duplicate ID in {path.name}")
            rows[key] = row
        return rows


def compare(left, right):
    wins = sum(a > b for a, b in zip(left, right, strict=True))
    ties = sum(a == b for a, b in zip(left, right, strict=True))
    return (wins + ties / 2) / len(left)


def summarize(full, flash):
    wins = sum(a > b for a, b in zip(full, flash, strict=True))
    ties = sum(a == b for a, b in zip(full, flash, strict=True))
    n = len(full)
    decided = n - ties
    return {
        "same_prompt_rank_accuracy": compare(full, flash),
        "different_prompt_rank_accuracy": (
            sum(compare([a] * (n - 1), flash[:i] + flash[i + 1:]) for i, a in enumerate(full)) / n
        ),
        "paired_wins": wins,
        "paired_ties": ties,
        "paired_sign_p_one_sided": sum(comb(decided, i) for i in range(wins, decided + 1)) / 2**decided,
        "full_recall_at_zero": compare(full, [0] * n),
        "flash_recall_at_zero": compare([0] * n, flash),
    }


def audit(train_path, full_path, flash_path):
    train, full, flash = (read_index(p) for p in (train_path, full_path, flash_path))
    if not train.keys() == full.keys() == flash.keys():
        raise ValueError("feature tables must cover the same training IDs")
    pairs = defaultdict(dict)
    for key, row in train.items():
        if int(full[key]["n_tok"]) != int(flash[key]["n_tok"]):
            raise ValueError("model feature token counts differ")
        label = row["model"]
        if label not in {FULL, FLASH}:
            continue
        pair = pairs[row["prompt_id"]]
        if label in pair:
            raise ValueError("multiple responses for one model within a prompt")
        a, b = full[key], flash[key]
        scores = {
            "mean_log_probability_ratio": float(a["lp_mean"]) - float(b["lp_mean"]),
            "bpc_ratio": float(b["bpc"]) - float(a["bpc"]),
            "full_likelihood_only": float(a["lp_mean"]),
            "flash_likelihood_only": -float(b["lp_mean"]),
        }
        if not all(map(isfinite, scores.values())):
            raise ValueError("nonfinite likelihood evidence")
        pair[label] = scores
    complete = [p for p in pairs.values() if p.keys() == {FULL, FLASH}]
    if len(complete) < 2:
        raise ValueError("need at least two matched prompts")
    return {
        "scope": "spent_official_train_glm_pair_only",
        "training_index_rows": len(train),
        "paired_prompts": len(complete),
        "unpaired_prompts": len(pairs) - len(complete),
        "scores": {name: summarize([p[FULL][name] for p in complete], [p[FLASH][name] for p in complete])
                   for name in complete[0][FULL]},
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("train", "full", "flash"):
        cli.add_argument(field, type=Path)
    args = cli.parse_args()
    print(json.dumps(audit(args.train, args.full, args.flash), indent=2, sort_keys=True))
