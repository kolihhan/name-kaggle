"""Read-only audit of exact normalized overlap between two publicly supplied external pools.

Reports aggregate lineage statistics only; does NOT download official competition data,
train a model, select a classifier, or publish source texts / prompt IDs / hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

EXPECTED_STAGE1_SHA = "37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec"
FAMILY_PREFIXES = ("claude-", "gemini-", "glm-", "gpt-", "kimi", "qwen")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def normalized_hash(text):
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = " ".join(value.split())
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pick_column(cols, choices):
    return next((col for col in choices if col in cols), None)


def family(model):
    return next((prefix for prefix in FAMILY_PREFIXES if str(model).startswith(prefix)), "<unknown>")


def source_label_gate(group, model_col):
    counts = group[model_col].astype(str).value_counts()
    eligible = [label for label, count in counts.items() if count >= 20]
    groups = Counter(family(label) for label in eligible)
    return {
        "labels_with_20_rows": len(eligible),
        "sibling_families_with_2_labels_20_rows": sum(n >= 2 for n in groups.values()),
    }


def analyze(stage1_path: Path, candidate_path: Path):
    sha1 = sha256_file(stage1_path)
    if sha1 != EXPECTED_STAGE1_SHA:
        raise RuntimeError(f"Stage-1 pool SHA mismatch: {sha1}")
    candidate_sha = sha256_file(candidate_path)
    base = pd.read_parquet(stage1_path)
    candidate = pd.read_parquet(candidate_path)
    if len(base) != 75783 or set(("text", "source", "model")) - set(base.columns):
        raise RuntimeError("unexpected Stage-1 corpus")
    columns = list(map(str, candidate.columns))
    source_col = pick_column(columns, ("source_dataset", "source", "dataset_ref", "dataset", "source_id", "source_name"))
    label_col = pick_column(columns, ("model", "label", "model_label", "generator_model"))
    text_col = pick_column(columns, ("text", "response", "passage", "output_text"))
    prompt_col = pick_column(columns, ("prompt_id", "prompt_key", "prompt_text", "prompt"))
    report = {
        "scope": "external_pools_source_overlap_recon_no_official_data_no_model",
        "stage1_sha256": sha1,
        "candidate_sha256": candidate_sha,
        "stage1_rows": len(base),
        "candidate_rows": len(candidate),
        "candidate_columns": columns,
        "detected_fields": {
            "source": source_col,
            "label": label_col,
            "text": text_col,
            "prompt": prompt_col,
        },
        "stage1_source_counts": {
            str(k): int(v) for k, v in base["source"].astype(str).value_counts().items()
        },
    }
    if not source_col or not label_col or not text_col:
        report["admission"] = "BLOCKED_SCHEMA_UNRESOLVED"
        return report

    base_hashes = set(base["text"].map(normalized_hash))
    cand_hashes = candidate[text_col].map(normalized_hash)
    source_vals = candidate[source_col].fillna("<missing>").astype(str)
    model_vals = candidate[label_col].fillna("<missing>").astype(str)
    stats = pd.DataFrame({
        "source": source_vals,
        "model": model_vals,
        "overlap": cand_hashes.isin(base_hashes),
        "hash": cand_hashes,
    })
    report["exact_normalized_text_overlap_rows"] = int(stats["overlap"].sum())
    report["unique_candidate_texts"] = int(stats["hash"].nunique())
    report["source_count"] = int(stats["source"].nunique())
    report["label_count"] = int(stats["model"].nunique())
    report["label_counts"] = {
        str(k): int(v) for k, v in model_vals.value_counts().items()
    }
    report["source_stats"] = {}
    for name, g in stats.groupby("source"):
        label_counts = {str(k): int(v) for k, v in g["model"].value_counts().items()}
        report["source_stats"][name] = {
            "rows": len(g),
            "unique_responses": int(g["hash"].nunique()),
            "overlap_with_stage1_rows": int(g["overlap"].sum()),
            "overlap_with_stage1_fraction": round(float(g["overlap"].mean()), 6),
            "label_count": len(label_counts),
            "label_counts": label_counts,
            **source_label_gate(g, "model"),
        }
    report["source_strings_matching_stage1_short_names"] = sorted(
        name for name in report["source_stats"]
        if name.lower() in set(base["source"].astype(str).str.lower())
    )
    if prompt_col:
        report["candidate_unique_prompts"] = int(candidate[prompt_col].astype(str).nunique())
        report["prompt_field_has_nulls"] = bool(candidate[prompt_col].isna().any())
    report["served_model_columns"] = sorted(
        c for c in columns if "served" in c.lower() or "route" in c.lower()
        or "model_version" in c.lower() or "provider" in c.lower()
    )
    report["admission"] = "NOT_ADMITTED_PRELIMINARY_TEXT_AND_SCHEMA_AUDIT_ONLY"
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = analyze(args.stage1, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    public = {k: v for k, v in report.items() if k != "source_stats"}
    public["top_source_stats"] = dict(sorted(report.get("source_stats", {}).items(),
             key=lambda item: -item[1]["rows"])[:20])
    print(json.dumps(public, indent=2, sort_keys=True)[:23000])


if __name__ == "__main__":
    main()
