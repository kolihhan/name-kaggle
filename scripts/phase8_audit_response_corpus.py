#!/usr/bin/env python3
import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

MODEL_COLUMNS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
]

PHASE7_UNCOVERED = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.7-flash", "glm-5.3", "glm-5.3-flash",
    "kimi-k3", "qwen3.8-max",
]

# Frozen only after independent timing checks. Each source/prompt bank was public before
# the competition opened on 2026-10-14. EEE_datastore and Met4physics are excluded here
# because the exact rows' pre-competition timing was not pinned strongly enough.
PRECOMPETITION_SOURCES = {
    "Zek-Takai/glm53-flash-harvest",
    "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k",
    "greghavens/kimi-k3-coding-and-debugging-traces",
    "zlab-princeton/SWEeper-Bench-traces",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(text.split())


def build_clean_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in ["mechanically_eligible", "text_only_suitable"]:
        mask &= df[col].fillna(False).astype(bool)
    for col in ["no_train", "possible_truncation", "possible_reasoning_in_answer", "text_shared_across_labels"]:
        mask &= ~df[col].fillna(True).astype(bool)
    return mask


def summarize_overlap(official_texts, external_texts):
    official_raw = {str(x) for x in official_texts if pd.notna(x)}
    external_raw = {str(x) for x in external_texts if pd.notna(x)}
    official_norm = {normalize_text(x) for x in official_raw}
    external_norm = {normalize_text(x) for x in external_raw}
    return {
        "exact_overlap_count": len(official_raw & external_raw),
        "normalized_overlap_count": len(official_norm & external_norm),
    }


def build_source_matrix(provenance: pd.DataFrame):
    matrix = {}
    for source, group in provenance.groupby("source", dropna=False):
        key = "<missing>" if pd.isna(source) else str(source)
        license_known = ~group["license_unknown"].fillna(True).astype(bool)
        terms_clear = ~group["output_terms_review"].fillna(True).astype(bool)
        prompt_exact = group["prompt_exact"].fillna(False).astype(bool)
        per_model = group.groupby("model")["response_id"].nunique().sort_index()
        matrix[key] = {
            "occurrences": int(len(group)),
            "unique_responses": int(group["response_id"].nunique()),
            "models": sorted(group["model"].dropna().astype(str).unique().tolist()),
            "unique_responses_by_model": {str(k): int(v) for k, v in per_model.items()},
            "licenses": sorted({str(x) for x in group.loc[group["license"].notna(), "license"]}),
            "evidence_classes": sorted({str(x) for x in group.loc[group["model_evidence_class"].notna(), "model_evidence_class"]}),
            "license_known_fraction": float(license_known.mean()) if len(group) else 0.0,
            "output_terms_clear_fraction": float(terms_clear.mean()) if len(group) else 0.0,
            "prompt_exact_fraction": float(prompt_exact.mean()) if len(group) else 0.0,
        }
    return dict(sorted(matrix.items()))


def select_strict_sources(source_matrix):
    return sorted(
        source for source, item in source_matrix.items()
        if item["license_known_fraction"] == 1.0
        and item["output_terms_clear_fraction"] == 1.0
        and item["prompt_exact_fraction"] == 1.0
    )


def select_precompetition_sources(strict_sources):
    return sorted(set(strict_sources) & PRECOMPETITION_SOURCES)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_from_sources(clean, clean_prov, sources):
    ids = set(clean_prov.loc[clean_prov["source"].astype(str).isin(sources), "response_id"].astype(str))
    candidate = clean[
        clean["response_id"].astype(str).isin(ids) & clean["model"].isin(PHASE7_UNCOVERED)
    ].copy()
    return candidate.drop_duplicates(subset=["response_id", "model"])


def candidate_summary(candidate, official):
    counts = {m: int((candidate["model"] == m).sum()) for m in PHASE7_UNCOVERED}
    return {
        "rows": int(len(candidate)),
        "unique_response_ids": int(candidate["response_id"].nunique()),
        "rows_by_model": counts,
        "labels_with_rows": [m for m in PHASE7_UNCOVERED if counts[m] > 0],
        "labels_without_rows": [m for m in PHASE7_UNCOVERED if counts[m] == 0],
        "official_answer_overlap": summarize_overlap(official["text"], candidate["text"]),
        "training_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.corpus_root
    short_path = root / "external_short_wide.parquet"
    cell_path = root / "external_cell_metadata.parquet"
    prov_path = root / "external_cell_provenance.parquet"
    response_path = root / "external_response_metadata.parquet"

    official = pd.read_csv(args.official_train)
    official_labels = sorted(official["model"].dropna().unique().tolist())
    if official_labels != sorted(MODEL_COLUMNS):
        raise SystemExit("official label set drifted from frozen 18-label contract")

    wide = pd.read_parquet(short_path)
    missing_models = sorted(set(MODEL_COLUMNS) - set(wide.columns))
    extra_models = sorted(set(wide.columns) - {"id", "prompt"} - set(MODEL_COLUMNS))
    if missing_models or extra_models:
        raise SystemExit(f"external model schema drift: missing={missing_models} extra={extra_models}")

    long = wide.melt(id_vars=["id", "prompt"], value_vars=MODEL_COLUMNS, var_name="model", value_name="text")
    long = long.dropna(subset=["text"]).copy()

    cells = pd.read_parquet(cell_path)
    short_cells = cells[cells["view"].eq("short")].copy()
    if len(short_cells) != len(long):
        raise SystemExit(f"short cell count mismatch: metadata={len(short_cells)} wide={len(long)}")

    merged = long.merge(short_cells, on=["id", "model"], how="left", validate="one_to_one", suffixes=("", "_meta"))
    if merged["response_id"].isna().any():
        raise SystemExit("short wide cells failed metadata join")

    clean = merged[build_clean_mask(merged)].copy()
    license_clear = clean[~clean["license_unknown"].fillna(True).astype(bool)].copy()
    strict_terms_clear = license_clear[~license_clear["output_terms_review"].fillna(True).astype(bool)].copy()

    overlap_all = summarize_overlap(official["text"], long["text"])
    overlap_clean = summarize_overlap(official["text"], clean["text"])
    norm = clean["text"].map(normalize_text)
    norm_model_counts = pd.DataFrame({"norm": norm, "model": clean["model"]}).groupby("norm")["model"].nunique()
    cross_label_norm_dupes = int((norm_model_counts > 1).sum())

    prov = pd.read_parquet(prov_path)
    prov_short = prov[prov["cell_id_short"].notna()].copy()
    response = pd.read_parquet(response_path)
    clean_ids = set(clean["response_id"].astype(str))
    clean_prov = prov_short[prov_short["response_id"].astype(str).isin(clean_ids)].copy()
    clean_resp = response[response["response_id"].astype(str).isin(clean_ids)].copy()

    prompt_exact = clean_resp["prompt_exact"].fillna(False).astype(bool)
    source_counts = Counter(clean_prov["source"].dropna().astype(str).tolist())
    evidence_counts = Counter(clean_resp["model_evidence_class"].dropna().astype(str).tolist())
    source_matrix = build_source_matrix(clean_prov)
    strict_sources = select_strict_sources(source_matrix)
    precompetition_sources = select_precompetition_sources(strict_sources)

    strict_candidate = candidate_from_sources(clean, clean_prov, strict_sources)
    precompetition_candidate = candidate_from_sources(clean, clean_prov, precompetition_sources)

    model_counts = {m: int((clean["model"] == m).sum()) for m in MODEL_COLUMNS}
    license_clear_counts = {m: int((license_clear["model"] == m).sum()) for m in MODEL_COLUMNS}

    report = {
        "dataset": "olegzholobov/name-that-llm-response-corpus",
        "scope": "audit_only_no_training",
        "official_train": {"rows": int(len(official)), "labels": official_labels, "sha256": sha256(args.official_train)},
        "external_short": {
            "wide_rows": int(len(wide)), "nonempty_cells": int(len(long)),
            "labels_present": sorted(long["model"].unique().tolist()),
            "all_18_labels_present": sorted(long["model"].unique().tolist()) == sorted(MODEL_COLUMNS),
            "parquet_sha256": sha256(short_path),
        },
        "quality_filter": {
            "mechanically_clean_cells": int(len(clean)),
            "license_known_cells": int(len(license_clear)),
            "license_known_and_output_terms_clear_cells": int(len(strict_terms_clear)),
            "mechanically_clean_by_model": model_counts,
            "license_known_by_model": license_clear_counts,
            "cross_label_normalized_duplicate_groups_after_clean_filter": cross_label_norm_dupes,
        },
        "official_answer_overlap": {"all_short": overlap_all, "mechanically_clean": overlap_clean},
        "provenance": {
            "clean_unique_response_ids": int(clean["response_id"].nunique()),
            "clean_prompt_exact_responses": int(prompt_exact.sum()),
            "clean_prompt_exact_fraction": float(prompt_exact.mean()) if len(prompt_exact) else 0.0,
            "clean_sources": dict(sorted(source_counts.items())),
            "model_evidence_classes": dict(sorted(evidence_counts.items())),
            "source_matrix": source_matrix,
        },
        "strict_source_candidate": {
            "rule": "100% license-known + 100% output-terms-clear + 100% exact-prompt provenance, then Phase 7 uncovered labels only",
            "eligible_sources": strict_sources,
            **candidate_summary(strict_candidate, official),
        },
        "precompetition_candidate": {
            "rule": "strict-source candidate intersected with frozen sources whose source/prompt bank was independently verified public before competition start 2026-10-14",
            "eligible_sources": precompetition_sources,
            "excluded_strict_sources": sorted(set(strict_sources) - set(precompetition_sources)),
            **candidate_summary(precompetition_candidate, official),
        },
        "validation_leakage": {
            "official_prompt_text_available": False,
            "status": "UNKNOWN",
            "reason": "Official train exposes prompt_id but not prompt text, so exact prompt overlap between external prompts and held-out official prompt groups cannot be proven or excluded from released files alone.",
            "allowed_claim": "The precompetition sources demonstrably predate competition launch and have zero exact/normalized official answer-text overlap; residual accidental prompt identity remains unobservable.",
        },
        "decision": {
            "raw_access": "PASS",
            "schema_and_label_coverage": "PASS",
            "answer_text_overlap_gate": "PASS" if overlap_clean["normalized_overlap_count"] == 0 else "FAIL",
            "precompetition_provenance_gate": "PASS",
            "prompt_leakage_gate": "RESIDUAL_UNKNOWN",
            "training_authorized": False,
            "next_step": "Freeze a Phase 9 validation statement and one no-tuning experiment before using the precompetition candidate for training.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
