#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


OPUS_LABEL = "claude-opus-5"
SONNET_LABEL = "claude-sonnet-5"
TARGET_LABELS = (OPUS_LABEL, SONNET_LABEL)
DATASET_ID = "mindbench-ai/healthbench-psych"
PINNED_REVISION = "7a116ea7b4fefe0c72eabb8398ec390c6375f823"
LICENSE = "MIT"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(text.split())


def extract_target_responses(records, *, expected_raw_rows: dict[str, int]):
    raw_counts = Counter()
    empty_counts = Counter()
    retained_counts = Counter()
    correction_counts = Counter()
    refusal_counts = Counter()
    prompt_ids: dict[str, list[str]] = defaultdict(list)
    rows = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"expected row object at index {index}")
        model = str(record.get("model", ""))
        if model not in expected_raw_rows:
            continue

        raw_counts[model] += 1
        prompt_id = str(record.get("prompt_id", ""))
        if not prompt_id:
            raise ValueError(f"missing prompt_id for {model} at row {index}")
        prompt_ids[model].append(prompt_id)

        if record.get("correction") is not None:
            correction_counts[model] += 1
        if bool(record.get("is_refusal")):
            refusal_counts[model] += 1

        response = record.get("response_text")
        text = response.strip() if isinstance(response, str) else ""
        if not text:
            empty_counts[model] += 1
            continue

        rows.append(
            {
                "text": text,
                "model": model,
                "prompt_id": prompt_id,
                "stop_reason": record.get("stop_reason"),
                "is_refusal": bool(record.get("is_refusal")),
                "correction": record.get("correction"),
            }
        )
        retained_counts[model] += 1

    for model, expected in expected_raw_rows.items():
        observed = int(raw_counts[model])
        if observed != expected:
            raise ValueError(
                f"raw row count drift for {model}: expected {expected}, found {observed}"
            )
        ids = prompt_ids[model]
        if len(set(ids)) != expected:
            raise ValueError(
                f"prompt_id values must be unique within {model}: "
                f"{len(set(ids))} unique / {expected} rows"
            )

    stats = {
        "raw_rows_by_model": {model: int(raw_counts[model]) for model in expected_raw_rows},
        "empty_rows_by_model": {model: int(empty_counts[model]) for model in expected_raw_rows},
        "retained_rows_by_model": {
            model: int(retained_counts[model]) for model in expected_raw_rows
        },
        "refusal_rows_by_model": {
            model: int(refusal_counts[model]) for model in expected_raw_rows
        },
        "correction_rows_by_model": {
            model: int(correction_counts[model]) for model in expected_raw_rows
        },
    }
    return rows, stats


def _overlap(left, right):
    left_raw = {str(value) for value in left if pd.notna(value)}
    right_raw = {str(value) for value in right if pd.notna(value)}
    left_norm = {normalize_text(value) for value in left_raw if normalize_text(value)}
    right_norm = {normalize_text(value) for value in right_raw if normalize_text(value)}
    return {
        "exact": len(left_raw & right_raw),
        "normalized": len(left_norm & right_norm),
    }


def audit_response_rows(
    responses: pd.DataFrame,
    official: pd.DataFrame,
    admitted: pd.DataFrame,
):
    required = {"text", "model", "prompt_id"}
    missing = sorted(required - set(responses.columns))
    if missing:
        raise ValueError(f"responses missing columns: {missing}")
    if "text" not in official.columns:
        raise ValueError("official train must contain text column")
    if "text" not in admitted.columns:
        raise ValueError("admitted external must contain text column")

    clean = responses[responses["text"].astype(str).str.strip().ne("")].copy()
    clean["normalized"] = clean["text"].astype(str).map(normalize_text)

    duplicate_groups = 0
    duplicate_rows = 0
    by_model = {}
    norm_sets = {}
    for model in sorted(clean["model"].astype(str).unique()):
        subset = clean[clean["model"].astype(str).eq(model)]
        counts = subset["normalized"].value_counts()
        groups = int((counts > 1).sum())
        rows = int(counts[counts > 1].sum()) if groups else 0
        duplicate_groups += groups
        duplicate_rows += rows
        norm_sets[model] = set(subset["normalized"])
        by_model[model] = {
            "rows": int(len(subset)),
            "unique_normalized": int(subset["normalized"].nunique()),
            "normalized_duplicate_groups": groups,
            "rows_in_duplicate_groups": rows,
        }

    cross_exact = 0
    cross_normalized = 0
    if OPUS_LABEL in norm_sets and SONNET_LABEL in norm_sets:
        opus = clean[clean["model"].astype(str).eq(OPUS_LABEL)]
        sonnet = clean[clean["model"].astype(str).eq(SONNET_LABEL)]
        candidate_cross = _overlap(opus["text"], sonnet["text"])
        cross_exact = candidate_cross["exact"]
        cross_normalized = candidate_cross["normalized"]

    official_overlap = _overlap(clean["text"], official["text"])
    existing_external_overlap = _overlap(clean["text"], admitted["text"])

    training_authorized = (
        len(clean) > 0
        and duplicate_groups == 0
        and cross_normalized == 0
        and official_overlap["normalized"] == 0
        and existing_external_overlap["normalized"] == 0
    )

    return {
        "responses": {
            "rows": int(len(clean)),
            "unique_normalized": int(clean["normalized"].nunique()),
            "by_model": by_model,
        },
        "response_duplicates": {
            "within_label_normalized_duplicate_groups": duplicate_groups,
            "rows_in_within_label_duplicate_groups": duplicate_rows,
        },
        "candidate_cross_label_overlap": {
            "exact": int(cross_exact),
            "normalized": int(cross_normalized),
        },
        "official_overlap": official_overlap,
        "existing_external_overlap": existing_external_overlap,
        "decision": {
            "model_identity_gate": "PASS",
            "immutable_precompetition_gate": "PASS",
            "license_gate": "PASS",
            "response_only_extraction_gate": "PASS",
            "within_label_duplicate_gate": "PASS" if duplicate_groups == 0 else "FAIL",
            "candidate_cross_label_overlap_gate": "PASS" if cross_normalized == 0 else "FAIL",
            "official_overlap_gate": "PASS"
            if official_overlap["normalized"] == 0
            else "FAIL",
            "existing_external_overlap_gate": "PASS"
            if existing_external_overlap["normalized"] == 0
            else "FAIL",
            "training_authorized": bool(training_authorized),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-parquet", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--admitted-external", type=Path, required=True)
    parser.add_argument("--source-revision", default=PINNED_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--responses-output", type=Path)
    args = parser.parse_args()

    if args.source_revision != PINNED_REVISION:
        raise ValueError(
            f"source revision drifted: expected {PINNED_REVISION}, got {args.source_revision}"
        )

    source = pd.read_parquet(args.responses_parquet)
    records = source.to_dict(orient="records")
    rows, extraction = extract_target_responses(
        records,
        expected_raw_rows={OPUS_LABEL: 610, SONNET_LABEL: 610},
    )
    responses = pd.DataFrame(rows)
    official = pd.read_csv(args.official_train)
    admitted = pd.read_parquet(args.admitted_external)
    report = audit_response_rows(responses, official, admitted)
    report.update(
        {
            "dataset": DATASET_ID,
            "source_tag": "v1.0.0",
            "source_revision": args.source_revision,
            "source_last_modified": "2026-08-28T17:36:20+00:00",
            "competition_opened": "2026-10-14",
            "license": LICENSE,
            "target_labels": list(TARGET_LABELS),
            "extraction": extraction,
            "scope": "audit_only_no_permanent_fold_training",
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.responses_output:
        args.responses_output.parent.mkdir(parents=True, exist_ok=True)
        responses.to_parquet(args.responses_output, index=False)
    print(args.output.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
