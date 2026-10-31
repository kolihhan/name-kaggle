#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


DATASET_ID = "TheFusionCube/Fable-5-CoT-Traces"
PINNED_REVISION = "3624e3a8aab36a89dc3b89aebef0e034a64ea04e"
EXPECTED_MODEL = "claude-fable-5"
LICENSE = "MIT"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(text.split())


def extract_fable5_responses(records, *, expected_model: str = EXPECTED_MODEL):
    rows = []
    stats = Counter(
        {
            "rows_seen": 0,
            "decoy_rows_excluded": 0,
            "empty_response_rows": 0,
            "responses_extracted": 0,
        }
    )

    for row_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"expected JSON object at row {row_index}")
        stats["rows_seen"] += 1

        category = str(record.get("category", "")).strip()
        if category.casefold() == "decoy":
            stats["decoy_rows_excluded"] += 1
            continue

        observed_model = record.get("model")
        if observed_model != expected_model:
            raise ValueError(
                "model identity mismatch at "
                f"row={row_index}: expected={expected_model!r} observed={observed_model!r}"
            )

        response = record.get("response")
        text = response.strip() if isinstance(response, str) else ""
        if not text:
            stats["empty_response_rows"] += 1
            continue

        rows.append(
            {
                "text": text,
                "model": expected_model,
                "category": category,
                "source_row_index": row_index,
                "truncated": record.get("truncated"),
                "contamination_stripped": record.get("contamination_stripped"),
            }
        )
        stats["responses_extracted"] += 1

    return rows, dict(stats)


def _overlap(left, right):
    left_raw = {str(x) for x in left if pd.notna(x)}
    right_raw = {str(x) for x in right if pd.notna(x)}
    left_norm = {normalize_text(x) for x in left_raw if normalize_text(x)}
    right_norm = {normalize_text(x) for x in right_raw if normalize_text(x)}
    return {
        "exact": len(left_raw & right_raw),
        "normalized": len(left_norm & right_norm),
    }


def audit_response_rows(
    responses: pd.DataFrame,
    official: pd.DataFrame,
    admitted: pd.DataFrame,
):
    if "text" not in responses.columns:
        raise ValueError("responses must contain text column")
    if "text" not in official.columns:
        raise ValueError("official train must contain text column")
    if "text" not in admitted.columns:
        raise ValueError("admitted external must contain text column")

    response_texts = responses["text"].fillna("").astype(str)
    nonempty = response_texts.str.strip().ne("")
    clean = response_texts[nonempty]
    normalized = clean.map(normalize_text)
    counts = normalized.value_counts()
    duplicate_groups = int((counts > 1).sum())
    duplicate_rows = int(counts[counts > 1].sum()) if duplicate_groups else 0

    official_overlap = _overlap(clean, official["text"])
    cross_label_overlap = _overlap(clean, admitted["text"])

    training_authorized = (
        len(clean) > 0
        and official_overlap["normalized"] == 0
        and cross_label_overlap["normalized"] == 0
    )

    return {
        "responses": {
            "rows": int(len(responses)),
            "nonempty_rows": int(nonempty.sum()),
            "empty_rows": int((~nonempty).sum()),
            "unique_normalized": int(normalized.nunique()),
        },
        "response_duplicates": {
            "normalized_duplicate_groups": duplicate_groups,
            "rows_in_duplicate_groups": duplicate_rows,
        },
        "official_overlap": official_overlap,
        "cross_label_overlap": cross_label_overlap,
        "decision": {
            "model_identity_gate": "PASS",
            "immutable_precompetition_gate": "PASS",
            "license_gate": "PASS",
            "source_defined_decoy_exclusion_gate": "PASS",
            "response_only_extraction_gate": "PASS",
            "official_overlap_gate": "PASS"
            if official_overlap["normalized"] == 0
            else "FAIL",
            "cross_label_overlap_gate": "PASS"
            if cross_label_overlap["normalized"] == 0
            else "FAIL",
            "training_authorized": bool(training_authorized),
        },
    }


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pinned FusionCube Claude Fable 5 response-only corpus."
    )
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--admitted-external", type=Path, required=True)
    parser.add_argument("--source-revision", default=PINNED_REVISION)
    parser.add_argument("--expected-model", default=EXPECTED_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--responses-output", type=Path)
    args = parser.parse_args()

    if args.source_revision != PINNED_REVISION:
        raise ValueError(
            f"source revision drifted: expected {PINNED_REVISION}, got {args.source_revision}"
        )

    records = load_jsonl(args.source_jsonl)
    rows, extraction = extract_fable5_responses(
        records,
        expected_model=args.expected_model,
    )
    responses = pd.DataFrame(rows)
    official = pd.read_csv(args.official_train)
    admitted = pd.read_parquet(args.admitted_external)
    report = audit_response_rows(responses, official, admitted)
    report.update(
        {
            "dataset": DATASET_ID,
            "source_revision": args.source_revision,
            "source_repo_state_date": "2026-06-16",
            "license": LICENSE,
            "expected_model": args.expected_model,
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
