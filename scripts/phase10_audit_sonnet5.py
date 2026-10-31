#!/usr/bin/env python3
import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(text.split())


def _overlap(official_texts, external_texts):
    official_raw = {str(x) for x in official_texts if pd.notna(x)}
    external_raw = {str(x) for x in external_texts if pd.notna(x)}
    official_norm = {normalize_text(x) for x in official_raw}
    external_norm = {normalize_text(x) for x in external_raw}
    return {
        "exact_overlap_count": len(official_raw & external_raw),
        "normalized_overlap_count": len(official_norm & external_norm),
    }


def audit_frames(external: pd.DataFrame, official: pd.DataFrame, expected_model: str):
    required = {"question", "answer"}
    missing = sorted(required - set(external.columns))
    if missing:
        raise ValueError(f"external dataset must contain question and answer columns; missing={missing}")
    if "text" not in official.columns:
        raise ValueError("official train must contain text column")

    answers = external["answer"]
    answer_strings = answers.fillna("").astype(str)
    nonempty_mask = answer_strings.str.strip().ne("")
    clean_answers = answer_strings[nonempty_mask]
    normalized = clean_answers.map(normalize_text)
    duplicate_groups = int((normalized.value_counts() > 1).sum())

    if "model" not in external.columns:
        model_identity = {
            "gate": "UNKNOWN",
            "reason": "raw rows have no model column; repository/card naming alone is not row-level identity evidence",
            "expected_model": expected_model,
        }
    else:
        models = external["model"].fillna("").astype(str)
        wrong = int((models != expected_model).sum())
        model_identity = {
            "gate": "PASS" if wrong == 0 else "FAIL",
            "expected_model": expected_model,
            "mismatched_rows": wrong,
            "observed_models": sorted(models.unique().tolist()),
        }

    overlap = _overlap(official["text"], clean_answers)
    overlap_gate = "PASS" if overlap["normalized_overlap_count"] == 0 else "FAIL"

    report = {
        "external": {
            "rows": int(len(external)),
            "columns": external.columns.tolist(),
            "null_or_empty_answers": int((~nonempty_mask).sum()),
            "unique_normalized_answers": int(normalized.nunique()),
            "normalized_duplicate_groups": duplicate_groups,
            "classifier_text_column": "answer",
            "response_only_extraction_gate": "PASS",
        },
        "official_answer_overlap": overlap,
        "model_identity": model_identity,
        "prompt_leakage": {
            "gate": "UNKNOWN",
            "reason": "official train exposes prompt_id but not prompt text, so external question identity against held-out official prompts is not observable from released files",
        },
        "decision": {
            "answer_text_overlap_gate": overlap_gate,
            "model_identity_gate": model_identity["gate"],
            "license_compatibility_gate": "AMBIGUOUS",
            "training_authorized": False,
            "reason": "Phase 10 requires row-level model identity and unambiguous usable output/license terms before any permanent-fold experiment",
        },
    }
    return report


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-jsonl", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-model", default="claude-sonnet-5")
    parser.add_argument("--source-revision", default=None)
    args = parser.parse_args()

    external = pd.read_json(args.external_jsonl, lines=True)
    official = pd.read_csv(args.official_train)
    report = audit_frames(external, official, args.expected_model)
    report["dataset"] = "Essentialia/claude-sonnet-5-2500x"
    report["source_revision"] = args.source_revision
    report["external_sha256"] = sha256(args.external_jsonl)
    report["official_train_sha256"] = sha256(args.official_train)
    report["scope"] = "audit_only_no_training"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
