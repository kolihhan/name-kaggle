#!/usr/bin/env python3
import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


SETTING_FOLDERS = {
    "high": "claude-opus-4-8-high",
    "medium": "claude-opus-4-8-medium",
    "max": "claude-opus-4-8-max",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(text.split())


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n\n".join(part.strip() for part in parts if part.strip()).strip()
    return ""


def extract_assistant_responses(records, setting: str, source_file: str, expected_model: str):
    rows = []
    stats = {
        "attempts_seen": 0,
        "responses_extracted": 0,
        "missing_assistant_response": 0,
    }

    if not isinstance(records, list):
        raise ValueError(f"expected top-level list in {source_file}")

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        for attempt_key, attempt in record.items():
            if not str(attempt_key).startswith("attempt_") or not isinstance(attempt, dict):
                continue
            stats["attempts_seen"] += 1
            metadata = attempt.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"model identity missing in {source_file} record={record_index} attempt={attempt_key}"
                )
            observed_model = metadata.get("model")
            if observed_model != expected_model:
                raise ValueError(
                    "model identity mismatch in "
                    f"{source_file} record={record_index} attempt={attempt_key}: "
                    f"expected={expected_model!r} observed={observed_model!r}"
                )

            assistant_parts = []
            choices = metadata.get("choices", [])
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message")
                    if not isinstance(message, dict) or message.get("role") != "assistant":
                        continue
                    text = _content_to_text(message.get("content"))
                    if text:
                        assistant_parts.append(text)

            text = "\n\n".join(assistant_parts).strip()
            if not text:
                stats["missing_assistant_response"] += 1
                continue

            rows.append(
                {
                    "text": text,
                    "setting": setting,
                    "source_file": source_file,
                    "attempt": str(attempt_key),
                    "model": expected_model,
                }
            )
            stats["responses_extracted"] += 1

    return rows, stats


def _overlap(left, right):
    left_raw = {str(x) for x in left if pd.notna(x)}
    right_raw = {str(x) for x in right if pd.notna(x)}
    left_norm = {normalize_text(x) for x in left_raw if normalize_text(x)}
    right_norm = {normalize_text(x) for x in right_raw if normalize_text(x)}
    return {
        "exact": len(left_raw & right_raw),
        "normalized": len(left_norm & right_norm),
    }


def audit_response_rows(responses: pd.DataFrame, official: pd.DataFrame, admitted: pd.DataFrame):
    required_response = {"text", "setting", "source_file"}
    if not required_response <= set(responses.columns):
        raise ValueError(f"responses missing columns: {sorted(required_response - set(responses.columns))}")
    if "text" not in official.columns:
        raise ValueError("official train must contain text column")
    if "text" not in admitted.columns:
        raise ValueError("admitted external data must contain text column")

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
            "by_setting": {str(k): int(v) for k, v in responses["setting"].value_counts().sort_index().items()},
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
            "response_only_extraction_gate": "PASS",
            "official_overlap_gate": "PASS" if official_overlap["normalized"] == 0 else "FAIL",
            "cross_label_overlap_gate": "PASS" if cross_label_overlap["normalized"] == 0 else "FAIL",
            "training_authorized": bool(training_authorized),
        },
    }


def load_snapshot(snapshot_root: Path, expected_model: str):
    all_rows = []
    per_setting = {}
    file_counts = {}

    for setting, folder_name in SETTING_FOLDERS.items():
        folder = snapshot_root / folder_name
        files = sorted(folder.glob("*.json"))
        file_counts[setting] = len(files)
        setting_stats = Counter()
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
            rows, stats = extract_assistant_responses(
                records,
                setting=setting,
                source_file=f"{folder_name}/{path.name}",
                expected_model=expected_model,
            )
            all_rows.extend(rows)
            setting_stats.update(stats)
        per_setting[setting] = dict(setting_stats)

    return pd.DataFrame(all_rows), per_setting, file_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--official-train", type=Path, required=True)
    parser.add_argument("--admitted-external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--responses-output", type=Path)
    parser.add_argument("--expected-model", default="claude-opus-4-8")
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    responses, extraction_stats, file_counts = load_snapshot(args.snapshot_root, args.expected_model)
    official = pd.read_csv(args.official_train)
    admitted = pd.read_parquet(args.admitted_external)
    report = audit_response_rows(responses, official, admitted)
    report.update(
        {
            "dataset": "arcprize/arc_agi_v2_public_eval",
            "source_revision": args.source_revision,
            "source_last_modified": "2026-06-04T16:45:03+00:00",
            "license": "MIT",
            "expected_model": args.expected_model,
            "source_files": file_counts,
            "extraction": extraction_stats,
            "scope": "audit_only_no_permanent_fold_training",
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.responses_output:
        args.responses_output.parent.mkdir(parents=True, exist_ok=True)
        responses.to_parquet(args.responses_output, index=False)
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
