#!/usr/bin/env python3
"""Strict, standalone Name That LLM score-file contract.

Designed for submission preflight and Kaggle hidden-rerun output verification.
Never calls Kaggle, uses GPU, fits a model, or submits a competition entry.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd

LABELS = frozenset({
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "claude-fable-5",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "qwen3.8-max", "glm-5.3",
    "glm-5.3-flash", "kimi-k3",
})

class SubmissionContractError(ValueError):
    pass

def _ids(frame: pd.DataFrame, name: str) -> list[str]:
    if "id" not in frame.columns:
        raise SubmissionContractError(f"{name}: missing id")
    if frame["id"].isna().any():
        raise SubmissionContractError(f"{name}: null id")
    ids = frame["id"].astype(str).tolist()
    if any(not x.strip() or x.lower() in ("nan", "none", "null") for x in ids):
        raise SubmissionContractError(f"{name}: blank id")
    if len(ids) != len(set(ids)):
        raise SubmissionContractError(f"{name}: duplicate id")
    return ids

def check(input_ids: pd.DataFrame, submission: pd.DataFrame) -> dict:
    """Verify every ID, exact labels and roundtrip-safe schema.

    ID order is deliberately required as an additional engineering invariant,
    though Kaggle's contract primarily specifies one row per ID.
    """
    if input_ids.empty:
        raise SubmissionContractError("input test has no rows")
    if submission.columns.tolist() != ["id", "model"]:
        raise SubmissionContractError("submission must have exactly ['id','model'] columns")
    expected = _ids(input_ids, "test")
    actual = _ids(submission, "submission")
    if len(expected) != len(actual) or expected != actual:
        raise SubmissionContractError("submission must match test IDs exactly in original order")
    if submission["model"].isna().any():
        raise SubmissionContractError("null model prediction")
    predicted = submission["model"].astype(str)
    illegal = set(predicted) - LABELS
    if illegal:
        raise SubmissionContractError("unrecognized model label(s): " + repr(sorted(illegal)))
    return {
        "passed": True,
        "rows": len(expected),
        "unique_test_ids": len(set(expected)),
        "unique_predictions": int(predicted.nunique()),
        "legal_labels": len(LABELS),
        "exact_id_match": True,
        "schema": ["id", "model"],
    }

def verify_files(test_csv: Path, submission_parquet: Path) -> dict:
    if not test_csv.is_file():
        raise SubmissionContractError("missing test CSV")
    if not submission_parquet.is_file():
        raise SubmissionContractError("missing submission parquet")
    if submission_parquet.name != "submission.parquet":
        raise SubmissionContractError("output filename must be submission.parquet")
    if submission_parquet.stat().st_size == 0:
        raise SubmissionContractError("zero-byte submission")
    test = pd.read_csv(test_csv, dtype={"id": str}, keep_default_na=False, usecols=["id"])
    try:
        pred = pd.read_parquet(submission_parquet)
    except Exception as exc:
        raise SubmissionContractError("invalid parquet data") from exc
    result = check(test, pred)
    result.update({
        "file_bytes": submission_parquet.stat().st_size,
        "sha256": hashlib.file_digest(submission_parquet.open("rb"), "sha256").hexdigest(),
    })
    return result

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_files(args.test_csv, args.submission)
    except (SubmissionContractError, OSError, ValueError) as exc:
        print("NTL_SCORE_CONTRACT_FAIL=" + json.dumps({"error": str(exc)}), flush=True)
        return 2
    print("NTL_SCORE_CONTRACT_PASS=" + json.dumps(result, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
