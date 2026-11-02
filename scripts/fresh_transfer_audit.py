from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

OFFICIAL_LABELS = {
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
}
GROUP_KEY_PRIORITY = ("prompt_id", "prompt_key", "prompt")
MIN_ROWS = 300
MIN_LABELS = 8
MIN_GROUPS = 30
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def raw_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def load_records(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if suffix in {".jsonl", ".ndjson"}:
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("rows", "data", "records"):
                if isinstance(obj.get(key), list):
                    return obj[key]
        raise RuntimeError(f"unsupported JSON record shape: {path}")
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
            return pd.read_parquet(path).to_dict(orient="records")
        except Exception as exc:
            raise RuntimeError(f"cannot read parquet {path}: {exc}") from exc
    raise RuntimeError(f"unsupported file type: {path}")


def choose_group_key(records: list[dict[str, object]]) -> str:
    for key in GROUP_KEY_PRIORITY:
        if records and all(str(row.get(key, "")).strip() for row in records):
            return key
    raise RuntimeError(
        "fresh transfer corpus lacks a complete natural prompt grouping; "
        f"required one of {GROUP_KEY_PRIORITY}"
    )


def collect_seen_hashes(paths: Iterable[Path]) -> tuple[set[str], set[str]]:
    raw: set[str] = set()
    normalized: set[str] = set()
    for path in paths:
        for row in load_records(path):
            text = str(row.get("text", ""))
            if not text.strip():
                continue
            raw.add(raw_hash(text))
            normalized.add(text_hash(text))
    return raw, normalized


def audit_fresh_corpus(candidate_path: Path, seen_paths: list[Path]) -> tuple[list[dict], dict]:
    rows = load_records(candidate_path)
    if not rows:
        raise RuntimeError("candidate corpus is empty")
    missing = sorted({"text", "model"} - set(rows[0]))
    if missing:
        raise RuntimeError(f"candidate missing required columns: {missing}")

    group_key = choose_group_key(rows)
    seen_raw, seen_norm = collect_seen_hashes(seen_paths)
    kept = []
    dropped = Counter()
    seen_candidate_norm: dict[str, str] = {}
    conflicts = []

    for index, row in enumerate(rows):
        text = str(row.get("text", ""))
        model = str(row.get("model", "")).strip()
        group = str(row.get(group_key, "")).strip()
        if not text.strip() or not group:
            dropped["empty_text_or_group"] += 1
            continue
        if model not in OFFICIAL_LABELS:
            dropped["non_competition_label"] += 1
            continue
        raw = raw_hash(text)
        normalized = text_hash(text)
        if raw in seen_raw:
            dropped["seen_exact_text"] += 1
            continue
        if normalized in seen_norm:
            dropped["seen_normalized_text"] += 1
            continue
        previous = seen_candidate_norm.get(normalized)
        if previous is not None:
            if previous != model:
                conflicts.append({"row": index, "model": model, "previous_model": previous, "hash": normalized})
            dropped["candidate_internal_duplicate"] += 1
            continue
        seen_candidate_norm[normalized] = model
        out = dict(row)
        out["_fresh_norm_sha256"] = normalized
        kept.append(out)

    if conflicts:
        raise RuntimeError(f"normalized-text label conflicts inside candidate: {conflicts[:5]}")

    labels = Counter(str(row["model"]) for row in kept)
    groups = Counter(str(row[group_key]) for row in kept)
    valid = len(kept) >= MIN_ROWS and len(labels) >= MIN_LABELS and len(groups) >= MIN_GROUPS
    reasons = []
    if len(kept) < MIN_ROWS:
        reasons.append(f"rows<{MIN_ROWS}")
    if len(labels) < MIN_LABELS:
        reasons.append(f"labels<{MIN_LABELS}")
    if len(groups) < MIN_GROUPS:
        reasons.append(f"groups<{MIN_GROUPS}")

    manifest = {
        "candidate_file": str(candidate_path),
        "candidate_file_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "group_key": group_key,
        "input_rows": len(rows),
        "fresh_rows": len(kept),
        "label_count": len(labels),
        "group_count": len(groups),
        "label_counts": dict(sorted(labels.items())),
        "dropped": dict(sorted(dropped.items())),
        "seen_corpora": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in seen_paths
        ],
        "normalization": "NFKC + casefold + whitespace-collapse; punctuation preserved",
        "valid_for_transfer_gate": valid,
        "invalid_reasons": reasons,
        "minimums": {"rows": MIN_ROWS, "labels": MIN_LABELS, "prompt_groups": MIN_GROUPS},
    }
    return kept, manifest


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--seen", type=Path, action="append", default=[])
    parser.add_argument("--out-clean", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()
    kept, manifest = audit_fresh_corpus(args.candidate, args.seen)
    write_csv(kept, args.out_clean)
    args.out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not manifest["valid_for_transfer_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
