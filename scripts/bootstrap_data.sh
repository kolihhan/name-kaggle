#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data"
ZIP_ARG="${1:-}"
EXPECTED_TRAIN_SHA256="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_TEST_SHA256="4545aa3a3459f4c0724f97ad5bd06c4caa3506bdfea5fab905e8203184fc0c57"
MIRROR_BASE="https://raw.githubusercontent.com/RohanExploit/kaggle-name-that-llm-top20/main/data"

mkdir -p "$DATA_DIR"

if [[ -n "$ZIP_ARG" ]]; then
  cp "$ZIP_ARG" "$DATA_DIR/name-that-llm.zip"
  unzip -o "$DATA_DIR/name-that-llm.zip" -d "$DATA_DIR" >/dev/null
elif [[ -f "$ROOT/name-that-llm.zip" ]]; then
  cp "$ROOT/name-that-llm.zip" "$DATA_DIR/name-that-llm.zip"
  unzip -o "$DATA_DIR/name-that-llm.zip" -d "$DATA_DIR" >/dev/null
elif command -v kaggle >/dev/null 2>&1; then
  kaggle competitions download -c name-that-llm -p "$DATA_DIR"
  unzip -o "$DATA_DIR/name-that-llm.zip" -d "$DATA_DIR" >/dev/null
elif command -v curl >/dev/null 2>&1; then
  echo "Kaggle CLI unavailable; using byte-verified public transport mirror."
  curl -fL "$MIRROR_BASE/train.csv" -o "$DATA_DIR/train.csv"
  curl -fL "$MIRROR_BASE/test.csv" -o "$DATA_DIR/test.csv"
else
  echo "No local zip, Kaggle CLI, or curl available." >&2
  echo "Usage: scripts/bootstrap_data.sh /path/to/name-that-llm.zip" >&2
  exit 2
fi

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch: $path" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 3
  fi
  echo "verified $(basename "$path"): $actual"
}

verify_sha256 "$DATA_DIR/train.csv" "$EXPECTED_TRAIN_SHA256"
verify_sha256 "$DATA_DIR/test.csv" "$EXPECTED_TEST_SHA256"

echo "Dataset ready in $DATA_DIR"
if [[ ! -f "$DATA_DIR/sample_submission.parquet" ]]; then
  echo "note: sample_submission.parquet is available only from the official Kaggle zip; it is not required for training/CV."
fi
