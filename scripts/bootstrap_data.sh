#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data"
ZIP_ARG="${1:-}"
EXPECTED_TRAIN_SHA256="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"

mkdir -p "$DATA_DIR"

if [[ -n "$ZIP_ARG" ]]; then
  cp "$ZIP_ARG" "$DATA_DIR/name-that-llm.zip"
elif [[ -f "$ROOT/name-that-llm.zip" ]]; then
  cp "$ROOT/name-that-llm.zip" "$DATA_DIR/name-that-llm.zip"
elif command -v kaggle >/dev/null 2>&1; then
  kaggle competitions download -c name-that-llm -p "$DATA_DIR"
else
  echo "No local zip supplied and Kaggle CLI is unavailable." >&2
  echo "Usage: scripts/bootstrap_data.sh /path/to/name-that-llm.zip" >&2
  exit 2
fi

unzip -o "$DATA_DIR/name-that-llm.zip" -d "$DATA_DIR" >/dev/null

actual="$(sha256sum "$DATA_DIR/train.csv" | awk '{print $1}')"
if [[ "$actual" != "$EXPECTED_TRAIN_SHA256" ]]; then
  echo "train.csv SHA-256 mismatch" >&2
  echo "expected: $EXPECTED_TRAIN_SHA256" >&2
  echo "actual:   $actual" >&2
  exit 3
fi

echo "Dataset ready in $DATA_DIR"
echo "train.csv SHA-256 verified: $actual"
