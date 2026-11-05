#!/usr/bin/env python3
"""Public Kaggle candidate metadata reconnaissance. No row-level data output."""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess

CANDIDATES = [
    "crtproduction/name-that-llm-open-pool",
    "olegzholobov/name-that-llm-response-corpus",
    "nishantdahal/ntl-handwriting-blend",
    "maniceet/name-that-llm-openrouter-openai-generations",
]


def run_cmd(args):
    env = os.environ.copy()
    result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=90)
    # Mask credentials and redact URLs before any shared/public logging.
    output = result.stdout.strip()
    stderr = result.stderr.strip()
    for secret in (env.get("KAGGLE_API_TOKEN"), env.get("KAGGLE_KEY")):
        if secret:
            stderr = stderr.replace(secret, "[REDACTED_SECRET]")
    stderr = re.sub(r"https?://[^\\s]+", "[REDACTED_URL]", stderr)
    stderr = re.sub(r"(?i)(token|key|authorization)([=: ]+)([^\\s,]+)", r"\\1\\2[REDACTED]", stderr)
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
        "stdout_excerpt": output[:6500] if result.returncode == 0 else "",
        "error_excerpt": stderr[:700] if result.returncode else "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = {"scope": "public_dataset_metadata_only_no_competition_data_no_training",
              "cli_help": run_cmd(["kaggle", "datasets", "files", "--help"]),
              "candidates": {}}
    for ref in CANDIDATES:
        result["candidates"][ref] = {
            "files": run_cmd(["kaggle", "datasets", "files", ref]),
        }
        print(ref, json.dumps(result["candidates"][ref]), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
