#!/usr/bin/env python3
"""Public Kaggle candidate metadata reconnaissance. No row-level data output."""
from __future__ import annotations
import argparse
import json
import os
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
    # Never print stderr: Kaggle error messages can occasionally echo credentials.
    output = result.stdout.strip()
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
        "stdout_excerpt": output[:6500] if result.returncode == 0 else "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = {"scope": "public_dataset_metadata_only_no_competition_data_no_training",
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
