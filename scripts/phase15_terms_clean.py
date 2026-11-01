from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPECTED_PHASE14_EXTERNAL_ROWS = 43_781
EXPECTED_TERMS_CLEAN_ROWS = 26_025
EXPECTED_BLOCKED_ROWS = 17_756

BLOCKED_EXTERNAL_LABELS = {
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
}

EXPECTED_RETAINED_LABEL_COUNTS = {
    "claude-fable-5": 1_193,
    "claude-opus-4-6": 1_352,
    "claude-opus-4-8": 1_364,
    "claude-opus-5": 1_240,
    "claude-sonnet-5": 1_350,
    "gpt-5.5": 3_525,
    "gpt-5.6-luna": 4_326,
    "gpt-5.6-sol": 3_879,
    "gpt-5.6-terra": 3_875,
    "gpt-6-astra": 3_044,
    "qwen3.8-max": 877,
}


def build_terms_clean_subset(
    phase14_external: pd.DataFrame,
    *,
    validate_full_contract: bool = True,
) -> pd.DataFrame:
    required = {"prompt_id", "text", "model"}
    missing = sorted(required - set(phase14_external.columns))
    if missing:
        raise RuntimeError(f"Phase 14 external frame missing required columns: {missing}")

    if validate_full_contract and len(phase14_external) != EXPECTED_PHASE14_EXTERNAL_ROWS:
        raise RuntimeError(
            "Phase 14 external row count drifted: "
            f"{len(phase14_external)} != {EXPECTED_PHASE14_EXTERNAL_ROWS}"
        )

    models = phase14_external["model"].astype(str)
    blocked_mask = models.isin(BLOCKED_EXTERNAL_LABELS)
    clean = phase14_external.loc[~blocked_mask].copy().reset_index(drop=True)

    if validate_full_contract:
        blocked_rows = int(blocked_mask.sum())
        if blocked_rows != EXPECTED_BLOCKED_ROWS:
            raise RuntimeError(
                f"blocked external row count drifted: {blocked_rows} != {EXPECTED_BLOCKED_ROWS}"
            )
        if len(clean) != EXPECTED_TERMS_CLEAN_ROWS:
            raise RuntimeError(
                f"terms-clean external row count drifted: {len(clean)} != {EXPECTED_TERMS_CLEAN_ROWS}"
            )
        observed_counts = {
            str(label): int(count)
            for label, count in clean["model"].astype(str).value_counts().sort_index().items()
        }
        if observed_counts != EXPECTED_RETAINED_LABEL_COUNTS:
            raise RuntimeError(
                "terms-clean retained label counts drifted: "
                f"{observed_counts} != {EXPECTED_RETAINED_LABEL_COUNTS}"
            )
        if set(clean["model"].astype(str).unique()) & BLOCKED_EXTERNAL_LABELS:
            raise RuntimeError("terms-clean subset still contains blocked provider labels")
        if clean["text"].isna().any() or clean["model"].isna().any():
            raise RuntimeError("terms-clean subset contains null text/model")
        if (clean["text"].astype(str).str.strip() == "").any():
            raise RuntimeError("terms-clean subset contains empty text")

    return clean


def audit_terms_clean_subset(phase14_external: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    clean = build_terms_clean_subset(phase14_external, validate_full_contract=True)
    report = {
        "phase": 15,
        "scope": "provider_terms_admission_only_no_training_no_permanent_folds",
        "phase14_external_rows": int(len(phase14_external)),
        "blocked_labels": sorted(BLOCKED_EXTERNAL_LABELS),
        "blocked_rows": int(len(phase14_external) - len(clean)),
        "retained_rows": int(len(clean)),
        "retained_prompt_ids": int(clean["prompt_id"].astype(str).nunique()),
        "retained_labels": {
            str(label): int(count)
            for label, count in clean["model"].astype(str).value_counts().sort_index().items()
        },
        "invariants": {
            "no_blocked_labels_remain": not bool(
                set(clean["model"].astype(str).unique()) & BLOCKED_EXTERNAL_LABELS
            ),
            "no_null_text": bool(clean["text"].notna().all()),
            "no_null_model": bool(clean["model"].notna().all()),
        },
        "decision": {
            "training_authorized_after_terms_filter": True,
            "permanent_folds_authorized": False,
        },
    }
    return report, clean


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit the frozen Phase 15 provider-terms-clean subset.")
    parser.add_argument("--phase14-external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset-output", type=Path, required=True)
    args = parser.parse_args()

    phase14_external = pd.read_parquet(args.phase14_external)
    report, clean = audit_terms_clean_subset(phase14_external)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.subset_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    clean.to_parquet(args.subset_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
