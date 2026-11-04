"""Frozen, no-training summary of independent public-source LoRA24 A1 predictions.

Input rows are public-source holdout predictions only; never official competition rows.
The summaries are descriptive: Stage-1 A1 is not the Stage-2 final 0.520 model.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

SOURCES = ("abdullah", "semenoff", "zakir")
EXPECTED_ELIGIBLE_ROWS = 3324
EXPECTED_ELIGIBLE_LABELS = 13
REFERENCE_OFFICIAL_STAGE2_WITHIN_FAMILY_WRONG = (701, 1223)


def family(label: str) -> str:
    prefixes = (
        ("claude-", "anthropic"),
        ("gemini-", "google"),
        ("gpt-", "openai"),
        ("glm-", "zai"),
        ("kimi-", "moonshot"),
        ("qwen", "qwen"),
    )
    for prefix, name in prefixes:
        if str(label).startswith(prefix):
            return name
    raise ValueError(f"unknown model label family: {label!r}")


def summarize_predictions(
    frame: pd.DataFrame,
    *,
    eligible_labels: Sequence[str],
    require_frozen_counts: bool = True,
) -> dict[str, object]:
    """Summarize plain-head error location, *without* fitting or tuning anything."""
    needed = {"row_index", "source", "prompt_id", "model", "prediction"}
    if missing := needed - set(frame.columns):
        raise ValueError(f"missing fields: {sorted(missing)}")
    if frame[list(needed)].isna().any().any():
        raise ValueError("null diagnostic fields")
    if frame["row_index"].duplicated().any():
        raise ValueError("duplicate source row index")
    source_values = frame["source"].astype(str)
    if set(source_values) != set(SOURCES):
        raise ValueError(f"unexpected source set: {sorted(set(source_values))}")
    labels = tuple(sorted(set(str(label) for label in eligible_labels)))
    if not labels:
        raise ValueError("eligible label set is empty")
    if not set(frame["model"].astype(str)).issubset(set(labels)):
        raise ValueError("non-eligible true labels present")
    if require_frozen_counts and (
        len(labels) != EXPECTED_ELIGIBLE_LABELS or len(frame) != EXPECTED_ELIGIBLE_ROWS
    ):
        raise ValueError("frozen eligible label / row count mismatch")
    if not len(frame):
        raise ValueError("no predictions")

    truths = frame["model"].astype(str).to_numpy()
    preds = frame["prediction"].astype(str).to_numpy()
    fam_true = np.asarray([family(x) for x in truths], dtype=object)
    fam_pred = np.asarray([family(x) for x in preds], dtype=object)
    good = truths == preds
    same_family_wrong = (~good) & (fam_true == fam_pred)
    different_family_wrong = (~good) & (fam_true != fam_pred)

    def count(mask: np.ndarray) -> dict[str, object]:
        n = int(mask.sum())
        wrong = int((~good & mask).sum())
        within = int((same_family_wrong & mask).sum())
        across = int((different_family_wrong & mask).sum())
        assert wrong == within + across
        return {
            "rows": n,
            "correct": n - wrong,
            "wrong": wrong,
            "within_family_wrong": within,
            "across_family_wrong": across,
            "within_family_fraction_of_wrong": None if wrong == 0 else within / wrong,
            "family_accuracy": float((fam_true[mask] == fam_pred[mask]).mean()),
            "plain_accuracy": float(good[mask].mean()),
        }

    by_source = {
        src: count(source_values.to_numpy() == src)
        for src in SOURCES
    }
    by_family = {
        fam: count(fam_true == fam)
        for fam in sorted(set(fam_true))
    }
    pooled = count(np.ones(len(frame), dtype=bool))
    counts = {str(x): int((truths == str(x)).sum()) for x in labels}
    recall = {
        name: float((preds[truths == name] == name).mean())
        for name in labels
    }
    pooled["balanced_accuracy"] = float(np.mean(list(recall.values())))
    assert sum(s["rows"] for s in by_source.values()) == len(frame)
    assert sum(s["wrong"] for s in by_source.values()) == pooled["wrong"]
    return {
        "scope": "stage1_A1_external_source_holdout_only",
        "no_training": True,
        "no_kaggle_competition_submission": True,
        "not_the_final_0520_stage2_model": True,
        "sources": list(SOURCES),
        "eligible_labels": list(labels),
        "pooled": pooled,
        "by_source": by_source,
        "by_true_family": by_family,
        "per_label_recall": recall,
        "per_label_count": counts,
        "reference_official_stage2_oof": {
            "within_family_wrong": REFERENCE_OFFICIAL_STAGE2_WITHIN_FAMILY_WRONG[0],
            "all_wrong": REFERENCE_OFFICIAL_STAGE2_WITHIN_FAMILY_WRONG[1],
            "directly_comparable": False,
        },
        "decision": "descriptive_only_no_promotion",
    }
