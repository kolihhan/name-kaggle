from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

REQUIRED_COLUMNS = {"source", "prompt_id", "prompt_key", "model", "text"}


def build_firewall_lock(
    pool: pd.DataFrame,
    *,
    consumed_sources: Iterable[str],
    min_label_rows: int = 20,
) -> dict[str, object]:
    missing = sorted(REQUIRED_COLUMNS - set(pool.columns))
    if missing:
        raise RuntimeError(f"pool missing required columns: {missing}")

    consumed = sorted({str(value) for value in consumed_sources})
    present = set(pool["source"].astype(str).unique())
    missing_consumed = sorted(set(consumed) - present)
    if missing_consumed:
        raise RuntimeError(f"consumed sources missing from pool: {missing_consumed}")

    firewall = sorted(present - set(consumed))
    if not firewall:
        raise RuntimeError("no unconsumed sources remain for firewall")

    source = pool["source"].astype(str)
    holdout = pool.loc[source.isin(firewall)].copy()
    train_before = pool.loc[source.isin(consumed)].copy()

    holdout_prompt_ids = set(holdout["prompt_id"].astype(str))
    holdout_prompt_keys = set(holdout["prompt_key"].astype(str))
    train_overlap = (
        train_before["prompt_id"].astype(str).isin(holdout_prompt_ids)
        | train_before["prompt_key"].astype(str).isin(holdout_prompt_keys)
    )
    train = train_before.loc[~train_overlap].copy()

    train_labels = set(train["model"].astype(str))
    holdout_counts = holdout["model"].astype(str).value_counts()
    eligible_labels = sorted(
        str(label)
        for label, count in holdout_counts.items()
        if int(count) >= int(min_label_rows) and str(label) in train_labels
    )
    eligible = holdout.loc[holdout["model"].astype(str).isin(eligible_labels)].copy()

    prompt_id_overlap = len(
        set(train["prompt_id"].astype(str)) & set(holdout["prompt_id"].astype(str))
    )
    prompt_key_overlap = len(
        set(train["prompt_key"].astype(str)) & set(holdout["prompt_key"].astype(str))
    )

    source_rows = {
        str(label): int(count)
        for label, count in pool["source"].astype(str).value_counts().sort_index().items()
    }
    source_prompts = {
        str(label): int(count)
        for label, count in pool.groupby(pool["source"].astype(str))["prompt_id"].nunique().sort_index().items()
    }
    source_labels = {
        str(label): int(count)
        for label, count in pool.groupby(pool["source"].astype(str))["model"].nunique().sort_index().items()
    }

    return {
        "consumed_sources": consumed,
        "firewall_sources": firewall,
        "all_sources": sorted(present),
        "source_rows": source_rows,
        "source_prompt_counts": source_prompts,
        "source_label_counts": source_labels,
        "heldout_rows": int(len(holdout)),
        "train_rows_before_leakage_removal": int(len(train_before)),
        "train_rows_after_leakage_removal": int(len(train)),
        "removed_training_rows_for_prompt_overlap": int(train_overlap.sum()),
        "eligible_labels": eligible_labels,
        "eligible_heldout_rows": int(len(eligible)),
        "training_labels_after_leakage_removal": sorted(train_labels),
        "prompt_id_overlap_after_removal": int(prompt_id_overlap),
        "prompt_key_overlap_after_removal": int(prompt_key_overlap),
        "model_scores_used_for_selection": False,
    }
