from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from scripts.run_two_stage_lora24_gate_a import (
    MIN_ELIGIBLE_HELDOUT_ROWS,
    MIN_ELIGIBLE_LABELS,
    select_holdout_sources,
    select_eligible_labels,
)

TRANSFER_HOLDOUT_SALT = "ntl-prompt-invariant-v2-transfer-holdout-v1"
TRANSFER_HOLDOUT_SOURCE_COUNT = 3


def select_transfer_holdout_sources(frame: pd.DataFrame) -> list[str]:
    """Choose a deterministic alternate source holdout without model scores.

    The original Gate-A holdout sources are excluded so this transfer check does
    not recycle the exact source set already used to admit v1.
    """
    original = set(select_holdout_sources(frame))
    candidates = sorted(set(frame["source"].astype(str)) - original)
    ranked = sorted(
        candidates,
        key=lambda source: hashlib.sha256(
            f"{TRANSFER_HOLDOUT_SALT}|{source}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ranked) < TRANSFER_HOLDOUT_SOURCE_COUNT:
        raise RuntimeError("not enough non-v1-holdout sources for v2 transfer split")
    return ranked[:TRANSFER_HOLDOUT_SOURCE_COUNT]


def build_transfer_split(frame: pd.DataFrame) -> dict[str, object]:
    holdout_sources = select_transfer_holdout_sources(frame)
    source_values = frame["source"].astype(str)
    heldout_mask = source_values.isin(holdout_sources)
    heldout_idx = np.flatnonzero(heldout_mask.to_numpy())

    heldout_prompt_ids = set(frame.loc[heldout_mask, "prompt_id"].astype(str))
    train_mask = ~heldout_mask
    train_mask &= ~frame["prompt_id"].astype(str).isin(heldout_prompt_ids)

    if "prompt_key" in frame.columns:
        heldout_prompt_keys = set(frame.loc[heldout_mask, "prompt_key"].astype(str))
        train_mask &= ~frame["prompt_key"].astype(str).isin(heldout_prompt_keys)

    train_idx = np.flatnonzero(train_mask.to_numpy())
    train = frame.iloc[train_idx]
    heldout = frame.iloc[heldout_idx]
    eligible_labels = select_eligible_labels(train, heldout)

    source_eligible_rows = {
        source: int(
            (
                (heldout["source"].astype(str) == source)
                & heldout["model"].astype(str).isin(eligible_labels)
            ).sum()
        )
        for source in holdout_sources
    }
    eligible_heldout_rows = int(heldout["model"].astype(str).isin(eligible_labels).sum())
    prompt_id_overlap = len(
        set(train["prompt_id"].astype(str)) & set(heldout["prompt_id"].astype(str))
    )
    prompt_key_overlap = 0
    if "prompt_key" in frame.columns:
        prompt_key_overlap = len(
            set(train["prompt_key"].astype(str)) & set(heldout["prompt_key"].astype(str))
        )

    valid = (
        len(eligible_labels) >= MIN_ELIGIBLE_LABELS
        and eligible_heldout_rows >= MIN_ELIGIBLE_HELDOUT_ROWS
        and all(value > 0 for value in source_eligible_rows.values())
        and prompt_id_overlap == 0
        and prompt_key_overlap == 0
    )
    return {
        "salt": TRANSFER_HOLDOUT_SALT,
        "holdout_sources": holdout_sources,
        "train_idx": train_idx,
        "heldout_idx": heldout_idx,
        "train_rows": int(len(train_idx)),
        "heldout_rows": int(len(heldout_idx)),
        "eligible_labels": eligible_labels,
        "eligible_label_count": int(len(eligible_labels)),
        "eligible_heldout_rows": eligible_heldout_rows,
        "source_eligible_rows": source_eligible_rows,
        "prompt_id_overlap": int(prompt_id_overlap),
        "prompt_key_overlap": int(prompt_key_overlap),
        "valid_for_transfer_gate": bool(valid),
    }
