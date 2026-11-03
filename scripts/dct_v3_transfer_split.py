from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd

from scripts.run_two_stage_lora24_gate_a import (
    MIN_ELIGIBLE_HELDOUT_ROWS,
    MIN_ELIGIBLE_LABELS,
    select_eligible_labels,
)

TRANSFER_HOLDOUT_SALT = "ntl-dct-temporal-v3-transfer-holdout-v1"
TRANSFER_HOLDOUT_SOURCE_COUNT = 3
EXCLUDED_HOLDOUT_SOURCES = frozenset(
    {
        # v1 Gate A
        "abdullah",
        "semenoff",
        "zakir",
        # rejected prompt-invariant-contrastive-v2 transfer gate
        "jadkutbi",
        "lucasboesen",
        "maniceet",
    }
)


def _build_for_sources(frame: pd.DataFrame, holdout_sources: list[str]) -> dict[str, object]:
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
    eligible_heldout_rows = int(
        heldout["model"].astype(str).isin(eligible_labels).sum()
    )
    prompt_id_overlap = len(
        set(train["prompt_id"].astype(str)) & set(heldout["prompt_id"].astype(str))
    )
    prompt_key_overlap = 0
    if "prompt_key" in frame.columns:
        prompt_key_overlap = len(
            set(train["prompt_key"].astype(str))
            & set(heldout["prompt_key"].astype(str))
        )

    valid = (
        len(eligible_labels) >= MIN_ELIGIBLE_LABELS
        and eligible_heldout_rows >= MIN_ELIGIBLE_HELDOUT_ROWS
        and all(value > 0 for value in source_eligible_rows.values())
        and prompt_id_overlap == 0
        and prompt_key_overlap == 0
    )
    return {
        "holdout_sources": list(holdout_sources),
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


def _combination_digest(sources: tuple[str, ...] | list[str]) -> str:
    ordered = sorted(str(value) for value in sources)
    payload = "|".join([TRANSFER_HOLDOUT_SALT, *ordered])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_ranked_combinations(frame: pd.DataFrame) -> list[tuple[str, tuple[str, ...]]]:
    candidates = sorted(
        set(frame["source"].astype(str)) - set(EXCLUDED_HOLDOUT_SOURCES)
    )
    if len(candidates) < TRANSFER_HOLDOUT_SOURCE_COUNT:
        raise RuntimeError("not enough unspent sources for v3 transfer split")

    valid: list[tuple[str, tuple[str, ...]]] = []
    for combo in itertools.combinations(candidates, TRANSFER_HOLDOUT_SOURCE_COUNT):
        stats = _build_for_sources(frame, list(combo))
        if bool(stats["valid_for_transfer_gate"]):
            valid.append((_combination_digest(combo), combo))
    if not valid:
        raise RuntimeError("no valid unspent three-source combination for v3 transfer gate")
    return sorted(valid, key=lambda item: (item[0], item[1]))


def select_transfer_holdout_sources(frame: pd.DataFrame) -> list[str]:
    """Select one valid unspent source triple without using model predictions."""
    return list(_valid_ranked_combinations(frame)[0][1])


def build_transfer_split(frame: pd.DataFrame) -> dict[str, object]:
    ranked = _valid_ranked_combinations(frame)
    selection_digest, selected = ranked[0]
    result = _build_for_sources(frame, list(selected))
    result.update(
        {
            "salt": TRANSFER_HOLDOUT_SALT,
            "excluded_holdout_sources": sorted(EXCLUDED_HOLDOUT_SOURCES),
            "selection_digest": selection_digest,
            "candidate_valid_combinations": int(len(ranked)),
            "selection_uses_model_scores": False,
        }
    )
    return result
