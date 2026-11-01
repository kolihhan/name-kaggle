from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd


SOURCE_HOLDOUT_SALT = "ntl-two-stage-lora24-v1-source-holdout"
MIN_ELIGIBLE_LABELS = 8
MIN_ELIGIBLE_HELDOUT_ROWS = 300
MIN_HELDOUT_ROWS_PER_LABEL = 20


def select_holdout_sources(frame: pd.DataFrame) -> list[str]:
    """Select the frozen three-source holdout by full SHA256 digest order."""
    sources = sorted(frame["source"].astype(str).unique().tolist())
    ranked = sorted(
        sources,
        key=lambda source: hashlib.sha256(
            f"{SOURCE_HOLDOUT_SALT}|{source}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:3]


def build_stage1_split(frame: pd.DataFrame) -> dict[str, object]:
    """Build the frozen source holdout and remove prompt-bank leakage from train."""
    holdout_sources = select_holdout_sources(frame)
    source_values = frame["source"].astype(str)
    heldout_mask = source_values.isin(holdout_sources)
    heldout_idx = np.flatnonzero(heldout_mask.to_numpy())

    heldout_prompt_ids = set(frame.loc[heldout_mask, "prompt_id"].astype(str))
    train_mask = ~heldout_mask
    train_mask &= ~frame["prompt_id"].astype(str).isin(heldout_prompt_ids)

    heldout_prompt_keys: set[str] = set()
    if "prompt_key" in frame.columns:
        heldout_prompt_keys = set(frame.loc[heldout_mask, "prompt_key"].astype(str))
        train_mask &= ~frame["prompt_key"].astype(str).isin(heldout_prompt_keys)

    train_idx = np.flatnonzero(train_mask.to_numpy())
    train = frame.iloc[train_idx]
    heldout = frame.iloc[heldout_idx]

    prompt_id_overlap = len(
        set(train["prompt_id"].astype(str)) & set(heldout["prompt_id"].astype(str))
    )
    prompt_key_overlap = 0
    if "prompt_key" in frame.columns:
        prompt_key_overlap = len(
            set(train["prompt_key"].astype(str))
            & set(heldout["prompt_key"].astype(str))
        )

    return {
        "holdout_sources": holdout_sources,
        "train_idx": train_idx,
        "heldout_idx": heldout_idx,
        "prompt_id_overlap": prompt_id_overlap,
        "prompt_key_overlap": prompt_key_overlap,
    }


def select_eligible_labels(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    min_heldout_rows: int = MIN_HELDOUT_ROWS_PER_LABEL,
) -> list[str]:
    """Return labels present in train with enough union-heldout examples."""
    train_labels = set(train["model"].astype(str))
    heldout_counts = heldout["model"].astype(str).value_counts()
    return sorted(
        label
        for label, count in heldout_counts.items()
        if label in train_labels and int(count) >= min_heldout_rows
    )


def compute_class_weights(
    labels: Sequence[object] | np.ndarray,
    *,
    classes: Sequence[str],
) -> dict[str, float]:
    """Inverse-frequency class weights normalized to arithmetic mean 1."""
    label_values = np.asarray(labels, dtype=object).astype(str)
    counts = {str(cls): int(np.sum(label_values == str(cls))) for cls in classes}
    missing = [label for label, count in counts.items() if count <= 0]
    if missing:
        raise ValueError(f"class weights require positive training counts: {missing}")
    raw = {label: 1.0 / count for label, count in counts.items()}
    mean_weight = float(np.mean(list(raw.values())))
    return {label: float(weight / mean_weight) for label, weight in raw.items()}


def build_prompt_triplets(
    frame: pd.DataFrame,
    rows: Sequence[int] | np.ndarray,
    *,
    seed: int,
) -> list[np.ndarray]:
    """Build deterministic microbatches of up to three rows from one prompt only."""
    rows_array = np.asarray(rows, dtype=int)
    rng = np.random.RandomState(seed)
    batches: list[np.ndarray] = []
    prompt_values = frame["prompt_id"].astype(str).to_numpy()

    for prompt_id in sorted(set(prompt_values[rows_array])):
        prompt_rows = rows_array[prompt_values[rows_array] == prompt_id].copy()
        prompt_rows = rng.permutation(prompt_rows)
        for start in range(0, len(prompt_rows), 3):
            batches.append(np.asarray(prompt_rows[start : start + 3], dtype=int))

    if batches:
        order = rng.permutation(len(batches))
        batches = [batches[int(i)] for i in order]
    return batches


def gate_a_decision(
    *,
    eligible_label_count: int,
    eligible_heldout_rows: int,
    pooled_delta: float,
    source_deltas: Sequence[float],
) -> dict[str, object]:
    """Apply the frozen Gate-A PASS/FAIL/INCONCLUSIVE rules exactly."""
    if len(source_deltas) != 3:
        raise ValueError("Gate A requires exactly three source deltas")

    if (
        eligible_label_count < MIN_ELIGIBLE_LABELS
        or eligible_heldout_rows < MIN_ELIGIBLE_HELDOUT_ROWS
    ):
        return {
            "status": "inconclusive",
            "official_oof_authorized": False,
            "reason": "insufficient_eligible_labels_or_rows",
        }

    deltas = np.asarray(source_deltas, dtype=float)
    passes = (
        float(pooled_delta) > 0.0
        and float(deltas.mean()) > 0.0
        and int(np.sum(deltas < 0.0)) <= 1
    )
    if passes:
        return {
            "status": "pass",
            "official_oof_authorized": True,
            "reason": "all_frozen_gate_a_rules_passed",
        }
    return {
        "status": "fail",
        "official_oof_authorized": False,
        "reason": "pooled_or_source_consistency_rule_failed",
    }
