from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_phase14_public_sparse import SEED_SOURCES


MIN_SOURCE_ROWS = 100
MIN_SOURCE_PROMPTS = 20
MIN_SHARED_LABELS = 3


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def build_prompt_residuals(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Center frozen embeddings within prompt groups and return the LDA-fit mask."""
    _require_columns(frame, {"prompt_id", "model"})
    values = np.asarray(embeddings)
    if values.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if len(values) != len(frame):
        raise ValueError("embedding rows must match frame rows")

    residuals = np.zeros(values.shape, dtype=np.float32)
    fit_mask = np.zeros(len(frame), dtype=bool)

    prompt_ids = frame["prompt_id"].astype(str).to_numpy()
    labels = frame["model"].astype(str).to_numpy()

    for prompt_id in pd.unique(prompt_ids):
        idx = np.flatnonzero(prompt_ids == prompt_id)
        group = values[idx].astype(np.float32, copy=False)
        centered = group - group.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        if np.any(nonzero):
            centered = centered.copy()
            centered[nonzero] /= norms[nonzero]
        residuals[idx] = centered

        if len(set(labels[idx])) >= 2:
            fit_mask[idx] = True

    return residuals, fit_mask


def build_source_holdout(
    frame: pd.DataFrame,
    source: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a source holdout while removing prompt-bank leakage from training."""
    _require_columns(frame, {"source", "prompt_id", "prompt_key", "model"})

    source_values = frame["source"].astype(str)
    heldout_mask = source_values.eq(str(source)).to_numpy()
    heldout_idx = np.flatnonzero(heldout_mask)
    if len(heldout_idx) == 0:
        raise ValueError(f"source has no rows: {source}")

    heldout = frame.iloc[heldout_idx]
    heldout_prompt_ids = set(heldout["prompt_id"].astype(str))
    heldout_prompt_keys = set(heldout["prompt_key"].astype(str))

    prompt_ids = frame["prompt_id"].astype(str)
    prompt_keys = frame["prompt_key"].astype(str)
    leakage_mask = prompt_ids.isin(heldout_prompt_ids) | prompt_keys.isin(heldout_prompt_keys)
    train_mask = (~heldout_mask) & (~leakage_mask.to_numpy())
    train_idx = np.flatnonzero(train_mask)

    train_labels = set(frame.iloc[train_idx]["model"].astype(str))
    heldout_labels = set(heldout["model"].astype(str))
    shared_labels = sorted(train_labels & heldout_labels)
    return train_idx, heldout_idx, shared_labels


def eligible_seed_sources(frame: pd.DataFrame) -> list[str]:
    """Return seed sources satisfying the contract-frozen Gate A eligibility rules."""
    _require_columns(frame, {"source", "prompt_id", "prompt_key", "model"})

    eligible: list[str] = []
    sources = frame["source"].astype(str)
    for source in sorted(SEED_SOURCES):
        source_frame = frame.loc[sources.eq(source)]
        if len(source_frame) < MIN_SOURCE_ROWS:
            continue
        if source_frame["prompt_id"].astype(str).nunique() < MIN_SOURCE_PROMPTS:
            continue
        _, _, shared_labels = build_source_holdout(frame, source)
        if len(shared_labels) < MIN_SHARED_LABELS:
            continue
        eligible.append(source)
    return eligible
