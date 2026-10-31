from __future__ import annotations

import numpy as np
import pandas as pd


def validate_fold_contract(train: pd.DataFrame) -> None:
    required = {"prompt_id", "fold_id"}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"missing fold columns: {missing}")

    fold_counts = train.groupby("prompt_id")["fold_id"].nunique()
    if (fold_counts > 1).any():
        raise ValueError("prompt_id appears in multiple folds")

    observed = set(int(x) for x in train["fold_id"].dropna().unique())
    if observed != {0, 1, 2, 3, 4}:
        raise ValueError(f"expected folds 0..4, found {sorted(observed)}")


def prepare_fold_indices(
    *,
    train_rows: int,
    external_rows: int,
    fold_ids: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(fold_ids) != train_rows:
        raise ValueError("fold_ids length must match train_rows")

    valid_idx = np.flatnonzero(fold_ids == fold)
    official_train_idx = np.flatnonzero(fold_ids != fold)
    external_idx = np.arange(train_rows, train_rows + external_rows)
    augmented_train_idx = np.concatenate([official_train_idx, external_idx])
    return official_train_idx, valid_idx, augmented_train_idx
