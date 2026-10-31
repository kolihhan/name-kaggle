from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def validate_test_frame(test: pd.DataFrame) -> None:
    required = {"id", "text"}
    missing = sorted(required - set(test.columns))
    if missing:
        raise RuntimeError(f"test frame missing columns: {missing}")
    if test["id"].isna().any() or test["text"].isna().any():
        raise RuntimeError("test frame contains null ids or text")
    if test["id"].duplicated().any():
        raise RuntimeError("test frame contains duplicate ids")


def build_submission_frame(
    test: pd.DataFrame,
    predictions: np.ndarray,
    *,
    allowed_labels: Iterable[str],
) -> pd.DataFrame:
    validate_test_frame(test)
    predictions = np.asarray(predictions, dtype=object)
    if len(predictions) != len(test):
        raise RuntimeError(
            f"prediction count mismatch: {len(predictions)} != {len(test)}"
        )
    if any(value is None for value in predictions.tolist()):
        raise RuntimeError("predictions contain null values")

    allowed = {str(label) for label in allowed_labels}
    observed = {str(label) for label in predictions.tolist()}
    outside = sorted(observed - allowed)
    if outside:
        raise RuntimeError(f"predictions outside official label set: {outside}")

    submission = pd.DataFrame(
        {
            "id": test["id"].astype(str).to_numpy(),
            "model": predictions.astype(str),
        }
    )
    if submission["model"].isna().any():
        raise RuntimeError("submission contains null model values")
    return submission
