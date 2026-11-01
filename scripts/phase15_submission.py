from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_phase14_public_sparse import build_fold_features, make_svc


def _require_columns(name: str, frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} frame missing required columns: {missing}")


def train_full_and_predict(
    official: pd.DataFrame,
    external: pd.DataFrame,
    test: pd.DataFrame,
    *,
    min_df: int = 2,
    feature_cap: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit the frozen Phase 15 recipe on all official + terms-clean external rows."""
    _require_columns("official", official, {"text", "model"})
    _require_columns("external", external, {"text", "model"})
    _require_columns("test", test, {"text"})

    if official["text"].isna().any() or official["model"].isna().any():
        raise RuntimeError("official training frame contains null text/model")
    if external["text"].isna().any() or external["model"].isna().any():
        raise RuntimeError("external training frame contains null text/model")
    if test["text"].isna().any():
        raise RuntimeError("test frame contains null text")

    official_labels = set(official["model"].astype(str).unique())
    external_labels = set(external["model"].astype(str).unique())
    if not external_labels:
        raise RuntimeError("external training frame has no labels")
    unknown_external = external_labels - official_labels
    if unknown_external:
        raise RuntimeError(
            f"external labels outside official label set: {sorted(unknown_external)}"
        )

    train_text = official["text"].astype(str).tolist() + external["text"].astype(str).tolist()
    train_y = official["model"].astype(str).tolist() + external["model"].astype(str).tolist()
    test_text = test["text"].astype(str).tolist()

    x_train, x_test, feature_meta = build_fold_features(
        train_text,
        test_text,
        min_df=min_df,
        feature_cap=feature_cap,
    )
    model = make_svc()
    model.fit(x_train, train_y)
    predictions = np.asarray(model.predict(x_test), dtype=object)

    if len(predictions) != len(test):
        raise RuntimeError(
            f"prediction count mismatch: predictions={len(predictions)} test={len(test)}"
        )
    predicted_labels = set(map(str, predictions))
    if not predicted_labels.issubset(official_labels):
        raise RuntimeError(
            f"predictions contain labels outside official label set: {sorted(predicted_labels - official_labels)}"
        )

    feature_meta.update(
        {
            "official_training_rows": int(len(official)),
            "external_training_rows": int(len(external)),
            "external_training_labels": sorted(external_labels),
            "total_training_rows": int(len(train_text)),
            "test_rows": int(len(test)),
        }
    )
    return predictions, feature_meta


def build_submission_frame(
    test: pd.DataFrame,
    predictions: np.ndarray,
    *,
    allowed_labels: set[str],
) -> pd.DataFrame:
    _require_columns("test", test, {"id"})
    if len(predictions) != len(test):
        raise RuntimeError("submission prediction count does not match test rows")

    submission = pd.DataFrame(
        {
            "id": test["id"].to_numpy(copy=True),
            "model": np.asarray(predictions, dtype=object),
        }
    )
    if submission["id"].isna().any() or submission["model"].isna().any():
        raise RuntimeError("submission contains null id/model")
    if submission["id"].duplicated().any():
        raise RuntimeError("submission contains duplicate ids")
    predicted = set(submission["model"].astype(str).unique())
    allowed = set(map(str, allowed_labels))
    if not predicted.issubset(allowed):
        raise RuntimeError(
            f"submission contains disallowed labels: {sorted(predicted - allowed)}"
        )
    return submission
