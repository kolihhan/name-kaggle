from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from scripts.run_strong_sparse_external import validate_external


EXPECTED_COVERED_LABELS = {
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
}


def gate_predictions(
    anchor_predictions: np.ndarray,
    specialist_predictions: np.ndarray,
    covered_labels: Iterable[str],
) -> np.ndarray:
    if len(anchor_predictions) != len(specialist_predictions):
        raise RuntimeError("anchor and specialist prediction lengths differ")

    covered = set(covered_labels)
    observed_specialist = set(str(x) for x in specialist_predictions.tolist())
    outside = sorted(observed_specialist - covered)
    if outside:
        raise RuntimeError(f"specialist predictions outside covered label set: {outside}")

    result = np.asarray(anchor_predictions, dtype=object).copy()
    anchor = np.asarray(anchor_predictions, dtype=object)
    specialist = np.asarray(specialist_predictions, dtype=object)
    mask = np.fromiter((str(label) in covered for label in anchor), dtype=bool, count=len(anchor))
    result[mask] = specialist[mask]
    return result


def specialist_training_rows(
    official_fit: pd.DataFrame,
    external: pd.DataFrame,
    *,
    covered_labels: Iterable[str],
) -> tuple[list[str], list[str]]:
    covered = set(covered_labels)
    official_covered = official_fit.loc[
        official_fit["model"].astype(str).isin(covered), ["text", "model"]
    ]
    texts = pd.concat(
        [official_covered["text"], external["text"]], ignore_index=True
    ).astype(str).tolist()
    labels = pd.concat(
        [official_covered["model"], external["model"]], ignore_index=True
    ).astype(str).tolist()
    return texts, labels


def assert_uncovered_recall_invariant(
    anchor_recalls: dict[str, float],
    ensemble_recalls: dict[str, float],
    *,
    covered_labels: Iterable[str],
) -> None:
    covered = set(covered_labels)
    if set(anchor_recalls) != set(ensemble_recalls):
        raise RuntimeError("anchor and ensemble recall label sets differ")
    for label, anchor_recall in anchor_recalls.items():
        if label in covered:
            continue
        ensemble_recall = ensemble_recalls[label]
        if not np.isclose(anchor_recall, ensemble_recall, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"uncovered recall drift for {label}: {anchor_recall} -> {ensemble_recall}"
            )
