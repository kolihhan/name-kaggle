from __future__ import annotations

import numpy as np
import pytest

from scripts import run_two_stage_lora24_gate_a as gate


def test_score_predictions_uses_mean_per_class_recall():
    report = gate.score_predictions(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
        labels=["a", "b"],
    )

    assert report["balanced_accuracy"] == pytest.approx(0.75)
    assert report["per_label_recall"] == {"a": 0.5, "b": 1.0}
    assert report["label_counts"] == {"a": 2, "b": 2}


def test_score_predictions_rejects_requested_label_without_support():
    with pytest.raises(ValueError, match="no evaluation rows"):
        gate.score_predictions(["a"], ["a"], labels=["a", "b"])


def test_pad_triplets_for_accumulation_is_deterministic_and_full_steps():
    batches = [
        np.array([0, 1, 2]),
        np.array([3, 4]),
        np.array([5]),
        np.array([6, 7, 8]),
        np.array([9]),
    ]

    first = gate.pad_triplets_for_accumulation(batches, accumulation=6, seed=20261101)
    second = gate.pad_triplets_for_accumulation(batches, accumulation=6, seed=20261101)

    assert len(first) == 6
    assert len(first) % 6 == 0
    for original, padded in zip(batches, first[: len(batches)]):
        np.testing.assert_array_equal(original, padded)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert any(np.array_equal(first[-1], original) for original in batches)


def test_pad_triplets_for_accumulation_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one triplet"):
        gate.pad_triplets_for_accumulation([], accumulation=6, seed=20261101)
