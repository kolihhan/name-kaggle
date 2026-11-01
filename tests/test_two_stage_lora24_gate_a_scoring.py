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


def test_group_triplets_for_accumulation_preserves_exact_one_epoch():
    batches = [
        np.array([0, 1, 2]),
        np.array([3, 4]),
        np.array([5]),
        np.array([6, 7, 8]),
        np.array([9]),
        np.array([10, 11]),
        np.array([12]),
    ]

    groups = gate.group_triplets_for_accumulation(batches, accumulation=6)

    assert [len(group) for group in groups] == [6, 1]
    flattened = [batch for group in groups for batch in group]
    assert len(flattened) == len(batches)
    for original, observed in zip(batches, flattened):
        np.testing.assert_array_equal(original, observed)


def test_group_triplets_for_accumulation_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one triplet"):
        gate.group_triplets_for_accumulation([], accumulation=6)


def test_group_triplets_for_accumulation_rejects_nonpositive_accumulation():
    with pytest.raises(ValueError, match="accumulation must be positive"):
        gate.group_triplets_for_accumulation([np.array([0])], accumulation=0)
