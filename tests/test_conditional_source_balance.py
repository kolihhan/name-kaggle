import numpy as np
import pandas as pd
import pytest

from scripts.run_conditional_source_balance import (
    conditional_weights,
    decision,
    paired_bootstrap,
    purge_overlap,
    shuffle_within_class,
    summarize_predictions,
)


def test_conditional_weights_equalize_source_mass_without_changing_class_mass():
    frame = pd.DataFrame({"model": ["a"] * 4 + ["b"] * 2,
                          "source": ["x", "x", "x", "y", "x", "y"]})
    weights = conditional_weights(frame)
    np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 2 / 3, 2, 1, 1])
    assert weights[:3].sum() == pytest.approx(2)
    assert weights[:4].sum() == pytest.approx(4)
    assert weights[4:].sum() == pytest.approx(2)


def test_shuffle_preserves_weight_multiset_within_each_label():
    labels = np.array(["a"] * 20 + ["b"] * 10)
    weights = np.arange(1, 31, dtype=float)
    result = shuffle_within_class(weights, labels)
    for label in ["a", "b"]:
        np.testing.assert_array_equal(np.sort(result[labels == label]),
                                      np.sort(weights[labels == label]))
    assert np.count_nonzero(result != weights) > 20
    np.testing.assert_array_equal(result, shuffle_within_class(weights, labels))


def test_purge_removes_prompt_and_normalized_answer_overlap_across_sources():
    train = pd.DataFrame({"prompt_id": ["p1", "p2", "p3", "p4"],
                          "prompt_key": ["k1", "k2", "k3", "k4"],
                          "text": ["ONE", "two", "ＦＯＯ\nBAR", "keep"]})
    test = pd.DataFrame({"prompt_id": ["p1", "other"],
                         "prompt_key": ["other", "k2"],
                         "text": ["foo  bar", "different"]})
    kept, report = purge_overlap(train, test)
    assert kept.text.tolist() == ["keep"]
    assert report["removed_train_rows"] == 3
    assert report["remaining_prompt_id_overlap"] == 0
    assert report["remaining_prompt_key_overlap"] == 0
    assert report["remaining_answer_overlap"] == 0


def test_summary_decomposes_macro_errors_and_reports_real_paired_changes():
    labels = np.array(["gpt-a", "gpt-a", "gpt-b", "gemini-a"])
    base = np.array(["gpt-b", "gpt-a", "gpt-a", "gpt-a"])
    candidate = np.array(["gpt-a", "gpt-b", "gpt-b", "gpt-a"])
    result = summarize_predictions(labels, candidate, base, sorted(set(labels)))
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["macro_sibling_error"] == pytest.approx(1 / 6)
    assert result["macro_provider_error"] == pytest.approx(1 / 3)
    assert result["rescues_vs_B"] == 2
    assert result["regressions_vs_B"] == 1


def test_bootstrap_uses_shared_prompt_draws_and_paired_row_errors():
    frame = pd.DataFrame({"source": ["s1"] * 4 + ["s2"] * 4,
                          "model": ["a", "b", "a", "b"] * 2,
                          "prompt_key": ["p", "p", "q", "q"] * 2,
                          "prompt_id": ["p", "p", "q", "q"] * 2})
    y = frame.model.to_numpy()
    right = y.copy()
    wrong = np.where(y == "a", "b", "a")
    eligible = {"s1": ["a", "b"], "s2": ["a", "b"]}
    positive = paired_bootstrap(frame, right, wrong, eligible, replicates=50)
    assert positive["retained_replicates"] == 50
    assert positive["ci95"] == pytest.approx([1, 1])
    zero = paired_bootstrap(frame, right, right, eligible, replicates=50)
    assert zero["ci95"] == pytest.approx([0, 0])


def test_gate_rejects_failed_control_and_missing_bootstrap_evidence():
    assert decision(0.02, 0.001, [0.02] * 3, [0.01, 0.03], 2000) == "REJECT"
    assert decision(0.02, 0.01, [0.02] * 3, [0.01, 0.03], 1800) == "INCONCLUSIVE"
    assert decision(0.02, 0.01, [0.02] * 3, [0.01, 0.03], 2000) == "PASS_NEXT_STAGE_ONLY"
