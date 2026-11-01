import numpy as np

from scripts.run_phase17_global_balance import (
    PHASE7_SCORE,
    NUMERIC_ATOL,
    promotion_decision,
    row_softmax,
    sinkhorn_uniform,
)


def test_row_softmax_is_finite_positive_and_normalized():
    margins = np.array(
        [
            [3.0, 1.0, -2.0],
            [1000.0, 999.0, 998.0],
            [-1000.0, -1001.0, -999.0],
        ],
        dtype=np.float64,
    )
    probs = row_softmax(margins)
    assert probs.shape == margins.shape
    assert np.isfinite(probs).all()
    assert (probs > 0).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-12, rtol=0)


def test_sinkhorn_uniform_preserves_rows_and_balances_columns():
    probs = row_softmax(
        np.array(
            [
                [3.0, 1.0, 0.0],
                [2.0, 0.0, 1.0],
                [0.0, 3.0, 1.0],
                [0.5, 2.0, 1.5],
                [1.0, 0.0, 3.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
    )
    balanced = sinkhorn_uniform(probs, iterations=100)
    assert balanced.shape == probs.shape
    assert np.isfinite(balanced).all()
    assert (balanced >= 0).all()
    assert np.allclose(balanced.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    assert np.allclose(balanced.sum(axis=0), 2.0, atol=1e-9, rtol=0)


def test_sinkhorn_uniform_does_not_mutate_input():
    probs = row_softmax(np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64))
    original = probs.copy()
    _ = sinkhorn_uniform(probs, iterations=100)
    assert np.array_equal(probs, original)


def test_phase17_promotion_requires_real_gain_over_phase7():
    tie = promotion_decision(PHASE7_SCORE)
    epsilon = promotion_decision(PHASE7_SCORE + NUMERIC_ATOL / 2)
    gain = promotion_decision(PHASE7_SCORE + NUMERIC_ATOL * 2)
    assert tie["promote_phase17"] is False
    assert epsilon["promote_phase17"] is False
    assert gain["promote_phase17"] is True
