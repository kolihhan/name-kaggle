import numpy as np
import pandas as pd
import pytest

from scripts.run_style_baseline import (
    blend_probabilities,
    extract_style_features,
    summarize_oof,
)


def test_style_features_are_finite_order_stable_and_typography_sensitive():
    texts = [
        'Plain text with an em dash — and curly quote “hello”.',
        "PLAIN text with a hyphen - and straight quote \"hello\"!",
    ]
    a = extract_style_features(texts)
    b = extract_style_features(list(reversed(texts)))[::-1]

    assert a.ndim == 2
    assert a.shape[0] == 2
    assert a.shape[1] >= 60
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a[0], a[1])


def test_geometric_blend_is_normalized_and_prefers_joint_signal():
    p1 = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=float)
    p2 = np.array([[0.6, 0.4], [0.1, 0.9]], dtype=float)
    p3 = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)

    blended = blend_probabilities([p1, p2, p3])
    np.testing.assert_allclose(blended.sum(axis=1), np.ones(2))
    assert blended[0, 0] > blended[0, 1]
    assert blended[1, 1] > blended[1, 0]


def test_geometric_blend_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        blend_probabilities([
            np.ones((2, 2)) / 2,
            np.ones((2, 3)) / 3,
        ])


def test_summarize_oof_scores_only_given_official_rows():
    train = pd.DataFrame({"model": ["a", "b", "a", "b"]})
    pred = np.array(["a", "b", "b", "b"], dtype=object)
    report = summarize_oof(train, pred, fold_scores=[1.0, 0.5])

    assert report["oof_rows"] == 4
    assert report["balanced_accuracy"] == pytest.approx(0.75)
    assert report["per_model_recall"] == {
        "a": pytest.approx(0.5),
        "b": pytest.approx(1.0),
    }
