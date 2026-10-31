import numpy as np

from scripts.run_reference_style_repro import extract_reference_style_features


def test_reference_style_feature_contract_is_exact_and_order_stable():
    texts = [
        'Alpha — text with “quotes” and contractions it\'s.',
        'Beta - TEXT with 123 digits and punctuation!',
    ]
    a = extract_reference_style_features(texts)
    b = extract_reference_style_features(list(reversed(texts)))[::-1]

    assert a.shape == (2, 108)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a[0], a[1])
