import numpy as np
import pandas as pd
import pytest

from scripts.run_honest_style_baseline import (
    EXPECTED_FOLD_FINGERPRINT,
    STYLE_FEATURE_NAMES,
    build_multiview_model,
    extract_style_features,
    summarize_oof,
)


def test_style_features_are_deterministic_finite_and_named():
    texts = [
        '“Hello,” she said — it\'s 42% done.\n\n## Note',
        'plain words only',
    ]
    first = extract_style_features(texts)
    second = extract_style_features(texts)

    assert first.shape == (2, len(STYLE_FEATURE_NAMES))
    assert len(STYLE_FEATURE_NAMES) == len(set(STYLE_FEATURE_NAMES))
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)


def test_style_features_preserve_typography_signals():
    fancy = extract_style_features(['“x” — y’s'])[0]
    plain = extract_style_features(['"x" - y\'s'])[0]
    index = {name: i for i, name in enumerate(STYLE_FEATURE_NAMES)}

    assert fancy[index['count_em_dash']] > plain[index['count_em_dash']]
    assert fancy[index['count_curly_apostrophe']] > plain[index['count_curly_apostrophe']]
    assert fancy[index['count_left_curly_quote']] > plain[index['count_left_curly_quote']]
    assert plain[index['count_ascii_apostrophe']] > fancy[index['count_ascii_apostrophe']]


def test_multiview_model_contract_is_frozen():
    model = build_multiview_model(seed=20261031)
    features = model.named_steps['features']
    classifier = model.named_steps['classifier']

    assert set(dict(features.transformer_list)) == {
        'char', 'word', 'boundary', 'punct', 'style'
    }
    assert features.transformer_weights == {'style': 0.2}
    assert classifier.C == pytest.approx(1.5)
    assert classifier.class_weight == 'balanced'


def test_summarize_oof_scores_only_official_rows():
    train = pd.DataFrame({'model': ['a', 'b', 'a', 'b']})
    pred = np.array(['a', 'b', 'b', 'b'], dtype=object)
    report = summarize_oof(train, pred, fold_scores=[1.0, 0.5])

    assert report['oof_rows'] == 4
    assert report['balanced_accuracy'] == pytest.approx(0.75)
    assert report['per_model_recall'] == {
        'a': pytest.approx(0.5),
        'b': pytest.approx(1.0),
    }


def test_phase1_fold_fingerprint_constant_is_reused():
    assert EXPECTED_FOLD_FINGERPRINT == (
        'bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d'
    )
