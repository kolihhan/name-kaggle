import numpy as np
import pandas as pd
from scipy import sparse

from scripts.run_phase14_public_sparse import (
    PHASE7_SCORE,
    SEED_SOURCES,
    SMALL_WORDS,
    VIEWS,
    build_conservative_subset,
    build_fold_features,
    make_svc,
    promotion_decision,
    shape_text,
    small_words_text,
)


def test_phase14_recipe_constants_are_frozen():
    assert VIEWS == [
        ("char", 3, 5, 250_000, 1.0),
        ("word", 1, 3, 180_000, 0.7),
        ("shape", 2, 6, 60_000, 0.5),
        ("small", 1, 3, 50_000, 0.5),
    ]
    assert "the" in SMALL_WORDS
    assert "with" in SMALL_WORDS
    assert "model" not in SMALL_WORDS
    assert PHASE7_SCORE == 0.3700521998508576


def test_phase14_text_transforms_match_public_recipe():
    assert shape_text("Hello WORLD 123!") == "X X 123!"
    assert shape_text("hello World-test") == "x X-x"
    assert small_words_text("The model writes with style!") == "the CONTENT CONTENT with CONTENT !"
    assert small_words_text("I can do it, now.") == "i can do it , now ."


def test_conservative_subset_inherits_only_exact_safe_prompt_ids():
    pool = pd.DataFrame(
        {
            "prompt_id": ["p1", "p2", "p1", "p3"],
            "prompt_key": ["seyitbank:g1", "keshav:k1", "unknown:u1", "unknown:u2"],
            "text": ["a", "b", "c", "d"],
            "model": ["m1", "m2", "m3", "m4"],
            "source": ["seyit", "keshav", "unknown", "unknown"],
            "setting": ["default"] * 4,
        }
    )
    safe = build_conservative_subset(pool, validate_full_contract=False)
    assert safe["text"].tolist() == ["a", "b", "c"]
    assert set(SEED_SOURCES) >= {"seyit", "keshav"}


def test_phase14_fold_features_are_sparse_aligned_and_finite():
    train_text = [
        "The model writes one short paragraph.",
        "Another model writes two short paragraphs!",
        "Style can differ, but punctuation stays.",
        "These examples are deliberately tiny.",
    ]
    valid_text = ["The unseen model writes differently!", "Another unseen example."]
    x_train, x_valid, meta = build_fold_features(
        train_text,
        valid_text,
        min_df=1,
        feature_cap=100,
    )
    assert sparse.isspmatrix_csr(x_train)
    assert sparse.isspmatrix_csr(x_valid)
    assert x_train.shape[0] == 4
    assert x_valid.shape[0] == 2
    assert x_train.shape[1] == x_valid.shape[1]
    assert np.isfinite(x_train.data).all()
    assert np.isfinite(x_valid.data).all()
    assert set(meta["widths"]) == {"char", "word", "shape", "small"}


def test_phase14_classifier_configuration_is_frozen():
    model = make_svc()
    assert model.C == 0.5
    assert model.class_weight == "balanced"
    assert model.random_state == 42
    assert model.max_iter == 5000


def test_phase14_promotion_rejects_numeric_tie_and_requires_real_gain():
    tie = promotion_decision(PHASE7_SCORE)
    epsilon = promotion_decision(PHASE7_SCORE + 5e-13)
    gain = promotion_decision(PHASE7_SCORE + 2e-12)
    assert tie["promote_phase14"] is False
    assert epsilon["promote_phase14"] is False
    assert gain["promote_phase14"] is True
