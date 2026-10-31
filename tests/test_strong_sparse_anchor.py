import numpy as np
from scipy import sparse

from scripts.run_strong_sparse_anchor import (
    build_fold_features,
    make_svc,
)


def test_build_fold_features_uses_same_fold_local_width_and_finite_values():
    train_text = [
        "Alpha text — one.",
        "Alpha text — two.",
        "Beta text! one.",
        "Beta text! two.",
    ]
    valid_text = ["Alpha text — unseen!", "Beta text! unseen?"]

    x_train, x_valid, meta = build_fold_features(
        train_text,
        valid_text,
        min_df=1,
        max_features=100,
    )

    assert sparse.isspmatrix_csr(x_train)
    assert sparse.isspmatrix_csr(x_valid)
    assert x_train.shape[0] == 4
    assert x_valid.shape[0] == 2
    assert x_train.shape[1] == x_valid.shape[1]
    assert x_train.shape[1] > meta["style_features"]
    assert np.isfinite(x_train.data).all()
    assert np.isfinite(x_valid.data).all()
    assert meta["style_weight"] == 0.2


def test_strong_sparse_classifier_configuration_is_frozen():
    model = make_svc(seed=42)
    assert model.C == 1.5
    assert model.class_weight == "balanced"
    assert model.max_iter == 15000
