import numpy as np
import pandas as pd
import pytest

from scripts.run_frozen_encoder_ablation import (
    prepare_fold_indices,
    run_embedding_ablation,
    validate_fold_contract,
)


def sample_train() -> pd.DataFrame:
    rows = []
    row_id = 0
    for prompt_id, fold_id in [("p0", 0), ("p1", 1), ("p2", 2), ("p3", 3), ("p4", 4)]:
        for model in ["m0", "m1"]:
            rows.append(
                {
                    "id": row_id,
                    "prompt_id": prompt_id,
                    "model": model,
                    "text": f"{prompt_id}-{model}",
                    "fold_id": fold_id,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


def test_validate_fold_contract_rejects_prompt_leakage():
    train = sample_train()
    train.loc[1, "fold_id"] = 1

    with pytest.raises(ValueError, match="prompt_id appears in multiple folds"):
        validate_fold_contract(train)


def test_prepare_fold_indices_keeps_validation_identical_and_external_train_only():
    train = sample_train()
    validate_fold_contract(train)

    official_train_idx, valid_idx, augmented_train_idx = prepare_fold_indices(
        train_rows=len(train),
        external_rows=3,
        fold_ids=train["fold_id"].to_numpy(),
        fold=2,
    )

    expected_valid = np.flatnonzero(train["fold_id"].to_numpy() == 2)
    expected_official_train = np.flatnonzero(train["fold_id"].to_numpy() != 2)
    external_idx = np.arange(len(train), len(train) + 3)

    np.testing.assert_array_equal(valid_idx, expected_valid)
    np.testing.assert_array_equal(official_train_idx, expected_official_train)
    np.testing.assert_array_equal(
        augmented_train_idx,
        np.concatenate([expected_official_train, external_idx]),
    )
    assert not np.intersect1d(valid_idx, augmented_train_idx).size


def test_run_embedding_ablation_scores_only_official_oof_rows():
    train = sample_train()
    official_embeddings = np.array(
        [[-2.0, 0.0] if model == "m0" else [2.0, 0.0] for model in train["model"]],
        dtype=np.float32,
    )
    external = pd.DataFrame(
        {
            "model": ["m0", "m1", "m0", "m1"],
            "text": ["e0", "e1", "e2", "e3"],
        }
    )
    external_embeddings = np.array(
        [[-2.5, 0.0], [2.5, 0.0], [-3.0, 0.0], [3.0, 0.0]],
        dtype=np.float32,
    )

    report = run_embedding_ablation(
        train=train,
        official_embeddings=official_embeddings,
        external=external,
        external_embeddings=external_embeddings,
    )

    experiments = {x["name"]: x for x in report["experiments"]}
    assert set(experiments) == {"official_only", "official_plus_external"}
    assert experiments["official_only"]["oof_rows"] == len(train)
    assert experiments["official_plus_external"]["oof_rows"] == len(train)
    assert experiments["official_only"]["balanced_accuracy"] == pytest.approx(1.0)
    assert experiments["official_plus_external"]["balanced_accuracy"] == pytest.approx(1.0)
