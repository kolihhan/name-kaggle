import numpy as np
import pandas as pd
import pytest

from scripts.run_multiview_external_ablation import (
    EXPECTED_EXTERNAL_LABELS,
    EXPECTED_EXTERNAL_PROMPTS,
    EXPECTED_EXTERNAL_ROWS,
    PHASE5_REFERENCE,
    balanced_external_slice,
    prepare_augmented_fold,
    validate_external_snapshot,
)


def sample_train() -> pd.DataFrame:
    rows = []
    for prompt_id, fold_id in [("p0", 0), ("p1", 1), ("p2", 2), ("p3", 3), ("p4", 4)]:
        for model in ["a", "b"]:
            rows.append(
                {
                    "id": f"{prompt_id}-{model}",
                    "prompt_id": prompt_id,
                    "text": f"official {prompt_id} {model}",
                    "model": model,
                    "fold_id": fold_id,
                }
            )
    return pd.DataFrame(rows)


def test_phase5_reference_is_frozen():
    assert PHASE5_REFERENCE == pytest.approx(0.36003977131493914)
    assert EXPECTED_EXTERNAL_ROWS == 2503
    assert EXPECTED_EXTERNAL_PROMPTS == 430
    assert len(EXPECTED_EXTERNAL_LABELS) == 8


def test_validate_external_snapshot_rejects_unknown_labels():
    frame = pd.DataFrame(
        {
            "text": ["x"],
            "model": ["unknown-model"],
            "prompt_id": ["e0"],
        }
    )
    with pytest.raises(ValueError, match="outside competition label set"):
        validate_external_snapshot(
            frame,
            official_labels={"a", "b"},
            enforce_canonical_shape=False,
        )


def test_balanced_external_slice_is_equal_and_deterministic():
    frame = pd.DataFrame(
        {
            "text": [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(2)],
            "model": ["a"] * 4 + ["b"] * 2,
            "prompt_id": [f"p{i}" for i in range(6)],
        }
    )
    first = balanced_external_slice(frame, seed=42)
    second = balanced_external_slice(frame, seed=42)
    assert first.equals(second)
    assert first["model"].value_counts().to_dict() == {"a": 2, "b": 2}


def test_prepare_augmented_fold_keeps_external_out_of_validation():
    train = sample_train()
    external = pd.DataFrame(
        {
            "text": ["e0", "e1", "e2"],
            "model": ["a", "b", "a"],
            "prompt_id": ["ep0", "ep1", "ep2"],
        }
    )
    fit_text, fit_y, valid_text, valid_y, valid_idx = prepare_augmented_fold(
        train, external, fold=2
    )

    expected_valid = train.loc[train.fold_id == 2]
    assert valid_text.tolist() == expected_valid["text"].tolist()
    assert valid_y.tolist() == expected_valid["model"].tolist()
    np.testing.assert_array_equal(valid_idx, expected_valid.index.to_numpy())
    assert len(fit_text) == int((train.fold_id != 2).sum()) + len(external)
    assert len(fit_y) == len(fit_text)
    assert not set(external["text"]).intersection(set(valid_text))
