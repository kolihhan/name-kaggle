import hashlib

import numpy as np
import pandas as pd
import pytest

from scripts.train_modernbert_cv import (
    EXPECTED_FOLD_FINGERPRINT,
    build_folds,
    fold_fingerprint,
    label_mapping,
    summarize_oof,
)


def sample_train() -> pd.DataFrame:
    rows = []
    for prompt_num in range(10):
        prompt_id = f"p{prompt_num:03d}"
        for model in ["model-b", "model-a"]:
            rows.append(
                {
                    "id": f"{prompt_num:03d}-{model}",
                    "prompt_id": prompt_id,
                    "text": f"text {prompt_num} {model}",
                    "model": model,
                }
            )
    return pd.DataFrame(rows)


def test_fold_fingerprint_is_order_stable_and_sensitive_to_mapping():
    frame = pd.DataFrame(
        {
            "id": ["b", "a"],
            "prompt_id": ["p1", "p0"],
            "model": ["m1", "m0"],
            "fold_id": [1, 0],
        }
    )
    expected_payload = "a,p0,m0,0\nb,p1,m1,1"
    expected = hashlib.sha256(expected_payload.encode()).hexdigest()
    assert fold_fingerprint(frame) == expected
    assert fold_fingerprint(frame.iloc[::-1]) == expected

    changed = frame.copy()
    changed.loc[0, "fold_id"] = 4
    assert fold_fingerprint(changed) != expected


def test_expected_phase1_fold_fingerprint_is_frozen():
    assert EXPECTED_FOLD_FINGERPRINT == (
        "bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d"
    )


def test_label_mapping_is_sorted_and_round_trips():
    label2id, id2label = label_mapping(["z", "a", "m", "a"])
    assert label2id == {"a": 0, "m": 1, "z": 2}
    assert id2label == {0: "a", 1: "m", 2: "z"}


def test_summarize_oof_scores_only_official_rows():
    train = pd.DataFrame({"model": ["a", "b", "a", "b"]})
    preds = np.array(["a", "b", "b", "b"], dtype=object)
    report = summarize_oof(train, preds, fold_scores=[1.0, 0.5])

    assert report["oof_rows"] == 4
    assert report["balanced_accuracy"] == pytest.approx(0.75)
    assert report["per_model_recall"] == {"a": pytest.approx(0.5), "b": pytest.approx(1.0)}


def test_build_folds_keeps_each_prompt_in_one_fold():
    folded = build_folds(sample_train(), n_splits=5)
    assert set(folded["fold_id"]) == {0, 1, 2, 3, 4}
    assert folded.groupby("prompt_id")["fold_id"].nunique().max() == 1
