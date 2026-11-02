import numpy as np
import pandas as pd
import pytest

from scripts.two_stage_lora24_submission import (
    EXPECTED_OFFICIAL_ROWS,
    build_submission_frame,
    build_submission_lineage,
    plain_test_batches,
    validate_final_fit_frames,
)


def official_frame(n=EXPECTED_OFFICIAL_ROWS):
    labels = [f"m{i % 18}" for i in range(n)]
    return pd.DataFrame({
        "id": [f"tr-{i}" for i in range(n)],
        "prompt_id": [f"p-{i // 18}" for i in range(n)],
        "text": [f"train text {i}" for i in range(n)],
        "model": labels,
    })


def test_validate_final_fit_frames_returns_all_official_indices_only():
    train = official_frame()
    test = pd.DataFrame({"id": ["te-1", "te-2"], "text": ["a", "b"]})

    fit_idx, labels = validate_final_fit_frames(train, test)

    assert fit_idx.tolist() == list(range(EXPECTED_OFFICIAL_ROWS))
    assert len(labels) == 18
    assert set(labels) == set(train["model"])
    assert not set(train["id"]) & set(test["id"])


def test_validate_final_fit_frames_rejects_duplicate_test_ids():
    train = official_frame()
    test = pd.DataFrame({"id": ["te-1", "te-1"], "text": ["a", "b"]})

    with pytest.raises(RuntimeError, match="duplicate"):
        validate_final_fit_frames(train, test)


def test_validate_final_fit_frames_rejects_null_test_text():
    train = official_frame()
    test = pd.DataFrame({"id": ["te-1"], "text": [None]})

    with pytest.raises(RuntimeError, match="null"):
        validate_final_fit_frames(train, test)


def test_plain_test_batches_cover_every_test_row_once():
    batches = list(plain_test_batches(7, batch_size=3))
    assert [batch.tolist() for batch in batches] == [[0, 1, 2], [3, 4, 5], [6]]


def test_build_submission_frame_preserves_ids_and_rejects_unknown_labels():
    test = pd.DataFrame({"id": ["a", "b"], "text": ["x", "y"]})
    submission = build_submission_frame(
        test,
        np.asarray(["m0", "m1"], dtype=object),
        allowed_labels={"m0", "m1"},
    )
    assert submission.columns.tolist() == ["id", "model"]
    assert submission.to_dict("records") == [
        {"id": "a", "model": "m0"},
        {"id": "b", "model": "m1"},
    ]

    with pytest.raises(RuntimeError, match="outside official label set"):
        build_submission_frame(
            test,
            np.asarray(["m0", "bad"], dtype=object),
            allowed_labels={"m0", "m1"},
        )


def test_build_submission_lineage_is_plain_only_and_contamination_free():
    payload = build_submission_lineage(
        official_train_sha256="abc",
        test_rows=4000,
        base_manifest_sha256="base-manifest",
        runtime_manifest_sha256="runtime-manifest",
    )
    assert payload["candidate"] == "two-stage-lora24-v1"
    assert payload["official_training_rows"] == EXPECTED_OFFICIAL_ROWS
    assert payload["stage2_epochs"] == 2
    assert payload["primary_head"] == "plain"
    assert payload["centered_head_used_for_submission"] is False
    assert payload["competition_test_rows_used_for_training"] == 0
    assert payload["competition_test_rows_used_for_model_selection"] == 0
    assert payload["public_lb_tuning"] is False
    assert payload["test_rows"] == 4000
    assert payload["official_train_sha256"] == "abc"
    assert payload["base_manifest_sha256"] == "base-manifest"
    assert payload["runtime_manifest_sha256"] == "runtime-manifest"
