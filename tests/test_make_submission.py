import numpy as np
import pandas as pd
import pytest

from scripts.make_submission import build_submission_frame, validate_test_frame


def test_build_submission_frame_preserves_ids_and_phase7_predictions():
    test = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "text": ["one", "two", "three"],
        }
    )
    predictions = np.array(["gpt-5.5", "gemini-3.8-flash", "claude-opus-5"], dtype=object)
    allowed = {"gpt-5.5", "gemini-3.8-flash", "claude-opus-5"}

    submission = build_submission_frame(test, predictions, allowed_labels=allowed)

    assert submission.columns.tolist() == ["id", "model"]
    assert submission["id"].tolist() == ["a", "b", "c"]
    assert submission["model"].tolist() == predictions.tolist()


def test_build_submission_frame_rejects_unknown_labels():
    test = pd.DataFrame({"id": ["a"], "text": ["one"]})

    with pytest.raises(RuntimeError, match="outside official label set"):
        build_submission_frame(
            test,
            np.array(["not-a-real-model"], dtype=object),
            allowed_labels={"gpt-5.5"},
        )


def test_validate_test_frame_rejects_duplicate_ids():
    test = pd.DataFrame({"id": ["a", "a"], "text": ["one", "two"]})

    with pytest.raises(RuntimeError, match="duplicate ids"):
        validate_test_frame(test)
