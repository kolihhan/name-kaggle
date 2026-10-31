from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.make_submission import (
    build_submission_frame,
    predict_phase7,
    validate_test_frame,
)
from scripts.run_gated_specialist import (
    EXPECTED_COVERED_LABELS,
    gate_predictions,
    specialist_training_rows,
)
from scripts.run_strong_sparse_anchor import DEFAULT_SEED, build_fold_features, make_svc


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


def test_predict_phase7_matches_reference_pipeline_on_committed_test():
    train = pd.read_csv(Path("data/train.csv"))
    external = pd.read_parquet(Path("data/external/gpt_gemini_generations.parquet"))
    test = pd.read_csv(Path("data/test.csv"))

    observed = predict_phase7(train, external, test, seed=DEFAULT_SEED)

    x_anchor, x_test, _ = build_fold_features(
        train["text"].astype(str).tolist(),
        test["text"].astype(str).tolist(),
    )
    anchor = make_svc(DEFAULT_SEED)
    anchor.fit(x_anchor, train["model"].astype(str))
    anchor_predictions = anchor.predict(x_test)

    specialist_text, specialist_y = specialist_training_rows(
        train,
        external,
        covered_labels=EXPECTED_COVERED_LABELS,
    )
    x_specialist, x_specialist_test, _ = build_fold_features(
        specialist_text,
        test["text"].astype(str).tolist(),
    )
    specialist = make_svc(DEFAULT_SEED)
    specialist.fit(x_specialist, specialist_y)
    specialist_predictions = specialist.predict(x_specialist_test)

    expected = gate_predictions(
        anchor_predictions,
        specialist_predictions,
        EXPECTED_COVERED_LABELS,
    )
    assert observed.tolist() == expected.tolist()
