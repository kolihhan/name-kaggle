import json

import pandas as pd

from scripts.build_phase15_kaggle_kernel import (
    COMPETITION,
    EXPECTED_RAW_SHA256,
    KERNEL_REF,
    PUBLIC_DATASET,
    build_notebook_source,
    write_kernel_bundle,
)
from scripts.phase15_submission import build_submission_frame, train_full_and_predict


def test_phase15_full_train_allows_external_subset_of_official_labels():
    official = pd.DataFrame(
        {
            "text": [
                "alpha one",
                "alpha two",
                "beta one",
                "beta two",
                "gamma one",
                "gamma two",
            ],
            "model": ["a", "a", "b", "b", "c", "c"],
        }
    )
    external = pd.DataFrame(
        {
            "text": ["alpha external", "beta external"],
            "model": ["a", "b"],
        }
    )
    test = pd.DataFrame(
        {"id": ["x", "y", "z"], "text": ["alpha three", "beta three", "gamma three"]}
    )

    pred, meta = train_full_and_predict(
        official,
        external,
        test,
        min_df=1,
        feature_cap=100,
    )
    assert len(pred) == 3
    assert set(pred) <= {"a", "b", "c"}
    assert meta["official_training_rows"] == 6
    assert meta["external_training_rows"] == 2
    assert meta["total_training_rows"] == 8
    assert meta["test_rows"] == 3

    submission = build_submission_frame(test, pred, allowed_labels={"a", "b", "c"})
    assert submission.columns.tolist() == ["id", "model"]
    assert submission["id"].tolist() == ["x", "y", "z"]


def test_phase15_kernel_contract_is_frozen():
    assert KERNEL_REF == "kolihhan/name-that-llm-phase15-terms-clean"
    assert PUBLIC_DATASET == "nishantdahal/ntl-handwriting-blend"
    assert COMPETITION == "name-that-llm"
    assert EXPECTED_RAW_SHA256 == "37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec"


def test_phase15_notebook_source_rebuilds_both_frozen_subsets():
    source = build_notebook_source(
        "RUNNER_SENTINEL = 1",
        "TERMS_SENTINEL = 2",
        "SUBMISSION_SENTINEL = 3",
    )
    assert "RUNNER_SENTINEL = 1" in source
    assert "TERMS_SENTINEL = 2" in source
    assert "SUBMISSION_SENTINEL = 3" in source
    assert EXPECTED_RAW_SHA256 in source
    assert "pool_v4.parquet" in source
    assert "build_conservative_subset" in source
    assert "build_terms_clean_subset" in source
    assert "EXPECTED_TERMS_CLEAN_ROWS" in source
    assert "train_full_and_predict" in source
    assert "submission.parquet" in source


def test_phase15_kernel_bundle_is_private_offline_and_has_exact_sources(tmp_path):
    runner = tmp_path / "runner.py"
    terms = tmp_path / "terms.py"
    submission = tmp_path / "submission.py"
    runner.write_text("RUNNER_SENTINEL = 1\n", encoding="utf-8")
    terms.write_text("TERMS_SENTINEL = 2\n", encoding="utf-8")
    submission.write_text("SUBMISSION_SENTINEL = 3\n", encoding="utf-8")

    out = tmp_path / "kernel"
    write_kernel_bundle(
        out,
        runner_path=runner,
        terms_path=terms,
        submission_path=submission,
    )
    metadata = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["id"] == KERNEL_REF
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is False
    assert metadata["dataset_sources"] == [PUBLIC_DATASET]
    assert metadata["competition_sources"] == [COMPETITION]
    assert (out / metadata["code_file"]).is_file()
