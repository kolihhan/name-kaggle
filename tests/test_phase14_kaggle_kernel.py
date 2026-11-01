import json

from scripts.build_phase14_kaggle_kernel import (
    COMPETITION,
    EXPECTED_RAW_SHA256,
    KERNEL_REF,
    PUBLIC_DATASET,
    build_notebook_source,
    write_kernel_bundle,
)


def test_phase14_kernel_contract_is_frozen():
    assert KERNEL_REF == "kolihhan/name-that-llm-phase14-public-sparse"
    assert PUBLIC_DATASET == "nishantdahal/ntl-handwriting-blend"
    assert COMPETITION == "name-that-llm"
    assert EXPECTED_RAW_SHA256 == "37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec"


def test_phase14_notebook_source_contains_required_submission_guards():
    source = build_notebook_source("RUNNER_SENTINEL = 1", "SUBMISSION_SENTINEL = 2")
    assert "RUNNER_SENTINEL = 1" in source
    assert "SUBMISSION_SENTINEL = 2" in source
    assert EXPECTED_RAW_SHA256 in source
    assert "pool_v4.parquet" in source
    assert "build_conservative_subset" in source
    assert "train_full_and_predict" in source
    assert "submission.parquet" in source


def test_phase14_kernel_bundle_is_private_offline_and_has_exact_sources(tmp_path):
    runner = tmp_path / "runner.py"
    submission = tmp_path / "submission.py"
    runner.write_text("RUNNER_SENTINEL = 1\n", encoding="utf-8")
    submission.write_text("SUBMISSION_SENTINEL = 2\n", encoding="utf-8")

    out = tmp_path / "kernel"
    write_kernel_bundle(out, runner_path=runner, submission_path=submission)
    metadata = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["id"] == KERNEL_REF
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is False
    assert metadata["dataset_sources"] == [PUBLIC_DATASET]
    assert metadata["competition_sources"] == [COMPETITION]
    assert (out / metadata["code_file"]).is_file()
