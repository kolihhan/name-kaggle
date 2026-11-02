import pytest

from scripts.materialize_two_stage_lora24_offline_inputs import (
    BASE_DATASET_REF,
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_DATASET_REF,
    WHEEL_SPECS,
    assert_dataset_ref_absent,
    build_base_lineage,
    build_file_manifest,
    build_runtime_lineage,
    validate_flat_package,
)


def test_file_manifest_is_sorted_and_byte_sensitive(tmp_path):
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    files, digest = build_file_manifest(tmp_path)
    assert [entry["path"] for entry in files] == ["a.txt", "b.txt"]
    assert len(digest) == 64

    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    _, changed_digest = build_file_manifest(tmp_path)
    assert changed_digest != digest


def test_base_lineage_pins_exact_frozen_model_and_excludes_competition_data():
    payload = build_base_lineage(files=[], manifest_sha256="abc")
    assert payload["model_id"] == MODEL_ID == "Qwen/Qwen3-8B-Base"
    assert payload["model_revision"] == MODEL_REVISION
    assert payload["dataset_ref"] == BASE_DATASET_REF
    assert payload["competition_train_rows_included"] == 0
    assert payload["competition_test_rows_included"] == 0
    assert payload["file_manifest_sha256"] == "abc"


def test_runtime_lineage_pins_scientific_runtime_versions_and_no_competition_data():
    payload = build_runtime_lineage(files=[], manifest_sha256="def")
    assert payload["dataset_ref"] == RUNTIME_DATASET_REF
    assert "transformers==4.57.1" in WHEEL_SPECS
    assert "peft==0.17.1" in WHEEL_SPECS
    assert payload["wheel_specs"] == list(WHEEL_SPECS)
    assert payload["competition_train_rows_included"] == 0
    assert payload["competition_test_rows_included"] == 0
    assert payload["file_manifest_sha256"] == "def"


def test_refuse_existing_dataset_ref_is_fail_closed():
    csv = "ref,title\nkolihhan/name-that-llm-qwen3-8b-base-49e3418f,existing\n"
    with pytest.raises(RuntimeError, match="refusing overwrite/version"):
        assert_dataset_ref_absent(csv, BASE_DATASET_REF)
    assert_dataset_ref_absent("ref,title\nother/x,y\n", BASE_DATASET_REF)


def test_dataset_package_refuses_nested_payload_files(tmp_path):
    (tmp_path / "root.whl").write_text("ok", encoding="utf-8")
    validate_flat_package(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "hidden.bin").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="nested payload"):
        validate_flat_package(tmp_path)
