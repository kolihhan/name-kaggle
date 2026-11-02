from __future__ import annotations

from pathlib import Path

# Recovery trigger is intentionally test-only; it must not create/version the dataset.
RUNTIME_WORKFLOW = Path(".github/workflows/two-stage-lora24-gate-b-runtime.yml")
GATE_B_WORKFLOW = Path(".github/workflows/two-stage-lora24-gate-b.yml")


def test_runtime_workflow_materializes_one_private_immutable_version1_dataset():
    assert RUNTIME_WORKFLOW.is_file()
    text = RUNTIME_WORKFLOW.read_text(encoding="utf-8")
    assert "[gate-b-runtime]" in text
    assert "name-that-llm-two-stage-lora24-v1-gate-b-runtime" in text
    assert "kaggle datasets create" in text
    assert "kaggle datasets version" not in text
    assert "Refuse to overwrite or version an existing runtime dataset" in text


def test_runtime_recovery_is_attach_only_and_locks_created_version1_bytes():
    text = RUNTIME_WORKFLOW.read_text(encoding="utf-8")
    assert "[gate-b-runtime-recover]" in text
    assert "recover-existing-runtime:" in text
    recovery = text.split("recover-existing-runtime:", 1)[1]
    assert "kaggle datasets create" not in recovery
    assert "kaggle datasets version" not in recovery
    assert "1d466f4c756394aea4b57f056238ff457f9f0852" in recovery
    assert "8d6c05d5ec29841d4a25cbf7c903c5d0414d3bf754b4bdd80b1cfdba64c1a933" in recovery
    assert "f818750300838089f3610347074da52a8340d74a57f54b698ce6c65ea68c3785" in recovery
    assert "dataset_create_called': False" in recovery
    assert "dataset_version_called': False" in recovery
    assert "gate_b_training_called': False" in recovery


def test_runtime_dataset_contains_only_frozen_gate_b_runtime_inputs():
    text = RUNTIME_WORKFLOW.read_text(encoding="utf-8")
    required = [
        "data/train.csv",
        "artifacts/phase1/folds.csv",
        "scripts/__init__.py",
        "scripts/run_two_stage_lora24_gate_b_gpu.py",
        "scripts/run_two_stage_lora24_gate_b.py",
        "scripts/run_two_stage_lora24_gate_a.py",
        "scripts/run_style_baseline.py",
        "scripts/run_strong_sparse_anchor.py",
    ]
    for path in required:
        assert path in text
    assert "data/test.csv" not in text
    assert "sample_submission" not in text
    assert "nishantdahal/ntl-handwriting-blend" not in text
    assert "kaggle competitions submit" not in text.lower()


def test_runtime_dataset_locks_official_hashes_and_source_commit():
    text = RUNTIME_WORKFLOW.read_text(encoding="utf-8")
    assert "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707" in text
    assert "bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d" in text
    assert "runtime_lineage.json" in text
    assert "file_manifest_sha256" in text
    assert "transport_sha256" in text


def test_gate_b_kernel_uses_runtime_dataset_instead_of_embedding_official_csvs():
    text = GATE_B_WORKFLOW.read_text(encoding="utf-8")
    assert "name-that-llm-two-stage-lora24-v1-gate-b-runtime" in text
    assert 'pathlib.Path("data/train.csv")' not in text
    assert 'pathlib.Path("artifacts/phase1/folds.csv")' not in text
    assert "gate-b-runtime.tar.gz" in text
    assert "runtime_lineage.json" in text


def test_gate_b_kernel_fails_closed_before_push_if_source_reaches_kaggle_limit():
    text = GATE_B_WORKFLOW.read_text(encoding="utf-8")
    assert "kernel_main_bytes" in text
    assert "1000000" in text
    assert "Kaggle kernel source must stay below 1,000,000 bytes" in text


def test_gate_b_still_attaches_exact_checkpoint_dataset_without_versioning_it():
    text = GATE_B_WORKFLOW.read_text(encoding="utf-8")
    assert "name-that-llm-two-stage-lora24-v1-a1-checkpoint" in text
    assert "current_version_number']) == 1" in text
    assert "kaggle datasets version" not in text
