from __future__ import annotations

import re
from pathlib import Path


WORKFLOW = Path(".github/workflows/two-stage-lora24-gate-b-checkpoint.yml")


def test_gate_b_checkpoint_workflow_is_exact_private_transport_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "41333631967" in text
    assert "13094069155" in text
    assert "sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66" in text
    assert "two-stage-lora24-gate-a-a1-checkpoint" in text
    assert "kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint" in text
    assert "[gate-b-checkpoint]" in text

    lowered = text.lower()
    assert "kaggle kernels push" not in lowered
    assert "run_two_stage_lora24_gate_a_gpu" not in text
    assert "kaggle datasets version" not in lowered
    assert "--public" not in text


def test_gate_b_checkpoint_workflow_installs_kaggle_runtime_dependency_six():
    text = WORKFLOW.read_text(encoding="utf-8")
    install_line = next(
        line
        for line in text.splitlines()
        if "pip install -r requirements.txt" in line and "kaggle==2.2.5" in line
    )
    assert "six" in install_line.split()


def test_gate_b_checkpoint_dataset_title_meets_kaggle_length_contract():
    text = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"'title': '([^']+)'", text)
    assert match is not None
    assert 6 <= len(match.group(1)) <= 50


def test_gate_b_checkpoint_workflow_verifies_lineage_before_dataset_create():
    text = WORKFLOW.read_text(encoding="utf-8")
    verify_pos = text.index("Verify exact Gate-A artifact and checkpoint lineage")
    create_pos = text.index("Create private immutable Kaggle checkpoint dataset")
    assert verify_pos < create_pos
    assert "validate_gate_a_checkpoint_meta" in text
    assert "checkpoint_lineage.json" in text
    assert "file_manifest_sha256" in text


def test_gate_b_checkpoint_workflow_fails_closed_if_dataset_slug_already_exists():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "Refuse to overwrite or version an existing checkpoint dataset" in text
    assert "kaggle datasets list" in text
    assert "exit 1" in text
