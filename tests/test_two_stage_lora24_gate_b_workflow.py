from __future__ import annotations

import re
from pathlib import Path


WORKFLOW = Path(".github/workflows/two-stage-lora24-gate-b.yml")


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _kaggle_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def test_gate_b_workflow_requires_dedicated_run_marker_for_heavy_job():
    text = _workflow_text()
    assert "[gate-b-run]" in text
    assert "contains(github.event.head_commit.message, '[gate-b-run]')" in text
    assert "kaggle-gpu:" in text


def test_gate_b_kernel_title_resolves_to_declared_kernel_id():
    text = _workflow_text()
    ref_match = re.search(r'KERNEL_REF="\$\{USERNAME\}/([^"]+)"', text)
    title_match = re.search(r'"title": "([^"]+)"', text)
    assert ref_match is not None
    assert title_match is not None
    assert title_match.group(1) == "Name That LLM Two-stage LoRA24 Gate B"
    assert _kaggle_slug(title_match.group(1)) == ref_match.group(1)


def test_gate_b_workflow_attaches_only_locked_checkpoint_dataset_and_no_competition_source():
    text = _workflow_text()
    assert '"competition_sources": []' in text
    assert '"dataset_sources": ["kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint"]' in text
    assert "nishantdahal/ntl-handwriting-blend" not in text
    assert "pool_v4.parquet" not in text


def test_gate_b_workflow_embeds_only_official_train_and_permanent_folds_not_test_data():
    text = _workflow_text()
    assert 'pathlib.Path("data/train.csv")' in text
    assert 'pathlib.Path("artifacts/phase1/folds.csv")' in text
    assert "data/test.csv" not in text
    assert "sample_submission" not in text


def test_gate_b_workflow_never_submits_to_competition():
    text = _workflow_text().lower()
    assert "kaggle competitions submit" not in text
    assert "competition_test_rows_used" in text
    assert "kaggle_submission_authorized" in text


def test_gate_b_workflow_verifies_complete_oof_and_locked_checkpoint_lineage():
    text = _workflow_text()
    assert "2699" in text
    assert "bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d" in text
    assert "c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0" in text
    assert "13094069155" in text
    assert "a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66" in text
    assert "gate_b_candidate_oof.csv" in text
    for fold in range(5):
        assert f"fold_{fold}.json" in text
        assert f"fold_{fold}_predictions.csv" in text


def test_gate_b_workflow_uploads_evidence_only_not_stage2_checkpoints():
    text = _workflow_text()
    assert "two-stage-lora24-gate-b-evidence" in text
    assert "gate_b_candidate_oof.csv" in text
    assert "stage2-checkpoint" not in text.lower()
    assert "save_pretrained" not in text
