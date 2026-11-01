from __future__ import annotations

import inspect
from pathlib import Path

from scripts import run_two_stage_lora24_gate_a_gpu as gpu


def test_gpu_runner_pins_exact_qwen3_base_revision():
    assert gpu.MODEL_ID == "Qwen/Qwen3-8B-Base"
    assert gpu.MODEL_REVISION == "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
    assert gpu.TRANSFORMERS_VERSION == "4.57.1"
    assert gpu.PEFT_VERSION == "0.17.1"


def test_gpu_runner_scope_is_external_only_and_never_submits():
    assert gpu.execution_scope() == {
        "scope": "external_transfer_gate_a_only_no_official_oof",
        "official_rows_scored": 0,
        "official_oof_runs": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "kaggle_submission_authorized": False,
    }


def test_gpu_runner_conditions_are_frozen_a0_a1_only():
    assert gpu.CONDITIONS == ("A0", "A1")


def test_gpu_runner_consumes_exactly_one_epoch_without_triplet_padding():
    source = inspect.getsource(gpu._run_condition_worker)
    assert "group_triplets_for_accumulation" in source
    assert "pad_triplets_for_accumulation" not in source


def test_gpu_runner_retains_pinned_qwen_stock_final_norm():
    source = inspect.getsource(gpu._run_condition_worker)
    assert "base.norm = nn.Identity()" not in source


def test_gate_a_recovery_attaches_to_existing_kernel_without_pushing_new_version():
    workflow = Path(".github/workflows/two-stage-lora24-gate-a.yml").read_text(encoding="utf-8")
    assert "[gate-a-recover]" in workflow
    assert "recover-existing-kaggle-gate-a:" in workflow
    recovery = workflow.split("recover-existing-kaggle-gate-a:", 1)[1]
    assert "kaggle kernels push" not in recovery
    assert 'kaggle kernels status "$KERNEL_REF"' in recovery
    assert 'kaggle kernels output "$KERNEL_REF"' in recovery
