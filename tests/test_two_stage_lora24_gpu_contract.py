from __future__ import annotations

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
