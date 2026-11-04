"""Opt-in worker-resume interface and frozen-architecture source invariants.

This is a CPU/static check. CUDA equivalence must be verified separately.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import scripts.run_two_stage_lora24_gate_a_gpu as control
from scripts.materialize_dct_v3_worker import materialize


def test_shared_v1_worker_defaults_preserve_frozen_path():
    signature = inspect.signature(control._run_condition_worker)
    assert signature.parameters["optimizer_boundary_path"].default is None
    assert signature.parameters["checkpoint_every_steps"].default == 0
    assert signature.parameters["stop_after_optimizer_steps"].default is None
    with pytest.raises(ValueError, match="require an optimizer_boundary_path"):
        control._run_condition_worker(
            condition="A1",
            safe_path=Path("not-read.parquet"),
            snapshot_path=Path("not-read-model"),
            worker_output=Path("not-written.json"),
            checkpoint_dir=Path("not-written"),
            stop_after_optimizer_steps=1,
        )


def test_materialized_v3_contains_identical_resume_protocol_and_only_frozen_dct_change():
    original = inspect.getsource(control)
    v3 = materialize(original)
    compile(v3, "<materialized-v3-worker>", "exec")
    assert original.count("from scripts.dct_v3_cuda_resume import") == 1
    assert v3.count("from scripts.dct_v3_cuda_resume import") == 1
    for marker in (
        "optimizer_boundary_path: Path | None = None",
        "save_optimizer_boundary(",
        "load_optimizer_boundary(",
        "frozen-run",  # Checked by helper, not script; replaced below
    )[:3]:
        assert marker in original
        assert marker in v3
    assert "masked_dct_pool" not in original
    assert "masked_dct_pool" in v3
    assert 'CANDIDATE = "two-stage-lora24-v1"' in original
    assert 'CANDIDATE = "dct-temporal-fingerprint-v3"' in v3


def test_resume_protocol_reaches_optimizer_boundary_before_any_evaluation():
    code = inspect.getsource(control._run_condition_worker)
    assert code.index("scaler.step(optimizer)") < code.index("save_boundary(")
    assert code.index("save_boundary(") < code.index('if optimizer_steps != total_steps:')
    assert code.index("heldout UNTOUCHED") < code.index("_score_condition(")
    assert "torch.cuda.set_rng_state_all" in Path(
        inspect.getfile(control)
    ).with_name("dct_v3_cuda_resume.py").read_text()
