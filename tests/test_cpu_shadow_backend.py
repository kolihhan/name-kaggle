from __future__ import annotations

import pytest

from scripts.cpu_shadow_backend import (
    backend_runtime_spec,
    backend_lineage_tag,
    prepare_fp32_head_input,
)


def test_fp16_backend_contract_is_explicit():
    spec = backend_runtime_spec("fp16")
    assert spec == {
        "name": "fp16",
        "model_dtype": "float16",
        "autocast": False,
        "autocast_dtype": None,
        "grad_scaler": False,
    }


def test_bf16_backend_contract_uses_cpu_autocast_without_grad_scaler():
    spec = backend_runtime_spec("bf16")
    assert spec == {
        "name": "bf16",
        "model_dtype": "bfloat16",
        "autocast": True,
        "autocast_dtype": "bfloat16",
        "grad_scaler": False,
    }


def test_backend_lineage_tags_are_distinct():
    assert backend_lineage_tag("fp16") != backend_lineage_tag("bf16")


def test_unknown_backend_fails_closed():
    with pytest.raises(ValueError, match="unsupported CPU shadow backend"):
        backend_runtime_spec("int8")


def test_classifier_head_input_is_promoted_to_fp32():
    class DummyTensor:
        def __init__(self):
            self.float_called = False

        def float(self):
            self.float_called = True
            return "fp32-view"

    tensor = DummyTensor()
    assert prepare_fp32_head_input(tensor) == "fp32-view"
    assert tensor.float_called is True
