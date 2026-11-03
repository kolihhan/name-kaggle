from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from typing import Any


_BACKENDS: dict[str, dict[str, object]] = {
    "fp16": {
        "name": "fp16",
        "model_dtype": "float16",
        "autocast": False,
        "autocast_dtype": None,
        "grad_scaler": False,
    },
    "bf16": {
        "name": "bf16",
        "model_dtype": "bfloat16",
        "autocast": True,
        "autocast_dtype": "bfloat16",
        "grad_scaler": False,
    },
}


def backend_runtime_spec(name: str) -> dict[str, object]:
    try:
        return dict(_BACKENDS[name])
    except KeyError as exc:
        raise ValueError(f"unsupported CPU shadow backend: {name}") from exc


def backend_lineage_tag(name: str) -> str:
    spec = backend_runtime_spec(name)
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def torch_model_dtype(torch: Any, name: str):
    spec = backend_runtime_spec(name)
    return torch.float16 if spec["model_dtype"] == "float16" else torch.bfloat16


def cpu_autocast_context(torch: Any, name: str):
    spec = backend_runtime_spec(name)
    if not bool(spec["autocast"]):
        return nullcontext()
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
