"""Exact optimizer-boundary checkpoint primitive for the frozen v3 CUDA path.

Infrastructure only: no dataset/architecture/optimizer/metric choices here.
A caller must build the same frozen model, optimizer and scheduler before
loading, supply a strict fingerprint, and resume at completed_steps.
Save only at a clean optimizer boundary, after GradScaler/scheduler updates
and optimizer.zero_grad(set_to_none=True), never within a triplet group.

Only load checkpoints produced by our own trusted workflow:
PyTorch checkpoints use pickle and are unsafe for untrusted input.
"""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np
import torch

SCHEMA_VERSION = 1


def _trainable(modules: Mapping[str, torch.nn.Module]) -> dict[str, torch.nn.Parameter]:
    named: dict[str, torch.nn.Parameter] = {}
    for module_name, module in modules.items():
        for param_name, parameter in module.named_parameters():
            if parameter.requires_grad:
                name = f"{module_name}.{param_name}"
                if name in named:
                    raise RuntimeError(f"Duplicate checkpoint parameter name {name}")
                named[name] = parameter
    if not named:
        raise RuntimeError("Refusing to checkpoint a model with no trainable parameters")
    return named


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda": None,
    }
    if state["cuda_available"]:
        state["cuda"] = [x.cpu().clone() for x in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    expected_cuda = bool(state["cuda_available"])
    if expected_cuda != bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA availability changed across checkpoint boundary")
    if expected_cuda:
        saved_states = state.get("cuda")
        if not isinstance(saved_states, list) or len(saved_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA device/RNG state count changed across checkpoint")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if expected_cuda:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_optimizer_boundary(
    path: Path,
    *,
    fingerprint: str,
    modules: Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    completed_steps: int,
    total_steps: int,
    training_stats: Mapping[str, int | float] | None = None,
) -> None:
    """Atomically store exact-resumable state. No partial accumulation allowed."""
    if not fingerprint or len(fingerprint) < 16:
        raise ValueError("Expected strict, non-empty frozen-run fingerprint")
    if not 0 <= completed_steps <= total_steps or total_steps <= 0:
        raise ValueError("Invalid completed/total optimizer step cursor")
    parameters = _trainable(modules)
    if any(param.grad is not None for param in parameters.values()):
        raise RuntimeError("Checkpoint requested mid-accumulation; gradients still present")
    payload = {
        "schema": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "completed_steps": int(completed_steps),
        "total_steps": int(total_steps),
        "training_stats": dict(training_stats or {}),
        "trainable_parameters": {
            name: parameter.detach().cpu().clone() for name, parameter in parameters.items()
        },
        "optimizer": _cpu_copy(optimizer.state_dict()),
        "scheduler": _cpu_copy(scheduler.state_dict()),
        "grad_scaler": _cpu_copy(scaler.state_dict()),
        "rng": _capture_rng(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as f:
            temp_path = Path(f.name)
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_optimizer_boundary(
    path: Path,
    *,
    expected_fingerprint: str,
    modules: Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    expected_total_steps: int,
    training_stats_out: MutableMapping[str, int | float] | None = None,
) -> int:
    """Restore trainables/optimizer/scheduler/scaler/RNG and return next cursor.

    Must call only on a newly reconstructed, identical frozen model.
    Does not restore nontrainable pretrained backbone weights; their revision
    must be independently pinned and included in the supplied fingerprint.
    """
    if not expected_fingerprint or len(expected_fingerprint) < 16:
        raise ValueError("Expected strict frozen-run fingerprint")
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA_VERSION:
        raise RuntimeError("Checkpoint schema mismatch")
    if payload.get("fingerprint") != expected_fingerprint:
        raise RuntimeError("Checkpoint lineage fingerprint mismatch")
    if payload.get("total_steps") != expected_total_steps:
        raise RuntimeError("Checkpoint total_steps mismatch")
    completed = payload.get("completed_steps")
    if type(completed) is not int or not 0 <= completed <= expected_total_steps:
        raise RuntimeError("Invalid checkpoint step cursor")
    saved_params = payload.get("trainable_parameters")
    params = _trainable(modules)
    if not isinstance(saved_params, dict) or saved_params.keys() != params.keys():
        raise RuntimeError("Trainable parameter names do not match checkpoint")
    for name, parameter in params.items():
        tensor = saved_params[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
            raise RuntimeError(f"Trainable parameter shape/dtype mismatch: {name}")
    # Validate CUDA count before mutating model/optimizer state.
    saved_rng = payload["rng"]
    if bool(saved_rng["cuda_available"]) != bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA availability mismatch")
    if torch.cuda.is_available() and len(saved_rng["cuda"]) != torch.cuda.device_count():
        raise RuntimeError("CUDA device count mismatch")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(saved_params[name].to(device=parameter.device))
            parameter.grad = None
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["grad_scaler"])
    _restore_rng(saved_rng)
    if training_stats_out is not None:
        stats = payload.get("training_stats", {})
        if not isinstance(stats, dict) or any(type(v) not in (int, float) for v in stats.values()):
            raise RuntimeError("Invalid training_stats payload")
        training_stats_out.clear()
        training_stats_out.update(stats)
    return completed
