from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

SCHEMA_VERSION = 2


def _named_parameters(modules: Mapping[str, nn.Module]) -> dict[str, nn.Parameter]:
    named: dict[str, nn.Parameter] = {}
    seen: set[int] = set()
    for module_name, module in modules.items():
        if not module_name:
            raise ValueError("module names must be non-empty")
        for parameter_name, parameter in module.named_parameters():
            full_name = f"{module_name}.{parameter_name}"
            if full_name in named:
                raise RuntimeError(f"duplicate parameter name: {full_name}")
            if id(parameter) in seen:
                raise RuntimeError(f"parameter shared across module mapping: {full_name}")
            named[full_name] = parameter
            seen.add(id(parameter))
    return named


def _parameter_contract(named: Mapping[str, nn.Parameter]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in named.items()
    }


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _validate_optimizer_covers_trainables(
    named: Mapping[str, nn.Parameter], optimizer: torch.optim.Optimizer
) -> None:
    trainable_ids = {id(parameter) for parameter in named.values() if parameter.requires_grad}
    optimizer_ids = _optimizer_parameter_ids(optimizer)
    if optimizer_ids != trainable_ids:
        missing = sorted(
            name
            for name, parameter in named.items()
            if parameter.requires_grad and id(parameter) not in optimizer_ids
        )
        unexpected = sum(1 for parameter_id in optimizer_ids if parameter_id not in trainable_ids)
        raise RuntimeError(
            "optimizer/trainable parameter contract drift: "
            f"missing={missing[:10]} unexpected_count={unexpected}"
        )


def _validate_progress(*, cursor: int, accumulation_position: int, accumulation_size: int) -> None:
    if int(cursor) < 0:
        raise ValueError("cursor must be non-negative")
    if int(accumulation_size) <= 0:
        raise ValueError("accumulation_size must be positive")
    if not 0 <= int(accumulation_position) < int(accumulation_size):
        raise ValueError(
            "accumulation_position must satisfy 0 <= position < accumulation_size"
        )


def capture_resume_state(
    *,
    modules: Mapping[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None = None,
    cursor: int,
    accumulation_position: int,
    accumulation_size: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_progress(
        cursor=cursor,
        accumulation_position=accumulation_position,
        accumulation_size=accumulation_size,
    )
    named = _named_parameters(modules)
    _validate_optimizer_covers_trainables(named, optimizer)
    trainable_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in named.items()
        if parameter.requires_grad
    }
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in named.items()
        if parameter.requires_grad and parameter.grad is not None
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": dict(metadata),
        "progress": {
            "cursor": int(cursor),
            "accumulation_position": int(accumulation_position),
            "accumulation_size": int(accumulation_size),
        },
        # Keep the full parameter contract to detect base-model or module drift,
        # but persist values only for parameters that training can mutate.
        "parameter_contract": _parameter_contract(named),
        "trainable_parameters": trainable_parameters,
        "gradients": gradients,
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": (
                [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
        },
    }


def save_resume_state(path: Path, state: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state), path)


def load_resume_state(path: Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("resume state must deserialize to a dictionary")
    return payload


def restore_resume_state(
    state: Mapping[str, Any],
    *,
    modules: Mapping[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if int(state.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("resume-state schema drift")

    metadata = state.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("resume state metadata is missing")
    for key, expected in dict(expected_metadata or {}).items():
        observed = metadata.get(key)
        if observed != expected:
            raise RuntimeError(
                f"resume metadata drift for {key}: {observed!r} != {expected!r}"
            )

    progress = state.get("progress")
    if not isinstance(progress, dict):
        raise RuntimeError("resume progress is missing")
    cursor = int(progress.get("cursor", -1))
    accumulation_position = int(progress.get("accumulation_position", -1))
    accumulation_size = int(progress.get("accumulation_size", -1))
    _validate_progress(
        cursor=cursor,
        accumulation_position=accumulation_position,
        accumulation_size=accumulation_size,
    )

    named = _named_parameters(modules)
    _validate_optimizer_covers_trainables(named, optimizer)
    expected_contract = state.get("parameter_contract")
    observed_contract = _parameter_contract(named)
    if expected_contract != observed_contract:
        raise RuntimeError("parameter contract drift")

    saved_trainable = state.get("trainable_parameters")
    if not isinstance(saved_trainable, dict):
        raise RuntimeError("checkpoint trainable parameter state is missing")
    expected_trainable_names = {
        name for name, parameter in named.items() if parameter.requires_grad
    }
    if set(saved_trainable) != expected_trainable_names:
        raise RuntimeError("trainable parameter state contract drift")
    with torch.no_grad():
        for name, tensor in saved_trainable.items():
            parameter = named[name]
            if list(tensor.shape) != list(parameter.shape):
                raise RuntimeError(f"trainable parameter shape drift for {name}")
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))

    optimizer.load_state_dict(state["optimizer"])
    saved_scheduler = state.get("scheduler")
    if scheduler is None:
        if saved_scheduler is not None:
            raise RuntimeError("checkpoint contains scheduler state but no scheduler was supplied")
    else:
        if saved_scheduler is None:
            raise RuntimeError("checkpoint is missing scheduler state")
        scheduler.load_state_dict(saved_scheduler)

    saved_scaler = state.get("scaler")
    if scaler is None:
        if saved_scaler is not None:
            raise RuntimeError("checkpoint contains scaler state but no scaler was supplied")
    else:
        if saved_scaler is None:
            raise RuntimeError("checkpoint is missing scaler state")
        scaler.load_state_dict(saved_scaler)

    for parameter in named.values():
        parameter.grad = None
    saved_gradients = state.get("gradients")
    if not isinstance(saved_gradients, dict):
        raise RuntimeError("checkpoint gradients are missing")
    unknown_gradients = set(saved_gradients) - set(named)
    if unknown_gradients:
        raise RuntimeError(f"checkpoint contains unknown gradients: {sorted(unknown_gradients)}")
    for name, gradient in saved_gradients.items():
        parameter = named[name]
        if not parameter.requires_grad:
            raise RuntimeError(f"checkpoint gradient targets frozen parameter: {name}")
        if list(gradient.shape) != list(parameter.shape):
            raise RuntimeError(f"gradient shape drift for {name}")
        parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype).clone()

    rng = state.get("rng")
    if not isinstance(rng, dict):
        raise RuntimeError("checkpoint RNG state is missing")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    cuda_rng = rng.get("torch_cuda")
    if cuda_rng is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_rng)

    return {
        "cursor": cursor,
        "accumulation_position": accumulation_position,
        "accumulation_size": accumulation_size,
        "metadata": dict(metadata),
        "restored_gradient_count": len(saved_gradients),
        "restored_trainable_parameter_count": len(saved_trainable),
        "scaler_restored": scaler is not None,
    }
