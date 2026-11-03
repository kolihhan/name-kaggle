from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

SCHEMA_VERSION = 1


def _named_parameters(
    modules: Mapping[str, torch.nn.Module], *, trainable_only: bool = False
):
    for module_name, module in modules.items():
        for param_name, parameter in module.named_parameters():
            if trainable_only and not parameter.requires_grad:
                continue
            yield f"{module_name}.{param_name}", parameter


def _capture_trainable_parameters(
    modules: Mapping[str, torch.nn.Module],
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in _named_parameters(modules, trainable_only=True)
    }


def _restore_trainable_parameters(
    modules: Mapping[str, torch.nn.Module],
    state: Mapping[str, torch.Tensor],
) -> None:
    parameters = dict(_named_parameters(modules, trainable_only=True))
    if set(state) != set(parameters):
        raise RuntimeError(
            "CPU shadow trainable parameter set mismatch: "
            f"saved={len(state)} current={len(parameters)}"
        )
    with torch.no_grad():
        for name, tensor in state.items():
            parameter = parameters[name]
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise RuntimeError(f"CPU shadow parameter shape mismatch: {name}")
            parameter.copy_(
                tensor.to(device=parameter.device, dtype=parameter.dtype)
            )


def _capture_gradients(
    modules: Mapping[str, torch.nn.Module],
) -> dict[str, torch.Tensor]:
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in _named_parameters(modules, trainable_only=True):
        if parameter.grad is not None:
            gradients[name] = parameter.grad.detach().cpu().clone()
    return gradients


def _restore_gradients(
    modules: Mapping[str, torch.nn.Module],
    gradients: Mapping[str, torch.Tensor],
) -> None:
    parameters = dict(_named_parameters(modules, trainable_only=True))
    unknown = sorted(set(gradients) - set(parameters))
    if unknown:
        raise RuntimeError(
            f"checkpoint gradients reference unknown parameters: {unknown[:5]}"
        )
    for parameter in parameters.values():
        parameter.grad = None
    for name, gradient in gradients.items():
        parameter = parameters[name]
        parameter.grad = gradient.to(
            device=parameter.device, dtype=parameter.dtype
        ).clone()


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def save_shadow_checkpoint(
    path: Path,
    *,
    lineage_fingerprint: str,
    modules: Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    progress: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "lineage_fingerprint": str(lineage_fingerprint),
        "trainable_parameters": _capture_trainable_parameters(modules),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "gradients": _capture_gradients(modules),
        "rng": _capture_rng_state(),
        "progress": dict(progress),
    }
    torch.save(payload, path)


def load_shadow_checkpoint(
    path: Path,
    *,
    expected_lineage_fingerprint: str,
    modules: Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("unsupported CPU shadow checkpoint schema")

    observed_lineage = str(payload.get("lineage_fingerprint", ""))
    if observed_lineage != str(expected_lineage_fingerprint):
        raise RuntimeError(
            "CPU shadow checkpoint lineage mismatch: "
            f"{observed_lineage!r} != {expected_lineage_fingerprint!r}"
        )

    _restore_trainable_parameters(
        modules, payload.get("trainable_parameters", {})
    )
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _restore_gradients(modules, payload.get("gradients", {}))
    _restore_rng_state(payload["rng"])
    return dict(payload.get("progress", {}))
