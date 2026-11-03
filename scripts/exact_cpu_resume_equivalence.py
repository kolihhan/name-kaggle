from __future__ import annotations

import copy
import json
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from scripts.exact_cpu_resume_state import (
    capture_resume_state,
    load_resume_state,
    restore_resume_state,
    save_resume_state,
)

ACCUMULATION_SIZE = 6
MICROBATCHES = 13
INTERRUPT_AFTER = 7
SEED = 20261103


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_training_objects(seed: int = SEED):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.GELU(),
        nn.Dropout(0.25),
        nn.Linear(8, 3),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 / (step + 1)
    )
    return model, optimizer, scheduler


def _microbatches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(987654)
    batches = []
    for index in range(MICROBATCHES):
        x = torch.randn(3, 4, generator=generator)
        y = torch.tensor(
            [index % 3, (index + 1) % 3, (index + 2) % 3], dtype=torch.long
        )
        batches.append((x, y))
    return batches


def _train_until(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    cursor: int,
    accumulation_position: int,
    stop_cursor: int,
) -> tuple[int, int]:
    if not 0 <= cursor <= stop_cursor <= len(batches):
        raise ValueError("invalid cursor range")
    if not 0 <= accumulation_position < ACCUMULATION_SIZE:
        raise ValueError("invalid accumulation position")

    model.train()
    while cursor < stop_cursor:
        group_start = (cursor // ACCUMULATION_SIZE) * ACCUMULATION_SIZE
        group_end = min(group_start + ACCUMULATION_SIZE, len(batches))
        group_size = group_end - group_start
        expected_position = cursor - group_start
        if accumulation_position != expected_position:
            raise RuntimeError(
                f"accumulation cursor drift: {accumulation_position} != {expected_position}"
            )
        if accumulation_position == 0:
            optimizer.zero_grad(set_to_none=True)

        x, y = batches[cursor]
        loss = torch.nn.functional.cross_entropy(model(x), y)
        (loss / group_size).backward()
        cursor += 1
        accumulation_position += 1

        if cursor == group_end:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            accumulation_position = 0

    return cursor, accumulation_position


def _nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _parameter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def run_equivalence_probe(work_dir: Path) -> dict[str, Any]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    batches = _microbatches()

    _seed_all(SEED)
    baseline_model, baseline_optimizer, baseline_scheduler = _build_training_objects()
    baseline_cursor, baseline_position = _train_until(
        model=baseline_model,
        optimizer=baseline_optimizer,
        scheduler=baseline_scheduler,
        batches=batches,
        cursor=0,
        accumulation_position=0,
        stop_cursor=len(batches),
    )
    baseline_parameters = _parameter_state(baseline_model)
    baseline_optimizer_state = copy.deepcopy(baseline_optimizer.state_dict())
    baseline_scheduler_state = copy.deepcopy(baseline_scheduler.state_dict())
    baseline_rng = (random.random(), float(np.random.rand()), torch.rand(5))

    _seed_all(SEED)
    first_model, first_optimizer, first_scheduler = _build_training_objects()
    cursor, position = _train_until(
        model=first_model,
        optimizer=first_optimizer,
        scheduler=first_scheduler,
        batches=batches,
        cursor=0,
        accumulation_position=0,
        stop_cursor=INTERRUPT_AFTER,
    )
    if cursor != INTERRUPT_AFTER or position != 1:
        raise RuntimeError("probe did not stop at the intended mid-accumulation point")

    checkpoint = capture_resume_state(
        modules={"model": first_model},
        optimizer=first_optimizer,
        scheduler=first_scheduler,
        cursor=cursor,
        accumulation_position=position,
        accumulation_size=ACCUMULATION_SIZE,
        metadata={"candidate": "synthetic-equivalence", "scope": "resume-proof"},
    )
    checkpoint_path = work_dir / "resume.pt"
    save_resume_state(checkpoint_path, checkpoint)

    second_model, second_optimizer, second_scheduler = _build_training_objects(seed=999)
    restored = restore_resume_state(
        load_resume_state(checkpoint_path),
        modules={"model": second_model},
        optimizer=second_optimizer,
        scheduler=second_scheduler,
        expected_metadata={"candidate": "synthetic-equivalence", "scope": "resume-proof"},
    )
    resumed_cursor, resumed_position = _train_until(
        model=second_model,
        optimizer=second_optimizer,
        scheduler=second_scheduler,
        batches=batches,
        cursor=int(restored["cursor"]),
        accumulation_position=int(restored["accumulation_position"]),
        stop_cursor=len(batches),
    )
    resumed_rng = (random.random(), float(np.random.rand()), torch.rand(5))

    parameters_equal = _nested_equal(baseline_parameters, _parameter_state(second_model))
    optimizer_equal = _nested_equal(
        baseline_optimizer_state, second_optimizer.state_dict()
    )
    scheduler_equal = _nested_equal(
        baseline_scheduler_state, second_scheduler.state_dict()
    )
    next_python_rng_equal = baseline_rng[0] == resumed_rng[0]
    next_numpy_rng_equal = baseline_rng[1] == resumed_rng[1]
    next_torch_rng_equal = torch.equal(baseline_rng[2], resumed_rng[2])

    result = {
        "microbatches": MICROBATCHES,
        "accumulation_size": ACCUMULATION_SIZE,
        "interrupted_after_microbatch": INTERRUPT_AFTER,
        "baseline_cursor": baseline_cursor,
        "baseline_accumulation_position": baseline_position,
        "final_cursor": resumed_cursor,
        "final_accumulation_position": resumed_position,
        "parameters_equal": parameters_equal,
        "optimizer_equal": optimizer_equal,
        "scheduler_equal": scheduler_equal,
        "next_python_rng_equal": next_python_rng_equal,
        "next_numpy_rng_equal": next_numpy_rng_equal,
        "next_torch_rng_equal": next_torch_rng_equal,
    }
    result["all_equal"] = bool(
        parameters_equal
        and optimizer_equal
        and scheduler_equal
        and next_python_rng_equal
        and next_numpy_rng_equal
        and next_torch_rng_equal
        and baseline_cursor == resumed_cursor == MICROBATCHES
        and baseline_position == resumed_position == 0
    )
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="exact-cpu-resume-") as tmp:
        result = run_equivalence_probe(Path(tmp))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_equal"]:
        raise SystemExit("exact resume equivalence failed")


if __name__ == "__main__":
    main()
