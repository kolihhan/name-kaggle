import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.exact_cpu_resume_state import (
    capture_resume_state,
    load_resume_state,
    restore_resume_state,
    save_resume_state,
)


def _build(seed: int = 17):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    return model, optimizer, scheduler


def _one_backward(model, x, y):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    return float(loss.detach())


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_mid_accumulation_round_trip_preserves_gradients_optimizer_scheduler_and_cursor(tmp_path: Path):
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    model, optimizer, scheduler = _build()
    x = torch.tensor([[0.1, 0.2, -0.3, 0.4], [0.5, -0.6, 0.7, 0.8]])
    y = torch.tensor([1, 2])

    _one_backward(model, x, y)
    state = capture_resume_state(
        modules={"model": model},
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=7,
        accumulation_position=1,
        accumulation_size=6,
        metadata={"candidate": "synthetic", "scope": "unit_test"},
    )
    path = tmp_path / "state.pt"
    save_resume_state(path, state)

    restored_model, restored_optimizer, restored_scheduler = _build(seed=999)
    summary = restore_resume_state(
        load_resume_state(path),
        modules={"model": restored_model},
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_metadata={"candidate": "synthetic", "scope": "unit_test"},
    )

    assert summary["cursor"] == 7
    assert summary["accumulation_position"] == 1
    assert summary["accumulation_size"] == 6
    for (name_a, param_a), (name_b, param_b) in zip(model.named_parameters(), restored_model.named_parameters()):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)
        assert param_a.grad is not None
        assert param_b.grad is not None
        assert torch.equal(param_a.grad, param_b.grad)
    _assert_nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    _assert_nested_equal(scheduler.state_dict(), restored_scheduler.state_dict())


def test_restore_replays_next_python_numpy_and_torch_rng_draws():
    random.seed(303)
    np.random.seed(303)
    torch.manual_seed(303)
    model, optimizer, scheduler = _build()
    state = capture_resume_state(
        modules={"model": model},
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=0,
        accumulation_position=0,
        accumulation_size=6,
        metadata={"candidate": "synthetic"},
    )
    expected = (random.random(), float(np.random.rand()), torch.rand(4))

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    restored_model, restored_optimizer, restored_scheduler = _build(seed=2)
    restore_resume_state(
        state,
        modules={"model": restored_model},
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_metadata={"candidate": "synthetic"},
    )
    observed = (random.random(), float(np.random.rand()), torch.rand(4))

    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])


def test_restore_rejects_parameter_shape_drift():
    model, optimizer, scheduler = _build()
    state = capture_resume_state(
        modules={"model": model},
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=1,
        accumulation_position=0,
        accumulation_size=6,
        metadata={"candidate": "synthetic"},
    )

    drifted = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 3))
    drifted_optimizer = torch.optim.AdamW(drifted.parameters(), lr=1e-2)
    drifted_scheduler = torch.optim.lr_scheduler.LambdaLR(drifted_optimizer, lambda step: 1.0)
    with pytest.raises(RuntimeError, match="parameter contract drift"):
        restore_resume_state(
            state,
            modules={"model": drifted},
            optimizer=drifted_optimizer,
            scheduler=drifted_scheduler,
            expected_metadata={"candidate": "synthetic"},
        )


def test_capture_rejects_invalid_accumulation_position():
    model, optimizer, scheduler = _build()
    with pytest.raises(ValueError, match="accumulation_position"):
        capture_resume_state(
            modules={"model": model},
            optimizer=optimizer,
            scheduler=scheduler,
            cursor=0,
            accumulation_position=6,
            accumulation_size=6,
            metadata={"candidate": "synthetic"},
        )


def test_checkpoint_payload_stores_only_trainable_parameter_values():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-2
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    state = capture_resume_state(
        modules={"model": model},
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=0,
        accumulation_position=0,
        accumulation_size=6,
        metadata={"candidate": "synthetic"},
    )

    assert "trainable_parameters" in state
    assert "modules" not in state
    assert sorted(state["trainable_parameters"]) == ["model.2.bias", "model.2.weight"]
    assert "model.0.weight" in state["parameter_contract"]
    assert state["parameter_contract"]["model.0.weight"]["requires_grad"] is False


def test_round_trip_preserves_cpu_grad_scaler_state():
    model, optimizer, scheduler = _build()
    scaler = torch.amp.GradScaler("cpu", init_scale=256.0)
    x = torch.tensor([[0.1, 0.2, -0.3, 0.4], [0.5, -0.6, 0.7, 0.8]])
    y = torch.tensor([1, 2])
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        loss = torch.nn.functional.cross_entropy(model(x), y)
    scaler.scale(loss).backward()

    state = capture_resume_state(
        modules={"model": model},
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        cursor=1,
        accumulation_position=1,
        accumulation_size=6,
        metadata={"candidate": "synthetic", "amp": "cpu-fp16"},
    )

    restored_model, restored_optimizer, restored_scheduler = _build(seed=999)
    restored_scaler = torch.amp.GradScaler("cpu", init_scale=1.0)
    restore_resume_state(
        state,
        modules={"model": restored_model},
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        expected_metadata={"candidate": "synthetic", "amp": "cpu-fp16"},
    )

    _assert_nested_equal(scaler.state_dict(), restored_scaler.state_dict())
    assert restored_scaler.get_scale() == 256.0
