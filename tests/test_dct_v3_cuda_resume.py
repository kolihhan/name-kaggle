"""CPU-only verification of exact optimizer-boundary resume protocol."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from scripts.dct_v3_cuda_resume import load_optimizer_boundary, save_optimizer_boundary

LINEAGE = "v3-paired-transfer|qwen3@49e3418|publicpool@37750e8|split@03edcd6|dct01|seed20261101"


def build(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 12), nn.Dropout(p=0.35), nn.GELU(), nn.Linear(12, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - 0.1 * step)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    return model, optimizer, scheduler, scaler


def run_step(model, optimizer, scheduler, scaler, step):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for batch in range(3):
        x = torch.tensor([[(i + 2 * step + batch) / 10 for i in range(4)]], dtype=torch.float32)
        y = torch.tensor([(step + batch) % 3])
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        scaler.scale(loss / 3).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def test_uninterrupted_equals_resumed_through_dropout_and_adamw(tmp_path):
    full = build()
    for step in range(5):
        run_step(*full, step)
    full_params = [p.detach().clone() for p in full[0].parameters()]
    full_random = [random.random(), float(np.random.rand()), float(torch.rand(1))]

    interrupted = build()
    for step in range(2):
        run_step(*interrupted, step)
    file = tmp_path / "checkpoint.pt"
    save_optimizer_boundary(
        file, fingerprint=LINEAGE, modules={"model": interrupted[0]},
        optimizer=interrupted[1], scheduler=interrupted[2], scaler=interrupted[3],
        completed_steps=2, total_steps=5,
    )
    assert file.is_file() and file.stat().st_size > 0
    resumed = build(seed=999)
    first_remaining = load_optimizer_boundary(
        file, expected_fingerprint=LINEAGE, modules={"model": resumed[0]},
        optimizer=resumed[1], scheduler=resumed[2], scaler=resumed[3], expected_total_steps=5,
    )
    assert first_remaining == 2
    for step in range(first_remaining, 5):
        run_step(*resumed, step)
    for expected, actual in zip(full_params, resumed[0].parameters()):
        assert torch.equal(expected, actual)
    assert full[2].state_dict() == resumed[2].state_dict()
    assert full[3].state_dict() == resumed[3].state_dict()
    assert [random.random(), float(np.random.rand()), float(torch.rand(1))] == full_random


def test_bad_lineage_is_rejected_before_mutation(tmp_path):
    original = build()
    file = tmp_path / "checkpoint.pt"
    save_optimizer_boundary(
        file, fingerprint=LINEAGE, modules={"model": original[0]},
        optimizer=original[1], scheduler=original[2], scaler=original[3],
        completed_steps=0, total_steps=5,
    )
    fresh = build(seed=9)
    before = [p.detach().clone() for p in fresh[0].parameters()]
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_optimizer_boundary(
            file, expected_fingerprint=LINEAGE + "drift", modules={"model": fresh[0]},
            optimizer=fresh[1], scheduler=fresh[2], scaler=fresh[3], expected_total_steps=5,
        )
    assert all(torch.equal(a, b) for a, b in zip(before, fresh[0].parameters()))


def test_refuse_mid_accumulation_or_invalid_step(tmp_path):
    model, optimizer, scheduler, scaler = build()
    nn.functional.cross_entropy(model(torch.ones(1, 4)), torch.tensor([0])).backward()
    with pytest.raises(RuntimeError, match="mid-accumulation"):
        save_optimizer_boundary(
            tmp_path / "bad.pt", fingerprint=LINEAGE, modules={"model": model},
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            completed_steps=0, total_steps=5,
        )
    optimizer.zero_grad(set_to_none=True)
    with pytest.raises(ValueError, match="cursor"):
        save_optimizer_boundary(
            tmp_path / "bad.pt", fingerprint=LINEAGE, modules={"model": model},
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            completed_steps=6, total_steps=5,
        )
