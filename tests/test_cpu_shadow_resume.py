import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from scripts.cpu_shadow_resume import load_shadow_checkpoint, save_shadow_checkpoint


LINEAGE = "dct-v3-cpu-shadow|qwen3-8b|rev-49e3418|spec-v1"


def build_system(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.Dropout(0.25),
        nn.GELU(),
        nn.Linear(8, 3),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 - 0.1 * step
    )
    return model, optimizer, scheduler


def backward_microbatch(model, x, y, group_size=2):
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    (loss / group_size).backward()
    return float(loss.detach())


def finish_step(model, optimizer, scheduler):
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def test_resume_mid_accumulation_is_bitwise_identical(tmp_path: Path):
    x1 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]])
    y1 = torch.tensor([0, 2])
    x2 = torch.tensor([[-2.0, 1.0, 0.0, 3.0], [4.0, 1.0, -1.0, 2.0]])
    y2 = torch.tensor([1, 0])

    full_model, full_opt, full_sched = build_system()
    full_model.train()
    backward_microbatch(full_model, x1, y1)
    backward_microbatch(full_model, x2, y2)
    finish_step(full_model, full_opt, full_sched)

    split_model, split_opt, split_sched = build_system()
    split_model.train()
    loss1 = backward_microbatch(split_model, x1, y1)
    ckpt = tmp_path / "shadow.pt"
    save_shadow_checkpoint(
        ckpt,
        lineage_fingerprint=LINEAGE,
        modules={"model": split_model},
        optimizer=split_opt,
        scheduler=split_sched,
        progress={
            "group_index": 0,
            "microbatch_index": 1,
            "optimizer_steps": 0,
            "loss_sum": loss1,
            "loss_count": 1,
        },
    )

    resumed_model, resumed_opt, resumed_sched = build_system(seed=999)
    progress = load_shadow_checkpoint(
        ckpt,
        expected_lineage_fingerprint=LINEAGE,
        modules={"model": resumed_model},
        optimizer=resumed_opt,
        scheduler=resumed_sched,
    )
    assert progress["microbatch_index"] == 1
    assert any(p.grad is not None for p in resumed_model.parameters())

    resumed_model.train()
    backward_microbatch(resumed_model, x2, y2)
    finish_step(resumed_model, resumed_opt, resumed_sched)

    for a, b in zip(full_model.parameters(), resumed_model.parameters()):
        assert torch.equal(a, b)
    assert full_sched.state_dict() == resumed_sched.state_dict()


def test_resume_rejects_wrong_lineage(tmp_path: Path):
    model, opt, sched = build_system()
    ckpt = tmp_path / "shadow.pt"
    save_shadow_checkpoint(
        ckpt,
        lineage_fingerprint=LINEAGE,
        modules={"model": model},
        optimizer=opt,
        scheduler=sched,
        progress={"group_index": 0, "microbatch_index": 0},
    )

    model2, opt2, sched2 = build_system()
    with pytest.raises(RuntimeError, match="lineage"):
        load_shadow_checkpoint(
            ckpt,
            expected_lineage_fingerprint="different",
            modules={"model": model2},
            optimizer=opt2,
            scheduler=sched2,
        )


def test_resume_restores_python_numpy_and_torch_rng(tmp_path: Path):
    model, opt, sched = build_system()
    ckpt = tmp_path / "shadow.pt"
    save_shadow_checkpoint(
        ckpt,
        lineage_fingerprint=LINEAGE,
        modules={"model": model},
        optimizer=opt,
        scheduler=sched,
        progress={"group_index": 0, "microbatch_index": 0},
    )
    expected = (random.random(), float(np.random.rand()), torch.rand(4))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    model2, opt2, sched2 = build_system(seed=999)
    load_shadow_checkpoint(
        ckpt,
        expected_lineage_fingerprint=LINEAGE,
        modules={"model": model2},
        optimizer=opt2,
        scheduler=sched2,
    )
    observed = (random.random(), float(np.random.rand()), torch.rand(4))

    assert expected[0] == observed[0]
    assert expected[1] == observed[1]
    assert torch.equal(expected[2], observed[2])


def test_checkpoint_saves_only_trainable_parameters(tmp_path: Path):
    model, opt, sched = build_system()
    model[0].weight.requires_grad = False
    model[0].bias.requires_grad = False
    ckpt = tmp_path / "shadow.pt"
    save_shadow_checkpoint(
        ckpt,
        lineage_fingerprint=LINEAGE,
        modules={"model": model},
        optimizer=opt,
        scheduler=sched,
        progress={"group_index": 0, "microbatch_index": 0},
    )
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "modules" not in payload
    assert "model.0.weight" not in payload["trainable_parameters"]
    assert "model.3.weight" in payload["trainable_parameters"]
