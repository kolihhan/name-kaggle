from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


MODULE = "scripts.run_prompt_residual_style_gate_a"


def _load_module():
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        pytest.fail(f"production module is missing: {exc}")


def test_build_prompt_residuals_centers_prompt_groups_and_masks_single_label_groups():
    mod = _load_module()
    assert hasattr(mod, "build_prompt_residuals")

    frame = pd.DataFrame(
        {
            "prompt_id": ["p1", "p1", "p2", "p2", "p3"],
            "model": ["a", "b", "a", "a", "c"],
        }
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 4.0],
            [5.0, 5.0],
        ],
        dtype=np.float32,
    )

    residuals, fit_mask = mod.build_prompt_residuals(frame, embeddings)

    np.testing.assert_allclose(residuals[0], [-1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(residuals[1], [1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(residuals[2], [0.0, -1.0], atol=1e-7)
    np.testing.assert_allclose(residuals[3], [0.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(residuals[4], [0.0, 0.0], atol=1e-7)
    assert np.isfinite(residuals).all()
    assert fit_mask.tolist() == [True, True, False, False, False]


def test_build_prompt_residuals_rejects_row_mismatch():
    mod = _load_module()
    frame = pd.DataFrame({"prompt_id": ["p1"], "model": ["a"]})
    with pytest.raises(ValueError, match="embedding rows"):
        mod.build_prompt_residuals(frame, np.zeros((2, 3), dtype=np.float32))


def test_build_source_holdout_removes_prompt_id_and_prompt_key_leakage():
    mod = _load_module()
    assert hasattr(mod, "build_source_holdout")

    frame = pd.DataFrame(
        {
            "source": ["hailey", "hailey", "other", "other", "other", "other"],
            "prompt_id": ["p1", "p2", "p1", "p3", "p4", "p5"],
            "prompt_key": ["k1", "k2", "kx", "k2", "k4", "k5"],
            "model": ["a", "b", "a", "b", "a", "c"],
        }
    )

    train_idx, heldout_idx, shared_labels = mod.build_source_holdout(frame, "hailey")

    assert heldout_idx.tolist() == [0, 1]
    assert train_idx.tolist() == [4, 5]
    assert shared_labels == ["a"]

    train = frame.iloc[train_idx]
    heldout = frame.iloc[heldout_idx]
    assert set(train["prompt_id"]).isdisjoint(set(heldout["prompt_id"]))
    assert set(train["prompt_key"]).isdisjoint(set(heldout["prompt_key"]))


def test_eligible_seed_sources_applies_frozen_row_prompt_and_shared_label_rules():
    mod = _load_module()
    assert hasattr(mod, "eligible_seed_sources")

    rows: list[dict[str, str]] = []
    labels = ["a", "b", "c"]

    for i in range(100):
        rows.append(
            {
                "source": "hailey",
                "prompt_id": f"h{i % 20}",
                "prompt_key": f"hk{i % 20}",
                "model": labels[i % 3],
            }
        )
    for i in range(100):
        rows.append(
            {
                "source": "keshav",
                "prompt_id": f"k{i % 20}",
                "prompt_key": f"kk{i % 20}",
                "model": labels[i % 3],
            }
        )
    for i in range(150):
        rows.append(
            {
                "source": "not-a-seed-source",
                "prompt_id": f"x{i % 30}",
                "prompt_key": f"xk{i % 30}",
                "model": labels[i % 3],
            }
        )

    eligible = mod.eligible_seed_sources(pd.DataFrame(rows))

    assert eligible == ["hailey", "keshav"]
