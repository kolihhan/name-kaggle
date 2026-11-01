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


def _small_style_fixture() -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, str]] = []
    vectors: list[list[float]] = []
    model_style = {
        "a": np.asarray([2.0, 0.0, 0.0, 0.0]),
        "b": np.asarray([0.0, 2.0, 0.0, 0.0]),
        "c": np.asarray([0.0, 0.0, 2.0, 0.0]),
    }
    for source_i, source in enumerate(["hailey", "keshav", "other"]):
        for prompt_i in range(3):
            prompt = np.asarray([0.0, 0.0, 0.0, 20.0 + 5.0 * prompt_i + source_i])
            for model in ["a", "b", "c"]:
                rows.append(
                    {
                        "source": source,
                        "prompt_id": f"{source}-p{prompt_i}",
                        "prompt_key": f"{source}-k{prompt_i}",
                        "model": model,
                    }
                )
                vectors.append((prompt + model_style[model]).tolist())
    return pd.DataFrame(rows), np.asarray(vectors, dtype=np.float32)


def _eligibility_fixture() -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, str]] = []
    vectors: list[list[float]] = []
    labels = ["a", "b", "c"]
    styles = {
        "a": np.asarray([2.0, 0.0, 0.0, 0.0]),
        "b": np.asarray([0.0, 2.0, 0.0, 0.0]),
        "c": np.asarray([0.0, 0.0, 2.0, 0.0]),
    }
    for source_i, source in enumerate(["hailey", "not-a-seed-source"]):
        for i in range(120):
            prompt_i = i % 20
            model = labels[i % 3]
            rows.append(
                {
                    "source": source,
                    "prompt_id": f"{source}-p{prompt_i}",
                    "prompt_key": f"{source}-k{prompt_i}",
                    "model": model,
                }
            )
            prompt = np.asarray([0.0, 0.0, 0.0, 10.0 + prompt_i + source_i])
            vectors.append((prompt + styles[model]).tolist())
    return pd.DataFrame(rows), np.asarray(vectors, dtype=np.float32)


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


def test_fit_style_projection_is_finite_and_has_discriminant_width():
    mod = _load_module()
    assert hasattr(mod, "fit_style_projection")
    frame, embeddings = _small_style_fixture()

    projection = mod.fit_style_projection(frame, embeddings)
    transformed = projection.transform(embeddings)

    assert transformed.shape == (len(frame), 2)
    assert np.isfinite(transformed).all()


def test_evaluate_source_holdout_reports_same_scope_for_raw_and_projection():
    mod = _load_module()
    assert hasattr(mod, "evaluate_source_holdout")
    frame, embeddings = _small_style_fixture()

    result = mod.evaluate_source_holdout(frame, embeddings, "hailey")

    assert result["source"] == "hailey"
    assert result["shared_labels"] == ["a", "b", "c"]
    assert result["heldout_rows"] == 9
    assert result["training_rows"] == 18
    assert result["heldout_prompt_ids"] == 3
    assert result["raw"]["scored_rows"] == result["projected"]["scored_rows"] == 9
    assert set(result["raw"]["per_label_recall"]) == {"a", "b", "c"}
    assert set(result["projected"]["per_label_recall"]) == {"a", "b", "c"}
    assert result["prompt_id_overlap"] == 0
    assert result["prompt_key_overlap"] == 0
    assert -1.0 <= result["delta_projected_minus_raw"] <= 1.0


def test_run_gate_a_is_inconclusive_with_only_one_eligible_seed_source():
    mod = _load_module()
    assert hasattr(mod, "run_gate_a")
    frame, embeddings = _eligibility_fixture()

    report = mod.run_gate_a(frame, embeddings)

    assert report["eligible_sources"] == ["hailey"]
    assert len(report["source_results"]) == 1
    assert report["decision"]["status"] == "inconclusive"
    assert report["decision"]["official_oof_authorized"] is False
    assert report["decision"]["reason"] == "fewer_than_two_eligible_sources"


def test_build_execution_report_freezes_data_model_and_safety_metadata():
    mod = _load_module()
    assert hasattr(mod, "build_execution_report")
    gate = {
        "eligible_sources": ["hailey", "keshav"],
        "source_results": [],
        "aggregate": {"mean_delta_projected_minus_raw": 0.01, "worse_sources": 0},
        "decision": {"status": "pass", "official_oof_authorized": True},
    }

    report = mod.build_execution_report(
        gate,
        raw_sha256="abc123",
        phase14_rows=43_781,
        phase15_rows=26_025,
        phase15_prompt_ids=2_801,
        phase15_labels=11,
        model_name="answerdotai/ModernBERT-base",
        max_length=256,
    )

    assert report["scope"] == "external_transfer_gate_a_only_no_official_oof"
    assert report["data"] == {
        "raw_pool_sha256": "abc123",
        "phase14_rows": 43_781,
        "phase15_rows": 26_025,
        "phase15_prompt_ids": 2_801,
        "phase15_labels": 11,
    }
    assert report["encoder"] == {
        "model": "answerdotai/ModernBERT-base",
        "frozen": True,
        "max_length": 256,
        "pooling": "attention_mask_mean_then_l2",
    }
    assert report["projection"]["type"] == "LinearDiscriminantAnalysis"
    assert report["projection"]["solver"] == "svd"
    assert report["classifier"] == {
        "type": "LinearSVC",
        "class_weight": "balanced",
        "C": 1.0,
    }
    assert report["official_rows_scored"] == 0
    assert report["kaggle_submission_authorized"] is False
    assert report["gate_a"] is gate
