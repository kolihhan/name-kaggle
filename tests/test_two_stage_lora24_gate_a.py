from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


MODULE = "scripts.run_two_stage_lora24_gate_a"


def _load_module():
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        pytest.fail(f"production module is missing: {exc}")


def test_select_holdout_sources_uses_frozen_hash_order():
    mod = _load_module()
    frame = pd.DataFrame(
        {
            "source": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
            "prompt_id": [f"p{i}" for i in range(6)],
            "prompt_key": [f"k{i}" for i in range(6)],
            "model": ["m0"] * 6,
            "text": ["x"] * 6,
        }
    )

    selected = mod.select_holdout_sources(frame)

    assert selected == ["gamma", "delta", "beta"]


def test_build_stage1_split_removes_source_and_prompt_bank_leakage():
    mod = _load_module()
    frame = pd.DataFrame(
        {
            "source": [
                "gamma", "delta", "beta",  # held-out by frozen hash rule
                "alpha", "alpha", "epsilon", "zeta", "zeta",
            ],
            "prompt_id": ["h1", "h2", "h3", "h1", "safe1", "safe2", "safe3", "safe4"],
            "prompt_key": ["hk1", "hk2", "hk3", "other", "hk2", "safe2", "safe3", "safe4"],
            "model": ["a", "b", "c", "a", "b", "a", "b", "c"],
            "text": [f"row-{i}" for i in range(8)],
        }
    )

    split = mod.build_stage1_split(frame)

    assert split["holdout_sources"] == ["gamma", "delta", "beta"]
    assert split["heldout_idx"].tolist() == [0, 1, 2]
    # row 3 leaks prompt_id=h1; row 4 leaks prompt_key=hk2; remaining rows are safe.
    assert split["train_idx"].tolist() == [5, 6, 7]

    train = frame.iloc[split["train_idx"]]
    heldout = frame.iloc[split["heldout_idx"]]
    assert set(train["source"]).isdisjoint(set(split["holdout_sources"]))
    assert set(train["prompt_id"]).isdisjoint(set(heldout["prompt_id"]))
    assert set(train["prompt_key"]).isdisjoint(set(heldout["prompt_key"]))
    assert split["prompt_id_overlap"] == 0
    assert split["prompt_key_overlap"] == 0


def test_select_eligible_labels_applies_training_presence_and_heldout_count():
    mod = _load_module()
    train = pd.DataFrame(
        {
            "model": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
        }
    )
    heldout = pd.DataFrame(
        {
            "model": ["a"] * 20 + ["b"] * 19 + ["c"] * 25 + ["d"] * 30,
        }
    )

    eligible = mod.select_eligible_labels(train, heldout)

    assert eligible == ["a", "c"]


def test_compute_class_weights_uses_only_training_counts_and_mean_one():
    mod = _load_module()
    labels = np.asarray(["a", "a", "a", "b", "c", "c"], dtype=object)

    weights = mod.compute_class_weights(labels, classes=["a", "b", "c"])

    assert set(weights) == {"a", "b", "c"}
    assert weights["b"] > weights["c"] > weights["a"]
    assert np.mean(list(weights.values())) == pytest.approx(1.0)


def test_build_prompt_triplets_never_mixes_prompt_groups_and_is_deterministic():
    mod = _load_module()
    frame = pd.DataFrame(
        {
            "prompt_id": ["p1"] * 5 + ["p2"] * 4 + ["p3"] * 2,
            "model": ["a", "b", "c", "d", "e", "a", "b", "c", "d", "a", "b"],
        }
    )
    rows = np.arange(len(frame))

    a = mod.build_prompt_triplets(frame, rows, seed=20261101)
    b = mod.build_prompt_triplets(frame, rows, seed=20261101)

    assert [x.tolist() for x in a] == [x.tolist() for x in b]
    assert sum(len(x) for x in a) == len(frame)
    assert all(1 <= len(x) <= 3 for x in a)
    assert sorted(np.concatenate(a).tolist()) == rows.tolist()
    for batch in a:
        assert frame.iloc[batch]["prompt_id"].nunique() == 1


def test_gate_a_decision_passes_only_on_frozen_consistency_rules():
    mod = _load_module()

    passed = mod.gate_a_decision(
        eligible_label_count=10,
        eligible_heldout_rows=900,
        pooled_delta=0.03,
        source_deltas=[0.02, 0.04, -0.01],
    )
    assert passed == {
        "status": "pass",
        "official_oof_authorized": True,
        "reason": "all_frozen_gate_a_rules_passed",
    }

    failed = mod.gate_a_decision(
        eligible_label_count=10,
        eligible_heldout_rows=900,
        pooled_delta=0.03,
        source_deltas=[0.02, -0.01, -0.02],
    )
    assert failed["status"] == "fail"
    assert failed["official_oof_authorized"] is False
    assert failed["reason"] == "pooled_or_source_consistency_rule_failed"

    inconclusive = mod.gate_a_decision(
        eligible_label_count=7,
        eligible_heldout_rows=1000,
        pooled_delta=0.50,
        source_deltas=[0.50, 0.50, 0.50],
    )
    assert inconclusive == {
        "status": "inconclusive",
        "official_oof_authorized": False,
        "reason": "insufficient_eligible_labels_or_rows",
    }


def test_gate_a_decision_requires_three_source_deltas():
    mod = _load_module()
    with pytest.raises(ValueError, match="three source deltas"):
        mod.gate_a_decision(
            eligible_label_count=10,
            eligible_heldout_rows=900,
            pooled_delta=0.1,
            source_deltas=[0.1, 0.1],
        )
