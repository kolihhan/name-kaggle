"""Synthetic-only tests; no official competition texts or trained model weights."""
import hashlib

import numpy as np
import pytest

from scripts.run_sibling_information_probe import (
    LABELS, PREFIXES, evaluate, family, fit_predict, load_embeddings,
    sha256, validate_sources,
)


def _dataset(prefix, source_names, prompts_per_source, seed):
    rng = np.random.default_rng(seed)
    labels, pids, sources, hashes, features = [], [], [], [], []
    for s in source_names:
        for p in range(prompts_per_source):
            for k, label in enumerate(LABELS):
                z = np.zeros(24, dtype=np.float64)
                z[k % 24] = 2.5
                z[18 + PREFIXES.index(family(label))] = 0.8
                z += rng.normal(0, 0.4, len(z))
                labels.append(label)
                pids.append(f"{prefix}/{s}/prompt{p}")
                sources.append(s)
                features.append(z)
                hashes.append(hashlib.sha256(f"{prefix}:{s}:{p}:{label}".encode()).hexdigest())
    return dict(
        embeddings=np.stack(features),
        labels=np.asarray(labels, dtype=str),
        prompt_ids=np.asarray(pids, dtype=str),
        sources=np.asarray(sources, dtype=str),
        text_hashes=np.asarray(hashes, dtype=str),
    )


def _lock():
    return {
        "train_sha256": "a" * 64,
        "holdout_sha256": "b" * 64,
        "checkpoint_manifest_sha256": "c" * 64,
        "train_sources": ["tr1", "tr2"],
        "holdout_sources": ["ev1", "ev2"],
        "encoder_fit_sources": ["tr1", "tr2"],
        "model_serving_provenance_verified": True,
        "near_duplicate_audit_verified": True,
        "holdout_never_selected_for_training_or_tuning": True,
    }


@pytest.fixture(scope="module")
def examples():
    train = _dataset("training", ("tr1", "tr2"), 8, 42)
    holdout = _dataset("holdout", ("ev1", "ev2"), 12, 43)
    return train, holdout


def test_all_18_labels_and_six_families():
    assert len(LABELS) == 18
    assert len(set(map(family, LABELS))) == 6
    assert sum(family(x) == "gpt-" for x in LABELS) == 5


def test_family_probability_mass_invariant(examples):
    train, holdout = examples
    base, conditional = fit_predict(train, holdout)
    assert base.shape == conditional.shape == (len(holdout["labels"]), 18)
    assert np.allclose(conditional.sum(axis=1), 1)
    for f in PREFIXES:
        idx = [i for i, label in enumerate(LABELS) if family(label) == f]
        assert np.allclose(base[:, idx].sum(axis=1), conditional[:, idx].sum(axis=1))


def test_source_and_prompt_admission(examples):
    train, holdout = examples
    eligible = validate_sources(train, holdout, _lock())
    assert len(eligible) == 16  # two singleton families excluded
    assert len(set(map(family, eligible))) == 4
    broken = _lock()
    broken["encoder_fit_sources"] = ["tr1", "ev1"]
    with pytest.raises(ValueError, match="never-trained"):
        validate_sources(train, holdout, broken)
    broken = _lock()
    broken["model_serving_provenance_verified"] = False
    with pytest.raises(ValueError, match="admission"):
        validate_sources(train, holdout, broken)
    broken = {k: v.copy() for k, v in holdout.items()}
    broken["prompt_ids"][0] = train["prompt_ids"][0]
    with pytest.raises(ValueError, match="prompt overlap"):
        validate_sources(train, broken, _lock())


def test_real_evaluation_route_synthetic_only(examples):
    train, holdout = examples
    result = evaluate(train, holdout, _lock())
    assert result["status"] in ("PASS", "FAIL")
    assert result["holdout_rows"] == len(holdout["labels"])
    assert result["bootstrap_valid_replicates"] >= 4750
    assert len(result["per_source"]) == 2
    assert len(result["per_class"]) == 16
    assert result["scope"].startswith("independent external")
    assert result["flat_error_breakdown"]["same_family_errors"] >= 0
    assert result["treatment_error_breakdown"]["cross_family_errors"] >= 0
    assert isinstance(result["top1_family_switches"], int)


def test_checksum_and_no_pickle_objects(tmp_path, examples):
    train, _ = examples
    path = tmp_path / "synthetic.npz"
    np.savez(path, **train)
    assert len(load_embeddings(path, sha256(path))["labels"]) == len(train["labels"])
    with pytest.raises(ValueError, match="checksum"):
        load_embeddings(path, "0" * 64)
    train["labels"] = train["labels"].astype(object)
    unsafe = tmp_path / "unsafe.npz"
    np.savez(unsafe, **train)
    with pytest.raises(ValueError):
        load_embeddings(unsafe, sha256(unsafe))


def test_family_mass_preservation_does_not_ensure_top1_family():
    flat = np.zeros(18)
    altered = np.zeros(18)
    gpt = [i for i, x in enumerate(LABELS) if family(x) == "gpt-"]
    gemini = [i for i, x in enumerate(LABELS) if family(x) == "gemini-"]
    flat[gpt[0]] = .255
    flat[gpt[1]] = .255
    flat[gemini[0]] = .49
    altered[gpt[0]] = .51
    altered[gemini[0]] = .49
    assert np.isclose(flat[gpt].sum(), altered[gpt].sum())
    assert np.isclose(flat[gemini].sum(), altered[gemini].sum())
    assert family(LABELS[flat.argmax()]) != family(LABELS[altered.argmax()])
