"""CPU-only, source-disjoint sibling readout probe. Does NOT compute LoRA embeddings.

Inputs are private, immutable NPZ arrays containing: embeddings (N,D), labels,
prompt_ids, sources, text_hashes. A separate audited exporter must supply them.
No official competition data, Kaggle API, model download, training, or submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

LABELS = tuple(sorted((
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
)))
PREFIXES = ("claude-", "gemini-", "glm-", "gpt-", "kimi", "qwen")
SEED = 20261105
REPS = 5000


def family(label: str) -> str:
    matches = [p for p in PREFIXES if label.startswith(p)]
    if len(matches) != 1 or label not in LABELS:
        raise ValueError(f"unsupported model label: {label}")
    return matches[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_embeddings(path: Path, expected_sha: str) -> dict:
    if not path.is_file() or sha256(path) != expected_sha:
        raise ValueError(f"missing or checksum-mismatched embeddings: {path}")
    with np.load(path, allow_pickle=False) as obj:
        required = {"embeddings", "labels", "prompt_ids", "sources", "text_hashes"}
        if set(obj.files) != required:
            raise ValueError(f"incorrect NPZ fields: {obj.files}")
        out = {name: np.asarray(obj[name]) for name in required}
    x = out["embeddings"]
    if x.ndim != 2 or x.shape[1] < 2 or not np.issubdtype(x.dtype, np.number):
        raise ValueError("embeddings must be a numeric N x D matrix")
    if not np.isfinite(x).all():
        raise ValueError("non-finite embedding")
    if any(out[k].ndim != 1 or len(out[k]) != len(x)
           for k in required - {"embeddings"}):
        raise ValueError("misaligned metadata columns")
    for key in ("labels", "prompt_ids", "sources", "text_hashes"):
        if out[key].dtype.kind not in "US":
            raise ValueError(f"{key} must be a non-object string array")
        out[key] = out[key].astype(str)
        if any(not v.strip() for v in out[key]):
            raise ValueError(f"empty {key}")
    if set(out["labels"]) - set(LABELS):
        raise ValueError("unrecognized labels")
    if any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
           for v in out["text_hashes"]):
        raise ValueError("text_hashes must be SHA-256 hex")
    if len(set(out["text_hashes"])) != len(x):
        raise ValueError("duplicate passage hashes")
    return out


def validate_sources(train: dict, holdout: dict, lock: dict) -> tuple[str, ...]:
    mandatory = {
        "train_sha256", "holdout_sha256", "checkpoint_manifest_sha256",
        "train_sources", "holdout_sources", "encoder_fit_sources",
        "model_serving_provenance_verified", "near_duplicate_audit_verified",
        "holdout_never_selected_for_training_or_tuning",
    }
    if not mandatory <= set(lock):
        raise ValueError(f"lock missing fields: {sorted(mandatory - set(lock))}")
    for field in ("checkpoint_manifest_sha256",):
        value = lock[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"invalid {field}")
    for field in ("model_serving_provenance_verified", "near_duplicate_audit_verified",
                  "holdout_never_selected_for_training_or_tuning"):
        if lock[field] is not True:
            raise ValueError(f"admission not verified: {field}")
    fit_sources = set(map(str, lock["encoder_fit_sources"]))
    train_sources = set(map(str, train["sources"]))
    hold_sources = set(map(str, holdout["sources"]))
    if train_sources != set(lock["train_sources"]) or hold_sources != set(lock["holdout_sources"]):
        raise ValueError("source identities differ from precommitted lock")
    if len(hold_sources) < 2 or hold_sources & (train_sources | fit_sources):
        raise ValueError("holdout must contain >=2 never-trained external sources")
    if any("official" in s.lower() or "competition" in s.lower() for s in train_sources | hold_sources):
        raise ValueError("official competition corpus forbidden")
    if set(train["prompt_ids"]) & set(holdout["prompt_ids"]):
        raise ValueError("prompt overlap")
    if set(train["text_hashes"]) & set(holdout["text_hashes"]):
        raise ValueError("exact-text overlap")
    if train["embeddings"].shape[1] != holdout["embeddings"].shape[1]:
        raise ValueError("representation width mismatch")
    if set(train["labels"]) != set(LABELS):
        raise ValueError("control must train on all 18 competition labels")
    counts = {label: int((holdout["labels"] == label).sum()) for label in LABELS}
    eligible = tuple(label for label, n in counts.items() if n >= 20 and
                     sum(1 for other in LABELS if family(other) == family(label) and
                         counts[other] >= 20) >= 2)
    if len(set(map(family, eligible))) < 3:
        raise ValueError("holdout needs >=3 independently represented sibling families")
    return eligible


def linear_classifier(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000,
        solver="lbfgs", random_state=SEED,
    ).fit(x, y)


def fit_predict(train: dict, holdout: dict, *, shuffle_siblings: bool = False):
    x, y = train["embeddings"], train["labels"]
    test = holdout["embeddings"]
    control_head = linear_classifier(x, y)
    raw = control_head.predict_proba(test)
    columns = [int(np.flatnonzero(control_head.classes_ == label)[0]) for label in LABELS]
    flat = raw[:, columns]
    conditional = flat.copy()
    rng = np.random.default_rng(SEED)
    for fam in PREFIXES:
        idx = np.array([j for j, label in enumerate(LABELS) if family(label) == fam])
        if len(idx) < 2:
            continue
        train_mask = np.array([family(label) == fam for label in y])
        yy = y[train_mask].copy()
        if shuffle_siblings:
            yy = rng.permutation(yy)
        clf = linear_classifier(x[train_mask], yy)
        probs = clf.predict_proba(test)
        order = [int(np.flatnonzero(clf.classes_ == LABELS[j])[0]) for j in idx]
        mass = flat[:, idx].sum(axis=1)
        conditional[:, idx] = mass[:, None] * probs[:, order]
    if not np.isfinite(conditional).all() or (conditional < -1e-12).any():
        raise AssertionError("invalid conditional probabilities")
    if not np.allclose(conditional.sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("total probability changed")
    for fam in PREFIXES:
        idx = [j for j, label in enumerate(LABELS) if family(label) == fam]
        if not np.allclose(conditional[:, idx].sum(axis=1),
                           flat[:, idx].sum(axis=1), atol=1e-10):
            raise AssertionError(f"family mass changed: {fam}")
    return flat, conditional


def metric(truth: np.ndarray, predictions: np.ndarray, eligible: tuple[str, ...]) -> float:
    values = []
    for label in eligible:
        mask = truth == label
        if not mask.any():
            raise ValueError(f"required evaluation class missing: {label}")
        values.append(float(np.mean(predictions[mask] == label)))
    return float(np.mean(values))


def bootstrap(truth, base, treatment, prompts, eligible):
    prompt_set, inv = np.unique(prompts, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(prompt_set))]
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(REPS):
        picked = rng.integers(len(groups), size=len(groups))
        rows = np.concatenate([groups[i] for i in picked])
        if any(not np.any(truth[rows] == label) for label in eligible):
            continue
        deltas.append(metric(truth[rows], treatment[rows], eligible)
                      - metric(truth[rows], base[rows], eligible))
    if len(deltas) < 4750:
        raise ValueError(f"too few valid bootstrap replicates: {len(deltas)}/{REPS}")
    return [float(x) for x in np.quantile(deltas, [0.025, 0.975])], len(deltas)


def evaluate(train: dict, holdout: dict, lock: dict) -> dict:
    eligible = validate_sources(train, holdout, lock)
    flat, conditional = fit_predict(train, holdout)
    _, shuffled = fit_predict(train, holdout, shuffle_siblings=True)
    truth = holdout["labels"]
    base = np.asarray(LABELS)[flat.argmax(axis=1)]
    treat = np.asarray(LABELS)[conditional.argmax(axis=1)]
    negative = np.asarray(LABELS)[shuffled.argmax(axis=1)]
    base_ba = metric(truth, base, eligible)
    treat_ba = metric(truth, treat, eligible)
    neg_ba = metric(truth, negative, eligible)
    ci, valid_replicates = bootstrap(truth, base, treat, holdout["prompt_ids"], eligible)
    by_source = {}
    for source in sorted(set(holdout["sources"])):
        mask = holdout["sources"] == source
        covered = tuple(l for l in eligible if int((truth[mask] == l).sum()) >= 5)
        if not covered:
            raise ValueError(f"no eligible labels in source {source}")
        by_source[source] = {
            "rows": int(mask.sum()),
            "eligible_labels": len(covered),
            "delta": metric(truth[mask], treat[mask], covered) - metric(truth[mask], base[mask], covered),
        }
    per_class = {}
    safety_veto = []
    for label in eligible:
        mask = truth == label
        before = float(np.mean(base[mask] == label))
        after = float(np.mean(treat[mask] == label))
        per_class[label] = {"flat_recall": before, "conditional_recall": after, "delta": after - before}
        if before >= .25 and after < before - .15:
            safety_veto.append(label)
    def error_breakdown(predictions):
        correct = predictions == truth
        family_correct = np.fromiter(
            (family(a) == family(b) for a, b in zip(predictions, truth)),
            dtype=bool, count=len(truth),
        )
        return {
            "exact_accuracy": float(np.mean(correct)),
            "family_accuracy": float(np.mean(family_correct)),
            "within_correct_family_accuracy": (
                float(np.mean(correct[family_correct])) if family_correct.any() else None
            ),
            "same_family_errors": int(np.sum(~correct & family_correct)),
            "cross_family_errors": int(np.sum(~family_correct)),
        }
    result = {
        "status": "PASS" if (treat_ba > base_ba and ci[0] > 0
                            and sum(s["delta"] > 0 for s in by_source.values()) >= 2
                            and treat_ba > neg_ba and not safety_veto) else "FAIL",
        "scope": "independent external source probe only; NOT an official OOF or leaderboard result",
        "train_rows": len(train["labels"]),
        "holdout_rows": len(truth),
        "eligible_labels": list(eligible),
        "control_ba": base_ba, "treatment_ba": treat_ba, "negative_control_ba": neg_ba,
        "delta": treat_ba - base_ba, "prompt_bootstrap_95ci": ci,
        "bootstrap_valid_replicates": valid_replicates,
        "per_source": by_source, "per_class": per_class, "safety_veto": safety_veto,
        "flat_error_breakdown": error_breakdown(base),
        "treatment_error_breakdown": error_breakdown(treat),
        "top1_family_switches": int(sum(
            family(a) != family(b) for a, b in zip(base, treat)
        )),
        "holdout_sha256": lock["holdout_sha256"],
        "checkpoint_manifest_sha256": lock["checkpoint_manifest_sha256"],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    train = load_embeddings(args.train, lock["train_sha256"])
    holdout = load_embeddings(args.holdout, lock["holdout_sha256"])
    result = evaluate(train, holdout, lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in
        ("status", "control_ba", "treatment_ba", "negative_control_ba", "delta", "prompt_bootstrap_95ci")}))


if __name__ == "__main__":
    main()
