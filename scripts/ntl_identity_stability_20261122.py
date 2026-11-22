"""Prompt-stability-gated identity features: actual 18-way, 5-fold CPU experiment.

DIFFERENT from PC-CL: no contrastive model training, no prompt centering at
inference. Within-prompt model-vs-mean offsets are used TRAIN-ONLY to select
ngram features that retain a consistent direction across 4 disjoint prompt
partitions. The test-time classifier sees a single response and no prompt IDs.

Scientific scope: reused official train folds; NOT private/Public Kaggle gain.
No raw text, row identifiers, or predictions are written to the repo/log.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot

SEED = 20261122
ARMS = ("all", "supervised_chi2", "stable_prompt", "shuffled_prompt")
FEATURES = 24000
KEEP = 8000
SLICES = 4
BOOT = 1200


def stable_feature_scores(x, y, groups, n_classes, *, seed=SEED):
    """Max signed-stable class-vs-prompt contrast, no validation access.

    group centroids are sparse; materialize only per-class feature means,
    avoiding dense (n_rows, n_features) transforms and leaked texts.
    """
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=int)
    group_names, inv = np.unique(groups, return_inverse=True)
    if len(group_names) < 4 * 6:
        raise ValueError("insufficient independent prompt groups")
    if x.shape[0] != len(y) or len(groups) != len(y):
        raise ValueError("misaligned rows")
    counts = np.bincount(inv, minlength=len(group_names))
    indicator = sparse.csr_matrix(
        (1.0 / counts[inv], (inv, np.arange(len(y)))),
        shape=(len(group_names), len(y)),
    )
    centers = indicator @ x
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(group_names))
    slices = np.empty(len(group_names), dtype=int)
    slices[order] = np.arange(len(group_names)) % SLICES
    panel = slices[inv]
    d = np.zeros((SLICES, n_classes, x.shape[1]), dtype=np.float32)
    for s in range(SLICES):
        for k in range(n_classes):
            ids = np.flatnonzero((panel == s) & (y == k))
            if len(ids) < 4:
                raise ValueError(f"slice {s} label {k} too small")
            class_mean = np.asarray(x[ids].mean(axis=0)).ravel()
            matched_center = np.asarray(centers[inv[ids]].mean(axis=0)).ravel()
            d[s, k, :] = class_mean - matched_center
    sign_sum = np.sign(d).sum(axis=0)
    stable = np.abs(sign_sum) == SLICES
    amplitude = np.abs(d.mean(axis=0))
    score = np.max(np.where(stable, amplitude, 0), axis=0)
    if not np.isfinite(score).all():
        raise ArithmeticError("nonfinite feature scores")
    return score


def chosen(x, y, groups, n_classes, *, seed):
    actual = stable_feature_scores(x, y, groups, n_classes, seed=seed)
    group_null = np.random.default_rng(seed + 9000).permutation(groups)
    null = stable_feature_scores(x, y, group_null, n_classes, seed=seed)
    supervised = chi2(x, y)[0]
    supervised = np.nan_to_num(supervised, nan=0, posinf=0)
    result = {}
    for key, scores in [("stable_prompt", actual),
                        ("shuffled_prompt", null),
                        ("supervised_chi2", supervised)]:
        # deterministic ranking and fixed feature-count. No tuning.
        ids = np.argsort(-scores, kind="stable")[:KEEP]
        result[key] = np.sort(ids)
    return result, {
        "stable_nonzero": int(np.count_nonzero(actual)),
        "stable_vs_null_topk_overlap": int(len(np.intersect1d(
            result["stable_prompt"], result["shuffled_prompt"]))),
        "stable_vs_chi2_topk_overlap": int(len(np.intersect1d(
            result["stable_prompt"], result["supervised_chi2"]))),
    }


def get_model():
    return LinearSVC(C=0.5, class_weight="balanced", dual="auto",
                     max_iter=3000, random_state=SEED)


def full_experiment(frame):
    classes, y = np.unique(frame.model.to_numpy(), return_inverse=True)
    if len(frame) != 2699 or len(classes) != 18 or frame.prompt_id.nunique() != 150:
        raise ValueError("incorrect official train cohort")
    prompt = frame.prompt_id.astype(str).to_numpy()
    folds = frame.fold_id.to_numpy()
    pred = {name: np.full(len(y), -1, dtype=int) for name in ARMS}
    diagnostics = []
    for fold in range(5):
        tr, va = folds != fold, folds == fold
        if set(prompt[tr]) & set(prompt[va]):
            raise ValueError("prompt leaked across train/valid")
        vectorizer = TfidfVectorizer(
            analyzer="char", lowercase=False, ngram_range=(2, 5),
            min_df=2, max_features=FEATURES, sublinear_tf=True,
            dtype=np.float32,
        )
        x_train = vectorizer.fit_transform(frame.loc[tr, "text"].astype(str))
        x_valid = vectorizer.transform(frame.loc[va, "text"].astype(str))
        selects, info = chosen(x_train, y[tr], prompt[tr], len(classes),
                               seed=SEED + fold)
        for arm in ARMS:
            ids = selects.get(arm)
            xx = x_train if ids is None else x_train[:, ids]
            yy = x_valid if ids is None else x_valid[:, ids]
            clf = get_model()
            clf.fit(xx, y[tr])
            pred[arm][va] = clf.predict(yy)
        diagnostics.append({"fold": fold, "rows": int(va.sum()),
                            "features": int(x_train.shape[1]), **info})
        print("NTL_IDENTITY_STABILITY_FINISHED_FOLD=" + str(fold),
              flush=True)
    if any(np.any(v < 0) for v in pred.values()):
        raise ValueError("missing OOF predictions")
    return classes, y, prompt, pred, diagnostics


def paired_interval(y, pred, groups, base, contender):
    """Bootstrap whole prompts and recompute all 18 class denominators."""
    unique, inv = np.unique(groups, return_inverse=True)
    nclass = len(np.unique(y))
    totals = np.zeros((len(unique), nclass), dtype=np.int32)
    diffs = np.zeros_like(totals)
    for i in range(len(y)):
        totals[inv[i], y[i]] += 1
        diffs[inv[i], y[i]] += int(pred[contender][i] == y[i]) - int(
            pred[base][i] == y[i])
    rng = np.random.default_rng(SEED)
    samples = rng.multinomial(
        len(unique), [1.0 / len(unique)] * len(unique), size=BOOT)
    denom = samples @ totals
    valid = np.all(denom > 0, axis=1)
    delta = ((samples[valid] @ diffs) / denom[valid]).mean(axis=1)
    if len(delta) < 0.95 * BOOT:
        raise ValueError("bootstrap class support collapsed")
    return [float(z) for z in np.quantile(delta, [0.025, 0.975])]


def report(frame):
    with threadpool_limits(limits=2):
        labels, y, groups, pred, info = full_experiment(frame)
    ba = {arm: float(balanced_accuracy_score(y, pred[arm])) for arm in ARMS}
    comparisons = {}
    for base in ("all", "supervised_chi2", "shuffled_prompt"):
        before = pred[base] == y
        after = pred["stable_prompt"] == y
        comparisons[base] = {
            "delta_ba": ba["stable_prompt"] - ba[base],
            "prompt_bootstrap_ci95": paired_interval(
                y, pred, groups, base, "stable_prompt"),
            "rescues": int((~before & after).sum()),
            "regressions": int((before & ~after).sum()),
        }
    # The gate is intentionally strict: beat all three comparisons with a
    # positive prompt-bootstrap interval. Old official folds are SPENT DEV.
    gate = all(v["delta_ba"] >= 0.015 and v["prompt_bootstrap_ci95"][0] > 0
               for v in comparisons.values())
    return {
        "scope": "historically reused official train 5-fold prompt OOF ONLY",
        "model": "single-response char TFIDF / frozen LinearSVC",
        "mechanism": "stable model-vs-prompt feature direction across four train panels",
        "rows": len(y), "prompts": len(np.unique(groups)), "classes": len(labels),
        "feature_budget": FEATURES, "selected_count": KEEP,
        "arms": ba, "comparisons": comparisons, "folds": info,
        "mechanism_gate_pass": bool(gate),
        "public_kaggle_score": None, "gemma_score": None,
        "new_competition_submission": False, "gpu": False,
    }


def synthetic_self_test():
    rng = np.random.default_rng(77)
    n_prompts, classes = 40, 3
    x = np.abs(rng.normal(0.02, 0.005, size=(n_prompts * classes, 9)))
    y = np.tile(np.arange(classes), n_prompts)
    groups = np.repeat(np.arange(n_prompts), classes).astype(str)
    for row, lab in enumerate(y):
        x[row, lab] += 2.0
        x[row, 6 + int(row // classes) % 3] += 0.5
    f = stable_feature_scores(sparse.csr_matrix(x), y, groups, classes)
    if not set(np.argsort(f)[-3:]) == {0, 1, 2}:
        raise AssertionError("missed injected identity signal")
    if not np.all(np.isfinite(f)):
        raise AssertionError("nonfinite synthetic scores")
    print("NTL_IDENTITY_STABILITY_SYNTHETIC=PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    synthetic_self_test()
    if args.synthetic_only:
        return
    frame = load_snapshot(args.train, args.folds)
    results = report(frame)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(results, indent=2) + "\n")
    print("NTL_IDENTITY_STABILITY_RESULT=" + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
