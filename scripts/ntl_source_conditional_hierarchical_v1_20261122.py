"""NTL source-conditioned generative model: hierarchical Bayesian token evidence.

This is a deliberately cheap first falsifier of a NEW OBJECTIVE, not a claim
that the underlying Bayesian model or source attribution concept is novel.
Learn P(tokens | source model) with a provider-family prior, rather than
discriminatively predict source from text. At test each row is scored alone.

C0 class-only generative, C1 provider-conditioned generative,
C2 false-provider prior, C3 a same-fold discriminative char-SVC anchor.
The exact official 2,699 texts / 150 prompt disjoint folds are spent DEV data.
No external generation, extra GPU, Kaggle submission or raw-content logs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot
from scripts.audit_lora24_sibling_holdout import family

SEED = 20261122
FEATURES = 60000
BETA = 800.0     # shared-language prior pseudo-count, frozen
ALPHA = 2500.0   # additional provider prior pseudo-count, frozen
BOOT = 1200
ARMS = ("class_only", "provider_shrink", "shuffled_provider", "char_svc")


def build_distribution(x: sparse.csr_matrix, targets: np.ndarray,
                       providers: np.ndarray, n_classes: int,
                       provider_weight: float, background_weight: float):
    """Smoothed P(token | model, provider); sum of token log-probabilities
    is a proper source-conditioned bag-of-terms likelihood.
    """
    if x.shape[0] != len(targets) or len(targets) != len(providers):
        raise ValueError("bad training alignment")
    if provider_weight < 0 or background_weight <= 0:
        raise ValueError("bad immutable pseudo-counts")
    v = x.shape[1]
    if v < 2:
        raise ValueError("not enough vocabulary")
    counts = np.stack([
        np.asarray(x[targets == c].sum(axis=0)).ravel()
        for c in range(n_classes)], axis=0).astype(np.float64)
    if np.min(counts.sum(axis=1)) <= 0:
        raise ValueError("empty source class")
    background = counts.sum(axis=0) + 1.
    background /= background.sum()

    model_family = np.array([
        str(np.unique(providers[targets == c])[0])
        if len(np.unique(providers[targets == c])) == 1
        else "INCONSISTENT" for c in range(n_classes)])
    if "INCONSISTENT" in model_family:
        raise ValueError("training model family must be stable")

    parent = np.empty_like(counts)
    for p in np.unique(model_family):
        ix = np.flatnonzero(model_family == p)
        pooled = counts[ix].sum(axis=0)
        pooled = pooled + background_weight * background
        pooled /= pooled.sum()
        parent[ix] = pooled
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = (
        counts + background_weight * background[None, :]
        + provider_weight * parent
    ) / (totals + background_weight + provider_weight)
    if np.any(probabilities <= 0) or not np.allclose(probabilities.sum(axis=1), 1.):
        raise ArithmeticError("invalid generative probabilities")
    return np.log(probabilities)


def stratified_prompt_ci(y: np.ndarray, a: np.ndarray, b: np.ndarray,
                         prompts: np.ndarray):
    uniq, inverse = np.unique(prompts, return_inverse=True)
    nclass = len(np.unique(y))
    denom = np.zeros((len(uniq), nclass), dtype=np.int32)
    diffs = np.zeros_like(denom)
    for i, k in enumerate(y):
        denom[inverse[i], k] += 1
        diffs[inverse[i], k] += int(b[i] == k) - int(a[i] == k)
    draws = np.random.default_rng(SEED).multinomial(
        len(uniq), np.full(len(uniq), 1./len(uniq)), size=BOOT)
    counts = draws @ denom
    valid = np.all(counts != 0, axis=1)
    if valid.sum() < .95 * BOOT:
        raise ValueError("unstable bootstrap class support")
    change = ((draws[valid] @ diffs) / counts[valid]).mean(axis=1)
    return [float(x) for x in np.quantile(change, [.025, .975])]


def train_oof(frame):
    classes, y = np.unique(frame.model.to_numpy(), return_inverse=True)
    if len(y) != 2699 or len(classes) != 18 or frame.prompt_id.nunique() != 150:
        raise ValueError("nonofficial training cohort")
    prompts = frame.prompt_id.astype(str).to_numpy()
    folds = frame.fold_id.to_numpy()
    group = np.array([family(name) for name in classes])
    pred = {arm: np.full(len(y), -1, dtype=np.int16) for arm in ARMS}

    for fold in range(5):
        train, valid = folds != fold, folds == fold
        if set(prompts[train]) & set(prompts[valid]):
            raise ValueError("prompt leakage")
        text_train = frame.loc[train, "text"].astype(str).tolist()
        text_valid = frame.loc[valid, "text"].astype(str).tolist()
        vec = CountVectorizer(
            lowercase=True, analyzer="word", ngram_range=(1,2),
            min_df=2, max_features=FEATURES, dtype=np.float32
        )
        xt = vec.fit_transform(text_train).tocsr()
        xv = vec.transform(text_valid).tocsr()
        families = group[y[train]]
        regular = build_distribution(
            xt, y[train], families, len(classes),
            provider_weight=0., background_weight=BETA)
        hierarchical = build_distribution(
            xt, y[train], families, len(classes),
            provider_weight=ALPHA, background_weight=BETA)
        shuffled = np.random.default_rng(SEED + fold).permutation(group)
        fake = build_distribution(
            xt, y[train], shuffled[y[train]], len(classes),
            provider_weight=ALPHA, background_weight=BETA)
        positions = np.flatnonzero(valid)
        for name, logp in [
            ("class_only", regular),
            ("provider_shrink", hierarchical),
            ("shuffled_provider", fake)]:
            pred[name][positions] = np.asarray(xv @ logp.T).argmax(axis=1)

        tf = TfidfVectorizer(
            analyzer="char", lowercase=False, ngram_range=(2,5),
            min_df=2, max_features=50000, sublinear_tf=True,
            dtype=np.float32,
        )
        char_t = tf.fit_transform(text_train)
        char_v = tf.transform(text_valid)
        clf = LinearSVC(C=0.5, class_weight="balanced",
                        max_iter=3000, dual="auto", random_state=SEED)
        clf.fit(char_t, y[train])
        pred["char_svc"][positions] = clf.predict(char_v)
        print("NTL_GENERATIVE_SOURCE_FOLD=" + str(fold), flush=True)
    if any((v < 0).any() for v in pred.values()):
        raise ValueError("missing actual fold predictions")
    return classes,y,prompts,pred


def synthetic():
    x = sparse.csr_matrix(np.array([
        [6,0,1,1], [5,1,1,0], [0,6,1,1], [1,5,0,1],
        [6,0,1,0], [5,1,0,1], [0,6,0,1], [1,5,1,0],
    ], dtype=float))
    y=np.array([0,1,2,3,0,1,2,3])
    groups=np.array(["A","A","B","B"])[y]
    logp=build_distribution(x,y,groups,4,provider_weight=100.,background_weight=5.)
    assert logp.shape==(4,4)
    assert np.isfinite(logp).all()
    assert np.allclose(np.exp(logp).sum(axis=1), 1.)
    print("NTL_GENERATIVE_SOURCE_SYNTHETIC=PASS")


def report(frame):
    with threadpool_limits(limits=2):
        classes,y,prompts,pred = train_oof(frame)
    ba = {a: float(balanced_accuracy_score(y,pred[a])) for a in ARMS}
    candidate = pred["provider_shrink"]
    comparisons = {}
    for control in ("class_only", "shuffled_provider", "char_svc"):
        prior = pred[control]
        comparisons[control] = {
            "delta_ba": ba["provider_shrink"] - ba[control],
            "prompt_ci95": stratified_prompt_ci(y,prior,candidate,prompts),
            "rescues": int(((prior != y) & (candidate == y)).sum()),
            "regressions": int(((prior == y) & (candidate != y)).sum()),
        }
    # A positive mechanism result must surpass a wrong-provider negative
    # control, class-only generative model AND strong LOCAL proxy.
    # It still would not imply the new model improves 0.715 hidden Public.
    gate = all(v["delta_ba"] >= .025 and v["prompt_ci95"][0] > 0
               for v in comparisons.values())
    by_family = {}
    for f in sorted(set(map(family,classes))):
        mask=np.array([family(classes[k])==f for k in y])
        by_family[f] = {
            "n":int(mask.sum()),
            "class_only_acc":float((pred["class_only"][mask]==y[mask]).mean()),
            "provider_shrink_acc":float((candidate[mask]==y[mask]).mean()),
        }
    return {
        "method":"source-conditioned multinomial token-likelihood with provider Dirichlet shrinkage",
        "scientific_goal":"test family-dependent generative evidence, not public-LB optimization",
        "scope":"reused official train, prompt-disjoint DEV, not fresh transfer",
        "rows":len(y), "prompts":len(set(prompts)), "classes":len(classes),
        "arms":ba, "paired_comparisons":comparisons,
        "by_family":by_family,
        "alpha":ALPHA,"beta":BETA,"features_max":FEATURES,
        "falsifier_pass":bool(gate),
        "not_equal_to_0p715_baseline":True,
        "official_submission":False,"new_gpu":False
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--synthetic-only",action="store_true")
    args=p.parse_args()
    synthetic()
    if args.synthetic_only:return
    d=load_snapshot(Path("data/train.csv"),Path("artifacts/phase1/folds.csv"))
    r=report(d)
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
    print("NTL_HIERARCHICAL_GENERATIVE_RESULT="+json.dumps(r,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
