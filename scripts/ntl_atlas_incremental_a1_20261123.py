"""NTL ATLAS A1: paired error audit + ONE prespecified incremental feature.

No new model forward, no QLoRA, no paid APIs, no Kaggle submission.
Historical official TRAIN: 300 GLM/Flash responses within 150 prompt groups.
The original 53-feature C_global is reproduced exactly and never replaced.
The only new fitted arm: C_global + the original fixed ATLAS prod margin,
with the same grouped 5 folds, LogisticRegression(C=1), and scaling.
This is exploratory repeated-data evidence, NOT prospective validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.ntl_atlas_exact_token_readout_a0_20261123 import (
    TRAIN_SHA, FOLDS_SHA, EXPECTED_C_GLOBAL, EXPECTED_TOTAL,
    fingerprint, fixed_prediction, original_global_heldout, compare, score,
    FULL, FLASH, STATS, SEED, fixed_summary, load_cache, paired_rows,
)

PAIR_NAMES = ("paired_mean_diff", "paired_win_fraction",
              "paired_mean_sign", "paired_p90_p10_spread", "paired_lp_corr")
FEATURE_NAMES = tuple("full_" + x for x in STATS) + tuple(
    "flash_" + x for x in STATS) + PAIR_NAMES
assert len(FEATURE_NAMES) == 53


def oof_fit(x, label, fold, collect_explanation=False):
    yy = (label == FULL).astype(int)
    pred = np.empty(len(label), dtype=object)
    logit = np.full(len(label), np.nan)
    attribution = np.full(x.shape, np.nan)
    for k in range(5):
        tr = fold != k
        va = fold == k
        scaler = StandardScaler()
        ztr = scaler.fit_transform(x[tr])
        zva = scaler.transform(x[va])
        clf = LogisticRegression(C=1., random_state=SEED,
                                 class_weight="balanced", max_iter=2000)
        clf.fit(ztr, yy[tr])
        pred[va] = np.where(clf.predict(zva) == 1, FULL, FLASH)
        logit[va] = clf.decision_function(zva)
        if collect_explanation:
            attribution[va] = zva * clf.coef_[0]
    if np.isnan(logit).any() or np.any(pred == None):
        raise RuntimeError("OOF_MISSING_HELD_OUT_FOLD")
    return pred, logit, attribution


def quant(arr):
    z = np.asarray(arr, dtype=float)
    if not len(z):
        return None
    return {key: round(float(val), 6) for key, val in
            (("mean", z.mean()), ("median", np.median(z)),
             ("p10", np.quantile(z, .1)),
             ("p90", np.quantile(z, .9)))}


def corr(x, y):
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 6)


def group_diagnostic(which, true_sign, orig_margin, prod_margin,
                     mean_margin, lengths, attribution):
    idx = np.flatnonzero(which)
    if not len(idx):
        return {"n": 0}
    signed = attribution[idx] * true_sign[idx, None]
    influence = np.mean(np.abs(signed), axis=0)
    top = np.argsort(-influence)[:7]
    return {
        "n": int(len(idx)),
        "true_global_logit": quant(orig_margin[idx] * true_sign[idx]),
        "true_ATLAS_margin": quant(prod_margin[idx] * true_sign[idx]),
        "true_mean_margin": quant(mean_margin[idx] * true_sign[idx]),
        "suffix_tokens": quant(lengths[idx]),
        "global_top_absolute_feature_contributions": [{
            "name": FEATURE_NAMES[int(j)],
            "mean_absolute_contribution": round(float(influence[j]), 5),
            "mean_label_direction_contribution":
                round(float(signed[:, j].mean()), 5)
        } for j in top],
    }


def run(args):
    fingerprint(args.train, TRAIN_SHA)
    fingerprint(args.folds, FOLDS_SHA)
    train = pd.read_csv(args.train, dtype={"id": str, "prompt_id": str, "model": str})
    folds = pd.read_csv(args.folds, dtype={"id": str, "prompt_id": str, "model": str})
    featfull = pd.read_csv(args.glm53_features, dtype={"id": str})
    featflash = pd.read_csv(args.flash_features, dtype={"id": str})
    if (len(train) != 2699 or len(folds) != len(train) or
            train.prompt_id.nunique() != 150 or train.id.duplicated().any()):
        raise ValueError("OFFICIAL_TRAIN_OR_FOLD_DRIFT")
    full = load_cache(args.glm53_scores, featfull)
    flash = load_cache(args.flash_scores, featflash)
    if (len(full["ids"]) != 2699 or
            not np.array_equal(full["ids"], flash["ids"]) or
            set(full["ids"]) != set(train.id)):
        raise ValueError("TOKEN_CACHE_ID_ALIGNMENT_FAILURE")
    joined = train.merge(
        folds[["id", "prompt_id", "model", "fold_id"]], on="id",
        suffixes=("", "_fold"), validate="one_to_one")
    if (len(joined) != 2699 or
            not (joined.prompt_id == joined.prompt_id_fold).all() or
            not (joined.model == joined.model_fold).all() or
            joined.groupby("prompt_id").fold_id.nunique().max() != 1):
        raise ValueError("PROMPT_FOLD_OR_LABEL_LEAKAGE")
    paired = list(paired_rows(full, flash))
    ids = [row[0] for row in paired]
    ordered = joined.set_index("id").loc[ids]
    eligible = np.isin(ordered.model.to_numpy(), [FULL, FLASH])
    if int(eligible.sum()) != EXPECTED_TOTAL:
        raise ValueError("BINARY_COHORT_DRIFT")
    ix = np.flatnonzero(eligible)
    label = ordered.model.to_numpy(dtype=str)[eligible]
    prompt = ordered.prompt_id.to_numpy(dtype=str)[eligible]
    fold = ordered.fold_id.to_numpy(dtype=int)[eligible]
    if set(fold) != set(range(5)) or len(set(prompt)) != 150:
        raise ValueError("FROZEN_PROMPT_FOLD_COVERAGE_DRIFT")
    a = featfull.set_index("id").loc[ids, list(STATS)].to_numpy(dtype=float)
    b = featflash.set_index("id").loc[ids, list(STATS)].to_numpy(dtype=float)
    summaries = np.asarray([fixed_summary(row[1], row[2]) for row in paired])
    x53 = np.column_stack([a, b, summaries])[eligible]
    if x53.shape != (300, 53) or not np.isfinite(x53).all():
        raise ValueError("INVALID_53_FEATURE_CONTRACT")
    prod_margin, mean_margin, lengths, atlas_pred = [], [], [], []
    for i in ix:
        one = fixed_prediction(paired[i][1], paired[i][2])
        prod_margin.append(one["prod_margin"])
        mean_margin.append(one["mean_margin"])
        lengths.append(one["n_tokens"])
        atlas_pred.append(one["C_prod"])
    prod_margin = np.asarray(prod_margin, dtype=float)
    mean_margin = np.asarray(mean_margin, dtype=float)
    lengths = np.asarray(lengths, dtype=int)
    atlas_pred = np.asarray(atlas_pred, dtype=str)

    # Exact previously reported historical 53-feature control is a hard gate.
    frozen = original_global_heldout(
        featfull, featflash, paired, ix, label, fold)
    base, global_logit, contributions = oof_fit(
        x53, label, fold, collect_explanation=True)
    if not np.array_equal(frozen, base) or not np.isclose(
            score(label, base)["BA"], EXPECTED_C_GLOBAL, atol=1e-7):
        raise RuntimeError("BASELINE_PREDICTION_REPRODUCTION_DRIFT")

    # THE ONLY NEW FIT: same training protocol, same 53 features + 1 fixed margin.
    x54 = np.column_stack([x53, prod_margin])
    augmented, augmented_logit, _ = oof_fit(x54, label, fold)
    baseline_correct = base == label
    atlas_correct = atlas_pred == label
    cats = {
        "both_correct": baseline_correct & atlas_correct,
        "global_only_correct": baseline_correct & ~atlas_correct,
        "ATLAS_only_correct": ~baseline_correct & atlas_correct,
        "both_wrong": ~baseline_correct & ~atlas_correct,
    }
    true_sign = np.where(label == FULL, 1., -1.)
    # Reproducible and aggregate-only: no source text, raw tokens or response IDs.
    diagnostics = {
        k: group_diagnostic(mask, true_sign, global_logit,
                            prod_margin, mean_margin, lengths, contributions)
        for k, mask in cats.items()
    }
    pair_feature_corr = {
        FEATURE_NAMES[j]: corr(prod_margin, x53[:, j])
        for j in range(48, 53)
    }
    top_abs_corr = sorted(
        ((FEATURE_NAMES[j], abs(corr(prod_margin, x53[:, j]) or 0.))
         for j in range(53)), key=lambda v: -v[1])[:8]

    scores = {"frozen_global53": score(label, base),
              "ATLAS_readout": score(label, atlas_pred),
              "global53_plus_ONE_ATLAS_margin": score(label, augmented)}
    direct = compare(label, base, augmented, prompt)
    per_fold = {
        str(k): {
            "n": int(np.sum(fold == k)),
            "global_correct": int(np.sum(baseline_correct[fold == k])),
            "augmented_correct": int(np.sum(augmented[fold == k] == label[fold == k])),
            "net_augmented_correct": int(np.sum(augmented[fold == k] == label[fold == k])
                                       - np.sum(baseline_correct[fold == k])),
        } for k in range(5)
    }
    return {
        "experiment": "NTL_ATLAS_A1_PAIRED_ERROR_AUDIT_AND_FIXED_54TH_FEATURE",
        "cohort": "spent_official_TRAIN_300_GLM_binary_150_prompts",
        "models_compared": [FULL, FLASH],
        "feature_counts": {"historical_strong": 53, "single_added": 1},
        "additional_feature": "ATLAS_fixed_T1_2_4_8_prod_margin",
        "no_other_feature_change": True,
        "baseline_exact_prediction_reproduced": True,
        "scores": scores,
        "frozen_vs_augmentation": direct,
        "frozen_vs_ATLAS": compare(label, base, atlas_pred, prompt),
        "paired_error_groups": diagnostics,
        "ATLAS_margin_correlations_with_original_pair_stats": pair_feature_corr,
        "ATLAS_margin_highest_absolute_feature_correlations": [
            {"feature": key, "absolute_pearson_r": round(float(r), 6)}
            for key, r in top_abs_corr
        ],
        "fold_counts": per_fold,
        "decision": {
            "exploratory_54_feature_gain_on_spent_GLMS": bool(
                scores["global53_plus_ONE_ATLAS_margin"]["BA"] >
                scores["frozen_global53"]["BA"]),
            "independent_source_validation_available": False,
            "full_18class_champion_0p715_pair_available": False,
            "promote_to_champion": False
        },
        "no_new_model_forward": True,
        "no_training_of_foundation_models": True,
        "no_paid_API": True,
        "no_kaggle_submission": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for n in ("train", "folds", "glm53_features", "flash_features",
              "glm53_scores", "flash_scores"):
        parser.add_argument("--" + n, required=True, type=Path)
    result = run(parser.parse_args())
    print("NTL_ATLAS_A1_REAL_COMPLEMENTARITY=" +
          json.dumps(result, sort_keys=True), flush=True)
