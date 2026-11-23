"""Paired-error attribution audit for ATLAS A0 vs the frozen GLM global control.

No model inference or hyperparameter search. Rebuilds the exact 53-feature
prompt-disjoint C_global OOF model, then attributes each held-out logit to its
standardized feature contributions. Reports aggregate diagnostics only:
no text, response IDs, token arrays, or per-row predictions are persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.ntl_glm_distributed_evidence_20261122 import (
    SEED, FULL, FLASH, STATS, fixed_summary, load_cache, paired_rows,
)
from scripts.ntl_atlas_exact_token_readout_a0_20261123 import (
    TRAIN_SHA, FOLDS_SHA, TEMPERATURES, fixed_prediction,
)

EXPECTED_GLOBAL_BA = 0.9433333333333334
EXPECTED_ATLAS_BA = 0.9133333333333334
EXPECTED_RESCUES = 6
EXPECTED_REGRESSIONS = 15

PAIR_NAMES = (
    "pair__mean_logprob_diff",
    "pair__token_win_fraction",
    "pair__mean_token_sign",
    "pair__p90_p10_spread",
    "pair__score_correlation",
)

def fingerprint(path: Path, expected: str):
    got = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    if got != expected:
        raise ValueError(f"PINNED_FILE_SHA_DRIFT:{path.name}:{got}")
    return got

def feature_names():
    return ([f"glm53__{x}" for x in STATS] +
            [f"flash__{x}" for x in STATS] + list(PAIR_NAMES))

def make_matrix(fullfeat, flashfeat, paired):
    ids = [row[0] for row in paired]
    a = fullfeat.set_index("id").loc[ids]
    b = flashfeat.set_index("id").loc[ids]
    base = np.array([
        np.r_[a.loc[key, list(STATS)].to_numpy(dtype=float),
              b.loc[key, list(STATS)].to_numpy(dtype=float)]
        for key in ids
    ], dtype=float)
    pair = np.array([
        fixed_summary(row[1], row[2]) for row in paired
    ], dtype=float)
    x = np.concatenate([base, pair], axis=1)
    if x.shape[1] != 53 or not np.isfinite(x).all():
        raise ValueError("EXPECTED_53_FINITE_GLOBAL_FEATURES")
    return x

def safe_corr(a, b, rank=False):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if rank:
        a = pd.Series(a).rank(method="average").to_numpy()
        b = pd.Series(b).rank(method="average").to_numpy()
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 6)

def qsummary(values):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None
    q = np.quantile(v, [0.1, 0.5, 0.9])
    return {
        "mean": round(float(np.mean(v)), 6),
        "median": round(float(q[1]), 6),
        "p10": round(float(q[0]), 6),
        "p90": round(float(q[2]), 6),
    }

def feature_rankings(contrib, names, mask, n=8):
    block = contrib[mask]
    if not len(block):
        return {"supporting_truth": [], "opposing_truth": []}
    mean = block.mean(axis=0)
    support_rate = (block > 0).mean(axis=0)
    median_abs = np.median(np.abs(block), axis=0)
    order_hi = np.argsort(-mean)[:n]
    order_lo = np.argsort(mean)[:n]
    def pack(indices):
        return [{
            "feature": names[int(j)],
            "mean_signed_logit_contribution": round(float(mean[j]), 6),
            "fraction_helping_true_label": round(float(support_rate[j]), 4),
            "median_abs_contribution": round(float(median_abs[j]), 6),
        } for j in indices]
    return {
        "supporting_truth": pack(order_hi),
        "opposing_truth": pack(order_lo),
    }

def group_report(name, mask, labels, folds, global_margin, atlas_margin,
                 mean_margin, contrib, names, intercept_signed):
    idx = np.flatnonzero(mask)
    y = labels[idx]
    return {
        "n": int(len(idx)),
        "true_class_counts": {
            FULL: int(np.sum(y == FULL)),
            FLASH: int(np.sum(y == FLASH)),
        },
        "fold_counts": {
            str(k): int(np.sum(folds[idx] == k)) for k in range(5)
        },
        "global_signed_margin": qsummary(global_margin[idx] *
            np.where(y == FULL, 1.0, -1.0)),
        "global_abs_margin": qsummary(np.abs(global_margin[idx])),
        "atlas_signed_margin": qsummary(atlas_margin[idx] *
            np.where(y == FULL, 1.0, -1.0)),
        "mean_logprob_signed_margin": qsummary(mean_margin[idx] *
            np.where(y == FULL, 1.0, -1.0)),
        "mean_signed_intercept": round(float(np.mean(intercept_signed[idx])), 6),
        "feature_contributions": feature_rankings(contrib, names, mask),
    }

def run(args):
    fingerprint(args.train, TRAIN_SHA)
    fingerprint(args.folds, FOLDS_SHA)
    train = pd.read_csv(args.train, dtype={"id": str, "prompt_id": str, "model": str})
    folds_df = pd.read_csv(args.folds, dtype={"id": str, "prompt_id": str, "model": str})
    if (len(train) != 2699 or train.prompt_id.nunique() != 150 or
        len(folds_df) != 2699 or train.id.duplicated().any()):
        raise ValueError("OFFICIAL_COHORT_DRIFT")

    fullfeat = pd.read_csv(args.glm53_features, dtype={"id": str})
    flashfeat = pd.read_csv(args.flash_features, dtype={"id": str})
    full = load_cache(args.glm53_scores, fullfeat)
    flash = load_cache(args.flash_scores, flashfeat)
    if (len(full["ids"]) != 2699 or len(flash["ids"]) != 2699 or
        not np.array_equal(full["ids"], flash["ids"]) or
        set(full["ids"]) != set(train.id)):
        raise ValueError("TOKEN_CACHE_ID_ALIGNMENT_FAILURE")

    joined = train.merge(
        folds_df[["id", "prompt_id", "model", "fold_id"]],
        on="id", suffixes=("", "_fold"), validate="one_to_one"
    )
    if (len(joined) != 2699 or not joined.id.is_unique or
        not (joined.prompt_id == joined.prompt_id_fold).all() or
        not (joined.model == joined.model_fold).all() or
        joined.groupby("prompt_id").fold_id.nunique().max() != 1):
        raise ValueError("PROMPT_GROUP_OR_LABEL_FOLD_DRIFT")

    indexed = joined.set_index("id").loc[list(full["ids"])]
    labels_all = indexed.model.to_numpy(dtype=str)
    eligible = np.isin(labels_all, [FULL, FLASH])
    if int(eligible.sum()) != 300:
        raise ValueError("GLM_BINARY_SAMPLE_DRIFT")
    labels = labels_all[eligible]
    prompt = indexed.prompt_id.to_numpy(dtype=str)[eligible]
    fold = indexed.fold_id.to_numpy(dtype=int)[eligible]
    if len(np.unique(prompt)) != 150:
        raise ValueError("EXPECTED_150_PROMPT_GROUPS")

    paired = list(paired_rows(full, flash))
    x_all = make_matrix(fullfeat, flashfeat, paired)
    x = x_all[eligible]
    names = feature_names()
    yy = (labels == FULL).astype(int)

    atlas_pred = []
    atlas_margin = []
    mean_margin = []
    for i, (_, a, b) in enumerate(paired):
        if not eligible[i]:
            continue
        row = fixed_prediction(a, b)
        atlas_pred.append(row["C_prod"])
        atlas_margin.append(row["prod_margin"])
        mean_margin.append(row["mean_margin"])
    atlas_pred = np.asarray(atlas_pred, dtype=str)
    atlas_margin = np.asarray(atlas_margin, dtype=float)
    mean_margin = np.asarray(mean_margin, dtype=float)

    # Exact historical C_global recipe, preserving its five prompt-held-out folds.
    global_pred = np.empty(300, dtype=int)
    global_margin = np.empty(300, dtype=float)
    contributions = np.empty((300, 53), dtype=float)
    intercept_signed = np.empty(300, dtype=float)
    fold_coefs = []
    fold_intercepts = []
    for k in range(5):
        tr, va = fold != k, fold == k
        scaler = StandardScaler()
        xtr = scaler.fit_transform(x[tr])
        xva = scaler.transform(x[va])
        clf = LogisticRegression(
            C=1.0, random_state=SEED, class_weight="balanced", max_iter=2000
        )
        clf.fit(xtr, yy[tr])
        margin = clf.decision_function(xva)
        global_margin[va] = margin
        global_pred[va] = (margin >= 0).astype(int)
        contributions[va] = xva * clf.coef_.reshape(1, -1)
        intercept_signed[va] = clf.intercept_[0] * np.where(yy[va] == 1, 1.0, -1.0)
        fold_coefs.append(clf.coef_.reshape(-1))
        fold_intercepts.append(float(clf.intercept_[0]))

    global_labels = np.where(global_pred == 1, FULL, FLASH)
    def ba(pred):
        return float(.5 * (
            np.mean(pred[labels == FULL] == FULL) +
            np.mean(pred[labels == FLASH] == FLASH)
        ))
    global_ba = ba(global_labels)
    atlas_ba = ba(atlas_pred)
    if not np.isclose(global_ba, EXPECTED_GLOBAL_BA, atol=1e-8):
        raise RuntimeError(f"GLOBAL_CONTROL_REPRODUCTION_FAILED:{global_ba}")
    if not np.isclose(atlas_ba, EXPECTED_ATLAS_BA, atol=1e-8):
        raise RuntimeError(f"ATLAS_A0_REPRODUCTION_FAILED:{atlas_ba}")

    global_ok = global_labels == labels
    atlas_ok = atlas_pred == labels
    rescue = (~global_ok) & atlas_ok
    regression = global_ok & (~atlas_ok)
    if int(rescue.sum()) != EXPECTED_RESCUES or int(regression.sum()) != EXPECTED_REGRESSIONS:
        raise RuntimeError("PAIRED_ERROR_COUNTS_DRIFT")
    signed = np.where(labels == FULL, 1.0, -1.0)
    signed_contrib = contributions * signed.reshape(-1, 1)
    rescue_margin = np.abs(global_margin[rescue])
    regression_margin = np.abs(global_margin[regression])

    coef = np.asarray(fold_coefs)
    coef_sign_agreement = np.abs(np.sign(coef).mean(axis=0))
    coef_summary = [{
        "feature": names[j],
        "mean_abs_standardized_coefficient": round(float(np.mean(np.abs(coef[:, j]))), 6),
        "fold_sign_agreement": round(float(coef_sign_agreement[j]), 4),
        "mean_coefficient": round(float(np.mean(coef[:, j])), 6),
    } for j in np.argsort(-np.mean(np.abs(coef), axis=0))[:12]]

    pair_corr = []
    for j, name in enumerate(PAIR_NAMES):
        pair_corr.append({
            "feature": name,
            "pearson_with_ATLAS_margin": safe_corr(atlas_margin, x[:, 48 + j]),
            "spearman_with_ATLAS_margin": safe_corr(atlas_margin, x[:, 48 + j], rank=True),
        })
    pair_corr.sort(key=lambda z: abs(z["spearman_with_ATLAS_margin"] or 0), reverse=True)

    rescue_abs_med = float(np.median(rescue_margin))
    regression_abs_med = float(np.median(regression_margin))
    record = {
        "experiment": "NTL_ATLAS_A0_CGLOBAL_PAIRED_ERROR_ATTRIBUTION",
        "scope": "reused_official_train_300_GLM_binary_rows_150_prompt_groups_only",
        "n_rows": 300,
        "n_prompts": 150,
        "feature_count": 53,
        "folds": 5,
        "temperatures": list(TEMPERATURES),
        "reproduced_BA": {
            "C_global": round(global_ba, 8),
            "C_ATLAS_prod": round(atlas_ba, 8),
        },
        "paired_error_counts": {
            "same_prediction": int(np.sum(global_labels == atlas_pred)),
            "C_global_correct_ATLAS_wrong_regressions": int(regression.sum()),
            "C_global_wrong_ATLAS_correct_rescues": int(rescue.sum()),
            "net_ATLAS_minus_global_correct": int(atlas_ok.sum() - global_ok.sum()),
        },
        "global_vs_ATLAS_margin_association": {
            "pearson": safe_corr(global_margin, atlas_margin),
            "spearman": safe_corr(global_margin, atlas_margin, rank=True),
            "pearson_vs_mean_logprob_margin": safe_corr(atlas_margin, mean_margin),
        },
        "ATLAS_margin_vs_existing_pair_features": pair_corr,
        "error_group_diagnostics": {
            "global_correct_ATLAS_wrong": group_report(
                "global_correct_ATLAS_wrong", regression, labels, fold,
                global_margin, atlas_margin, mean_margin, signed_contrib,
                names, intercept_signed
            ),
            "global_wrong_ATLAS_correct": group_report(
                "global_wrong_ATLAS_correct", rescue, labels, fold,
                global_margin, atlas_margin, mean_margin, signed_contrib,
                names, intercept_signed
            ),
        },
        "global_model_feature_coefficient_stability_top12": coef_summary,
        "margin_diagnostic": {
            "median_abs_global_margin_on_ATLAS_rescues": round(rescue_abs_med, 6),
            "median_abs_global_margin_on_ATLAS_regressions": round(regression_abs_med, 6),
            "rescues_have_weaker_global_margin_median": bool(rescue_abs_med < regression_abs_med),
            "interpretation": "descriptive_only_small_spent_sample",
        },
        "data_handling": {
            "raw_response_text_persisted": False,
            "row_ids_or_per_row_predictions_persisted": False,
            "token_arrays_persisted": False,
            "new_model_forward": False,
            "paid_api": False,
            "new_training": False,
            "new_kaggle_submission": False,
        },
        "decision": {
            "promote_ATLAS": False,
            "fit_or_tune_on_these_21_cases": False,
            "next_gate": "inspect whether the 6 rescue cases carry stable conditional information before any one-feature nested grouped-fold test",
        },
    }
    print("NTL_ATLAS_PAIRED_ERROR_AUDIT=" + json.dumps(record, sort_keys=True), flush=True)
    return record

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("train", "folds", "glm53_features", "flash_features", "glm53_scores", "flash_scores"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    run(args)
