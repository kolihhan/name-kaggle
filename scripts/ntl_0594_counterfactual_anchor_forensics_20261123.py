"""Counterfactual 0.715-anchor diagnostic of NTL 0.594 joint-decoder failure.

A true high-accuracy champion's per-row predictions on the official 2699
train rows are NOT available. Build labelled synthetic 0.715-BA anchors
ONLY for mechanism sensitivity (never report as a real classifier score).
Use real prompt-disjoint OOF 18-class *weak* char LinearSVC score vectors,
and blind lexical clusters from held-out responses.

Control A: shipped bonus+1.5 & forced 18-way assignment.
Control B: same assignment using true prompt groups (NON-DEPLOYABLE ORACLE).
Control C: lexicographic minimum-label-changes under blind group uniqueness,
            only uses weaker scores to resolve collisions, never wholesale
            reshuffles a group already unique.
Control D: baseline anchor unchanged.
Two predeclared correlated-error structures, 5 predeclared seeds:
random mismatches and weak-model-correlated mismatches. No threshold tuning,
Kaggle submission, private test, model inference, or GPU.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot
from scripts.ntl_semantic_set_decoding_20261122 import lexical_cluster,assign

SEEDS=(1001,1002,1003,1004,1005)
TARGET=.715
BONUS=1.5
K=18


def weak_scores_and_blind_groups(d):
    labels,y=np.unique(d.model.astype(str),return_inverse=True)
    folds=d.fold_id.to_numpy()
    weak=np.full((len(d),len(labels)),np.nan,dtype=float)
    clusters=np.full(len(d),-1,dtype=int)
    for fold in range(5):
        train=np.flatnonzero(folds!=fold)
        val=np.flatnonzero(folds==fold)
        texts_train=d.text.iloc[train].astype(str).tolist()
        texts_val=d.text.iloc[val].astype(str).tolist()
        cv=TfidfVectorizer(analyzer="char",lowercase=False,ngram_range=(2,5),
             max_features=45000,min_df=2,sublinear_tf=True)
        x=cv.fit_transform(texts_train)
        clf=LinearSVC(C=.5,class_weight="balanced",random_state=20261122,
                      max_iter=3000,dual="auto").fit(x,y[train])
        weak[val]=clf.decision_function(cv.transform(texts_val))
        wv=TfidfVectorizer(analyzer="word",ngram_range=(1,2),
            stop_words="english",max_features=50000,min_df=2,
            sublinear_tf=True)
        wv.fit(texts_train)
        group=lexical_cluster(wv.transform(texts_val))
        clusters[val]=group+fold*10000
        print("NTL_0594_ANCHOR_SENSITIVITY_OOF_FOLD_DONE="+str(fold),flush=True)
    if not np.isfinite(weak).all() or (clusters<0).any():
        raise ValueError("OOF weak scores incomplete")
    return y,weak,clusters


def make_anchor(y,weak_argmax,seed,scenario):
    """Synthetic exact per-class ~.715 BA; declared dependence alternatives."""
    rng=np.random.default_rng(seed)
    result=np.array(y,copy=True)
    for cls in range(K):
        rows=np.flatnonzero(y==cls)
        bad_num=len(rows)-int(np.floor(TARGET*len(rows)+.5))
        bad=rng.choice(rows,size=bad_num,replace=False)
        if scenario=="uniform_errors":
            result[bad]=(y[bad]+rng.integers(1,K,size=len(bad)))%K
        elif scenario=="weak_correlated_errors":
            wrong=weak_argmax[bad]
            replace=wrong==y[bad]
            wrong[replace]=(y[bad][replace]+rng.integers(
                        1,K,size=int(replace.sum())))%K
            result[bad]=wrong
        else:raise ValueError("unknown scenario")
    return result


def anchored(weak,anchor,bonus=BONUS):
    z=(weak-weak.mean(axis=1,keepdims=True))/np.maximum(
        weak.std(axis=1,keepdims=True),1e-6)
    z[np.arange(len(anchor)),anchor]+=bonus
    return z


def lexicographic_collision_only(weak,anchor,groups):
    """Among assignments changing the FEWEST anchor labels, choose max weak score.

    1000 ensures that retaining one more original anchor label is worth
    more than all finite weak-score deviations in groups up to 18.
    It has no confidence parameter tuned to Kaggle Public BA.
    """
    out=anchor.copy()
    z=(weak-weak.mean(axis=1,keepdims=True))/np.maximum(
        weak.std(axis=1,keepdims=True),1e-6)
    for group in np.unique(groups):
        idx=np.flatnonzero(groups==group)
        m=len(idx)
        if not(2<=m<=K):continue
        old=anchor[idx]
        if len(np.unique(old))==len(old):
            continue
        score=z[idx]+1000*(np.arange(K)[None,:]==old[:,None])
        row,col=linear_sum_assignment(-score)
        out[idx[row]]=col
        min_changes=m-len(np.unique(old))
        if np.count_nonzero(out[idx]!=old)!=min_changes:
            raise ValueError("lexicographic minimal-label-change invariant failed")
    return out


def record(y,anchor,pred):
    base=anchor==y
    after=pred==y
    return {
        "anchor_BA18":round(float(balanced_accuracy_score(y,anchor)),7),
        "output_BA18":round(float(balanced_accuracy_score(y,pred)),7),
        "delta_BA18":round(float(balanced_accuracy_score(y,pred)-
                               balanced_accuracy_score(y,anchor)),7),
        "changed_rows":int(np.count_nonzero(pred!=anchor)),
        "rescues":int((~base&after).sum()),
        "regressions":int((base&~after).sum()),
    }


def main():
    d=load_snapshot(Path("data/train.csv"),Path("artifacts/phase1/folds.csv"))
    with threadpool_limits(limits=2):
        y,weak,blind=weak_scores_and_blind_groups(d)
    true_groups=d.prompt_id.astype(str).to_numpy()
    scenarios={}
    for scenario in ("uniform_errors","weak_correlated_errors"):
        rows=[]
        for seed in SEEDS:
            anchor=make_anchor(y,weak.argmax(axis=1),seed,scenario)
            scores=anchored(weak,anchor)
            trial={
                "shipped_bonus1p5_blind_hungarian":assign(scores,blind),
                "shipped_bonus1p5_oracle_hungarian":assign(scores,true_groups),
                "collision_only_blind_hungarian":lexicographic_collision_only(
                    weak,anchor,blind),
                "collision_only_oracle_hungarian":lexicographic_collision_only(
                    weak,anchor,true_groups),
                "synthetic_anchor_unchanged":anchor,
            }
            row={"seed":seed,"scenario":scenario,
                 "arms":{k:record(y,anchor,pred) for k,pred in trial.items()}}
            rows.append(row)
        arms=rows[0]["arms"]
        scenarios[scenario]={
            "trials":len(rows),
            "mean":{a:{metric:round(float(np.mean([
                r["arms"][a][metric] for r in rows])),7)
                for metric in arms[a]}
                for a in arms},
            "repeats":rows,
        }
    print("NTL_0594_COUNTERFACTUAL_0715_ROOTCAUSE="+json.dumps({
        "scope":"official train 2699 prompt-disjoint OOF weak scores + synthetic 0.715 label anchor",
        "strong_model_predictions_are_synthetic":True,
        "does_not_approximate_real_champion_error_correlations":True,
        "not_a_score_or_submittable_model":True,
        "claim":"isolates harmful weak-score overrides at increasing anchor strength",
        "real_official_old_score":0.715,
        "real_official_joint_score":0.594,
        "synthetic_anchor_target_balanced_accuracy":TARGET,
        "scenarios":scenarios,
        "no_Kaggle_submission":True,"no_gpu":True
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
