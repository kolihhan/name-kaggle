"""NTL Source-Capacity Assignment (SCA) — separate inference mechanism.

Hard per-prompt model uniqueness is potentially wrong on hidden test. The
soft capacity projection replaces each class by slots with incremental
reuse penalty, permitting duplicate source classes only when score evidence
overcomes the penalty. This is *not* a new classifier or another verifier.

Paired controls: exact historical hard Hungarian, fixed soft SCA, and
unchanged rowwise raw on same 2699 rows/folds/pretrained embedding. 18-class
full-OOF BA and fold-mean k5 BA. No tuned lambda, no target labels used by
the decoder, no official Kaggle claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot
import scripts.ntl_semantic_set_decoding_20261122 as prior

LAMBDA=2.0
SEED=20261122
MAX_M=18
ARMS=("raw","joint_semantic","joint_lexical","joint_random","joint_oracle")

def soft_source_capacity(scores,clusters,*,max_group=18):
    """Soft no-repeat prior by replicated label slots, unit-free row zscores.

    For each class c, slots r=1..group_size have score
    zscore(row)[c] - (r-1)*2.0. Hungarian solves injectively over slots,
    meaning labels CAN repeat, but only when evidence pays the penalty.
    """
    if max_group!=18:
        raise ValueError("frozen group limit changed")
    result=scores.argmax(axis=1).copy()
    n_classes=scores.shape[1]
    if n_classes!=18: # Synthetic unit test supports small class count.
        if not (2<=n_classes<=18):
            raise ValueError("invalid class dimension")
    for group in np.unique(clusters):
        index=np.flatnonzero(clusters==group)
        m=len(index)
        if m<2 or m>MAX_M:continue
        sub=np.asarray(scores[index],dtype=float)
        rowstd=np.maximum(sub.std(axis=1,keepdims=True),1e-6)
        normalized=(sub-sub.mean(axis=1,keepdims=True))/rowstd
        reuse=np.tile(np.arange(m,dtype=float),n_classes)
        slots=np.repeat(np.arange(n_classes),m)
        matrix=normalized[:,slots]-LAMBDA*reuse[None,:]
        rows,selected=linear_sum_assignment(-matrix)
        result[index[rows]]=slots[selected]
    return result


def self_test():
    # Both rows strongly prefer class 0; hard constraint forces one to
    # relabel, but the soft source-capacity model allows two class-0 rows.
    scores=np.array([[18.,0.,-2.],[17.,0.,-2.]])
    group=np.array([4,4])
    hard=prior.assign(scores,group)
    soft=soft_source_capacity(scores,group)
    assert len(set(hard))==2 and list(soft)==[0,0],(hard,soft)
    # No-op on singleton groups; exactly same base argmax.
    assert np.array_equal(
        soft_source_capacity(scores,np.array([0,1])),
        np.array([0,0])
    )
    print("NTL_SOFT_SOURCE_CAPACITY_SYNTHETIC=PASS",flush=True)


def fold_mean(result,group):
    rows=result["details"][group]
    if len(rows)!=5:
        raise RuntimeError("five folds expected")
    if group=="k5" and sum(x["rows"] for x in rows)!=750:
        raise ValueError("missing exactly 750 sparse rows")
    return {
        arm:float(np.mean([v["ba"][arm] for v in rows]))
        for arm in ARMS
    }


def run():
    self_test()
    frame=load_snapshot(Path("data/train.csv"),Path("artifacts/phase1/folds.csv"))
    with threadpool_limits(limits=2):
        z=prior.embed(frame.text.astype(str).tolist())
        hard=prior.evaluate(frame,z)
        # Ensures exact original prior baseline, not accidental retuning.
        if (abs(hard["full_ba"]["raw"]-0.31595824011931395)>1e-8
             or abs(hard["full_ba"]["joint_lexical"]-0.39117573949788714)>1e-8):
            raise ValueError("source benchmark has drifted")
        original=prior.assign
        try:
            prior.assign=soft_source_capacity
            soft=prior.evaluate(frame,z)
        finally:
            prior.assign=original
    assert soft["full_ba"]["raw"]==hard["full_ba"]["raw"], "wrong control"
    full_h,full_s=hard["full_ba"],soft["full_ba"]
    sparse_h=fold_mean(hard,"k5")
    sparse_s=fold_mean(soft,"k5")
    # All five full and k5 folded scores and groups are independently
    # computed by exactly the same original pipeline; only assignment varies.
    if abs(sparse_s["raw"]-sparse_h["raw"])>1e-12:
        raise ValueError("k5 denominator or classifier drift")
    delta_full=full_s["joint_lexical"]-full_h["joint_lexical"]
    delta_sparse=sparse_s["joint_lexical"]-sparse_h["joint_lexical"]
    return {
        "method":"source-capacity assignment with duplicate label slots",
        "lambda":LAMBDA,
        "predeclared_noninferiority_gate":{
            "full_loss_tolerance_BA":-.005,
            "k5_loss_tolerance_BA":-.005,
        },
        "score":"local GROUPED 5-fold DEV only; does not compare to champion 0.715",
        "full_BA18":{"hard":full_h,"soft":full_s},
        "k5_fold_mean_BA18":{"hard":sparse_h,"soft":sparse_s},
        "lexical_soft_minus_hard_BA18_full":delta_full,
        "lexical_soft_minus_hard_BA18_k5":delta_sparse,
        "soft_capacity_gate_pass": bool(delta_full>=-.005 and delta_sparse>=-.005),
        "hard_k5_lexical_prompt_pair_mean_precision":float(
            np.mean([r["lexical"]["precision"] for r in hard["details"]["k5"]])),
        "hard_k5_lexical_prompt_pair_mean_recall":float(
            np.mean([r["lexical"]["recall"] for r in hard["details"]["k5"]])),
        "test_label_class_repetition_unverified":True,
        "hidden_prompt_uniqueness_unverified":True,
        "no_competition_submit":True,"no_gpu":True
    }


if __name__=="__main__":
    print("NTL_SOFT_CAPACITY_REAL_RESULT="+json.dumps(run(),sort_keys=True),flush=True)
