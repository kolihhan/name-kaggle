#!/usr/bin/env python3
"""Falsify the *need* for new test-time same-prompt uniqueness rescoring.

Frozen post-hoc evaluation only, existing 1200-row authentic champion predictions.
Everything is computed from unlabeled response text and predicted model labels,
then evaluated against the historical TRAIN truth. Never export rows/text/preds.
No supervised training, no LLM inference, no changes to current candidate.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

EXPECTED_TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_PRED_SHA="926dffc6ab0909f63767e8fcec7ca9488f7d1dc7c49656b0eaa8cf50064e0826"
CONTRACT="ntl-blind-collision-error-triage-20261116-v1"
CUTOFF=0.90
N=1200
EXPECTED_WRONG=51


def collision_mask(group, predicted):
    by_pair=defaultdict(list)
    for i,(g,label) in enumerate(zip(group,predicted)):
        by_pair[(str(g),str(label))].append(i)
    flagged=np.zeros(len(predicted), dtype=bool)
    collision_groups=set()
    for (g,_),indices in by_pair.items():
        if len(indices)>1:
            flagged[indices]=True
            collision_groups.add(g)
    return flagged, collision_groups


def stats(flagged,wrong):
    caught=int((flagged&wrong).sum())
    flagged_n=int(flagged.sum())
    missed=int((~flagged&wrong).sum())
    clear=int((~flagged).sum())
    flag_rate=caught/flagged_n if flagged_n else 0
    rest_rate=missed/clear if clear else 0
    return {
        "flagged_responses":flagged_n,
        "flagged_wrong":caught,
        "flagged_right":flagged_n-caught,
        "missed_wrong":missed,
        "flagged_error_rate":round(flag_rate,6),
        "unflagged_error_rate":round(rest_rate,6),
        "wrong_recall":round(caught/int(wrong.sum()),6),
        "error_enrichment":round(flag_rate/rest_rate,6) if rest_rate>0 else None,
        "reviews_per_caught_error":round(flagged_n/caught,4) if caught else None,
    }


def audit(train:Path,predictions:Path):
    if hashlib.sha256(train.read_bytes()).hexdigest()!=EXPECTED_TRAIN_SHA:
        raise ValueError("not exact frozen historical official training snapshot")
    if hashlib.sha256(predictions.read_bytes()).hexdigest()!=EXPECTED_PRED_SHA:
        raise ValueError("not exact authenticated existing 1200 frozen predictions")
    frame=pd.read_csv(train, usecols=["id","prompt_id","model","text"],dtype=str).iloc[:N].copy()
    preds=pd.read_parquet(predictions)
    if len(frame)!=N or len(preds)!=N or preds.columns.tolist()!=["id","model"]:
        raise ValueError("frozen true rows/predictions shape drift")
    if preds.id.astype(str).tolist()!=["scale1200_%04d"%i for i in range(N)]:
        raise ValueError("predicted text/row order drift")
    labels=preds.model.astype(str).to_numpy()
    truth=frame.model.astype(str).to_numpy()
    wrong=(labels!=truth)
    if wrong.sum()!=EXPECTED_WRONG:
        raise ValueError("historical error-count fingerprint mismatch")
    vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,2),
                               min_df=1,sublinear_tf=True,norm="l2",max_features=100000)
    x=vectorizer.fit_transform(frame.text.fillna("").astype(str).tolist())
    dis=cosine_distances(x)
    np.clip(dis,0,2,out=dis)
    blind=AgglomerativeClustering(n_clusters=None,distance_threshold=CUTOFF,
                                 metric="precomputed",linkage="average").fit_predict(dis)
    oracle=frame.prompt_id.to_numpy()
    blind_flags,blind_collision_groups=collision_mask(blind,labels)
    oracle_flags,oracle_collision_groups=collision_mask(oracle,labels)
    b=stats(blind_flags,wrong);o=stats(oracle_flags,wrong)
    # Clear binary go/no-go. This *only* licenses investigation on a truly
    # independent full-18 cohort, never model promotion or LB submission.
    gate=(b["flagged_wrong"]>=15 and b["error_enrichment"] is not None
          and b["error_enrichment"]>=3 and
          b["reviews_per_caught_error"] is not None
          and b["reviews_per_caught_error"]<=10)
    return {
        "protocol":CONTRACT,"data":"historical_spent_1200_official_train",
        "same_model_twice_hypothesis":"High-precision blind same-prompt clusters can triage model-collision errors without fitting labels",
        "alternative":"Sparse blind clusters undercapture errors or review too many correct rows",
        "true_wrong_count":int(wrong.sum()),
        "truth_prompt_groups":int(frame.prompt_id.nunique()),
        "blind_groups":int(len(set(blind))),
        "blind_collision_clusters":len(blind_collision_groups),
        "oracle_collision_prompt_groups":len(oracle_collision_groups),
        "blind":b,"oracle_upper_reference":o,
        "preregistered_go_gate":"blind flag >=15 of 51 errors, enrichment >=3x, <=10 reviewed responses per captured error",
        "gate_pass":bool(gate),
        "decision":"EXPLORE_ON_INDEPENDENT_SOURCE_ONLY" if gate else "STOP_LABEL_COLLISION_HEURISTIC",
        "important":"oracle uses true TRAIN prompt grouping solely as diagnostic upper reference, not a deployable feature; no independent performance evidence",
        "new_model_training":False,"model_inference":False,"balanced_accuracy_measured":False,
        "official_kaggle_submission":False,
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",type=Path,required=True)
    parser.add_argument("--predictions",type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.train,args.predictions)
    print("NTL_PRO_REAL_FROZEN_COLLISION_TRIAGE="+json.dumps(result,sort_keys=True),flush=True)
