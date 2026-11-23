"""Mechanism-only investigation of the failed NTL 0.594 official experiment.

Question: Does the original unlabeled TF-IDF + average-linkage threshold .90
create different-prompt groups at larger inference batch sizes, which force a
perfect per-response source classifier to make errors under one-of-each-label
Hungarian assignment?

No hidden labels, Public leaderboard tuning, GPU, competition submissions,
new predictions, or model changes. The test is on public 2699/150 official
training rows with PREVIOUSLY FROZEN prompt fold manifest. Source group IDs
are used exclusively AFTER blinded clustering to compute evaluation metrics.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path
from collections import Counter
from itertools import permutations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot

SEED=20261123
THRESHOLD=.90  # byte-for-byte source inference threshold, NOT retuned
MAX_GROUP=18
SIZES=(1,2,3,4,5)  # 5 official folds: 540, 1080, 1620, 2160, 2699 rows
TRIAL_FOLD_ORDERS=(
    (0,1,2,3,4),
    (1,3,0,4,2),
    (2,4,1,0,3),
    (3,0,4,2,1),
    (4,2,3,1,0),
)


def choose2(x: int)->int:
    return x*(x-1)//2


def blind_lexical_cluster(x):
    """Exact algorithm in kaggle/ntl_0715_label_only_joint_final_20261122.py."""
    if x.shape[0]<2:return np.zeros(x.shape[0],dtype=int)
    similarity=(x @ x.T).toarray()
    dist=np.clip(1.0-similarity,0.0,2.0)
    np.fill_diagonal(dist,0.0)
    return AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=THRESHOLD,
        linkage="average",
        metric="precomputed"
    ).fit_predict(dist)


def metric_groups(pred, gold_prompt, gold_model):
    """Group purity and lower bound on errors FORCED by the hard constraint."""
    n=len(pred)
    if n!=len(gold_prompt) or n!=len(gold_model):
        raise ValueError("row alignment differs")
    by_cluster={}
    by_truth={}
    contingency=Counter()
    for i,(p,t,m) in enumerate(zip(pred,gold_prompt,gold_model)):
        by_cluster.setdefault(int(p),[]).append(i)
        by_truth[t]=by_truth.get(t,0)+1
        contingency[(int(p),t)]+=1
    predicted_pairs=sum(choose2(len(v)) for v in by_cluster.values())
    actual_pairs=sum(choose2(v) for v in by_truth.values())
    tp=sum(choose2(v) for v in contingency.values())
    oversized=sum(len(v) for v in by_cluster.values() if len(v)>MAX_GROUP)
    scored_groups=[v for v in by_cluster.values() if 2<=len(v)<=MAX_GROUP]
    scored_rows=sum(len(v) for v in scored_groups)
    cross_prompt_rows=sum(len(v) for v in scored_groups
                          if len({gold_prompt[i] for i in v})>1)
    impossible_count=sum(
        len(v)-len({gold_model[i] for i in v}) for v in scored_groups
    )
    forced_group_count=sum(
        len(v)!=len({gold_model[i] for i in v}) for v in scored_groups
    )
    return {
        "rows":n,
        "true_prompt_count":len(by_truth),
        "predicted_groups":len(by_cluster),
        "processed_rows_hard_assignment":scored_rows,
        "rows_in_groups_oversized_skipped":oversized,
        "rows_in_mixed_prompt_processed_groups":cross_prompt_rows,
        "groups_with_duplicate_true_sources":forced_group_count,
        "minimum_wrong_rows_even_if_all_start_correct":impossible_count,
        "minimum_wrong_fraction_all_rows":round(impossible_count/n,7),
        "pair_precision":round(tp/predicted_pairs,7) if predicted_pairs else 1.,
        "pair_recall":round(tp/actual_pairs,7) if actual_pairs else 0.,
        "pair_F1":round(2*tp/(predicted_pairs+actual_pairs),7)
                   if predicted_pairs+actual_pairs else 0.,
    }


def self_test():
    # A mixed group containing two different prompts but the same true
    # source necessarily induces one false relabel for a perfect classifier.
    v=metric_groups(
        np.array([0,0,1,1]),
        np.array(["promptA","promptB","promptA","promptA"]),
        np.array(["modelX","modelX","modelX","modelY"])
    )
    assert v["minimum_wrong_rows_even_if_all_start_correct"]==1,v
    assert v["groups_with_duplicate_true_sources"]==1,v
    # True prompt grouping cannot force a wrong label if its model grid
    # contains one observation per class and prompt.
    ideal=metric_groups(
        np.array([0,1,0,0]),
        np.array(["promptA","promptB","promptA","promptA"]),
        np.array(["modelX","modelX","modelY","modelZ"])
    )
    assert ideal["minimum_wrong_rows_even_if_all_start_correct"]==0,ideal
    print("NTL_0594_GROUP_FORCE_ERROR_SYNTHETIC=PASS",flush=True)


def main():
    self_test()
    d=load_snapshot(Path("data/train.csv"),Path("artifacts/phase1/folds.csv"))
    gold_prompt=d.prompt_id.astype(str).to_numpy()
    gold_model=d.model.astype(str).to_numpy()
    folds=d.fold_id.to_numpy()
    assert len(d)==2699 and len(set(gold_prompt))==150
    counts=d.groupby("prompt_id").model.nunique()
    assert counts.min()>=17 and counts.max()==18
    assert not d.duplicated(["prompt_id","model"]).any()
    with threadpool_limits(limits=2):
        tf=TfidfVectorizer(analyzer="word",ngram_range=(1,2),
            stop_words="english",max_features=50000,min_df=2,
            sublinear_tf=True)
        X=tf.fit_transform(d.text.astype(str).tolist())
        report=[]
        for trial,order in enumerate(TRIAL_FOLD_ORDERS):
            for howmany in SIZES:
                # Pooled fully disjoint Prompt folds, no label used to cluster.
                indices=np.flatnonzero(np.isin(folds,order[:howmany]))
                # Full N=2699 same corpus regardless fold order: do it once.
                if howmany==5 and trial>0:continue
                pred=blind_lexical_cluster(X[indices])
                stats=metric_groups(
                    pred,gold_prompt[indices],gold_model[indices])
                report.append({"trial":trial,"fold_count":howmany,**stats})
                print("NTL_0594_GROUP_SCALE_SLICE="+json.dumps({
                    "trial":trial,"fold_count":howmany,
                    "rows":stats["rows"],
                    "pair_precision":stats["pair_precision"],
                    "rows_in_mixed_prompt_processed_groups":
                        stats["rows_in_mixed_prompt_processed_groups"],
                    "minimum_wrong_rows_even_if_all_start_correct":
                        stats["minimum_wrong_rows_even_if_all_start_correct"]
                },sort_keys=True),flush=True)
    aggregate={}
    for size in SIZES:
        cohort=[r for r in report if r["fold_count"]==size]
        columns=("rows","pair_precision","pair_recall","pair_F1",
                 "rows_in_groups_oversized_skipped",
                 "rows_in_mixed_prompt_processed_groups",
                 "minimum_wrong_rows_even_if_all_start_correct",
                 "minimum_wrong_fraction_all_rows",
                 "predicted_groups","groups_with_duplicate_true_sources")
        aggregate[str(size)]={k:round(float(np.mean([r[k] for r in cohort])),7)
                  for k in columns}
        aggregate[str(size)]["trials"]=len(cohort)
    output={
        "experiment":"NTL 0.594 root-cause: hard source uniqueness structural feasibility vs group batch scale",
        "algorithm":"original word TFIDF cosine average-linkage distance .90, Hungarian groups <=18",
        "data":"official train only, 2699 rows, 150 prompts, five fold manifest",
        "not_hidden_kaggle_test":True,
        "no_live_champion_inference":True,
        "threshold_optimized":False,
        "seeds_preset":True,
        "class_prior_train":"one response per (prompt,model), no other source repeats per prompt",
        "meaning":"minimum_wrong_rows_even_if_all_start_correct is deterministic floor for an oracle classifier forced into incorrect discovered clusters, not observed 0.715 errors",
        "aggregates":aggregate,
        "raw_repeats":report,
        "official_score_old":0.715,
        "official_score_joint":0.594,
        "can_assign_exact_causality_to_hidden_test":False,
        "new_GPU":False,
        "new_submission":False,
    }
    print("NTL_0594_GROUP_SCALE_FINAL="+json.dumps(output,sort_keys=True),flush=True)


if __name__=="__main__":main()
