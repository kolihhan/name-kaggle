"""One frozen sparse-set evaluation of the original semantic-set joint scorer.

Re-executes exact source inference and five-fold predictions with NO changes
to the published algorithm or fitting. Only extracts the unreported k=5
per-fold outcome and the complete-prompt comparator from the original
evaluate() output. No hyperparameter search, no public LB tuning.

Important: mean per-fold BA is NOT a global OOF BA, so label accordingly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot
from scripts.ntl_semantic_set_decoding_20261122 import (
    embed, evaluate, MODEL,
)

ARMS=("raw","joint_semantic","joint_lexical","joint_random","joint_oracle")
SEED=20261122


def summarize(result):
    full=result["details"]["full"]
    sparse=result["details"]["k5"]
    if len(full)!=5 or len(sparse)!=5:
        raise ValueError("must have 5 real GroupKFold panels")
    if [x["fold"] for x in sparse]!=list(range(5)):
        raise ValueError("sparse fold mismatch")
    for k in sparse:
        if k["rows"]!=150:
            raise ValueError("sparse cohort must have 5 examples x 30 prompts")
        if set(k["ba"])!=set(ARMS):
            raise ValueError("missing fixed arm")
    if any(set(k["ba"])!=set(ARMS) for k in full):
        raise ValueError("full cohort arms missing")

    def avg(rows,name):
        return float(np.mean([r[name] for r in rows]))
    def bars(rows):
        return {arm:float(np.mean([r["ba"][arm] for r in rows])) for arm in ARMS}
    k5=bars(sparse); large=bars(full)
    ci_known=result["comparisons"]
    # This k5 statistic is a mean over 5 fold BA18, not an OOF bootstrap CI.
    return {
        "protocol":"ntl-set-decoder-frozen-sparse-report-20261122",
        "scientific_scope":"already-spent official train GroupKFold",
        "representation":MODEL,
        "initial_mechanism_threshold_changed":False,
        "repeated_training_or_tuning":False,
        "full_fold_mean_ba18":large,
        "k5_fold_mean_ba18":k5,
        "k5_delta_lexical_vs_raw":k5["joint_lexical"]-k5["raw"],
        "k5_delta_semantic_vs_raw":k5["joint_semantic"]-k5["raw"],
        "k5_delta_random_vs_raw":k5["joint_random"]-k5["raw"],
        "k5_delta_oracle_vs_raw":k5["joint_oracle"]-k5["raw"],
        "k5_prompt_pair_f1_lexical":avg([r["lexical"] for r in sparse],"pair_f1"),
        "k5_prompt_pair_precision_lexical":avg([r["lexical"] for r in sparse],"precision"),
        "k5_prompt_pair_recall_lexical":avg([r["lexical"] for r in sparse],"recall"),
        "k5_prompt_pair_f1_semantic":avg([r["semantic"] for r in sparse],"pair_f1"),
        "k5_prompt_pair_precision_semantic":avg([r["semantic"] for r in sparse],"precision"),
        "k5_prompt_pair_recall_semantic":avg([r["semantic"] for r in sparse],"recall"),
        "k5_rows":sum(x["rows"] for x in sparse),
        "full_cluster_pair_f1_lexical":avg([r["lexical"] for r in full],"pair_f1"),
        "full_cluster_pair_precision_lexical":avg([r["lexical"] for r in full],"precision"),
        "full_cluster_pair_recall_lexical":avg([r["lexical"] for r in full],"recall"),
        "full_cluster_pair_f1_semantic":avg([r["semantic"] for r in full],"pair_f1"),
        "full_cluster_pair_precision_semantic":avg([r["semantic"] for r in full],"precision"),
        "full_cluster_pair_recall_semantic":avg([r["semantic"] for r in full],"recall"),
        "fold_details":[{
            "fold":int(r["fold"]),
            "rows":int(r["rows"]),
            "ba":r["ba"],
            "lex_pair_f1":float(r["lexical"]["pair_f1"]),
            "sem_pair_f1":float(r["semantic"]["pair_f1"])
        } for r in sparse],
        "full_oof_ba18":result["full_ba"],
        "full_prompt_cluster_ci95_for_lexical":ci_known["joint_lexical"]["prompt_ci95"],
        "sparse_prompt_cluster_ci95":"NOT_COMPUTED__NO_ROWWISE_PREDICTIONS_STORED",
        "cannot_assume_hidden_test_unique_model_per_prompt":True,
        "0p715_champion_logits_present":False,
        "competition_submission":False,
        "gpu":False,
    }


def main():
    frame=load_snapshot(Path("data/train.csv"),Path("artifacts/phase1/folds.csv"))
    with threadpool_limits(limits=2):
        z=embed(frame.text.astype(str).tolist())
        result=evaluate(frame,z)
        report=summarize(result)
    print("NTL_SPARSE_SET_REAL_K5="+json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
