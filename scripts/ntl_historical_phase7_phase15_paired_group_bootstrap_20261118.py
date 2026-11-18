"""Real historical Phase 7 vs Phase 15 same-row paired prompt bootstrap.

No new training, model inference, Kaggle submission or model selection.
Two OLD already-adaptively-used OOF predictions are retrieved from original
authenticated GitHub Actions artifacts and checked against current frozen
official training labels and fold IDs. Cluster bootstrap examines historical
estimator volatility; it CANNOT make prefit-contaminated folds honest.

Predeclared:
  N=2699 official rows, 150 prompt clusters, 18 classes, 5 frozen folds.
  Observed historical candidate Delta BA=0.0352697 vs historical public Delta -0.012.
  Illustrative practical target (not a universal medal threshold): +0.05 BA;
  detectability to be estimated from prompt-cluster paired differences.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE=0.3700521998508576
CHALLENGER=0.4053218990802883
SEED=20261118
BOOT=8000
TARGET=.05


def main():
    p=argparse.ArgumentParser()
    for a in ("--train","--folds","--phase7","--phase15","--out"):
        p.add_argument(a,required=True,type=Path)
    a=p.parse_args()
    train=pd.read_csv(a.train,dtype={"id":str,"prompt_id":str})
    f=pd.read_csv(a.folds,dtype={"id":str,"prompt_id":str})
    p7=pd.read_csv(a.phase7,dtype={"id":str,"prompt_id":str})
    p15=pd.read_csv(a.phase15,dtype={"id":str,"prompt_id":str})
    cols=["id","prompt_id","model","fold_id"]
    for d,name in ((train,"train"),(f,"folds"),(p7,"phase7"),(p15,"phase15")):
        if len(d)!=2699 or d.id.duplicated().any():
            raise RuntimeError(name+" INDEX_FAIL")
    if not ({"prediction"}<=set(p7) and {"prediction"}<=set(p15)):
        raise RuntimeError("OLD_OOF_PREDICTIONS_NOT_PRESENT")
    if set(train.prompt_id)!=set(f.prompt_id) or train.prompt_id.nunique()!=150:
        raise RuntimeError("150_PROMPT_GROUPS_DRIFT")
    true=train[["id","prompt_id","model"]].merge(
        f[["id","fold_id"]],on="id",validate="one_to_one")
    if len(true)!=2699:
        raise RuntimeError("OFFICIAL_TRUE_JOIN_DRIFT")
    for original,name in ((p7,"p7"),(p15,"p15")):
        chk=true[cols].sort_values("id",kind="stable").reset_index(drop=True)
        got=original[cols].sort_values("id",kind="stable").reset_index(drop=True)
        if not chk.astype(str).equals(got.astype(str)):
            raise RuntimeError("OOF_META_NOT_EXACTLY_MATCHED_"+name)
        if original.prediction.isna().any():
            raise RuntimeError("EMPTY_OOF_"+name)
    tr=true.merge(p7[["id","prediction"]].rename(columns={"prediction":"p7"}),
                  on="id",validate="one_to_one")
    tr=tr.merge(p15[["id","prediction"]].rename(columns={"prediction":"p15"}),
                  on="id",validate="one_to_one")
    labs=sorted(tr.model.astype(str).unique())
    if len(labs)!=18 or not set(tr.p7).issubset(labs) or not set(tr.p15).issubset(labs):
        raise RuntimeError("OOF_LABEL_SET_DRIFT")
    true_y=tr.model.to_numpy(dtype=str)
    y7=tr.p7.to_numpy(dtype=str)
    y15=tr.p15.to_numpy(dtype=str)
    c7=true_y==y7
    c15=true_y==y15
    scores=lambda good: float(np.mean([np.mean(good[true_y==k]) for k in labs]))
    bs7=scores(c7);bs15=scores(c15)
    if abs(bs7-BASELINE)>1e-9 or abs(bs15-CHALLENGER)>1e-9:
        raise RuntimeError("HISTORIC_OOF_BA_FINGERPRINT_DRIFT")
    groups=sorted(tr.prompt_id.astype(str).unique())
    counts=np.empty((len(groups),len(labs)),dtype=int)
    wins7=np.empty_like(counts);wins15=np.empty_like(counts)
    prompt=tr.prompt_id.astype(str).to_numpy()
    for gi,g in enumerate(groups):
        for ci,model in enumerate(labs):
            sel=(prompt==g)&(true_y==model)
            counts[gi,ci]=int(sel.sum())
            wins7[gi,ci]=int((sel&c7).sum())
            wins15[gi,ci]=int((sel&c15).sum())
    if int(counts.sum())!=2699 or len(groups)!=150:
        raise RuntimeError("GROUP_COUNTER_FAIL")
    rng=np.random.default_rng(SEED)
    diffs=[]
    for _ in range(BOOT):
        idx=rng.integers(len(groups),size=len(groups))
        n=counts[idx].sum(axis=0)
        if not np.all(n):
            continue
        d=((wins15[idx].sum(axis=0)-wins7[idx].sum(axis=0))/n).mean()
        diffs.append(float(d))
    if len(diffs)<7000:raise RuntimeError("BOOTSTRAP_CLASS_DROPOUT")
    sample=np.array(diffs)
    low,high=np.quantile(sample,[.025,.975])
    std=float(sample.std(ddof=1))
    mde=(1.959963984540054+0.8416212335729143)*std
    per_fold={}
    fold=tr.fold_id.to_numpy()
    for i in range(5):
        ix=fold==i
        by_class=[float(np.mean(c15[ix&(true_y==lab)]))-float(np.mean(c7[ix&(true_y==lab)])) for lab in labs]
        per_fold[str(i)]={"rows":int(ix.sum()),
                          "paired_macro_BA_delta":float(np.mean(by_class)),
                          "rescues":int((~c7[ix]&c15[ix]).sum()),
                          "regressions":int((c7[ix]&~c15[ix]).sum())}
    per_label={lab:{"rows":int((true_y==lab).sum()),
                    "C0_recall":float(c7[true_y==lab].mean()),
                    "C1_recall":float(c15[true_y==lab].mean()),
                    "rescues":int(((~c7)&c15&(true_y==lab)).sum()),
                    "regressions":int((c7&(~c15)&(true_y==lab)).sum())}
                for lab in labs}
    q=float((c7!=c15).mean())
    result={
        "experiment":"HISTORICAL_PHASE7_VS_PHASE15_EXACT_PAIR_150_CLUSTER_BOOTSTRAP",
        "sample_rows":len(tr),
        "distinct_prompt_clusters":len(groups),
        "frozen_fold_count":5,
        "all_18_truth_classes":True,
        "official_prompt_group_disjoint":True,
        "C0_phase7_OOF_BA":round(bs7,6),
        "C1_phase15_OOF_BA":round(bs15,6),
        "historical_OOF_delta":round(bs15-bs7,6),
        "historical_public_delta":round(.357-.369,6),
        "rescues":int((~c7&c15).sum()),
        "regressions":int((c7&~c15).sum()),
        "discordance_rate":round(q,6),
        "per_prompt_cluster_paired_bootstrap_95_CI":[round(float(low),6),round(float(high),6)],
        "retrospective_optimistic_paired_MDE_80pct":round(mde,6),
        "preregistered_practical_delta_target":TARGET,
        "prereg_expected_historical_delta":round(CHALLENGER-BASELINE,6),
        "per_fold":per_fold,
        "per_label":per_label,
        "source_lineage_leakage_untested_by_bootstrap":True,
        "adaptive_reuse_of_150_groups":True,
        "not_new_independent_validation":True,
        "lightning_0691_honest_OOF_available":False,
        "new_model_training":False,
        "model_inference":False,
        "new_competition_submission":False,
        "raw_text_exported":False,
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("NTL_HISTORIC_150GROUP_REAL_PAIRED_CV="+json.dumps({
        k:result[k] for k in
        ("sample_rows","distinct_prompt_clusters","C0_phase7_OOF_BA",
         "C1_phase15_OOF_BA","historical_OOF_delta",
         "historical_public_delta","rescues","regressions","discordance_rate",
         "per_prompt_cluster_paired_bootstrap_95_CI",
         "retrospective_optimistic_paired_MDE_80pct","per_fold")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
