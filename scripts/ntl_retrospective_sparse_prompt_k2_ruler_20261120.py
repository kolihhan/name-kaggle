"""Retrospective sparse-prompt validation ruler: one historical predeclared pair.

Question: can blindly taking k=2 per prompt make Phase7/15 grouped OOF
prefer the Public-winning Phase7, when full grouped OOF prefers Phase15?
NO new trained models, no label-informed selection, no Kaggle submission.
Training-source and champion full-ensemble OOF limitations remain.

The complete expected outcome (full OOF: +0.035270, Public -0.012)
was established historically; this is a *spent, retrospective diagnostics*,
not a calibration claim or a new independent evaluation benchmark.

Each simulation samples k=1,2,4 rows without replacement *within each of
150 prompt groups*, making no change to the individual heldout predictions.
The model predictions are row-independent at this evaluation step; changed
scores reflect sample selection and estimator variance, not model improvement.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED=20261120
SIMULATIONS=5000
KS=(1,2,4)
EXPECTED7=.3700521998508576
EXPECTED15=.4053218990802883

def build_frame(train_path,folds_path,p7_path,p15_path):
    d=[]
    for path in (train_path,folds_path,p7_path,p15_path):
        d.append(pd.read_csv(path,dtype={"id":str,"prompt_id":str}))
    train,folds,p7,p15=d
    assert all(len(x)==2699 and not x.id.duplicated().any() for x in d)
    assert len(set(train.prompt_id))==150
    expected=train[["id","prompt_id","model"]].merge(folds[["id","fold_id"]],
                                                       on="id",validate="one_to_one")
    cols=["id","prompt_id","model","fold_id"]
    for got in [p7,p15]:
        if "prediction" not in got.columns:raise RuntimeError("MISSING_OLD_PREDICTIONS")
        a=got[cols].sort_values("id",kind="stable").reset_index(drop=True)
        b=expected[cols].sort_values("id",kind="stable").reset_index(drop=True)
        if not a.astype(str).equals(b.astype(str)):
            raise RuntimeError("ORIGINAL_OOF_PROVENANCE_MISMATCH")
    merged=expected.merge(p7[["id","prediction"]].rename(columns={"prediction":"p7"}),
                          on="id",validate="one_to_one")
    merged=merged.merge(p15[["id","prediction"]].rename(columns={"prediction":"p15"}),
                        on="id",validate="one_to_one")
    merged=merged.sort_values(["prompt_id","id"],kind="stable").reset_index(drop=True)
    return merged

def macro_ba(truth,pred,classes):
    scores=[float((truth[truth==k]==pred[truth==k]).mean()) for k in classes]
    return float(np.mean(scores))

def main():
    p=argparse.ArgumentParser()
    for name in ("train","folds","phase7","phase15","out"):
        p.add_argument("--"+name,required=True,type=Path)
    args=p.parse_args()
    df=build_frame(args.train,args.folds,args.phase7,args.phase15)
    labels=sorted(df.model.unique().tolist())
    if len(labels)!=18:raise RuntimeError("UNEXPECTED_LABEL_COUNT")
    yy=df.model.to_numpy(dtype=str)
    z7=df.p7.to_numpy(dtype=str)
    z15=df.p15.to_numpy(dtype=str)
    if not (set(z7)<=set(labels) and set(z15)<=set(labels)):
        raise RuntimeError("INVALID_LABEL_PREDICTIONS")
    baseline=macro_ba(yy,z7,labels)
    challenger=macro_ba(yy,z15,labels)
    if abs(baseline-EXPECTED7)>1e-10 or abs(challenger-EXPECTED15)>1e-10:
        raise RuntimeError("HISTORICAL_PAIRED_OOF_FINGERPRINT_DRIFT")
    # Prompt groups: 149 groups of size 18, one group of size 17.
    group_indices=[np.asarray(v.index,dtype=int) for _,v in df.groupby("prompt_id",sort=True)]
    if len(group_indices)!=150 or sorted(set(map(len,group_indices)))!=[17,18]:
        raise RuntimeError("GROUP_SIZE_DRIFT")
    classidx=pd.Categorical(df.model,categories=labels).codes.astype(np.int16)
    correct7=(z7==yy).astype(np.int8)
    correct15=(z15==yy).astype(np.int8)
    r=np.random.default_rng(SEED)
    report={}
    for k in KS:
        chosen=np.empty((SIMULATIONS,len(group_indices),k),dtype=np.int32)
        for g,ids in enumerate(group_indices):
            # One independent uniform k-combination from each prompt for each
            # simulation; same row subset is used for BOTH contenders.
            rand=r.random((SIMULATIONS,len(ids)),dtype=np.float64)
            local=np.argpartition(rand,kth=k-1,axis=1)[:,:k]
            chosen[:,g,:]=ids[local]
        pick=chosen.reshape(SIMULATIONS,-1)
        rid=np.broadcast_to(np.arange(SIMULATIONS,dtype=np.int32)[:,None],pick.shape)
        n=np.zeros((SIMULATIONS,len(labels)),dtype=np.int32)
        win7=np.zeros_like(n)
        win15=np.zeros_like(n)
        ix=(rid,classidx[pick])
        np.add.at(n,ix,1)
        np.add.at(win7,ix,correct7[pick])
        np.add.at(win15,ix,correct15[pick])
        valid=(n>0).all(axis=1)
        if valid.sum()<int(SIMULATIONS*.93):
            raise RuntimeError("TOO_FEW_ALL_CLASS_SPARSE_DRAWS")
        ba7=(win7[valid]/n[valid]).mean(axis=1)
        ba15=(win15[valid]/n[valid]).mean(axis=1)
        diff=ba15-ba7
        report[str(k)]={
            "rows_per_draw":int(k*len(group_indices)),
            "valid_draws":int(valid.sum()),
            "empty_class_draws":int((~valid).sum()),
            "phase7_mean_BA":round(float(ba7.mean()),6),
            "phase15_mean_BA":round(float(ba15.mean()),6),
            "phase15_minus_phase7_mean":round(float(diff.mean()),6),
            "diff_quantiles_2p5_50_97p5":[round(float(v),6) for v in np.quantile(diff,[.025,.5,.975])],
            "draws_preferring_public_winner_phase7":int((diff<0).sum()),
            "fraction_preferring_public_winner_phase7":round(float((diff<0).mean()),6),
            "fraction_tie":round(float((diff==0).mean()),6),
            "conditional_only_on_existing_OOF":True,
        }
    result={
        "experiment":"NTL_RETROSPECTIVE_K2_PROMPT_RULER_PHASE7_PHASE15",
        "seed":SEED,"simulations_per_k":SIMULATIONS,"groups":150,
        "official_train_rows":2699,
        "baseline_full_18class_OOF":round(baseline,6),
        "challenger_full_18class_OOF":round(challenger,6),
        "full_OOF_delta":round(challenger-baseline,6),
        "historical_public_baseline":.369,
        "historical_public_challenger":.357,
        "historical_public_delta":-.012,
        "k_paired_subsample_diagnostics":report,
        "only_two_historical_pipelines":True,
        "cannot_calibrate_cross_pipeline_ranking_or_pearson_correlation":True,
        "no_inference_retraining_GPU_or_submission":True,
        "no_0702_champion_true_OOF_claim":True,
        "no_new_independent_data":True,
        "source_provenance_generalization_unresolved":True,
    }
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NTL_SPARSITY_K2_PAIRED_OOF_RESULT="+json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
