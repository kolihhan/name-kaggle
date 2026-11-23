"""Nested prompt-grouped OOF test of learned fusion: 53 global + token evidence.

No model forwards. Reuses the exact SHA-pinned GLM/Flash token likelihood caches.
Outer evaluation is the frozen five-fold prompt-disjoint official TRAIN subset
(300 genuine GLM sibling-labelled rows / 150 prompts). Within each outer fold,
choose only the L2 regularization C and token-block scale alpha by inner
StratifiedGroupKFold. All scaling and fitting happen inside folds.

C0: historical 53-feature global-only logistic regression, fixed C=1.
C1: global + token features, fixed C=1, token scale=1.
C2: token-only, fixed C=1 (diagnostic).
C3: global + token, inner-CV-selected C/scale (primary).

This is a small, spent official TRAIN cohort. It cannot validate 18-class
transfer or predict the hidden leaderboard. No post-hoc threshold tuning.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from scripts.ntl_glm_distributed_evidence_20261122 import (
    SEED, FULL, FLASH, STATS, load_cache, paired_rows, fixed_summary
)
from scripts.ntl_atlas_exact_token_readout_a0_20261123 import original_paper_prod

TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLDS_SHA="94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"
EXPECTED_BASE_BA=0.9433333333333334
CANDIDATES=[(1.0,0.0),(0.1,0.5),(0.1,1.0),(1.0,0.25),
            (1.0,0.5),(1.0,1.0),(10.0,0.25),(10.0,0.5),(10.0,1.0)]

def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def token_features(a,b):
    """Small, predeclared token-evidence block; no labels or fitting."""
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    if a.ndim!=1 or b.ndim!=1 or len(a)!=len(b) or len(a)<16:
        raise ValueError("ALIGNED_TOKEN_STREAMS_REQUIRED")
    d=a-b
    q=np.percentile(d,[10,25,50,75,90])
    clipped=np.clip(d,q[0],q[4])
    # The exact ATLAS bounded product readout is retained as a learned feature,
    # rather than used as a standalone decision.
    prod=original_paper_prod(np.stack([a,b]))
    prod_margin=float(np.log(prod[0]+1e-300)-np.log(prod[1]+1e-300))
    quarters=np.array_split(d,4)
    return np.asarray([
        float(np.median(d)), float(d.std()), float(q[0]), float(q[1]),
        float(q[3]), float(q[4]), float(np.mean(np.abs(d))),
        float(clipped.mean()), prod_margin,
        float(quarters[0].mean()), float(quarters[1].mean()),
        float(quarters[2].mean()), float(quarters[3].mean())
    ],dtype=np.float64)

def make_global(fullfeat,flashfeat,ids,paired):
    f=fullfeat.set_index("id").loc[ids]
    l=flashfeat.set_index("id").loc[ids]
    xbase=np.asarray([
        np.r_[f.loc[k,list(STATS)].to_numpy(dtype=float),
              l.loc[k,list(STATS)].to_numpy(dtype=float)]
        for k in ids],dtype=np.float64)
    pair=np.asarray([fixed_summary(a,b) for _,a,b in paired],dtype=np.float64)
    x=np.c_[xbase,pair]
    if x.shape[1]!=53 or not np.isfinite(x).all():
        raise ValueError("EXPECTED_53_FINITE_GLOBAL_FEATURES")
    return x

def fit_predict(xtr,ytr,xva,C,alpha,n_global=53):
    mu=xtr.mean(axis=0)
    sd=xtr.std(axis=0)
    sd[sd<1e-10]=1.0
    ztr=(xtr-mu)/sd
    zva=(xva-mu)/sd
    if alpha!=1.0:
        ztr[:,n_global:]*=alpha
        zva[:,n_global:]*=alpha
    clf=LogisticRegression(C=float(C),random_state=SEED,
                           class_weight="balanced",max_iter=3000)
    clf.fit(ztr,ytr)
    return clf.predict(zva)

def ba(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return float(.5*(np.mean(p[y==1]==1)+np.mean(p[y==0]==0)))

def paired_summary(y,base,new,groups,reps=2000):
    y=np.asarray(y);base=np.asarray(base);new=np.asarray(new)
    keys=np.unique(groups)
    members=[np.flatnonzero(groups==k) for k in keys]
    rng=np.random.default_rng(20261123)
    deltas=[]
    for _ in range(reps):
        ix=np.concatenate([members[j] for j in rng.integers(len(members),size=len(members))])
        if len(np.unique(y[ix]))<2: continue
        deltas.append(ba(y[ix],new[ix])-ba(y[ix],base[ix]))
    return {
        "changed":int(np.sum(base!=new)),
        "rescues":int(np.sum((base!=y)&(new==y))),
        "regressions":int(np.sum((base==y)&(new!=y))),
        "net_correct":int(np.sum(new==y)-np.sum(base==y)),
        "delta_BA":round(ba(y,new)-ba(y,base),8),
        "prompt_bootstrap_95":[round(float(v),7) for v in np.quantile(deltas,[.025,.975])],
        "bootstrap_reps":len(deltas)
    }

def run(args):
    if sha256(args.train)!=TRAIN_SHA or sha256(args.folds)!=FOLDS_SHA:
        raise ValueError("PINNED_TRAIN_OR_FOLD_SHA_DRIFT")
    train=pd.read_csv(args.train,dtype={"id":str,"prompt_id":str,"model":str})
    folds=pd.read_csv(args.folds,dtype={"id":str,"prompt_id":str,"model":str})
    ff=pd.read_csv(args.glm53_features,dtype={"id":str})
    lf=pd.read_csv(args.flash_features,dtype={"id":str})
    full=load_cache(args.glm53_scores,ff)
    flash=load_cache(args.flash_scores,lf)
    if len(train)!=2699 or train.id.duplicated().any() or train.prompt_id.nunique()!=150:
        raise ValueError("OFFICIAL_TRAIN_COHORT_DRIFT")
    if not np.array_equal(full["ids"],flash["ids"]) or set(full["ids"])!=set(train.id):
        raise ValueError("TOKEN_CACHE_ID_COVERAGE_OR_ORDER_MISMATCH")
    merged=train.merge(folds[["id","prompt_id","model","fold_id"]],
                       on="id",suffixes=("","_fold"),validate="one_to_one")
    if (merged.prompt_id!=merged.prompt_id_fold).any() or (merged.model!=merged.model_fold).any():
        raise ValueError("FROZEN_FOLD_LABEL_MISMATCH")
    ix=merged.set_index("id").loc[list(full["ids"])]
    labels=ix.model.to_numpy(dtype=str)
    prompts=ix.prompt_id.to_numpy(dtype=str)
    fold=ix.fold_id.to_numpy(dtype=int)
    allowed=np.isin(labels,[FULL,FLASH])
    if int(allowed.sum())!=300 or len(set(prompts[allowed]))!=150:
        raise ValueError("EXPECTED_300_GLMS_150_PROMPTS")
    y=(labels[allowed]==FULL).astype(int)
    g=prompts[allowed]
    outer=fold[allowed]
    if set(outer)!=set(range(5)):
        raise ValueError("MISSING_FROZEN_OUTER_FOLD")
    for p in np.unique(g):
        if len(np.unique(outer[g==p]))!=1:
            raise ValueError("PROMPT_LEAKAGE_ACROSS_OUTER_FOLDS")
    paired=list(paired_rows(full,flash))
    ids=list(full["ids"])
    Xg_all=make_global(ff,lf,ids,paired)
    Xt_all=np.asarray([token_features(a,b) for _,a,b in paired],dtype=np.float64)
    Xg=Xg_all[allowed]; Xt=Xt_all[allowed]
    if Xt.shape!=(300,13) or not np.isfinite(Xt).all():
        raise ValueError("TOKEN_FEATURE_BLOCK_INVALID")
    X=np.c_[Xg,Xt]
    pred0=np.full(300,-1,dtype=int)
    pred1=np.full(300,-1,dtype=int)
    pred2=np.full(300,-1,dtype=int)
    pred3=np.full(300,-1,dtype=int)
    selected={}
    inner_scores={}
    for k in range(5):
        tr=outer!=k; va=outer==k
        # Historical baseline: identical feature matrix, fixed C=1.
        pred0[va]=fit_predict(Xg[tr],y[tr],Xg[va],1.0,0.0,n_global=53)
        # Direct concat, fixed C=1.
        pred1[va]=fit_predict(X[tr],y[tr],X[va],1.0,1.0,n_global=53)
        # Token-only diagnostic.
        pred2[va]=fit_predict(Xt[tr],y[tr],Xt[va],1.0,1.0,n_global=0)
        # Nested prompt-grouped selection: only outer-train rows enter selection.
        inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SEED+k)
        scores={}
        for C,alpha in CANDIDATES:
            vals=[]
            for it,iv in inner.split(X[tr],y[tr],groups=g[tr]):
                phat=fit_predict(X[tr][it],y[tr][it],X[tr][iv],C,alpha,n_global=53)
                vals.append(ba(y[tr][iv],phat))
            scores[(C,alpha)]=float(np.mean(vals))
        # Stable simplicity tie-break: higher mean inner BA, then smaller alpha,
        # then smaller C. Candidate set is frozen above before outer predictions.
        best=sorted(scores,key=lambda z:(-scores[z],z[1],z[0]))[0]
        selected[str(k)]={"C":best[0],"token_block_scale":best[1],
                          "inner_mean_BA":round(scores[best],7)}
        inner_scores[str(k)]={f"C{c}_A{a}":round(v,7) for (c,a),v in scores.items()}
        pred3[va]=fit_predict(X[tr],y[tr],X[va],best[0],best[1],n_global=53)
    for name,p in (("C0_global",pred0),("C1_global_plus_token_fixed",pred1),
                   ("C2_token_only",pred2),("C3_global_plus_token_nested",pred3)):
        if (p<0).any():raise ValueError("INCOMPLETE_OOF:"+name)
    scores={name:{"BA":round(ba(y,p),8),"correct":int((p==y).sum()),
                  "recall_full":round(float(np.mean(p[y==1]==1)),7),
                  "recall_flash":round(float(np.mean(p[y==0]==0)),7)}
            for name,p in (("C0_global",pred0),("C1_global_plus_token_fixed",pred1),
                           ("C2_token_only",pred2),("C3_global_plus_token_nested",pred3))}
    if not np.isclose(scores["C0_global"]["BA"],EXPECTED_BASE_BA,atol=1e-8):
        raise RuntimeError("HISTORICAL_53_FEATURE_BASELINE_NOT_REPRODUCED:"+str(scores["C0_global"]))
    fold_delta={str(k):{
        "fixed_concat_accuracy_delta":round(float(np.mean(pred1[outer==k]==y[outer==k])-np.mean(pred0[outer==k]==y[outer==k])),7),
        "nested_fusion_accuracy_delta":round(float(np.mean(pred3[outer==k]==y[outer==k])-np.mean(pred0[outer==k]==y[outer==k])),7)
    } for k in range(5)}
    record={
      "experiment":"NTL_GLOBAL53_PLUS_TOKEN13_NESTED_PROMPT_GROUPED_OOF",
      "official_train_rows":2699,"evaluated_real_glm_rows":300,
      "prompt_groups":150,"outer_folds":5,"inner_folds":4,
      "global_feature_count":53,"token_feature_count":13,
      "token_feature_names":["delta_median","delta_std","delta_q10","delta_q25","delta_q75","delta_q90","delta_mean_abs","delta_trimmed_mean","atlas_prod_log_margin","delta_quarter_1","delta_quarter_2","delta_quarter_3","delta_quarter_4"],
      "scores":scores,
      "paired_vs_global":{
        "fixed_concat":paired_summary(y,pred0,pred1,g),
        "nested_fusion":paired_summary(y,pred0,pred3,g),
        "token_only":paired_summary(y,pred0,pred2,g)
      },
      "outer_fold_delta":fold_delta,
      "selected_hyperparameters_by_outer_fold":selected,
      "inner_grid_scores":inner_scores,
      "primary_gate":{
        "nested_beats_global_by_at_least_1pp":bool(scores["C3_global_plus_token_nested"]["BA"]>=EXPECTED_BASE_BA+0.01),
        "nested_prompt_bootstrap_lower_bound_gt_zero":bool(paired_summary(y,pred0,pred3,g)["prompt_bootstrap_95"][0]>0),
        "nested_positive_outer_folds_at_least_4_of_5":bool(sum(v["nested_fusion_accuracy_delta"]>0 for v in fold_delta.values())>=4),
        "promote":False
      },
      "scope":"reused_spent_official_train_GLm_sibling_subset_only_not_prospective",
      "no_new_model_forward":True,"no_paid_api":True,"no_qlora":True,
      "no_gpu":True,"no_kaggle_submission":True,
      "official_0p715_18way_comparator_unavailable":True
    }
    print("NTL_GLOBAL_TOKEN_NESTED_FUSION="+json.dumps(record,sort_keys=True),flush=True)
    return record

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("train","folds","glm53_features","flash_features","glm53_scores","flash_scores"):
        p.add_argument("--"+name,required=True,type=Path)
    run(p.parse_args())
