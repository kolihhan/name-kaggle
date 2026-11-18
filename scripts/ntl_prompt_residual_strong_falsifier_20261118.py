"""NTL Phase D falsifier: prompt residual effect vs strong lexical model, and shuffled groups.

Single frozen weight (.25); no train/validation prompt reuse. Explicitly tests whether
the apparent uplift depends on true content-driven grouping rather than adding
any independently trained model's log-probabilities.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from scripts.ntl_prompt_residual_sparse_v1_20261118 import (
    SEED, WEIGHT, blind_clusters, center, fixed_subset, grouped_ci, load
)

PREVIOUS_STRONG_BA = 0.32078299776286356
SHUFFLES = 5

def standard(a):
    centered=a-a.mean(axis=1,keepdims=True)
    return centered/np.maximum(centered.std(axis=1,keepdims=True),1e-6)

def blend(raw,residual):
    return ((1-WEIGHT)*standard(raw)+WEIGHT*standard(residual)).argmax(axis=1)

def run():
    d=load()
    labels,y=np.unique(d.model.to_numpy(),return_inverse=True)
    groups=d.prompt_id.to_numpy()
    folds=d.fold_id.to_numpy()
    n=len(d)
    arms={name:np.full(n,-1,dtype=np.int16) for name in
          ("strong","blind_blend","oracle_blend","null_blend_0","null_blend_1",
           "null_blend_2","null_blend_3","null_blend_4")}
    sparse=[]
    ari=[]
    rng=np.random.default_rng(SEED+1)
    with threadpool_limits(limits=2):
      for fold in range(5):
        tr=folds!=fold
        te=~tr
        tx=d.text[tr].astype(str).tolist()
        vx=d.text[te].astype(str).tolist()
        tfidf=TfidfVectorizer(analyzer="char",lowercase=False,ngram_range=(2,5),
            min_df=2,max_features=50000,sublinear_tf=True,norm="l2")
        mat=tfidf.fit_transform(tx)
        test=tfidf.transform(vx)
        strong=LogisticRegression(C=3.0,max_iter=500,solver="lbfgs",tol=1e-4)
        strong.fit(mat,y[tr])
        sprob=strong.predict_log_proba(test)
        svd=TruncatedSVD(n_components=128,n_iter=5,random_state=SEED)
        tr_embed=normalize(svd.fit_transform(mat))
        te_embed=normalize(svd.transform(test))
        residual=LogisticRegression(C=3.0,max_iter=500,solver="lbfgs")
        residual.fit(center(tr_embed,groups[tr]),y[tr])
        blind=blind_clusters(vx)
        ari.append(float(adjusted_rand_score(groups[te],blind)))
        pred={
          "strong":sprob.argmax(axis=1),
          "blind_blend":blend(sprob,residual.predict_log_proba(center(te_embed,blind))),
          "oracle_blend":blend(sprob,residual.predict_log_proba(center(te_embed,groups[te]))),
        }
        for i in range(SHUFFLES):
            # Independent random partition of SAME cluster sizes; label IDs alone
            # are meaningless, we permute row assignment rather than group names.
            shuffled=blind[rng.permutation(len(blind))]
            pred[f"null_blend_{i}"]=blend(
                sprob,residual.predict_log_proba(center(te_embed,shuffled)))
        indices=np.flatnonzero(te)
        for arm in pred:
            arms[arm][indices]=pred[arm]
        sub=fixed_subset(d.id[te].to_numpy(),groups[te],5)
        sparseblind=blind_clusters([vx[k] for k in sub])
        sparsehat=blend(sprob[sub],residual.predict_log_proba(
            center(te_embed[sub],sparseblind)))
        sparse.append((indices[sub],sparsehat,
               float(adjusted_rand_score(groups[te][sub],sparseblind))))
        print("NTL_PROMPT_FALSIFIER_FOLD",fold,"rows",len(indices),
              "cluster_ari",round(ari[-1],3),flush=True)
    if any(np.any(v<0) for v in arms.values()):
        raise RuntimeError("Missing OOF predictions")
    bas=float(balanced_accuracy_score(y,arms["strong"]))
    # Exact replication of prior stronger OOF linear baseline, not just comparable CV.
    if abs(bas-PREVIOUS_STRONG_BA)>1e-12:
        raise RuntimeError(f"Strong OOF reproduction drift: {bas}")
    result={}
    for arm,hat in arms.items():
        if arm=="strong":continue
        value=float(balanced_accuracy_score(y,hat))
        result[arm]={
           "ba":value,"delta_vs_strong":value-bas,
           "ci95":grouped_ci(y,d.id.to_numpy(),groups,arms["strong"],hat),
           "rescued":int(((arms["strong"]!=y)&(hat==y)).sum()),
           "regressed":int(((arms["strong"]==y)&(hat!=y)).sum())
        }
    idx=np.concatenate([x[0] for x in sparse])
    hat=np.concatenate([x[1] for x in sparse])
    sy=y[idx]
    base=arms["strong"][idx]
    sparse_ba=float(balanced_accuracy_score(sy,hat))
    sparse_bas=float(balanced_accuracy_score(sy,base))
    result["sparse_k5"]={
      "n":len(idx),"strong_ba":sparse_bas,"blind_blend_ba":sparse_ba,
      "delta_vs_strong":sparse_ba-sparse_bas,
      "ci95":grouped_ci(sy,d.id.to_numpy()[idx],groups[idx],base,hat),
      "mean_cluster_ari":float(np.mean([x[2] for x in sparse])),
      "rescued":int(((base!=sy)&(hat==sy)).sum()),
      "regressed":int(((base==sy)&(hat!=sy)).sum()),
    }
    nulls=[result[f"null_blend_{i}"]["delta_vs_strong"] for i in range(SHUFFLES)]
    ctrl=result["blind_blend"]
    sparse_ctrl=result["sparse_k5"]
    # Single preregistered gate for mechanism on stronger baseline.
    gate=(ctrl["ci95"][0]>0 and sparse_ctrl["ci95"][0]>0
          and ctrl["delta_vs_strong"]>max(nulls))
    out={
      "version":"ntl-prompt-falsifier-strong-v1",
      "strong_oof_ba":bas,
      "weight":WEIGHT,
      "rows":n,"prompts":len(set(groups)),"folds":5,"labels":len(labels),
      "source_sha":"pinned by imported loader",
      "arms":result,
      "group_ari_full_mean":float(np.mean(ari)),
      "random_partition_deltas":nulls,
      "preregistered_gate":"blind_ci_low>0 and sparse_k5_ci_low>0 and blind_delta>max(5_random_partition_deltas)",
      "gate_pass":bool(gate),
      "model_sources":"frozen grouped official train; fold-local all fits",
      "metric_caveat":"OOF CV baseline 0.3208 is far below strongest Public BA 0.691; this does not establish marginal gain over that model",
      "gpu":False,"kaggle_submission":False
    }
    Path("result").mkdir(exist_ok=True)
    Path("result/ntl-prompt-falsifier-strong-v1.json").write_text(json.dumps(out,indent=2))
    print("NTL_STRONG_PROMPT_FALSIFIER_RESULT="+json.dumps(out,sort_keys=True),flush=True)
    return out

if __name__=="__main__":run()
