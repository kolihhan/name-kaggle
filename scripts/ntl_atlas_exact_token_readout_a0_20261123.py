"""ATLAS 2026's *argmax-prod readout* applied to REAL aligned GLM token scores.

This is an original-paper readout ablation, NOT faithful ATLAS model training:
our GLM/Flash token probabilities were independently produced by the two
native models, not a shared ATLAS LM under learned source-control prefixes.

Frozen BEFORE seeing this experiment's scores:
  C_mean: mean_t p(model|token) via average next-token log-likelihood.
  C_prod: exact paper main readout, mean over tokens and T in {1,2,4,8},
          using exp(-sum_j relu(s_j - s_k) / T).
  C_global: existing strong, five-fold, prompt-disjoint GLM 2-way model
            using 2x24 published globals + the 5 global comparison features.
  C_sign: unsupervised token-win proportion, descriptive only.
All arms use precisely the same two source token score streams and 300 genuine
GLM-labelled official TRAIN rows. Zero model forwards, zero paid APIs, no new
fitting except reproduction of previously-scored C_global; no hidden inference.

Scientific LIMIT: repeated official TRAIN / historical spent corpus. Not
prospective transfer, and no 0.715 final logits, so NO Public score claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.ntl_glm_distributed_evidence_20261122 import (
    SEED, FULL, FLASH, STATS, fixed_summary, load_cache, paired_rows,
)

TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLDS_SHA="94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"
TEMPERATURES=(1.0,2.0,4.0,8.0)
EXPECTED_C_GLOBAL=0.9433333333333334
EXPECTED_TOTAL=300

def fingerprint(path:Path,expected:str):
    with path.open("rb") as f:
        sha=hashlib.file_digest(f,"sha256").hexdigest()
    if sha!=expected:
        raise ValueError(f"PINNED_FILE_SHA_DRIFT:{path.name}:expected:{expected}:got:{sha}")
    return sha

def original_paper_prod(log_probs:np.ndarray,temps=TEMPERATURES)->np.ndarray:
    """Exact official src/atlas/model.py:_prod_readout semantics.

    log_probs: [K, L]; token losses aligned across all K conditions.
    Returns [K] averaged across valid tokens and prespecified temperatures.
    """
    lp=np.asarray(log_probs,dtype=np.float64)
    if lp.ndim!=2 or lp.shape[0]<2 or lp.shape[1]<1 or not np.isfinite(lp).all():
        raise ValueError("REAL_ALIGNED_LOGPROB_MATRIX_REQUIRED")
    k,t=lp.shape
    # Official: d[k,j,t] = s_k-s_j; exp[-sum_j relu(s_j-s_k)/T].
    diffs=lp[:,None,:]-lp[None,:,:]
    opponents=np.maximum(-diffs,0.0).sum(axis=1)
    out=np.array([np.exp(-opponents/temp).mean(axis=1)
                  for temp in temps],dtype=np.float64).mean(axis=0)
    if out.shape!=(k,) or not np.isfinite(out).all():
        raise ValueError("ATLAS_READOUT_NOT_FINITE")
    if (out<0).any() or (out>1+1e-12).any():
        raise ValueError("ATLAS_READOUT_NOT_BOUNDED")
    return out

def fixed_prediction(full_scores,flash_scores):
    a=np.asarray(full_scores,dtype=np.float64)
    b=np.asarray(flash_scores,dtype=np.float64)
    if a.ndim!=1 or b.ndim!=1 or len(a)!=len(b) or not len(a):
        raise ValueError("GLM_TOKEN_STREAM_SHAPES_MISMATCH")
    logprobs=np.stack([a,b])
    paper=original_paper_prod(logprobs)
    prior=logprobs.mean(axis=1)
    win=np.array([(a>b).mean(),(b>a).mean()])
    return {
        "C_mean":FULL if prior[0]>=prior[1] else FLASH,
        "C_prod":FULL if paper[0]>=paper[1] else FLASH,
        "C_sign":FULL if win[0]>=win[1] else FLASH,
        "mean_margin":float(prior[0]-prior[1]),
        "prod_margin":float(paper[0]-paper[1]),
        "n_tokens":len(a),
        "mean_tie":bool(np.isclose(prior[0],prior[1],atol=1e-12,rtol=0)),
        "prod_tie":bool(np.isclose(paper[0],paper[1],atol=1e-12,rtol=0))
    }

def score(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if set(y)!={FULL,FLASH} or len(y)!=len(p):
        raise ValueError("BINARY_LABEL_MISMATCH")
    per={c:float(np.mean(p[y==c]==c)) for c in (FULL,FLASH)}
    return {"BA":round(float(np.mean(list(per.values()))),8),
            "correct":int(np.sum(y==p)),
            "recall":{k:round(v,7) for k,v in per.items()}}

def compare(y,base,new,prompt,reps=1000):
    y=np.asarray(y);base=np.asarray(base);new=np.asarray(new)
    prompts=np.asarray(prompt,dtype=str)
    if len(y)!=len(base) or len(y)!=len(new) or len(y)!=len(prompts):
        raise ValueError("ROW_ALIGNMENT_BAD")
    c0=base==y;c1=new==y
    keys=np.unique(prompts)
    groups=[np.flatnonzero(prompts==q) for q in keys]
    rng=np.random.default_rng(20261123)
    diffs=[]
    for _ in range(reps):
        ids=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        if set(y[ids])!={FULL,FLASH}:continue
        diff=score(y[ids],new[ids])["BA"]-score(y[ids],base[ids])["BA"]
        diffs.append(float(diff))
    return {
       "changed_labels":int(np.sum(base!=new)),
       "rescues":int(np.sum((~c0)&c1)),
       "regressions":int(np.sum(c0&(~c1))),
       "net_correct":int(c1.sum()-c0.sum()),
       "delta_BA":round(score(y,new)["BA"]-score(y,base)["BA"],8),
       "bootstrap_prompt_count":int(len(keys)),
       "bootstrap_conditional95":[round(float(v),7)
          for v in np.quantile(diffs,[.025,.975])] if diffs else None,
       "spent_prompt_clusters_bootstrap_not_prospective_evidence":True
    }

def original_global_heldout(featfull,featflash,paired,indices,y,fold):
    """Reproduce fixed historical global C0 only (not an ATLAS fitted head)."""
    ids=[row[0] for row in paired]
    a=featfull.set_index("id").loc[ids]
    b=featflash.set_index("id").loc[ids]
    xbase=np.array([
      np.r_[a.loc[k,list(STATS)].to_numpy(dtype=float),
            b.loc[k,list(STATS)].to_numpy(dtype=float)]
      for k in ids],dtype=float)
    diff=np.array([fixed_summary(row[1],row[2]) for row in paired],dtype=float)
    x=np.concatenate([xbase,diff],axis=1)[indices]
    yy=(y==FULL).astype(int)
    if not np.isfinite(x).all() or set(fold)!=set(range(5)):
        raise ValueError("INVALID_STRONG_GLOBAL_FEATURES_OR_FOLDS")
    preds=np.empty(len(y),dtype=object)
    for k in range(5):
        train=fold!=k;valid=fold==k
        clf=make_pipeline(StandardScaler(),
               LogisticRegression(C=1.,random_state=SEED,
                                  class_weight="balanced",max_iter=2000))
        clf.fit(x[train],yy[train])
        preds[valid]=np.where(clf.predict(x[valid])==1,FULL,FLASH)
    measured=score(y,preds)["BA"]
    if not np.isclose(measured,EXPECTED_C_GLOBAL,atol=1e-6):
        raise RuntimeError("HISTORICAL_STRONG_C0_REPRODUCTION_FAILED:"+str(measured))
    return preds

def run(args):
    fingerprint(args.train,TRAIN_SHA)
    fingerprint(args.folds,FOLDS_SHA)
    train=pd.read_csv(args.train,dtype={"id":str,"prompt_id":str,"model":str})
    folds=pd.read_csv(args.folds,dtype={"id":str,"prompt_id":str,"model":str})
    if (len(train)!=2699 or train.prompt_id.nunique()!=150 or
        len(folds)!=2699 or train.id.duplicated().any()):
        raise ValueError("OFFICIAL_COHORT_DRIFT")
    fullfeat=pd.read_csv(args.glm53_features,dtype={"id":str})
    flashfeat=pd.read_csv(args.flash_features,dtype={"id":str})
    full=load_cache(args.glm53_scores,fullfeat)
    flash=load_cache(args.flash_scores,flashfeat)
    if (len(full["ids"])!=2699 or len(flash["ids"])!=2699 or
        not np.array_equal(full["ids"],flash["ids"]) or
        set(full["ids"])!=set(train.id)):
        raise ValueError("TOKEN_CACHE_ID_ALIGNMENT_FAILURE")
    joined=train.merge(folds[["id","prompt_id","model","fold_id"]],
                       on="id",suffixes=("","_fold"),validate="one_to_one")
    if (len(joined)!=2699 or not joined.id.is_unique or
        not (joined.prompt_id==joined.prompt_id_fold).all() or
        not (joined.model==joined.model_fold).all() or
        joined.groupby("prompt_id").fold_id.nunique().max()!=1):
        raise ValueError("PROMPT_GROUP_OR_TRAIN_LABEL_DRIFT")
    indexed=joined.set_index("id").loc[list(full["ids"])]
    labels=indexed.model.to_numpy(dtype=str)
    eligible=np.isin(labels,[FULL,FLASH])
    if int(eligible.sum())!=EXPECTED_TOTAL:
        raise ValueError("GLM_BINARY_SAMPLE_DRIFT")
    label=labels[eligible]
    prompt=indexed.prompt_id.to_numpy(dtype=str)[eligible]
    fold=indexed.fold_id.to_numpy(dtype=int)[eligible]
    paired=list(paired_rows(full,flash))
    # The joined subset must contain one or both GLM responses per prompt,
    # never train on an externally generated or label-leaked substitute.
    pred_by_arm={k:[] for k in ("C_mean","C_prod","C_sign")}
    margins={"mean":[],"prod":[]}
    lengths=[]
    tied={"mean":0,"prod":0}
    for i,(key,a,b) in enumerate(paired):
        if not eligible[i]:continue
        result=fixed_prediction(a,b)
        for k in pred_by_arm:pred_by_arm[k].append(result[k])
        margins["mean"].append(result["mean_margin"])
        margins["prod"].append(result["prod_margin"])
        lengths.append(result["n_tokens"])
        tied["mean"]+=result["mean_tie"]
        tied["prod"]+=result["prod_tie"]
    pred={k:np.asarray(v,dtype=str) for k,v in pred_by_arm.items()}
    pred["C_global"]=original_global_heldout(fullfeat,flashfeat,paired,np.flatnonzero(eligible),
                                               label,fold)
    stats={k:score(label,v) for k,v in pred.items()}
    pairs={
       "C_prod_vs_mean":compare(label,pred["C_mean"],pred["C_prod"],prompt),
       "C_prod_vs_global":compare(label,pred["C_global"],pred["C_prod"],prompt),
       "C_mean_vs_global":compare(label,pred["C_global"],pred["C_mean"],prompt),
       "C_sign_vs_mean":compare(label,pred["C_mean"],pred["C_sign"],prompt)
    }
    byfold={str(k):{"rows":int((fold==k).sum()),
                    "prod_minus_mean_accuracy":round(
                        float((pred["C_prod"][fold==k]==label[fold==k]).mean()-
                              (pred["C_mean"][fold==k]==label[fold==k]).mean()),7),
                    "prod_minus_global_accuracy":round(
                        float((pred["C_prod"][fold==k]==label[fold==k]).mean()-
                              (pred["C_global"][fold==k]==label[fold==k]).mean()),7)}
                    for k in range(5)}
    gates={
      "paper_readout_has_mean_BA_increment":
        bool(stats["C_prod"]["BA"]>stats["C_mean"]["BA"]),
      "paper_readout_beats_historical_strong_global_BA":
        bool(stats["C_prod"]["BA"]>stats["C_global"]["BA"]),
      "paper_readout_increment_prompt_bootstrap_CI_gt_zero":
        bool(pairs["C_prod_vs_mean"]["bootstrap_conditional95"][0]>0),
      "promote_to_champion":False
    }
    record={
      "experiment":"ATLAS_2026_exact_argmax_prod_readout_TWO_GLM_ONLY",
      "original_ATLAS_source":"https://github.com/komseok0109/ATLAS/blob/main/src/atlas/model.py",
      "original_paper_source_conditioning_retrained":False,
      "official_labelled_rows":2699,"tested_binary_glm_rows":len(label),
      "tested_prompt_groups":len(set(prompt)),
      "fixed_temperatures":list(TEMPERATURES),
      "score_cache_rows":len(full["ids"]),
      "original_token_alignment_confirmed":True,
      "full_lp_column":full["lp_column"],
      "flash_lp_column":flash["lp_column"],
      "token_count_min":min(lengths),"token_count_max":max(lengths),
      "token_count_median":int(np.median(lengths)),
      "approx_ties":tied,
      "scores":stats,"paired":pairs,
      "fold_diagnostics":byfold,
      "decision":gates,
      "scope":"repeatedly_consumed_official_train_prompt_heldout_glm_binary_mechanism_only",
      "no_full_18_0p715_pair":True,
      "no_external_source_transfer":True,
      "no_model_forward":True,
      "no_paid_API":True,
      "no_training_except_reproduced_historical_C0":True,
      "no_Kaggle_competition_submission":True
    }
    print("NTL_ATLAS_REAL_GLM_TOKEN_A0="+json.dumps(record,sort_keys=True),flush=True)
    return record

def smoke():
    # Formula must match exact torch tensor semantics of ATLAS original for K=2
    # on balanced typical and outlier samples. No synthetic claims of accuracy.
    cases=[
       [[-1.,-2.,-3.],[-2.,-2.,-5.]],
       [[-1.,-1.,-1.,-1.],[-1.2,-1.2,-1.2,-1.2]],
       [[-1.,-1.,-1.,-10.],[-2.,-2.,-2.,-2.]],
       [[-1.,-6.,-4.,-3.],[-5.,-2.,-2.,-4.]]
    ]
    for sample in cases:
        lp=np.asarray(sample,dtype=float)
        observed=original_paper_prod(lp)
        expected=[]
        for temp in TEMPERATURES:
            matrix=[]
            for k in range(len(lp)):
                token=[]
                for t in range(lp.shape[1]):
                    penalty=sum(max(float(lp[j,t]-lp[k,t]),0.) for j in range(len(lp)))
                    token.append(np.exp(-penalty/temp))
                matrix.append(np.mean(token))
            expected.append(matrix)
        assert np.allclose(observed,np.mean(expected,axis=0),rtol=0,atol=1e-14)
        assert (observed>=0).all() and (observed<=1).all()
    # When every token is won by full, paper readout must vote full.
    z=fixed_prediction([-1.,-2.,-1.],[-4.,-3.,-5.])
    assert z["C_prod"]==FULL and z["C_mean"]==FULL
    # A common score translation per token must cancel in cross-model product;
    # this property does not hold for raw nonaligned per-model token sequences.
    lp=np.asarray(cases[2])
    offset=np.asarray([2.,4.,-5.,1.])
    assert np.allclose(original_paper_prod(lp),original_paper_prod(lp+offset))
    print("NTL_ATLAS_CPU_SMOKE="+json.dumps({
       "pass":True,
       "formula_matched_original_source_exact":True,
       "different_score_scenarios":len(cases),
       "source_control_training":False,
       "real_label_results":None,
       "paid_API":False},sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("train","folds","glm53_features","flash_features","glm53_scores","flash_scores"):
        p.add_argument("--"+n,required=False,type=Path)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    smoke()
    if not args.smoke:
        if any(getattr(args,k) is None for k in
              ("train","folds","glm53_features","flash_features","glm53_scores","flash_scores")):
            p.error("real label/cache paths required unless --smoke")
        run(args)
