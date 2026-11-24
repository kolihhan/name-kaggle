"""Cross-publisher C1 and C2 replication for frozen 14-feature prompt-rank fingerprints.

This is an observational test on already-used publisher-labeled panels,
NOT a hidden-test classifier, not model training, and no new GPU/submit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata

from scripts.ntl_18source_fingerprint_invariance_20261124 import (
    FEATURES, F, SEED, OFFICIAL_SHA, featurize, sha,
)

C1_SHA="8746c4b77c2b09b9abafaeecc381a39321b9919214c005a3657042051c51b177"
C2_SHA="bf25ef1e953d5c37b177f27b841a04098ae9b35f45fc43fa11036d91e8831e02"

def load(path,expected,rows,prompts):
    if sha(path)!=expected:raise ValueError("data hash mismatch")
    z=pd.read_csv(path,dtype={"prompt_id":str,"label":str})
    if len(z)!=rows or z.prompt_id.nunique()!=prompts or z.label.nunique()!=12:
        raise ValueError("publisher rows, prompts or labels drift")
    if z.text.isna().any():raise ValueError("missing text")
    return z.rename(columns={"label":"model"})

def build_blocks(z,labels):
    out=[]
    used=set()
    for prompt,g in z.groupby("prompt_id"):
        if len(g)!=len(labels) or set(g.model)!=set(labels) or g.model.duplicated().any():
            continue
        by=g.set_index("model")
        out.append(np.stack([featurize(by.loc[l,"text"]) for l in labels]))
        used.add(prompt)
    if not out:return np.empty((0,len(labels),F)),len(z)-len(used)*len(labels)
    return np.stack(out),len(z)-len(used)*len(labels)

def rank_block(raw):
    count=raw.shape[1]
    assert count==12
    r=np.stack([rankdata(raw[:,:,k],axis=1,method="average") for k in range(F)],axis=2)
    return (r-(count+1)/2)/((count-1)/2)

def corr_by_feature(a,b):
    aa=a.mean(0);bb=b.mean(0)
    vals=[]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(F):
            q=spearmanr(aa[:,k],bb[:,k]).statistic
            vals.append(float(q) if np.isfinite(q) else 0.)
    return np.array(vals),aa,bb

def similarity(official,external):
    x=official/(np.linalg.norm(official,axis=1,keepdims=True)+1e-12)
    y=external/(np.linalg.norm(external,axis=1,keepdims=True)+1e-12)
    return y@x.T

def comparison(a,b,labels,reps=2000):
    assert a.shape[1:]==b.shape[1:]==(12,F)
    obs,oa,bb=corr_by_feature(a,b)
    s=similarity(oa,bb)
    ranked=np.argsort(-s,axis=1)
    matches=[labels[i] for i in range(12) if ranked[i,0]==i]
    misses={labels[i]:labels[ranked[i,0]] for i in range(12) if ranked[i,0]!=i}
    top3=sum(i in ranked[i,:3] for i in range(12))
    rng=np.random.default_rng(SEED)
    nullmax=[];nullcorrect=[]
    for j in range(reps):
        shuffled=np.stack([row[rng.permutation(12)] for row in b])
        v,_,means=corr_by_feature(a,shuffled)
        nullmax.append(np.max(np.abs(v)))
        nullcorrect.append(int((similarity(oa,means).argmax(1)==np.arange(12)).sum()))
    nullmax=np.asarray(nullmax);nullcorrect=np.asarray(nullcorrect)
    ci=np.zeros((500,F))
    for j in range(500):
        x=a[rng.integers(0,len(a),len(a))]
        y=b[rng.integers(0,len(b),len(b))]
        ci[j]=corr_by_feature(x,y)[0]
    items=[]
    for k,name in enumerate(FEATURES):
        low,high=np.quantile(ci[:,k],[.025,.975])
        items.append({
          "feature":name,
          "rho":round(float(obs[k]),4),
          "bootstrap_95":[round(float(low),4),round(float(high),4)],
          "maxT_FWER_p":round(float((1+(nullmax>=abs(obs[k])).sum())/(1+reps)),4),
          "mean_abs_prompt_rank_delta":round(float(np.mean(np.abs(oa[:,k]-bb[:,k]))),4),
        })
    return {"prompt_blocks_A":len(a),"prompt_blocks_B":len(b),"true_class_count":len(labels),
      "centroid_label_top1":len(matches),"centroid_label_top3":int(top3),
      "null_p_top1":round(float((1+(nullcorrect>=len(matches)).sum())/(reps+1)),4),
      "matching_labels":matches,"mismatched_label_nearest":misses,
      "features":items,"uses_privileged_per_prompt_classes":True,"no_hidden_BA_estimate":True}

def main(paths):
    start=time.monotonic()
    if sha(paths.official)!=OFFICIAL_SHA:raise ValueError("official hash drift")
    t=pd.read_csv(paths.official,dtype={"id":str,"prompt_id":str,"model":str})
    if len(t)!=2699 or t.model.nunique()!=18 or t.prompt_id.nunique()!=150 or not t.id.is_unique:
        raise ValueError("official train drift")
    c1=load(paths.c1,C1_SHA,360,30)
    c2=load(paths.c2,C2_SHA,470,50)
    labels=sorted(c1.model.unique())
    if not set(labels)<=set(t.model) or set(c2.model)!=set(labels):
        raise ValueError("panel class coverage mismatch")
    oa,unofficial=build_blocks(t,labels)
    c1a,c1unused=build_blocks(c1,labels)
    c2a,c2unused=build_blocks(c2,labels)
    if not len(oa) in (149,150):raise ValueError("original complete prompt count drift")
    if len(c1a)!=30:raise ValueError("C1 complete prompt count drift")
    data={"official_train_blocks12":len(oa),"c1_blocks12":len(c1a),"c1_unusable_rows":c1unused,
          "c2_blocks12":len(c2a),"c2_unusable_rows":c2unused,
          "exact_label_names":labels,"publisher_C1_C2_shared":True,
          "publisher_identity_not_independently_authenticated":True,
          "methods":"frozen 14 nontrainable style cues, within-prompt ranks",
          "no_train_no_gpu_no_competition_submission":True}
    ra=rank_block(oa);rb=rank_block(c1a)
    data["official_vs_C1"]=comparison(ra,rb,labels)
    if len(c2a)>=5:
        rc=rank_block(c2a)
        data["official_vs_C2"]=comparison(ra,rc,labels)
    else:
        data["C2_not_matched"]={"reason":"<5 full12 prompt blocks; never treat partial blocks as paired"}
    data["seconds"]=round(time.monotonic()-start,2)
    print("NTL_CROSSPUBLISHER_C1_C2_REAL="+json.dumps(data,sort_keys=True),flush=True)

def smoke():
    vals=rank_block(np.tile(np.arange(12)[None,:,None],(3,1,F)))
    assert vals.shape==(3,12,F) and np.allclose(vals.mean(axis=1),0)
    q=featurize("I'm ready — no doubt.")
    assert len(q)==F and q[6]>0
    print("NTL_CROSSPUBLISHER_SELFTEST=PASS",flush=True)

if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--smoke",action="store_true")
    for x in ("official","c1","c2"):pa.add_argument("--"+x,type=Path)
    args=pa.parse_args()
    smoke()
    if not args.smoke:
        if any(getattr(args,k) is None for k in ("official","c1","c2")):pa.error("all inputs required")
        main(args)
