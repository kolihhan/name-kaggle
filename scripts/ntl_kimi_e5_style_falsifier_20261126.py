#!/usr/bin/env python3
"""No-GPU diagnostic: exact same 70/40/80 paired cohort as frozen E5.
Compare explicitly fixed style/length controls, not a leaderboard candidate.
"""
from __future__ import annotations
import hashlib, json, re
import numpy as np
import pandas as pd
from ntl_kimi_e5_frozen_transfer_20261126 import (
    A, B, SEED, N_TRAIN, N_VALID, N_EXTERNAL, datasets, pair, log
)

FEATURES=("emdash","curly_apostrophe","curly_quote","semicolon","comma",
          "colon","digit","longword","avg_token_chars","word_count")

def features(t):
    words=str(t).split()
    n=max(1,len(words))
    alpha=re.findall(r"[A-Za-z]+",str(t))
    return np.array([
        100*t.count("—")/n,
        100*t.count("’")/n,
        100*(t.count("“")+t.count("”"))/n,
        100*t.count(";")/n,
        100*t.count(",")/n,
        100*t.count(":")/n,
        100*len(re.findall(r"\d",t))/n,
        sum(len(x)>=9 for x in alpha)/max(1,len(alpha)),
        sum(len(x) for x in words)/n,
        len(words)
    ],dtype=np.float64)

def evaluate(a,b,d,mid):
    sa=(a-mid)@d
    sb=(b-mid)@d
    return {"balanced_accuracy":round(float((np.mean(sa>0)+np.mean(sb<0))/2),6),
            "kimi_recall":round(float(np.mean(sa>0)),6),
            "gpt_recall":round(float(np.mean(sb<0)),6),
            "paired_ranking_accuracy":round(float(np.mean(sa>sb)),6),
            "n_pairs":len(sa)}

def main():
    off, ext=datasets()
    oa,ob,o=pair(off)
    ea,eb,e=pair(ext)
    train=o[:N_TRAIN];valid=o[N_TRAIN:N_TRAIN+N_VALID]; external=e[:N_EXTERNAL]
    assert (len(train),len(valid),len(external))==(70,40,80)
    sets={}
    for tag,a,b,ids in (("train",oa,ob,train),("official_holdout",oa,ob,valid),
                        ("external_firstparty",ea,eb,external)):
        sets[tag]=(
            np.stack([features(str(a.loc[x,"text"])) for x in ids]),
            np.stack([features(str(b.loc[x,"text"])) for x in ids])
        )
    ta,tb=sets["train"]
    means=(ta.mean(0)+tb.mean(0))/2
    scale=np.concatenate([ta,tb]).std(0)
    scale=np.where(scale>1e-8,scale,1.0)
    # No search, no seed tuning: three prespecified simple proxy controls.
    controls={
        "emdash_only":(0,),
        "wordcount_only":(9,),
        "ten_style_stats":tuple(range(10))
    }
    for name,index in controls.items():
        idx=list(index)
        train_a,train_b=ta[:,idx]/scale[idx],tb[:,idx]/scale[idx]
        direction=train_a.mean(0)-train_b.mean(0)
        midpoint=(train_a.mean(0)+train_b.mean(0))/2
        if float(np.linalg.norm(direction))<1e-10:
            log("NTL_E5_STYLE_CONTROL",name=name,result="ZERO_DISCRIMINATIVE_DIRECTION");continue
        row={"control":name,"fit_pairs":len(train),"features":[FEATURES[i] for i in idx]}
        for tag in ("official_holdout","external_firstparty"):
            a,b=sets[tag]
            row[tag]=evaluate(a[:,idx]/scale[idx],b[:,idx]/scale[idx],direction,midpoint)
        log("NTL_E5_STYLE_CONTROL",**row)
    a,b=sets["external_firstparty"]
    dash_none=(a[:,0]==0)&(b[:,0]==0)
    similar_len=np.abs(a[:,9]-b[:,9])<=10
    log("NTL_E5_FALSIFICATION_STRATA",
        external_pairs=len(external),
        neither_has_emdash=int(dash_none.sum()),
        both_have_emdash=int(((a[:,0]>0)&(b[:,0]>0)).sum()),
        roughly_same_word_count=int(similar_len.sum()),
        neither_emdash_and_similar_len=int((dash_none&similar_len).sum()),
        exact_same_selection_as_e5=True)
    log("NTL_E5_STYLE_NO_TRAINING_NO_GPU_NO_OFFICIAL_SUBMISSION",success=True)

if __name__=="__main__":
    main()
