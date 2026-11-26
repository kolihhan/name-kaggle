#!/usr/bin/env python3
"""Same-row E5-vs-style diagnostic, frozen E5 protocol and first-party source.

Strict no new model fit except already frozen 70-prompt class centroids,
no GPU, no Kaggle code competition submit or kernel push.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from ntl_kimi_e5_frozen_transfer_20261126 import (
    datasets, pair, embed, SEED, N_TRAIN, N_VALID, N_EXTERNAL, metrics, log
)
from ntl_kimi_e5_style_falsifier_20261126 import features

def decision(a,b,d,mid):
    return (a-mid)@d>0,(b-mid)@d>0, (a-b)@d>0

def test():
    off,ext=datasets()
    oa,ob,oi=pair(off)
    ea,eb,ei=pair(ext)
    tr=oi[:N_TRAIN]
    val=oi[N_TRAIN:N_TRAIN+N_VALID]
    ex=ei[:N_EXTERNAL]
    assert len(tr)==70 and len(val)==40 and len(ex)==80
    traintexts=[str(oa.loc[i,"text"]) for i in tr]+[str(ob.loc[i,"text"]) for i in tr]
    valtexts=[str(oa.loc[i,"text"]) for i in val]+[str(ob.loc[i,"text"]) for i in val]
    extexts=[str(ea.loc[i,"text"]) for i in ex]+[str(eb.loc[i,"text"]) for i in ex]
    sha=lambda s:hashlib.sha256(" ".join(s.casefold().split()).encode()).hexdigest()
    assert not set(map(sha,traintexts+valtexts)) & set(map(sha,extexts))
    n=len(tr); v=len(val); e=len(ex)
    rows=embed(traintexts+valtexts+extexts)
    assert rows.shape==(2*(n+v+e),1024)
    a,b=rows[:n],rows[n:2*n]
    ea_,eb_=rows[2*n+2*v:2*n+2*v+e],rows[2*n+2*v+e:]
    direction=a.mean(0)-b.mean(0)
    center=(a.mean(0)+b.mean(0))/2
    ek,eg,ep=decision(ea_,eb_,direction,center)
    eBA=float((np.mean(ek)+np.mean(~eg))/2)
    assert abs(eBA-.7)<1e-5,('frozen_E5_baseline_drift',eBA)

    # Frozen style-only controls: exact same definitions and supervised 70-prompt
    # centroids as successful style falsifier, not re-tuned to E5 results.
    af=np.stack([features(s) for s in traintexts[:n]])
    bf=np.stack([features(s) for s in traintexts[n:]])
    fx=np.stack([features(s) for s in extexts[:e]])
    fy=np.stack([features(s) for s in extexts[e:]])
    sigma=np.concatenate([af,bf]).std(0)
    sigma=np.where(sigma>1e-8,sigma,1.0)
    results={}
    joint={}
    for name,idx in [('emdash_only',[0]),('ten_style_stats',list(range(10))),('wordcount_only',[9])]:
        a,b=af[:,idx]/sigma[idx],bf[:,idx]/sigma[idx]
        x,y=fx[:,idx]/sigma[idx],fy[:,idx]/sigma[idx]
        direction=a.mean(0)-b.mean(0); mid=(a.mean(0)+b.mean(0))/2
        k,g,p=decision(x,y,direction,mid)
        bacc=float((np.mean(k)+np.mean(~g))/2)
        base_known={'emdash_only':.7,'ten_style_stats':.61875,'wordcount_only':.5}
        assert abs(bacc-base_known[name])<1e-5,('frozen_style_baseline_drift',name,bacc)
        # Pooled per-example Kimi and GPT correctness with correct truth.
        ec=np.concatenate([ek,~eg])
        sc=np.concatenate([k,~g])
        e_only=np.logical_and(ec,~sc)
        s_only=np.logical_and(sc,~ec)
        vals={
            'style_ba':round(bacc,6),
            'e5_only_correct':int(e_only.sum()),
            'style_only_correct':int(s_only.sum()),
            'both_correct':int(np.logical_and(ec,sc).sum()),
            'both_wrong':int(np.logical_and(~ec,~sc).sum()),
            'e5_only_correct_kimi':int(e_only[:e].sum()),
            'e5_only_correct_gpt':int(e_only[e:].sum()),
            'style_only_correct_kimi':int(s_only[:e].sum()),
            'style_only_correct_gpt':int(s_only[e:].sum()),
            'pair_rank_style':float(np.mean(p)),
            'pair_rank_e5':float(np.mean(ep))
        }
        assert vals['e5_only_correct']-vals['style_only_correct']==int(
            round((eBA-bacc)*2*e))
        results[name]=vals
        # Paired cluster bootstrap, draw prompt ID together for both labels.
        rng=np.random.default_rng(SEED)
        gaps=[]
        rankgaps=[]
        for _ in range(2000):
            ix=rng.integers(0,e,e)
            delta=(np.mean(ek[ix])+np.mean(~eg[ix])-
                   np.mean(k[ix])-np.mean(~g[ix]))/2
            gaps.append(delta)
            rankgaps.append(np.mean(ep[ix])-np.mean(p[ix]))
        results[name]['e5_minus_style_BA_95CI']=[float(v) for v in np.quantile(gaps,[.025,.975])]
        results[name]['e5_minus_style_pairrank_95CI']=[float(v) for v in np.quantile(rankgaps,[.025,.975])]
        log("NTL_E5_SAME_ROW_CONTROL",control=name,**results[name])
        if name=='emdash_only':
            # Subgroup with same em dash presence, genuinely hard for simple style.
            neutral=(fx[:,0]==0)&(fy[:,0]==0)
            both=(fx[:,0]>0)&(fy[:,0]>0)
            diff=~(neutral|both)
            for title,mask in [('both_no_emdash',neutral),('both_have_emdash',both),('one_has_emdash',diff)]:
                if mask.sum()==0:continue
                log('NTL_E5_STYLE_NEUTRAL_STRATUM',stratum=title,pairs=int(mask.sum()),
                    e5_binary_BA=float((np.mean(ek[mask])+np.mean(~eg[mask]))/2),
                    dash_binary_BA=float((np.mean(k[mask])+np.mean(~g[mask]))/2),
                    e5_pairrank=float(np.mean(ep[mask])),
                    dash_pairrank=float(np.mean(p[mask])))
    log("NTL_E5_SAME_ROW_CONCLUSION",
        scope="binary Kimi versus Luna only, no 18-class incumbent rescues measured",
        e5_ba=eBA,source_rows=e*2,control_rows_verified=True,
        no_kaggle_submission=True,no_gpu=True,no_encoder_training=True)

if __name__=="__main__":test()
