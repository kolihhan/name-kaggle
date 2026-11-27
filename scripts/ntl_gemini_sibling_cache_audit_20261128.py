"""Gemini sibling mechanism audit of the existing pinned historical OOF cache.
No fitting, inference, or candidate selection. Existing report's reference
score must reproduce. Historical rows are spent; not current v36 logits.
"""
import json
import numpy as np
import pandas as pd
from scripts import ntl_decompose_historical_oof_20261127 as src

def audit(arrays,train,report,scenarios):
    ids=arrays["ids"].astype(str);labels=arrays["labels"].astype(str)
    assert len(ids)==2699 and len(set(ids))==2699 and set(ids)==set(train.id)
    assert len(labels)==18 and len(set(labels))==18 and set(labels)==set(train.model)
    idx=pd.Index(ids).get_indexer(train.id)
    y=pd.Index(labels).get_indexer(train.model)
    assert np.array_equal(labels[arrays["y"].astype(int)][idx],train.model.to_numpy())
    names=arrays["component_names"].astype(str).tolist()
    assert len(names)==len(set(names)) and names==report["component_names"]
    assert arrays["components"].shape==(len(names),3,2699,18)
    google=np.flatnonzero(np.char.startswith(labels,"gemini-"))
    assert len(google)==4
    gmask=np.isin(y,google)
    flash=np.array([j for j in google if "flash" in labels[j]])
    assert len(flash)==3
    def ba(p):return float(np.mean([np.mean(p[y==k]==k) for k in range(18)]))
    def family(name):
        for k in ("qwen14","qwen8","q35","deberta","roberta","style","svm"):
            if name.startswith(k):return k
        return name.split("_")[0]
    for si,sc in enumerate(scenarios):
        a=arrays["reference"][si][idx]
        h=arrays["components"][:,si][:,idx,:]
        assert np.isfinite(a).all() and np.isfinite(h).all()
        p=a.argmax(1)
        expected=report["scenario_scores"]["fixed_reference"][str(sc)]
        assert min(abs(ba(p)-expected),abs(np.mean(p==y)-expected))<1e-8
        # Restrict to true Google rows for descriptive conditional oracle.
        # No oracle labels enter a deployed inference rule.
        cond=google[a[:,google].argmax(1)]
        hc=google[h[:,:,google].argmax(2)]
        valid=np.ptp(h[:,:,google],axis=2)>1e-10
        hit=(hc==y[None,:])&valid
        groups=sorted(set(map(family,names)))
        group_hit=np.array([hit[[family(n)==g for n in names]].any(0) for g in groups])
        per={}
        for k in google:
            m=y==k;miss=m&(cond!=y); fullmiss=m&(p!=y)
            per[str(labels[k])]={"n":int(m.sum()),"full_correct":int((m&(p==y)).sum()),
                "family_wrong":int((m&~np.isin(p,google)).sum()),
                "sibling_wrong":int((m&np.isin(p,google)&(p!=y)).sum()),
                "conditional_correct":int((m&(cond==y)).sum()),
                "conditional_miss":int(miss.sum()),
                "any_existing_head_recovers_conditional_miss":int((miss&hit.any(0)).sum()),
                "all_existing_heads_miss":int((miss&~hit.any(0)).sum()),
                "any_existing_head_correct_on_full_miss":int((fullmiss&hit.any(0)).sum()),
                "at_least_two_backbone_groups_correct_on_conditional_miss":int((miss&(group_hit.sum(0)>=2)).sum()),
                "conditional_confusion":{str(labels[j]):int((m&(cond==j)).sum()) for j in google}}
        missed=gmask&(cond!=y)
        # A positive head vote is an oracle upper bound, never a proposed routing rule.
        pairs={}
        for i,k in enumerate(flash):
            for j in flash[i+1:]:
                pair=np.isin(y,[k,j])
                chosen=np.where(a[:,k]>=a[:,j],k,j)
                pairs[labels[k]+" vs "+labels[j]]={"n":int(pair.sum()),"fixed_reference_pair_BA":float(np.mean([np.mean(chosen[y==z]==z) for z in (k,j)]))}
        src.log("GEMINI_SIBLING_CACHE_AUDIT",{"scenario":sc,"heads":len(names),"backbone_groups":groups,
            "reference_full18_BA":ba(p),"google_rows":int(gmask.sum()),"conditional_miss":int(missed.sum()),
            "oracle_any_head_recovers":int((missed&hit.any(0)).sum()),
            "oracle_all_heads_miss":int((missed&~hit.any(0)).sum()),
            "oracle_recovery_by_group":{g:int((missed&z).sum()) for g,z in zip(groups,group_hit)},
            "per_label":per,"flash_pair_controls":pairs,
            "scope":"spent historical publisher OOF; no current v36 row-aligned logits, no routing recommendation",
            "gpu":False,"training":False,"submission":False})
if __name__=="__main__":
    src.analyze=audit
    src.main()
