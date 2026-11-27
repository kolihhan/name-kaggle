"""Test whether existing sibling evidence is observable without oracle routing.
Same pinned historical cache, no fits or hyperparameter sweeps. Diagnostic only.
"""
import numpy as np
import pandas as pd
from scripts import ntl_decompose_historical_oof_20261127 as src

def evaluate(a,h,y,google,names):
    groups={}
    for i,name in enumerate(names):
        g=next((x for x in ("qwen14","qwen8","q35","deberta","roberta","style","svm") if name.startswith(x)),name.split("_")[0])
        groups.setdefault(g,[]).append(i)
    n=len(y)
    gmask=np.isin(y,google)
    # Four-class scores only; family detection is explicitly not assumed solved.
    a4=a[:,google]
    cond=google[a4.argmax(1)]
    p=a.argmax(1)
    v=h[:,:,google]
    valid=np.ptp(v,axis=2)>1e-10
    winners=v.argmax(2)
    # Equal total weight per backbone group, not per repeated correlated head.
    counts=np.stack([((winners==k)&valid).astype(float) for k in range(4)],axis=2)
    total=np.zeros((n,4));num=np.zeros(n)
    for indices in groups.values():
        k=valid[indices].sum(0)
        total+=counts[indices].sum(0)/np.maximum(k[:,None],1)
        num+=(k>0)
    shares=total/np.maximum(num[:,None],1)
    # Ties keep original decision, no new arbitrary class-order bias.
    top=shares.max(1)
    unique=(shares==top[:,None]).sum(1)==1
    vote=np.where(unique,google[shares.argmax(1)],cond)
    # A single fixed comparison: family-conditional equal-backbone vote.
    # Full-18 counterfactual only if the baseline already predicts Google.
    revised=np.where(np.isin(p,google),vote,p)
    def paired(mask,b,c):
        x=b[mask]==y[mask];z=c[mask]==y[mask]
        return {"n":int(mask.sum()),"baseline_correct":int(x.sum()),"vote_correct":int(z.sum()),
                "rescues":int((~x&z).sum()),"regressions":int((x&~z).sum())}
    truth_index=np.searchsorted(google,y).clip(0,3)
    true_share=shares[np.arange(n),truth_index]
    wrong_shares=shares.copy()
    wrong_shares[np.arange(n),truth_index]=-1
    wrong_max=wrong_shares.max(1)
    misses=gmask&(cond!=y)
    # Continuous evidence: does correct-class vote mass outrank all alternatives?
    # Also count strong support on wrong alternatives, not just good votes.
    support={"misses":int(misses.sum()),
        "truth_unique_vote_winner":int((misses&(true_share>wrong_max)).sum()),
        "truth_tied_vote_winner":int((misses&(true_share==wrong_max)).sum()),
        "false_vote_winner":int((misses&(true_share<wrong_max)).sum()),
        "truth_in_reference_top2":int((misses&np.any(google[np.argsort(-a4,axis=1,kind="stable")[:,:2]]==y[:,None],axis=1)).sum()),
        "truth_in_vote_top2":int((misses&np.any(google[np.argsort(-shares,axis=1,kind="stable")[:,:2]]==y[:,None],axis=1)).sum())}
    byclass={str(k):paired(y==k,cond,vote) for k in google}
    # Count disagreement false positives on baseline-correct vs wrong rows.
    for k,val in byclass.items():assert val["vote_correct"]-val["baseline_correct"]==val["rescues"]-val["regressions"]
    return {"conditional":paired(gmask,cond,vote),"full18":paired(np.ones(n,bool),p,revised),
        "full18_BA_base":float(np.mean([np.mean(p[y==k]==k) for k in range(18)])),
        "full18_BA_vote":float(np.mean([np.mean(revised[y==k]==k) for k in range(18)])),
        "support":support,"per_class_index":byclass,"groups":list(groups),"vote_ties_true_google":int((gmask&~unique).sum())}

def audit(arrays,train,report,scenarios):
    ids=arrays["ids"].astype(str);labels=arrays["labels"].astype(str)
    assert len(ids)==len(set(ids))==len(train)==2699 and set(ids)==set(train.id)
    assert len(labels)==len(set(labels))==18 and set(labels)==set(train.model)
    idx=pd.Index(ids).get_indexer(train.id)
    y=pd.Index(labels).get_indexer(train.model)
    assert np.array_equal(labels[arrays["y"].astype(int)][idx],train.model.to_numpy())
    names=arrays["component_names"].astype(str).tolist()
    assert names==report["component_names"] and len(names)==len(set(names))
    assert arrays["components"].shape==(len(names),3,2699,18)
    google=np.flatnonzero(np.char.startswith(labels,"gemini-"))
    assert len(google)==4
    for si,sc in enumerate(scenarios):
        a=arrays["reference"][si][idx];h=arrays["components"][:,si][:,idx,:]
        assert np.isfinite(a).all() and np.isfinite(h).all()
        p=a.argmax(1)
        ba=float(np.mean([np.mean(p[y==k]==k) for k in range(18)]))
        exp=report["scenario_scores"]["fixed_reference"][str(sc)]
        assert min(abs(ba-exp),abs((p==y).mean()-exp))<1e-8
        r=evaluate(a,h,y,google,names)
        r["per_class"]={labels[int(k)]:v for k,v in r.pop("per_class_index").items()}
        src.log("GEMINI_OBSERVABLE_VOTE_DIAGNOSTIC",{"scenario":sc,"heads":len(names),**r,
            "scope":"spent historical cache; fixed non-fitted control, not a proposed candidate or v36 transfer result",
            "gpu":False,"training":False,"submission":False})
if __name__=="__main__":
    src.analyze=audit
    src.main()
