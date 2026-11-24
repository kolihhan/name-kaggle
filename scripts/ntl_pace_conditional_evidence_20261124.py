"""PACE v0: source-conditioned native likelihood evidence; historical 18-way OOF.

No new model forward, raw text output, pretrained-weight training or submission.
All histogram boundaries and class estimators fit within each frozen prompt fold.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scripts.ntl_atlas_exact_token_readout_a0_20261123 import TRAIN_SHA,FOLDS_SHA,fingerprint
from scripts.ntl_glm_distributed_evidence_20261122 import STATS,SEED,load_cache,paired_rows,fixed_summary

BINS=8
EPS=1e-12
ALPHA=1.0
ARMS=("A_marginal","B_conditional","C_joint","D_global53")
def edges(x):
    return np.unique(np.quantile(x,np.arange(1,BINS)/BINS))
def code(a,b,ed,es):
    d=np.asarray(a)-np.asarray(b)
    s=-(np.asarray(a)+np.asarray(b))/2
    return np.searchsorted(ed,d,side="right"),np.searchsorted(es,s,side="right")
def hist(a,b,ed,es):
    di,si=code(a,b,ed,es)
    ns,nd=len(es)+1,len(ed)+1
    return np.bincount(si*nd+di,minlength=ns*nd).reshape(ns,nd)/len(di)
def distributions(rows,classes,y,tr,ed,es):
    ns,nd=len(es)+1,len(ed)+1
    H=np.full((len(classes),ns,nd),ALPHA/(ns*nd),dtype=float)
    idx={c:i for i,c in enumerate(classes)}
    for i in np.flatnonzero(tr):
        _,a,b=rows[i]
        H[idx[y[i]]]+=hist(a,b,ed,es)
    joint=H/H.sum(axis=(1,2),keepdims=True)
    conditional=H/H.sum(axis=2,keepdims=True)
    marginal=H.sum(axis=1)/H.sum(axis=(1,2))[:,None]
    assert np.allclose(joint.sum((1,2)),1) and np.allclose(conditional.sum(2),1)
    assert np.allclose(marginal.sum(1),1)
    return np.log(marginal),np.log(conditional),np.log(joint)
def metrics(y,p,classes):
    return {"BA":float(balanced_accuracy_score(y,p)),"correct":int((y==p).sum()),
            "recall":{str(c):float((p[y==c]==c).mean()) for c in classes}}
def delta(y,p0,p1,prompt,classes):
    a,b=p0==y,p1==y
    groups=[np.flatnonzero(prompt==g) for g in np.unique(prompt)]
    # Efficient cluster bootstrap: exact class denominators after prompt resampling.
    n=np.array([[np.sum(y[ix]==c) for c in classes] for ix in groups])
    e=np.array([[np.sum((b[ix].astype(int)-a[ix].astype(int))[y[ix]==c]) for c in classes] for ix in groups])
    rng=np.random.default_rng(20261124)
    weights=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=2000)
    den=weights@n;num=weights@e
    valid=(den>0).all(1)
    ds=(num[valid]/den[valid]).mean(1)
    return {"delta_BA":float(balanced_accuracy_score(y,p1)-balanced_accuracy_score(y,p0)),
      "rescues":int((~a&b).sum()),"regressions":int((a&~b).sum()),
      "changed":int((p0!=p1).sum()),"prompt_bootstrap95":[float(v) for v in np.quantile(ds,[.025,.975])]}
def run(a):
    started=time.monotonic()
    fingerprint(a.train,TRAIN_SHA);fingerprint(a.folds,FOLDS_SHA)
    train=pd.read_csv(a.train,dtype={"id":str,"prompt_id":str,"model":str})
    folds=pd.read_csv(a.folds,dtype={"id":str,"prompt_id":str,"model":str})
    assert len(train)==len(folds)==2699 and train.id.is_unique and folds.id.is_unique
    joined=train.merge(folds[["id","prompt_id","model","fold_id"]],on="id",suffixes=("","_fold"),validate="one_to_one")
    assert len(joined)==2699 and (joined.prompt_id==joined.prompt_id_fold).all() and (joined.model==joined.model_fold).all()
    assert joined.groupby("prompt_id").fold_id.nunique().max()==1 and joined.prompt_id.nunique()==150
    fa=pd.read_csv(a.glm53_features,dtype={"id":str});fb=pd.read_csv(a.flash_features,dtype={"id":str})
    full=load_cache(a.glm53_scores,fa);flash=load_cache(a.flash_scores,fb)
    rows=list(paired_rows(full,flash));ids=[r[0] for r in rows]
    assert len(ids)==2699 and set(ids)==set(train.id)
    frame=joined.set_index("id").loc[ids]
    y=frame.model.to_numpy(str);prompt=frame.prompt_id.to_numpy(str);f=frame.fold_id.to_numpy(int)
    classes=np.unique(y);assert len(classes)==18 and set(f)==set(range(5))
    X=np.column_stack([fa.set_index("id").loc[ids,list(STATS)].to_numpy(float),
        fb.set_index("id").loc[ids,list(STATS)].to_numpy(float),
        np.asarray([fixed_summary(r[1],r[2]) for r in rows])])
    assert X.shape==(2699,53) and np.isfinite(X).all()
    preds={k:np.full(len(y),"",dtype=object) for k in ARMS}
    shapes=[]
    for k in range(5):
        tr=f!=k;va=f==k
        assert set(y[tr])==set(classes) and set(y[va])==set(classes)
        assert not set(prompt[tr])&set(prompt[va])
        d=np.concatenate([rows[i][1]-rows[i][2] for i in np.flatnonzero(tr)])
        s=np.concatenate([-(rows[i][1]+rows[i][2])/2 for i in np.flatnonzero(tr)])
        ed,es=edges(d),edges(s);shapes.append([len(ed)+1,len(es)+1])
        lm,lc,lj=distributions(rows,classes,y,tr,ed,es)
        for i in np.flatnonzero(va):
            _,u,v=rows[i];di,si=code(u,v,ed,es)
            for name,z in zip(ARMS[:3],[lm[:,di],lc[:,si,di],lj[:,si,di]]):
                preds[name][i]=classes[np.argmax(z.mean(axis=1))]
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,random_state=SEED,class_weight="balanced",max_iter=2000))
        clf.fit(X[tr],y[tr]);preds["D_global53"][va]=clf.predict(X[va])
    assert all(np.isin(p,classes).all() for p in preds.values())
    scores={k:metrics(y,p,classes) for k,p in preds.items()}
    contrasts={name:delta(y,p,preds["B_conditional"],prompt,classes)
        for name,p in preds.items() if name!="B_conditional"}
    passed=(all(contrasts[n]["delta_BA"]>=.02 and contrasts[n]["prompt_bootstrap95"][0]>0 for n in ("A_marginal","C_joint"))
             and contrasts["D_global53"]["delta_BA"]>=.01)
    report={"experiment":"PACE_v0_18_class_native_conditional_likelihood",
        "historical_development_only":True,"rows":len(y),"classes":len(classes),"prompt_groups":len(set(prompt)),
        "aligned_native_token_pairs":sum(len(r[1]) for r in rows),"bins_per_fold":shapes,
        "scores":scores,"conditional_minus_controls":contrasts,
        "fold_BA":{k:[float(balanced_accuracy_score(y[f==i],p[f==i])) for i in range(5)] for k,p in preds.items()},
        "development_gate_pass":bool(passed),"transfer_verified":False,"promote":False,
        "runtime_s":round(time.monotonic()-started,3),"no_new_model_forward":True,
        "no_foundation_training":True,"no_submission":True}
    print("NTL_PACE_REAL_18WAY="+json.dumps(report,sort_keys=True),flush=True)
def smoke():
    a=np.array([-1.,-2.,-3.,-5.]);b=np.array([-2.,-1.,-2.,-6.])
    ed=edges(a-b);es=edges(-(a+b)/2)
    h=hist(a,b,ed,es);assert abs(h.sum()-1)<EPS
    rows=[("a",a,b),("b",b,a)];y=np.array(["A","B"]);classes=np.unique(y)
    lm,lc,lj=distributions(rows,classes,y,np.ones(2,dtype=bool),ed,es)
    di,si=code(a,b,ed,es);assert np.isfinite(lc[:,si,di]).all()
    assert lc[:,si,di].shape==(2,4)
    print("NTL_PACE_SMOKE=PASS",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("train","folds","glm53_features","flash_features","glm53_scores","flash_scores"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args();smoke()
    if not a.smoke:
        assert all(getattr(a,k) is not None for k in ("train","folds","glm53_features","flash_features","glm53_scores","flash_scores"))
        run(a)
