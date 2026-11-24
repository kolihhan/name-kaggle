"""CPR v0: prompt-matched counterfactual patch retrieval with external falsifier.

Training: only original official 18-way train, excluding frozen Fold0.
Validation: held-out Fold0 (18), independent-prompt C1 and C2 (12 each).
No foundation-model inference/training, GPU or competition submission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from scripts.run_strong_sparse_anchor import build_fold_features,make_svc
from scripts.ntl_glm_distributed_evidence_20261122 import SEED as PREV_SEED

SEED=20261124
TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLDS_SHA="94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"
PANEL_SHA={"c1":"8746c4b77c2b09b9abafaeecc381a39321b9919214c005a3657042051c51b177",
           "c2":"bf25ef1e953d5c37b177f27b841a04098ae9b35f45fc43fa11036d91e8831e02"}
EXPECTED={"c1":(360,30),"c2":(470,50)}
LENGTH=40
MAX_PATCHES=4
FOLD=0
WEIGHT_FRACTION=.9

def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def patchify(text):
    words=str(text or "").split()
    if not words:return [""]
    return [" ".join(words[i:i+LENGTH]) for i in range(0,min(len(words),LENGTH*MAX_PATCHES),LENGTH)]

def patches(frame):
    texts=[];docs=[];ys=[];prompts=[]
    for i,(text,y,g) in enumerate(zip(frame.text,frame.model,frame.prompt_id)):
        for s in patchify(text):
            texts.append(s);docs.append(i);ys.append(str(y));prompts.append(str(g))
    return texts,np.asarray(docs,dtype=np.int32),np.asarray(ys),np.asarray(prompts)

def prompt_weights(x,labels,groups):
    out=np.ones(x.shape[0],np.float32)
    for g in np.unique(groups):
        ix=np.flatnonzero(groups==g)
        if len(ix)<2:continue
        sim=(x[ix]@x[ix].T).toarray()
        eligible=labels[ix,None]!=labels[None,ix]
        if not eligible.any():continue
        sim[~eligible]=0
        mx=sim.max(axis=1)
        out[ix]=np.square(1-WEIGHT_FRACTION*np.clip(mx,0,1))
    if not (np.isfinite(out).all() and (out>0).all() and (out<=1).all()):
        raise ValueError("invalid distinctiveness weights")
    return out

def shuffled_weights(original,groups):
    rng=np.random.default_rng(SEED)
    v=original.copy()
    for g in np.unique(groups):
        ix=np.flatnonzero(groups==g)
        v[ix]=rng.permutation(v[ix])
    assert np.allclose(np.sort(original),np.sort(v))
    return v

def retrieval_predict(z,valid_doc,x,train_labels,weights,classes,chunk=48):
    # Full 18-way predictions. No external labels observed at test time.
    groups=[np.flatnonzero(train_labels==c) for c in classes]
    assert all(len(g) for g in groups)
    result=np.zeros((int(valid_doc.max())+1,len(classes)),np.float64)
    counts=np.bincount(valid_doc,minlength=len(result))
    assert np.all(counts>0)
    for start in range(0,z.shape[0],chunk):
        end=min(start+chunk,z.shape[0])
        sim=(z[start:end]@x.T).toarray()
        # Per class, each query patch takes its best matching training patch.
        # The only intervention is downweighting training patches.
        scores=np.column_stack([np.max(sim[:,g]*weights[g][None,:],axis=1) for g in groups])
        np.add.at(result,valid_doc[start:end],scores)
    result/=counts[:,None]
    return np.asarray(classes)[result.argmax(1)]

def transfer_ba(y,p,classes):
    # Unsupported-class predictions remain errors; no closed-12 reranking.
    return float(np.mean([(p[y==c]==c).mean() for c in classes]))

def paired(y,a,b,classes):
    ca=a==y;cb=b==y
    return {"A":transfer_ba(y,a,classes),"B":transfer_ba(y,b,classes),
            "delta":transfer_ba(y,b,classes)-transfer_ba(y,a,classes),
            "rescues":int((~ca&cb).sum()),"regressions":int((ca&~cb).sum()),
            "changed":int((a!=b).sum())}

def grouped_bootstrap(slices,classes,reps=2000):
    # Each slice=(true,prompt,old,new), SOURCE is part of grouping key.
    pieces=[]
    for name,(y,g,old,new) in slices.items():
        for unique in np.unique(g):
            idx=np.flatnonzero(g==unique)
            den=np.asarray([(y[idx]==c).sum() for c in classes])
            delta=np.asarray([((new[idx]==y[idx])&(y[idx]==c)).sum()
                           -((old[idx]==y[idx])&(y[idx]==c)).sum() for c in classes])
            pieces.append((den,delta))
    den=np.stack([v[0] for v in pieces]);delta=np.stack([v[1] for v in pieces])
    rng=np.random.default_rng(SEED)
    draw=rng.multinomial(len(pieces),np.full(len(pieces),1/len(pieces)),size=reps)
    d=draw@den;n=draw@delta
    keep=np.all(d>0,axis=1)
    out=(n[keep]/d[keep]).mean(1)
    if len(out)<.95*reps: raise ValueError("insufficient_valid_bootstrap_draws")
    return [float(x) for x in np.quantile(out,[.025,.975])]

def selftest():
    p=patchify(" ".join(str(i) for i in range(250)))
    assert len(p)==4 and [len(x.split()) for x in p]==[40]*4
    y=np.array(["A","B","A","B"])
    g=np.array(["p","p","q","q"])
    from scipy.sparse import csr_matrix
    x=csr_matrix(np.eye(4,dtype=np.float32))
    w=prompt_weights(x,y,g)
    assert np.allclose(w,1.)
    w2=shuffled_weights(np.array([1,.6,.4,.2],np.float32),g)
    assert np.array_equal(np.sort(w2),np.array([.2,.4,.6,1.],np.float32))
    assert list(retrieval_predict(x,np.arange(4),x,y,np.ones(4),["A","B"]))==["A","B","A","B"]
    print("NTL_CPR_CPU_SMOKE=PASS",flush=True)

def official_and_panels(paths):
    for p,expected in ((paths.train,TRAIN_SHA),(paths.folds,FOLDS_SHA),
                       (paths.c1,PANEL_SHA["c1"]),(paths.c2,PANEL_SHA["c2"])):
        if sha(p)!=expected:raise ValueError("pinned input hash mismatch: "+p.name)
    train=pd.read_csv(paths.train,dtype={"id":str,"prompt_id":str,"model":str})
    folds=pd.read_csv(paths.folds,dtype={"id":str,"prompt_id":str,"model":str})
    if len(train)!=len(folds)!=2699:raise ValueError("train/fold count mismatch")
    if len(train)!=2699 or train.id.duplicated().any() or train.prompt_id.nunique()!=150 or train.model.nunique()!=18:
        raise ValueError("official validation mismatch")
    d=train.merge(folds[["id","prompt_id","model","fold_id"]],
                  on="id",suffixes=("","_fold"),validate="one_to_one")
    if not (len(d)==2699 and (d.model==d.model_fold).all()
            and (d.prompt_id==d.prompt_id_fold).all()
            and d.groupby("prompt_id").fold_id.nunique().max()==1):
        raise ValueError("official fold leakage")
    panels={}
    labels=set(d.model)
    for k in ("c1","c2"):
        r=pd.read_csv(getattr(paths,k),dtype={"prompt_id":str,"label":str})
        n,p=EXPECTED[k]
        if len(r)!=n or r.prompt_id.nunique()!=p or "text" not in r or "label" not in r:
            raise ValueError("panel data counts drift "+k)
        if r.text.isna().any() or (r.text.astype(str).str.strip()=="").any():
            raise ValueError("panel empty responses")
        if r.label.nunique()!=12 or not set(r.label)<=labels:
            raise ValueError("panel classes drift")
        panels[k]=r.rename(columns={"label":"model"})
    if set(panels["c1"].model)!=set(panels["c2"].model):
        raise ValueError("C1 and C2 classes differ")
    # This detects only byte-exact contamination, not semantic or upstream exposure.
    normal=lambda x:set(str(t).strip().lower() for t in x)
    if normal(train.text)&normal(panels["c1"].text) or normal(train.text)&normal(panels["c2"].text):
        raise ValueError("official and external exact text overlap")
    if set(panels["c1"].prompt_id)&set(panels["c2"].prompt_id):
        raise ValueError("C1 C2 original prompt overlap")
    for k in panels:
        if set(d.prompt_id)&set(panels[k].prompt_id):
            raise ValueError("external prompt_id collision with official")
    return d,panels

def run(args):
    began=time.monotonic()
    train,panels=official_and_panels(args)
    tr=train[train.fold_id!=FOLD].reset_index(drop=True)
    val={"official_fold0":train[train.fold_id==FOLD].reset_index(drop=True),**panels}
    labels=np.sort(train.model.unique())
    pt,train_doc,train_y,train_g=patches(tr)
    vec=TfidfVectorizer(analyzer="char",ngram_range=(2,5),lowercase=False,
                        sublinear_tf=True,min_df=2,max_features=120000,
                        dtype=np.float32)
    with threadpool_limits(limits=2):
        x=vec.fit_transform(pt).tocsr()
        weights=prompt_weights(x,train_y,train_g)
        placebo=shuffled_weights(weights,train_g)
        predictions={};mins={}
        for name,frame in val.items():
            texts,doc,_,_=patches(frame)
            z=vec.transform(texts).tocsr()
            arms={
               "A_unweighted":retrieval_predict(z,doc,x,train_y,np.ones(len(train_y),np.float32),labels),
               "B_counterfactual":retrieval_predict(z,doc,x,train_y,weights,labels),
               "C_placebo":retrieval_predict(z,doc,x,train_y,placebo,labels),
            }
            y=frame.model.to_numpy(str)
            predictions[name]=(y,frame.prompt_id.to_numpy(str),arms)
            mins[name]={"rows":len(frame),"prompts":frame.prompt_id.nunique(),
                       "true_classes":len(np.unique(y)),"predicted_classes":{k:len(np.unique(v)) for k,v in arms.items()}}
            print("NTL_CPR_RETRIEVAL_PANEL="+json.dumps({"name":name,**mins[name]},sort_keys=True),flush=True)
        # Strong exact 4-view control fitted ONLY on official Fold0 training.
        fulltrain=tr.text.astype(str).tolist()
        allval=pd.concat(list(val.values()),ignore_index=True)
        vx=allval.text.astype(str).tolist()
        mx,mz,meta=build_fold_features(fulltrain,vx)
        clf=make_svc(SEED)
        clf.fit(mx,tr.model.to_numpy(str))
        strong=clf.predict(mz)
        cursor=0
        for name,frame in val.items():
            predictions[name][2]["D_strong_sparse"]=strong[cursor:cursor+len(frame)]
            cursor+=len(frame)
    output={}
    for name,(y,prompt,arms) in predictions.items():
        target=np.unique(y)
        output[name]={"n":len(y),"classes_scored":len(target),"scores":{
               key:float(transfer_ba(y,pred,target)) for key,pred in arms.items()},
             "B_vs_A":paired(y,arms["A_unweighted"],arms["B_counterfactual"],target),
             "B_vs_C":paired(y,arms["C_placebo"],arms["B_counterfactual"],target),
             "B_vs_D":paired(y,arms["D_strong_sparse"],arms["B_counterfactual"],target)}
    common=np.unique(panels["c1"].model.to_numpy(str))
    pooled={k:(predictions[k][0],predictions[k][1],
               predictions[k][2]["A_unweighted"],predictions[k][2]["B_counterfactual"])
            for k in ("c1","c2")}
    ci=grouped_bootstrap(pooled,common)
    passing=(all(output[k]["B_vs_A"]["delta"]>=.02 for k in ("c1","c2"))
             and ci[0]>0
             and np.mean([output[k]["scores"]["B_counterfactual"] for k in ("c1","c2")])
              >=np.mean([output[k]["scores"]["D_strong_sparse"] for k in ("c1","c2")])
             and output["official_fold0"]["B_vs_A"]["delta"]>0)
    result={"experiment":"CPR_v0_prompt_matched_counterfactual_patch_retrieval",
            "evaluation":output,"c1_c2_pooled_prompt_bootstrap95":ci,
            "train_rows":len(tr),"train_labels":len(labels),"train_patches":len(pt),
            "mean_original_patch_weight":float(np.mean(weights)),
            "feature_count":int(x.shape[1]),"strong_baseline_features":meta["total_features"],
            "continuation_gate_pass":bool(passing),"promote":False,
            "source_provenance":"publisher_claim_not_independently_authenticated",
            "full18_external_validation":False,"historical_official_fold0_only":True,
            "no_GPU":True,"no_competition_submission":True,
            "runtime_seconds":round(time.monotonic()-began,2)}
    print("NTL_CPR_REAL_RESULT="+json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    for k in ("train","folds","c1","c2"):ap.add_argument("--"+k,type=Path)
    a=ap.parse_args()
    selftest()
    if not a.smoke:
        if any(getattr(a,k) is None for k in ("train","folds","c1","c2")):ap.error("real source files required")
        run(a)
