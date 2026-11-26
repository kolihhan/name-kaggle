"""Frozen StyleDistance CPU patch-retrieval diagnostic, not HALvest reproduction.
Official training prompts only; no model fit, no hidden data, no submission.
"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

MODEL="StyleDistance/styledistance"
REV="b7df5f0"
N_PROMPTS=12

def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def score(queries,refs,labels):
    # Each text pair: average of best cosine for each query patch.
    classes=sorted(set(labels))
    out=np.empty((len(queries),len(classes)))
    for i,q in enumerate(queries):
        pair=np.array([(q@r.T).max(axis=1).mean() for r in refs])
        # average all reference-document similarities, identical per-class counts
        for j,c in enumerate(classes):
            out[i,j]=pair[labels==c].mean()
    return np.array(classes)[out.argmax(axis=1)]

def selftest():
    e=np.eye(2)
    assert list(score([e[:1],e[1:]],[e[:1],e[1:]],np.array(["a","b"])))==["a","b"]
    assert np.allclose(normalize(np.array([[3.,4.]])),[[.6,.8]])

def main(path):
    import torch
    from sentence_transformers import SentenceTransformer
    selftest()
    torch.set_num_threads(4)
    assert not torch.cuda.is_available(), "CPU only"
    start=time.monotonic()
    raw=Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest()=="40a22ce8d88c5f46feb4637a099db815f713ffe53e89dbe1b7b21de77707"
    df=pd.read_csv(path,dtype={"prompt_id":str})
    assert len(df)==2699 and df.model.nunique()==18
    train,test=next(GroupKFold(5).split(df,groups=df.prompt_id))
    def select(d):
        counts=d.groupby("prompt_id").model.agg(["size","nunique"])
        complete=sorted(counts[(counts["size"]==18)&(counts["nunique"]==18)].index)
        assert len(complete)>=N_PROMPTS
        # deterministic, not score-selected
        rng=np.random.default_rng(20261127)
        picked=rng.choice(complete,N_PROMPTS,replace=False)
        return d[d.prompt_id.isin(picked)].sort_values(["prompt_id","model"]).reset_index(drop=True)
    r,q=select(df.iloc[train]),select(df.iloc[test])
    assert not set(r.prompt_id)&set(q.prompt_id)
    assert not set(r.text)&set(q.text)
    rows=pd.concat([r,q],ignore_index=True)
    model=SentenceTransformer(MODEL,revision=REV,device="cpu",trust_remote_code=False,model_kwargs={"use_safetensors":True})
    model.eval()
    full=normalize(model.encode(rows.text.tolist(),batch_size=12,show_progress_bar=False,convert_to_numpy=True))
    chunks=[]
    counts=[]
    for text in rows.text:
        words=text.split()
        # contiguous nonoverlapping balanced spans, 2-4 patches, no tiny tail
        n=min(4,max(2,int(np.ceil(len(words)/40))))
        parts=[" ".join(x.tolist()) for x in np.array_split(words,n)]
        assert all(parts)
        chunks.extend(parts);counts.append(n)
    vectors=normalize(model.encode(chunks,batch_size=16,show_progress_bar=False,convert_to_numpy=True))
    assert np.isfinite(full).all() and np.isfinite(vectors).all()
    patches=np.split(vectors,np.cumsum(counts)[:-1])
    n=len(r);y=q.model.to_numpy(); labels=r.model.to_numpy()
    full_pred=score([v[None,:] for v in full[n:]],[v[None,:] for v in full[:n]],labels)
    patch_pred=score(patches[n:],patches[:n],labels)
    pooled=np.array([normalize(x.mean(axis=0)) for x in patches])
    pool_pred=score([v[None,:] for v in pooled[n:]],[v[None,:] for v in pooled[:n]],labels)
    shuffled=np.random.default_rng(20261127).permutation(labels)
    null_pred=score(patches[n:],patches[:n],shuffled)
    def metrics(pred):
        return {"BA":float(balanced_accuracy_score(y,pred)),"correct":int((y==pred).sum()),"recall":{c:float((pred[y==c]==c).mean()) for c in sorted(set(y))}}
    report={"scope":"spent official Fold0 DEV subsample; not independent or public LB",
            "model":MODEL,"revision":REV,"n_ref":n,"n_query":len(q),"patches":len(chunks),
            "whole":metrics(full_pred),"patch_mean":metrics(pool_pred),"patch_maxsim":metrics(patch_pred),"shuffled":metrics(null_pred),
            "rescue_vs_whole":int(((full_pred!=y)&(patch_pred==y)).sum()),"regress_vs_whole":int(((full_pred==y)&(patch_pred!=y)).sum()),
            "seconds":round(time.monotonic()-start,2),"gpu":False,"training":False,"submission":False}
    print("NTL_STYLE_PATCH_CPU="+json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    if sys.argv[1]=="--selftest": selftest();print("SELFTEST PASS")
    else: main(sys.argv[1])
