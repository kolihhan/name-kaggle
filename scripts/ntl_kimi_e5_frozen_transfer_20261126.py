#!/usr/bin/env python3
"""CPU-only, frozen E5-large-v2 Kimi/GPT transfer on real prompt-paired sources.
No retraining, Kaggle kernel push, GPU usage, or competition submission.
This is NOT an incumbent 0.718 paired comparison.
"""
from __future__ import annotations
import hashlib, io, json, os, subprocess, tempfile, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

A, B = "kimi-k3", "gpt-5.6-luna"
MODEL = "intfloat/e5-large-v2"
REVISION = "d94960066016d2064f2a3a2dbc65107a2b48230d"
SEED = 20261126
N_TRAIN, N_VALID, N_EXTERNAL = 70, 40, 80
BATCH, MAX_TOKENS = 4, 192

def log(key, **data):
    print(key+"="+json.dumps(data,sort_keys=True,default=str),flush=True)

def api(*args):
    p=subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=180)
    if p.returncode: raise RuntimeError("READ_ONLY_KAGGLE_FETCH_FAILED:"+str(p.returncode))

def datasets():
    with tempfile.TemporaryDirectory() as root:
        root=Path(root)
        api("competitions","download","name-that-llm","-p",str(root))
        z=list(root.glob("*.zip"))
        assert len(z)==1
        with zipfile.ZipFile(z[0]) as arc:
            official=pd.read_csv(io.BytesIO(arc.read("train.csv")),usecols=["prompt_id","model","text"])
        api("datasets","download","-d","prvsiyan/prvsiyans-extended-name-that-llm-dataset",
            "-f","passages_clean.csv","-p",str(root),"--unzip")
        external=pd.read_csv(root/"passages_clean.csv",low_memory=False)
    assert len(official)==2699 and official.prompt_id.nunique()==150
    assert len(external)==5070 and external.model.nunique()==18
    external=external.loc[external.split.isin(["new","new2"])].copy()
    external=external.loc[
        ((external.model==A)&(external.served_provider=="Moonshot AI"))|
        ((external.model==B)&(external.served_provider=="OpenAI (first-party API)"))
    ].copy()
    return official,external

def pair(df):
    df=df.loc[df.model.isin([A,B]),["prompt_id","model","text"]].copy()
    assert not df.duplicated(["prompt_id","model"]).any()
    a=df.loc[df.model==A].set_index("prompt_id")
    b=df.loc[df.model==B].set_index("prompt_id")
    ids=sorted(set(a.index)&set(b.index),
               key=lambda s:hashlib.sha256((str(s)+str(SEED)).encode()).hexdigest())
    return a,b,ids

def embed(texts):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    enc=AutoModel.from_pretrained(MODEL,revision=REVISION,use_safetensors=True).eval().to("cpu")
    assert not enc.training
    out=[]
    started=time.monotonic()
    with torch.inference_mode():
        for i in range(0,len(texts),BATCH):
            x=tok(["passage: "+s for s in texts[i:i+BATCH]],
                  max_length=MAX_TOKENS,truncation=True,padding=True,return_tensors="pt")
            hidden=enc(**x).last_hidden_state
            mask=x["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)
            out.append(F.normalize(pooled,p=2,dim=1).numpy().astype(np.float32))
            if (i//BATCH+1)%20==0 or i+BATCH>=len(texts):
                log("NTL_E5_CPU_PROGRESS",embedded=min(i+BATCH,len(texts)),
                    total=len(texts),elapsed_seconds=round(time.monotonic()-started,1))
    return np.concatenate(out,axis=0)

def metrics(a,b,d,mid):
    sa=(a-mid)@d
    sb=(b-mid)@d
    ra=float(np.mean(sa>0))
    rb=float(np.mean(sb<0))
    return {"ba":(ra+rb)/2,"kimi_recall":ra,"gpt_recall":rb,
            "matched_pair_ranking":float(np.mean(sa>sb)),"n_pairs":len(sa)}

def main():
    assert os.getenv("KAGGLE_API_TOKEN")
    official,external=datasets()
    oa,ob,ids=pair(official)
    ea,eb,eids=pair(external)
    assert len(ids)==150 and len(eids)>=150
    tr=ids[:N_TRAIN]
    val=ids[N_TRAIN:N_TRAIN+N_VALID]
    ex=eids[:N_EXTERNAL]
    assert len(set(tr)&set(val))==0
    trtext=[str(oa.loc[k,"text"]) for k in tr]+[str(ob.loc[k,"text"]) for k in tr]
    vtext=[str(oa.loc[k,"text"]) for k in val]+[str(ob.loc[k,"text"]) for k in val]
    etext=[str(ea.loc[k,"text"]) for k in ex]+[str(eb.loc[k,"text"]) for k in ex]
    digest=lambda s:hashlib.sha256(" ".join(s.casefold().split()).encode()).hexdigest()
    assert not set(map(digest,trtext+vtext))&set(map(digest,etext))
    log("NTL_E5_FROZEN_CONTRACT",model=MODEL,revision=REVISION,
        official_train_pairs=len(tr),official_unseen_pairs=len(val),external_firstparty_pairs=len(ex),
        input_passages=len(trtext+vtext+etext),no_shared_exact_text=True,
        fitted_encoder=False,source_training_provenance_incumbent_unknown=True)
    z=embed(trtext+vtext+etext)
    nt,nv,ne=len(tr),len(val),len(ex)
    assert z.shape==(len(trtext+vtext+etext),1024) and np.isfinite(z).all()
    ta,tb=z[:nt],z[nt:2*nt]
    va,vb=z[2*nt:2*nt+nv],z[2*nt+nv:2*nt+2*nv]
    ea_,eb_=z[2*nt+2*nv:2*nt+2*nv+ne],z[2*nt+2*nv+ne:]
    ma,mb=ta.mean(axis=0),tb.mean(axis=0)
    direction=ma-mb
    midpoint=(ma+mb)/2
    assert np.linalg.norm(direction)>0
    known=metrics(va,vb,direction,midpoint)
    actual=metrics(ea_,eb_,direction,midpoint)
    # Frozen word-length-only negative reference, not a submission heuristic.
    wa=np.array([len(t.split()) for t in trtext[:nt]])
    wb=np.array([len(t.split()) for t in trtext[nt:]])
    xa=np.array([len(t.split()) for t in etext[:ne]])
    xb=np.array([len(t.split()) for t in etext[ne:]])
    sign=1 if wa.mean()>wb.mean() else -1
    midpoint_words=(wa.mean()+wb.mean())/2
    lenba=.5*(np.mean(sign*(xa-midpoint_words)>0)+np.mean(sign*(xb-midpoint_words)<0))
    rng=np.random.default_rng(SEED)
    # Same paired prompt with random A/B assignment; exact representation held fixed.
    delta=ta-tb
    null=[]
    for _ in range(200):
        signs=rng.choice([-1,1],size=(nt,1))
        null.append(metrics(ea_,eb_,(delta*signs).mean(axis=0),midpoint)["ba"])
    null_p=(1+sum(x>=actual["ba"] for x in null))/(1+len(null))
    boot=[]
    for _ in range(2000):
        ix=rng.integers(0,ne,ne)
        boot.append(metrics(ea_[ix],eb_[ix],direction,midpoint)["ba"])
    log("NTL_E5_FROZEN_TRANSFER_REAL_RESULT",
        official_holdout=known,external_holdout=actual,
        external_ba_boot95=[round(float(x),5) for x in np.quantile(boot,[.025,.975])],
        external_length_only_ba=float(lenba),
        external_shuffled_label_null_mean=float(np.mean(null)),
        shuffled_null_tail_probability=float(null_p),
        metric_scope="binary frozen encoder readout; not 18-class BA",
        no_official_submission=True,no_training=True,no_gpu=True)

if __name__=="__main__":
    main()
