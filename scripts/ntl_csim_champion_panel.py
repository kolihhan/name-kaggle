"""Freeze a bounded five-class diagnostic cohort for 18-output champion inference.

Never train on publisher outputs. Never promote this cohort as independent.
A mutable Open Pool snapshot is pinned by bytes at run time; this check cannot
attest opaque checkpoint exposure or historical provider identity.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import pandas as pd

PUBLISHER_SHA="b581463c90354a57e693f8c7066a8fdd9b12b7cadcab9a9466b1e85c370ea86b"
OFFICIAL_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
ROUTES={
    "gemini-3.7-flash":"google/gemini-3.7-flash",
    "glm-5.3":"z-ai/glm-5.3",
    "glm-5.3-flash":"z-ai/glm-5.3-flash",
    "kimi-k3":"moonshotai/kimi-k3",
    "qwen3.8-max":"qwen/qwen3.8-max-0902",
}
N=100
SEED="ntl-csim-champion-panel-20261128-v1"

def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def normhash(s):
    return hashlib.sha256(" ".join(str(s).casefold().split()).encode()).digest()

def freeze(publisher,official,pool):
    if sha(publisher)!=PUBLISHER_SHA or sha(official)!=OFFICIAL_SHA:
        raise ValueError("pinned input changed")
    a=pd.read_csv(publisher,dtype=str,usecols=["id","prompt_id","model","served_model","text"])
    b=pd.read_csv(official,dtype=str,usecols=["model","text"])
    if len(a)!=4964 or len(b)!=2699 or b.model.nunique()!=18 or set(a.model)!=set(ROUTES):
        raise ValueError("unexpected source shape")
    if a.isna().any().any() or b.text.isna().any() or not a.id.is_unique:
        raise ValueError("invalid inputs")
    if not a.served_model.eq(a.model.map(ROUTES)).all():
        raise ValueError("publisher route mismatch")
    g=a.groupby("prompt_id").model.agg(["size","nunique"])
    complete=set(g.index[(g["size"]==5)&(g["nunique"]==5)])
    if len(complete)!=964:raise ValueError("matched source count changed")
    a=a[a.prompt_id.isin(complete)].copy()
    a["hash"]=a.text.map(normhash)
    off=set(b.text.map(normhash))
    # Read only response text and filter flag, never export raw source or hashes.
    q=pd.read_parquet(pool,columns=["text","flag_train_prompt_regen"])
    if q.text.isna().any():raise ValueError("invalid upstream text")
    allpool=set(q.text.map(normhash))
    trainpool=set(q.loc[~q.flag_train_prompt_regen.astype(bool),"text"].map(normhash))
    x=a["hash"].isin(off)
    y=a["hash"].isin(allpool)
    z=a["hash"].isin(trainpool)
    dup=a["hash"].duplicated(keep=False)
    blocked=set(a.loc[x|y|dup,"prompt_id"])
    elig=complete-blocked
    ordered=sorted(elig,key=lambda v:hashlib.sha256((SEED+"|"+str(v)).encode()).digest())
    report={"source_sha":PUBLISHER_SHA,"official_sha":OFFICIAL_SHA,"pool_sha":sha(pool),
            "pool_rows":len(q),"pool_training_eligible_rows":int((~q.flag_train_prompt_regen.astype(bool)).sum()),
            "complete_prompt_groups":len(complete),"official_exact_overlap_rows":int(x.sum()),
            "pool_exact_overlap_rows":int(y.sum()),"pool_training_eligible_overlap_rows":int(z.sum()),
            "publisher_duplicate_rows":int(dup.sum()),"blocked_groups":len(blocked),
            "eligible_groups":len(elig),"target_groups":N,
            "truth_classes":5,"scorer_output_classes":18,
            "source_independence":False,"opaque_prefit_exposure_verified":False,
            "diagnostic_only":True,"promotion":False}
    if len(ordered)<N:return None,dict(report,status="BLOCKED_INSUFFICIENT_GROUPS")
    panel=a[a.prompt_id.isin(ordered[:N])].sort_values(["prompt_id","model"]).reset_index(drop=True)
    assert len(panel)==5*N and panel["hash"].is_unique and panel.model.value_counts().eq(N).all()
    panel.insert(0,"diagnostic_id",[f"csim-probe-{i:04d}" for i in range(len(panel))])
    test=panel[["diagnostic_id","text"]].rename(columns={"diagnostic_id":"id"})
    truth=panel[["diagnostic_id","model","prompt_id"]].rename(columns={"diagnostic_id":"id"})
    digest=hashlib.sha256(test.to_csv(index=False,lineterminator="\n").encode()).hexdigest()
    report.update(status="DIAGNOSTIC_COHORT_READY",rows=len(panel),cohort_sha=digest)
    return (test,truth),report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("publisher","official","pool"):p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    _,report=freeze(args.publisher,args.official,args.pool)
    # Only aggregate receipt. No dataset, IDs, row predictions or text is retained.
    print("CHAMPION_PANEL="+json.dumps(report,sort_keys=True),flush=True)
