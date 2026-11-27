"""ID-aligned v36/RCLA inference bridge, no model or dataset loading.
A scoring callback is required; absence/time exhaustion preserves exact output.
Never reads labels or official test truth. Writes only via atomic replace.
"""
from pathlib import Path
import os
import time
import tempfile
import numpy as np
import pandas as pd
from scripts.ntl_rcla_arbitration import arbitrate,select_candidates

def apply(out_path, native_path, test_path, scorer, deadline):
    out_path=Path(out_path)
    summary={"applied":False,"reason":"not_started","eligible":0,"changed":0}
    # Budget deadline is an absolute monotonic time supplied by the outer runner.
    if scorer is None or time.monotonic()>=deadline:
        summary["reason"]="no_scorer_or_budget";return summary
    base=pd.read_parquet(out_path)
    test=pd.read_csv(test_path,dtype={"id":str},usecols=["id","text"])
    if base.columns.tolist()!=["id","model"] or len(base)!=len(test):
        raise ValueError("original output shape mismatch")
    b=base.id.astype(str).to_numpy()
    t=test.id.astype(str).to_numpy()
    if len(set(b))!=len(b) or len(set(t))!=len(t) or set(b)!=set(t):
        raise ValueError("test/output ID mismatch")
    if test.text.isna().any():raise ValueError("missing test text")
    with np.load(native_path,allow_pickle=False) as z:
        ids=z["ids"].astype(str);labels=z["labels"].astype(str);logits=z["logits"]
    if ids.ndim!=1 or len(ids)!=len(b) or len(set(ids))!=len(ids) or set(ids)!=set(b):
        raise ValueError("native logit ID mismatch")
    index=pd.Index(ids).get_indexer(b)
    if np.any(index<0):raise ValueError("missing logit IDs")
    logits=logits[index]
    inc=base.model.astype(str).to_numpy()
    candidates=select_candidates(b,labels,logits,inc)
    summary["eligible"]=len(candidates)
    if not candidates:summary["reason"]="no_eligible";return summary
    byid=test.set_index("id").text
    # Frozen original order, no outcome-dependent priority or labels.
    scores={}
    try:
        for c in candidates:
            if time.monotonic()>=deadline:break
            result=scorer(c, str(byid.loc[c.row_id]), deadline)
            # Discard incomplete scoring when deadline was exceeded.
            if time.monotonic()>=deadline:break
            if result is not None:scores[c.row_id]=result
    except (RuntimeError,TimeoutError,OSError):
        # No partial output mutation on scorer failure; keep original bytes.
        summary["reason"]="scorer_failed";return summary
    pred,stats=arbitrate(b,labels,logits,inc,scores)
    summary.update(stats)
    if not stats["changed"]:summary["reason"]="no_changes";return summary
    if time.monotonic()>=deadline:summary["reason"]="deadline_before_write";summary["changed"]=0;return summary
    final=base.copy()
    final["model"]=pred
    assert final.id.equals(base.id) and final.columns.tolist()==["id","model"]
    # No mutation outside the label-blind eligibility set.
    touched=set(b[pred!=inc])
    assert touched<={c.row_id for c in candidates}
    tmp=None
    try:
        fd,tmp=tempfile.mkstemp(prefix=".rcla-",suffix=".parquet",dir=out_path.parent)
        os.close(fd)
        final.to_parquet(tmp,index=False)
        reread=pd.read_parquet(tmp)
        if not reread.equals(final):raise RuntimeError("output roundtrip mismatch")
        if time.monotonic()>=deadline:summary["reason"]="deadline_before_replace";summary["changed"]=0;return summary
        os.replace(tmp,out_path);tmp=None
        summary["applied"]=True;summary["reason"]="applied"
    finally:
        if tmp is not None:Path(tmp).unlink(missing_ok=True)
    return summary
