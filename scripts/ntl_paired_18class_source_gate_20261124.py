#!/usr/bin/env python3
"""Paired 18-class source-transfer gate. Reads local private prediction CSVs only.
Required columns: row_id,prompt_id,label,incumbent_pred,candidate_pred,source.
Never prints passages, row identifiers or individual predictions.
"""
import argparse
import json
import numpy as np
import pandas as pd

REQUIRED = {"row_id","prompt_id","label","incumbent_pred","candidate_pred","source"}

def balanced(y, p, classes):
    return float(np.mean([(p[y == c] == c).mean() for c in classes]))

def evaluate(frame, seed=20261124, boots=2000):
    missing = REQUIRED - set(frame.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    if frame[list(REQUIRED)].isna().any().any(): raise ValueError("nulls in required fields")
    if frame.row_id.duplicated().any(): raise ValueError("duplicate row_id")
    classes = sorted(frame.label.unique())
    if len(classes) != 18: raise ValueError(f"require all 18 true classes, found {len(classes)}")
    for col in ("incumbent_pred","candidate_pred"):
        if not set(frame[col]).issubset(set(classes)): raise ValueError(f"invalid predictions in {col}")
    y=frame.label.to_numpy()
    a=frame.incumbent_pred.to_numpy()
    b=frame.candidate_pred.to_numpy()
    base=balanced(y,a,classes); cand=balanced(y,b,classes)
    rescues=int(((a!=y)&(b==y)).sum())
    regressions=int(((a==y)&(b!=y)).sum())
    rng=np.random.default_rng(seed)
    groups=frame.prompt_id.astype(str).to_numpy()
    uniq=np.unique(groups)
    indexes={g:np.flatnonzero(groups==g) for g in uniq}
    deltas=[]
    for _ in range(boots):
        chosen=rng.choice(uniq,size=len(uniq),replace=True)
        ix=np.concatenate([indexes[g] for g in chosen])
        yy=y[ix]
        if len(np.unique(yy))!=18: continue
        deltas.append(balanced(yy,b[ix],classes)-balanced(yy,a[ix],classes))
    if len(deltas)<100: raise ValueError("insufficient full-18 bootstrap replicates")
    slices={}
    for source, part in frame.groupby("source"):
        labels=set(part.label)
        slices[str(source)]={"n":len(part),"classes":len(labels),
            "rescues":int(((part.incumbent_pred!=part.label)&(part.candidate_pred==part.label)).sum()),
            "regressions":int(((part.incumbent_pred==part.label)&(part.candidate_pred!=part.label)).sum())}
        if len(labels)==18:
            slices[str(source)]["delta_ba"]=round(balanced(part.label.to_numpy(),part.candidate_pred.to_numpy(),classes)-balanced(part.label.to_numpy(),part.incumbent_pred.to_numpy(),classes),6)
    recalls={}
    for c in classes:
        ix=y==c
        recalls[str(c)]={"n":int(ix.sum()),"baseline":round(float((a[ix]==c).mean()),6),
                          "candidate":round(float((b[ix]==c).mean()),6)}
    return {"n":len(frame),"prompt_groups":len(uniq),"classes":18,
            "incumbent_ba":round(base,6),"candidate_ba":round(cand,6),
            "delta_ba":round(cand-base,6),"rescues":rescues,"regressions":regressions,
            "prompt_cluster_bootstrap_95ci":[round(float(v),6) for v in np.quantile(deltas,[.025,.975])],
            "valid_bootstraps":len(deltas),"source_slices":slices,"class_recall":recalls,
            "note":"Local authentic scored-panel evidence only; not official Public BA."}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",required=True,help="PRIVATE authentic same-row 18-class predictions")
    parser.add_argument("--output",required=True,help="local aggregate JSON; do not publish without review")
    args=parser.parse_args()
    result=evaluate(pd.read_csv(args.csv))
    with open(args.output,"w") as f: json.dump(result,f,indent=2)
    print("Validated private same-row paired evaluation; aggregate JSON written.")
