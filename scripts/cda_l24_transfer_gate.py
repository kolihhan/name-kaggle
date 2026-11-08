"""Frozen C0 versus C1 CDA-L24 source-transfer gate.

Combines private per-row predictions after both separately trained Stage-A
arms complete. A single paired bootstrap over ORIGINAL source prompt groups,
not new splits of repeatedly consumed official training data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED=20261108
REPS=5000
MIN_GAIN=.02
MAX_PROVIDER_DROP=.01
REQUIRED_PROMPTS=100


def provider(label:str)->str:
    if label.startswith("gpt-"):return "OpenAI"
    if label.startswith("gemini-"):return "Google"
    if label.startswith("claude-"):return "Anthropic"
    if label.startswith("glm-"):return "ZAI"
    if label.startswith("qwen"):return "Qwen"
    if label.startswith("kimi"):return "Moonshot"
    # Used only for synthetic unit-testing with arbitrary labels.
    return label.split("-")[0]


def assess(c0:pd.DataFrame,c1:pd.DataFrame,
           *, reps:int=REPS, seed:int=SEED)->dict:
    required=["row_index","source","prompt_group","label","prediction"]
    for frame in (c0,c1):
        if not set(required)<=set(frame):
            raise ValueError("private prediction schema invalid")
        if frame[required].isna().any().any() or frame.row_index.duplicated().any():
            raise ValueError("null or duplicate row in predictions")
    if len(c0)!=len(c1) or len(c0)<300:
        raise ValueError("C0/C1 transfer rows invalid")
    a=c0.sort_values("row_index").reset_index(drop=True)
    b=c1.sort_values("row_index").reset_index(drop=True)
    for col in ("row_index","source","prompt_group","label"):
        if not a[col].equals(b[col]):
            raise ValueError(f"unpaired C0/C1 source evidence: {col}")
    if a.source.nunique()!=1 or a.prompt_group.nunique()<REQUIRED_PROMPTS:
        raise ValueError("insufficient original prompt-group transfer evidence")
    y=a.label.astype(str).to_numpy()
    p0=a.prediction.astype(str).to_numpy()
    p1=b.prediction.astype(str).to_numpy()
    labels=sorted(set(y))
    if len(labels)<8:
        raise ValueError("fewer than eight eligible model labels")
    counts=np.array([(y==name).sum() for name in labels])
    if (counts<20).any():
        raise ValueError("one or more true labels have fewer than 20 rows")
    hit0=np.array([(p0[y==name]==name).mean() for name in labels])
    hit1=np.array([(p1[y==name]==name).mean() for name in labels])
    recall0=dict(zip(labels,map(float,hit0)))
    recall1=dict(zip(labels,map(float,hit1)))
    ba0=float(hit0.mean());ba1=float(hit1.mean())
    # Macro averaged provider correctness per true model class (not provider
    # frequency-weighted): labels predicted to another provider are wrong.
    prov0=float(np.mean([
      np.mean([provider(s)==provider(name) for s in p0[y==name]])
      for name in labels]))
    prov1=float(np.mean([
      np.mean([provider(s)==provider(name) for s in p1[y==name]])
      for name in labels]))
    groups=sorted(set(a.prompt_group.astype(str)))
    group_index={key:i for i,key in enumerate(groups)}
    g=np.array([group_index[key] for key in a.prompt_group.astype(str)])
    totals=np.zeros((len(groups),len(labels)),dtype=np.float64)
    fixes=np.zeros_like(totals)
    for j,name in enumerate(labels):
        ids=np.where(y==name)[0]
        np.add.at(totals[:,j],g[ids],1)
        np.add.at(fixes[:,j],g[ids],
                  (p1[ids]==name).astype(float)-(p0[ids]==name).astype(float))
    rng=np.random.default_rng(seed)
    bootstrap=[]
    for _ in range(reps):
        chosen=rng.integers(0,len(groups),size=len(groups))
        weight=np.bincount(chosen,minlength=len(groups))
        den=weight@totals
        if np.any(den==0):
            continue
        delta=(weight@fixes)/den
        bootstrap.append(float(delta.mean()))
    if len(bootstrap)<int(reps*.95):
        raise ValueError("unreliable bootstrap: too many absent-label resamples")
    lo,hi=np.quantile(bootstrap,[.025,.975]).tolist()
    gain=ba1-ba0
    decision=("PROMOTE_TO_GROUPED_OFFICIAL_OOF" if gain>=MIN_GAIN and
              lo>0 and (prov0-prov1)<=MAX_PROVIDER_DROP else
              "REJECT_CDA_L24_SOURCE_TRANSFER")
    return {
       "decision":decision,
       "matched_rows":len(a),"matched_prompt_groups":len(groups),
       "eligible_true_labels":len(labels),
       "C0_balanced_accuracy":ba0,
       "C1_balanced_accuracy":ba1,
       "C1_minus_C0":gain,
       "prompt_bootstrap_CI95":[lo,hi],
       "bootstrap_repetitions":len(bootstrap),
       "provider_macro_C0":prov0,"provider_macro_C1":prov1,
       "provider_macro_delta":prov1-prov0,
       "recalls_C0":recall0,"recalls_C1":recall1,
       "C0_only_correct":int(((p0==y)&(p1!=y)).sum()),
       "C1_only_correct":int(((p1==y)&(p0!=y)).sum()),
       "both_correct":int(((p1==y)&(p0==y)).sum()),
       "both_wrong":int(((p1!=y)&(p0!=y)).sum()),
       "official_competition_OOF":False,
       "kaggle_public_score":None,
       "no_submission_authorized_by_script":True,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--c0-private-predictions",type=Path,required=True)
    ap.add_argument("--c1-private-predictions",type=Path,required=True)
    ap.add_argument("--aggregate-json",type=Path,required=True)
    a=ap.parse_args()
    report=assess(pd.read_csv(a.c0_private_predictions),
                  pd.read_csv(a.c1_private_predictions))
    a.aggregate_json.parent.mkdir(parents=True,exist_ok=True)
    a.aggregate_json.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("CDA_L24_FROZEN_TRANSFER_GATE",
          json.dumps({k:report[k] for k in (
            "decision","matched_rows","eligible_true_labels",
            "C0_balanced_accuracy","C1_balanced_accuracy",
            "C1_minus_C0","prompt_bootstrap_CI95",
            "provider_macro_delta")},sort_keys=True))

if __name__=="__main__":main()
