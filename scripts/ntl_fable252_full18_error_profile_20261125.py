"""Actual 18-class error bottleneck of completed Fable252 v2, inference-free.

Purely observational. Uses only Kaggle private-v2 output submission.parquet
and SHA-pinned publisher-labelled external source; no new model, no submission.
Report aggregate labels and counts only, never texts, prompts or row IDs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PANEL_SHA = "bce34670f4d1170375e7baf85eec5577af4282d5f0ca6cf1573ce46e81c5a45c"
TOPICS = {"short":(98,25,65,83,16,61,68),"median":(77,54,99,37),"long":(72,63,87)}
EXPECTED_CORRECT = 151
EXPECTED_LABELS = 18
EXPECTED_ROWS = 252

def provider(model: str) -> str:
    if model.startswith(("gpt-", "gpt-6")):return "OpenAI"
    if model.startswith("gemini-"):return "Google"
    if model.startswith("claude-"):return "Anthropic"
    if model.startswith("glm-"):return "Z.ai"
    if model.startswith("kimi-"):return "Moonshot"
    if model.startswith("qwen"):return "Qwen"
    raise ValueError("unexpected official model label: "+model)

def load_truth_and_predictions(panelpath: Path, predictions: Path, aggregate: Path):
    with panelpath.open("rb") as h:
        if hashlib.file_digest(h,"sha256").hexdigest()!=PANEL_SHA:
            raise ValueError("external publisher data SHA changed")
    raw=pd.read_csv(panelpath)
    for k in ("version","topic_id","model","text"):
        if k not in raw:raise ValueError("panel lacks original field: "+k)
    blocks=[]
    for version,ids in TOPICS.items():
        for topic in ids:
            g=raw.loc[(raw.version==version)&(raw.topic_id==topic)].copy()
            if len(g)!=18 or g.model.nunique()!=18:
                raise ValueError("external source missing one of 18 model IDs for requested topic")
            blocks.append(g.sort_values("model",kind="stable"))
    truth=pd.concat(blocks,ignore_index=True)
    truth.insert(0,"id",[f"fable-observation-{i:03d}" for i in range(len(truth))])
    if len(truth)!=EXPECTED_ROWS or truth.model.nunique()!=18 or not truth.model.value_counts().eq(14).all():
        raise ValueError("truth panel balancing invalid")
    df=pd.read_parquet(predictions)
    if df.columns.tolist()!=["id","model"] or len(df)!=252 or df.id.astype(str).duplicated().any():
        raise ValueError("private scored artifact missing expected 252-row schema")
    if set(df.id.astype(str))!=set(truth.id):
        raise ValueError("private predictions do not match original frozen test IDs")
    df=df.assign(id=df.id.astype(str),model=df.model.astype(str))
    data=truth[["id","model","version","topic_id"]].merge(
        df,on="id",suffixes=("_true","_pred"),validate="one_to_one")
    if set(data.model_true)!=set(data.model_pred)|set(data.model_true):
        raise ValueError("unrecognized class label in predictions")
    if not set(data.model_pred)<=set(data.model_true):
        raise ValueError("predicted label outside frozen truth classes")
    parsed=json.loads(aggregate.read_text())
    if parsed.get("source_rows")!=252 or parsed.get("truth_rows_per_label")!=14:
        raise ValueError("pinned completed aggregate mismatch")
    count=int((data.model_true==data.model_pred).sum())
    if count!=EXPECTED_CORRECT or abs(parsed["new_protected_balanced_accuracy"]-count/252)>1e-9:
        raise ValueError("private predictions disagree with previously verified 151/252 result")
    return data

def report(data: pd.DataFrame) -> dict:
    labels=sorted(set(data.model_true))
    true=data.model_true.to_numpy(str)
    pred=data.model_pred.to_numpy(str)
    tf=np.array([provider(x) for x in true])
    pf=np.array([provider(x) for x in pred])
    exact=true==pred
    family_ok=tf==pf
    within=(~exact)&family_ok
    cross=(~exact)&(~family_ok)
    n=int(len(true))
    if n!=252 or exact.sum()!=151:raise AssertionError("baseline changed")
    grouped=[]
    conf=Counter((str(a),str(b)) for a,b in zip(true,pred) if a!=b)
    for label in labels:
        mask=true==label
        wrong=[{"pred":other,"count":int(howmany)} for (tru,other),howmany
               in conf.items() if tru==label]
        wrong.sort(key=lambda a:(-a["count"],a["pred"]))
        grouped.append({"model":label,"provider":provider(label),
             "n":int(mask.sum()),"correct":int((exact&mask).sum()),
             "recall":round(float(np.mean(exact[mask])),6),
             "same_provider_wrong":int((within&mask).sum()),
             "cross_provider_wrong":int((cross&mask).sum()),
             "top_confusions":wrong[:4]})
    providers=[]
    for name in sorted(set(tf)):
        mask=tf==name
        providers.append({"provider":name,"n":int(mask.sum()),
            "exact_correct":int(np.sum(exact&mask)),
            "within_wrong":int(np.sum(within&mask)),
            "cross_wrong":int(np.sum(cross&mask)),
            "family_correct":int(np.sum(family_ok&mask)),
            "exact_ba_given_its_labels":round(float(np.mean(exact[mask])),6)})
    slices=[]
    for version in TOPICS:
        mask=data.version.to_numpy(str)==version
        slices.append({"length_group":version,"n":int(mask.sum()),
            "exact_correct":int(np.sum(exact&mask)),
            "family_wrong":int(np.sum((~family_ok)&mask)),
            "sibling_wrong":int(np.sum(within&mask))})
    top_conf=[{"true":a,"pred":b,"count":int(k),"same_provider":provider(a)==provider(b)}
              for (a,b),k in conf.most_common(20)]
    assert sum(x["n"] for x in grouped)==252
    assert sum(x["exact_correct"] for x in providers)==151
    return {"scope":"executed_0.718_lineage_v36_Fable252_v2_external_publisher_labels",
       "historically_spent_external_source":True,
       "source_provider_provenance_authenticated":False,
       "same_rows_as_existing_Fable252_v2":True,
       "official_kaggle_public_ba_after_run":None,
       "official_kaggle_submission_made":False,
       "new_training":False,"new_inference":False,"new_gpu":False,
       "n":n,"balanced_classes":len(labels),
       "exact_correct":int(exact.sum()),"exact_BA":round(float(exact.mean()),7),
       "exact_wrong":int((~exact).sum()),
       "family_correct":int(family_ok.sum()),
       "same_provider_wrong":int(within.sum()),
       "cross_provider_wrong":int(cross.sum()),
       "oracle_if_all_within_provider_errors_fixed_unrealizable_BA":
          round(float((exact.sum()+within.sum())/n),7),
       "oracle_if_all_cross_provider_errors_fixed_unrealizable_BA":
          round(float((exact.sum()+cross.sum())/n),7),
       "per_label":grouped,
       "by_provider":providers,"by_length":slices,
       "top_offdiag":top_conf,
       "notes":"Oracle uses true labels and cannot run at competition inference. This is source-specific diagnosis, not a deployable proposal or an official 0.718 BA."}

def selftest():
    assert provider("gpt-5.6-sol")=="OpenAI"
    assert provider("claude-fable-5")=="Anthropic"
    assert provider("qwen3.8-max")=="Qwen"
    assert len(TOPICS["short"])+len(TOPICS["median"])+len(TOPICS["long"])==14
    assert sum([14]*18)==252
    x=pd.DataFrame({"model_true":["gpt-5.6-sol"]*14+["gpt-5.6-luna"]*14,
                    "model_pred":["gpt-5.6-sol"]*12+["gpt-5.6-terra"]*2+
                                 ["gpt-5.6-luna"]*10+["gemini-3.8-flash"]*4})
    p=[provider(z) for z in x.model_true]
    q=[provider(z) for z in x.model_pred]
    assert sum(a==b for a,b in zip(p,q))==24
    print("NTL_FABLE_FULL18_ERROR_PROFILE_SYNTHETIC_SELFTEST=PASS",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--panel",type=Path)
    p.add_argument("--submission",type=Path)
    p.add_argument("--aggregate",type=Path)
    p.add_argument("--selftest",action="store_true")
    args=p.parse_args()
    selftest()
    if args.selftest:return
    if not (args.panel and args.submission and args.aggregate):
        p.error("original panel, real private submission and aggregate required")
    evidence=load_truth_and_predictions(args.panel,args.submission,args.aggregate)
    result=report(evidence)
    print("NTL_FABLE_V2_REAL_FULL18_BOTTLENECK="+json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":main()
