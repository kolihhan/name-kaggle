"""External-only, CPU-trained 18-class source-shift control.

Purpose: falsify source-balanced character-frequency representation, not to
claim Kaggle score. All rows, model IDs and frozen partitions are external.
C0 / C1 share exactly the same features and classifier hyperparameters.
The sole change is label-only vs label-and-source-balanced sample weights.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

SEED=20261117
MIN_GAIN=0.03

def macro(true,pred,labels):
    return float(np.mean([np.mean(pred[true==k]==k) for k in labels]))

def make_weights(df, source_balanced):
    counts=df.groupby("model").size()
    group=df.groupby(["model","source"]).size()
    if source_balanced:
        w=np.array([1./group[(m,s)] for m,s in zip(df.model,df.source)],dtype=float)
    else:
        w=np.array([1./counts[m] for m in df.model],dtype=float)
    return w/w.mean()

def run(root):
    manifest=json.loads((root/"manifest.json").read_text())
    train=pd.read_parquet(root/"train.parquet")
    valid=pd.read_parquet(root/"valid.parquet")
    labels=list(manifest["labels"])
    if len(labels)!=18 or set(train.model)!=set(labels) or set(valid.model)!=set(labels):
        raise RuntimeError("NOT_COMPLETE_18_LABEL_EXTERNAL_DATA")
    if set(train.source)&set(valid.source) or set(train.prompt_id)&set(valid.prompt_id):
        raise RuntimeError("SOURCE_PROMPT_CONTAMINATION")
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),
        min_df=3,max_features=180000,sublinear_tf=True,dtype=np.float32)
    x=vec.fit_transform(train.text.astype(str))
    y=valid.model.to_numpy()
    z=vec.transform(valid.text.astype(str))
    if x.shape[1]<2000:raise RuntimeError("FEATURE_SPACE_COLLAPSED")
    summaries=[]
    predictions=[]
    for arm,weighted in (("C0_label_balanced",False),("C1_label_source_balanced",True)):
        model=LinearSVC(C=0.5,random_state=SEED,dual="auto",max_iter=3000)
        model.fit(x,train.model.to_numpy(),sample_weight=make_weights(train,weighted))
        pred=model.predict(z)
        predictions.append(pred)
        overall=macro(y,pred,labels)
        per_source={}
        for source in sorted(valid.source.unique()):
            mask=valid.source.eq(source).to_numpy()
            sc=sorted(set(y[mask]))
            per_source[source]={"rows":int(mask.sum()),"labels":len(sc),
                 "macro_present_labels":round(macro(y[mask],pred[mask],sc),6)}
        summaries.append(dict(arm=arm,validation_BA=round(overall,6),
            n_train=len(train),n_valid=len(valid),n_label=18,
            n_features=x.shape[1],by_source=per_source,
            per_label_recall={key:round(float(np.mean(pred[y==key]==key)),6) for key in labels}))
    pred0,pred1=predictions
    source_delta={s:round(summaries[1]["by_source"][s]["macro_present_labels"]-
                         summaries[0]["by_source"][s]["macro_present_labels"],6)
                  for s in summaries[0]["by_source"]}
    delta=summaries[1]["validation_BA"]-summaries[0]["validation_BA"]
    rescues=int(((pred0!=y)&(pred1==y)).sum())
    regressions=int(((pred0==y)&(pred1!=y)).sum())
    passed=bool(delta>=MIN_GAIN and sum(v>0 for v in source_delta.values())>=2 and rescues>regressions)
    return {"dataset":"public_openpool_source_verification_tier",
        "frozen_data_sha":manifest["data_sha256"],
        "data_release_included":False,
        "model_trained_cpu":True,
        "competition_submission":False,
        "source_served_id_independently_proven":False,
        "champion_same_row_comparison_available":False,
        "arms":summaries,
        "source_balancing_delta_BA":round(delta,6),
        "rescues":rescues,"regressions":regressions,
        "by_source_delta":source_delta,
        "decision":"KEEP_FOR_NEW_SOURCE_TEST_ONLY" if passed else "REJECT_CPU_SOURCE_BALANCE",
        "not_kaggle_public_BA":True}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=run(args.data_dir)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True))
    print("NTL_ACTUAL_CPU_SOURCE_DISJOINT_TRAIN_RESULT="+json.dumps({
       "C0_BA":result["arms"][0]["validation_BA"],
       "C1_BA":result["arms"][1]["validation_BA"],
       "delta_BA":result["source_balancing_delta_BA"],
       "rescues":result["rescues"],"regressions":result["regressions"],
       "by_source_delta":result["by_source_delta"],
       "decision":result["decision"],
    },sort_keys=True),flush=True)
if __name__=="__main__":main()
