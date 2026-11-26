#!/usr/bin/env python3
"""One-shot *spent* 252-row compatibility falsifier for frozen E5 Kimi-vs-Luna.

Uses previously completed private v36 lineage 18-class predictions.
CPU inference on frozen E5 only; ZERO Kaggle notebook pushes/competition submits/GPU.
Never interpret this historically spent panel as fresh independent validation.
"""
from __future__ import annotations
import hashlib, json, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from ntl_kimi_e5_frozen_transfer_20261126 import datasets, pair, embed, SEED
from ntl_kimi_e5_frozen_transfer_20261126 import N_TRAIN, A, B, log

PANEL_REF="nischaydnk/ntl-topic-synthetic-18llms"
KERNEL_REF="kolihhan/ntl-fable-preservation-252-v2-20261125"
PANEL_SHA="bce34670f4d1170375e7baf85eec5577af4282d5f0ca6cf1573ce46e81c5a45c"
PRED_SHA="1a5496f0f42c745a78ec37348828bf829310e9b64a34e8957e125e3a92b3f767"
TOPICS={"short":(98,25,65,83,16,61,68),"median":(77,54,99,37),"long":(72,63,87)}

def cli(*a):
    p=subprocess.run(["kaggle",*a],capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError("READ_ONLY_SOURCE_DOWNLOAD_FAILED:"+str(a[:2]))
    return p

def load_spent():
    with tempfile.TemporaryDirectory(prefix="ntl-e5-spent252-") as td:
        root=Path(td)
        cli("datasets","download","-d",PANEL_REF,"-f","generations.csv","-p",td,"--unzip")
        path=root/"generations.csv"
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==PANEL_SHA
        raw=pd.read_csv(path)
        rows=[]
        for version,ids in TOPICS.items():
            for topic in ids:
                z=raw[(raw.version==version)&(raw.topic_id==topic)].copy()
                assert len(z)==18 and z.model.nunique()==18
                rows.append(z.sort_values("model",kind="stable"))
        truth=pd.concat(rows,ignore_index=True)
        truth.insert(0,"id",[f"fable-observation-{i:03d}" for i in range(len(truth))])
        assert len(truth)==252 and truth.model.nunique()==18 and truth.model.value_counts().eq(14).all()
        cli("kernels","output",KERNEL_REF,"-p",td,"--file-pattern","^submission.parquet$","-o")
        f=list(root.rglob("submission.parquet"))
        assert len(f)==1 and hashlib.sha256(f[0].read_bytes()).hexdigest()==PRED_SHA
        pred=pd.read_parquet(f[0])
        assert pred.columns.tolist()==["id","model"] and len(pred)==252
    joined=truth[["id","model","text"]].merge(pred,how="left",on="id",suffixes=("_truth","_baseline"),validate="one_to_one",sort=False)
    assert len(joined)==252 and joined.model_baseline.notna().all()
    assert int((joined.model_truth==joined.model_baseline).sum())==151
    return joined

def main():
    base=load_spent()
    official,_=datasets()
    oa,ob,pids=pair(official)
    assert len(pids)==150
    used=pids[:N_TRAIN]
    train=[str(oa.loc[k,"text"]) for k in used]+[str(ob.loc[k,"text"]) for k in used]
    out=base.text.astype(str).tolist()
    # Source is historically spent: this is an execution compatibility check,
    # not fresh inference evidence against the pretrained 0.718 model.
    all_embeddings=embed(train+out)
    n=N_TRAIN
    assert all_embeddings.shape==(2*n+252,1024) and np.isfinite(all_embeddings).all()
    ma=all_embeddings[:n].mean(0);mb=all_embeddings[n:2*n].mean(0)
    direction=ma-mb
    mid=(ma+mb)/2
    e5_decision=(all_embeddings[2*n:]-mid)@direction
    e5_binary=np.where(e5_decision>0,A,B)
    pred=base.model_baseline.astype(str).to_numpy()
    truth=base.model_truth.astype(str).to_numpy()
    assert len(pred)==len(e5_binary)==252
    eligible=np.isin(pred,[A,B])
    new_pred=pred.copy()
    new_pred[eligible]=e5_binary[eligible]
    changed=(new_pred!=pred)
    original_correct=(pred==truth)
    new_correct=(new_pred==truth)
    rescue=np.logical_and(~original_correct,new_correct)
    regression=np.logical_and(original_correct,~new_correct)
    noncorrect=np.logical_and(changed,~rescue & ~regression)
    assert sum(new_correct)==151+sum(rescue)-sum(regression)
    assert not np.any(changed & ~eligible)
    per_truth={}
    for label in [A,B]:
        m=truth==label
        per_truth[label]={"n":int(m.sum()),"base_correct":int((original_correct&m).sum()),
                          "counterfactual_correct":int((new_correct&m).sum()),
                          "rescues":int((rescue&m).sum()),"regressions":int((regression&m).sum())}
    log("NTL_E5_SPENT_252_18CLASS_COMPATIBILITY",
        matched_baseline_sha256=PRED_SHA,
        test_rows=252,classes=18,balanced_labels=14,
        frozen_e5_model="intfloat/e5-large-v2",
        train_official_prompt_pairs=N_TRAIN,
        threshold_tuned=False,
        gate="only if original 18-class prediction Kimi K3 or GPT 5.6 Luna",
        baseline_correct=int(original_correct.sum()),
        counterfactual_correct=int(new_correct.sum()),
        baseline_18class_BA=float(original_correct.mean()),
        counterfactual_18class_BA=float(new_correct.mean()),
        eligible_predictions=int(eligible.sum()),
        changed_predictions=int(changed.sum()),
        rescues=int(rescue.sum()),regressions=int(regression.sum()),
        changed_but_still_incorrect=int(noncorrect.sum()),
        per_truth=per_truth,
        test_panel_historically_spent=True,
        not_independent_and_not_promotable=True,
        no_gpu=True,no_kaggle_submission=True,no_new_model_training=True)

if __name__=="__main__":main()
