"""Frozen Short-TRACE 8B real/shuffled source-transfer pilot.

No official competition rows, training of LLM, submission or raw response export.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.short_trace_core import make_features, evaluator_ranks

POOL_SHA="37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec"
HOLDOUT=("alperen","hailey","willrice")
SEED_SOURCES=frozenset(("abhinav","hailey","kaplinskiy","keshav",
                        "lucasboesen","maniceet","prvsiyan","prvsiyan_v1",
                        "semenoff","seyit","willrice","zakir","zbigniew"))
MAX_TOKENS=384
TRAIN_PER_LABEL=20
TEST_PER_LABEL=12
SEED=20261107
BASELINE_WIDTH=9
FULL_WIDTH=90


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def source_split(pool):
    """Byte-pinned Phase14 safe mask; existing locked DCT holdout source names."""
    needed={"source","prompt_id","prompt_key","text","model","setting"}
    if not needed.issubset(pool.columns) or len(pool)!=75783:
        raise ValueError("Unexpected source data rows/schema")
    seed=pool["source"].astype(str).isin(SEED_SOURCES)
    prompt_ids=set(pool.loc[seed,"prompt_id"].astype(str))
    safe=pool.loc[seed | pool["prompt_id"].astype(str).isin(prompt_ids)].copy()
    safe=safe.reset_index(drop=True)
    if len(safe)!=43781:
        raise ValueError("Frozen Phase14 safe subset drift")
    heldout_mask=safe["source"].astype(str).isin(HOLDOUT)
    holdout=safe.loc[heldout_mask].copy()
    train_mask=~heldout_mask
    train_mask &= ~safe["prompt_id"].astype(str).isin(set(holdout["prompt_id"].astype(str)))
    train_mask &= ~safe["prompt_key"].astype(str).isin(set(holdout["prompt_key"].astype(str)))
    train=safe.loc[train_mask].copy()
    if len(train)!=14502 or len(holdout)!=14506:
        raise ValueError(f"Locked source split drift {len(train)}, {len(holdout)}")
    if set(train["prompt_id"].astype(str)) & set(holdout["prompt_id"].astype(str)):
        raise ValueError("Train/holdout prompt_id overlap")
    if set(train["prompt_key"].astype(str)) & set(holdout["prompt_key"].astype(str)):
        raise ValueError("Train/holdout prompt_key overlap")
    counts=holdout["model"].astype(str).value_counts()
    labels=sorted(l for l,n in counts.items() if n>=20 and l in set(train["model"].astype(str)))
    if len(labels)!=17:raise ValueError(f"Expected 17 eligible labels, got {len(labels)}")
    return train,holdout,labels


def sample_rows(frame,labels,n):
    sampled=[]
    for label in labels:
        sub=frame.loc[frame["model"].astype(str).eq(label)].copy()
        sub["_k"]=sub.apply(lambda r:hashlib.sha256(
             ("short-trace-8b-v1|"+label+"|"+str(r["prompt_id"])+"|"+
              str(r["source"])+"|"+str(r["text"])).encode("utf8")).hexdigest(),axis=1)
        sub=sub.sort_values("_k").drop_duplicates("prompt_id")
        if len(sub)<n:raise ValueError(f"{label} has {len(sub)} distinct prompts, need {n}")
        sampled.append(sub.head(n).drop(columns=["_k"]))
    out=pd.concat(sampled,ignore_index=True)
    if len(out)!=len(labels)*n:raise ValueError("Sampling cardinality drift")
    return out


def balanced_accuracy(true,pred):
    labels=np.unique(true)
    return float(np.mean([(np.asarray(pred)[np.asarray(true)==label]==label).mean()
                          for label in labels]))


def prompt_bootstrap(y, a,b, prompts, iterations=1000):
    groups=np.unique(prompts)
    locations={g:np.flatnonzero(prompts==g) for g in groups}
    random=np.random.default_rng(SEED)
    delta=[]
    for _ in range(iterations):
        draws=random.choice(groups,size=len(groups),replace=True)
        idx=np.concatenate([locations[g] for g in draws])
        delta.append(balanced_accuracy(y[idx],a[idx])-balanced_accuracy(y[idx],b[idx]))
    return list(np.quantile(delta,[0.025,0.975]).astype(float))


def run_readout(train,test,features_train,features_test,labels):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    ytr=train["model"].to_numpy(dtype=str)
    yte=test["model"].to_numpy(dtype=str)
    outputs={}
    for arm in ("H","S","T"):
        xtr=np.stack([f[arm] for f in features_train])
        xte=np.stack([f[arm] for f in features_test])
        if (arm=="H" and xtr.shape[1]!=BASELINE_WIDTH) or (
                arm!="H" and xtr.shape[1]!=FULL_WIDTH):
            raise RuntimeError("Unexpected fixed feature width")
        learner=make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0,class_weight="balanced",
                               max_iter=2000,random_state=SEED))
        learner.fit(xtr,ytr)
        outputs[arm]=learner.predict(xte)
    by_source=[]
    for source,group in test.groupby("source",sort=True):
        ix=group.index.to_numpy(dtype=int)
        by_source.append({"source":str(source),"rows":len(ix),
          "eligible_labels":len(set(yte[ix])),
          "H":balanced_accuracy(yte[ix],outputs["H"][ix]),
          "S":balanced_accuracy(yte[ix],outputs["S"][ix]),
          "T":balanced_accuracy(yte[ix],outputs["T"][ix])})
    ba={arm:balanced_accuracy(yte,outputs[arm]) for arm in outputs}
    prompts=test["prompt_id"].astype(str).to_numpy()
    ci=prompt_bootstrap(yte,outputs["T"],outputs["S"],prompts)
    pass_gate=(ba["T"]-ba["S"]>=0.03 and ba["T"]-ba["H"]>=0.03 and
               ci[0]>0 and all(z["T"]-z["S"]>=-0.05 for z in by_source))
    return {
      "scores":ba,
      "delta_T_minus_S":ba["T"]-ba["S"],
      "delta_T_minus_H":ba["T"]-ba["H"],
      "prompt_bootstrap_T_minus_S_95":ci,
      "source_scores":by_source,
      "rescues_regressions":{
        arm:{"rescues":int(np.sum((outputs[arm]!=yte)&(outputs["T"]==yte))),
             "regressions":int(np.sum((outputs[arm]==yte)&(outputs["T"]!=yte)))}
        for arm in ("H","S")},
      "per_label_recall":{
        label:{arm:float((outputs[arm][yte==label]==label).mean()) for arm in ("H","S","T")}
        for label in labels},
      "pilot_gate":"PASS_FOR_LARGER_8B_VALIDATION_ONLY" if pass_gate else "NO_ADVANCE",
      "independent_fresh_holdout":False,
      "14B_approved":False,
      "kaggle_submission_approved":False,
    }


def load_model(root):
    import torch
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    if not torch.cuda.is_available():raise RuntimeError("Short-TRACE 8B requires GPU")
    c=AutoConfig.from_pretrained(str(root),local_files_only=True)
    if (c.model_type,int(c.hidden_size),int(c.num_hidden_layers))!=("qwen3",4096,36):
        raise RuntimeError("Not frozen Qwen3-8B-Base shape")
    tokenizer=AutoTokenizer.from_pretrained(str(root),local_files_only=True)
    config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(
        str(root),local_files_only=True,torch_dtype=torch.float16,
        device_map="auto",quantization_config=config,attn_implementation="sdpa")
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        model.requires_grad_(False)
    return model,tokenizer


def run(pool_path,model_root,output_path):
    if file_sha256(pool_path)!=POOL_SHA:raise ValueError("Public source hash drift")
    source=pd.read_parquet(pool_path)
    train,test,labels=source_split(source)
    sampled_train=sample_rows(train,labels,TRAIN_PER_LABEL)
    sampled_test=sample_rows(test,labels,TEST_PER_LABEL)
    model,tokenizer=load_model(model_root)
    ftr=[];fte=[]
    totals={"seen":0,"truncated":0,"tie_count":0}
    for frame,out in ((sampled_train,ftr),(sampled_test,fte)):
        for row in frame.itertuples():
            ranks,stats=evaluator_ranks(model,tokenizer,row.text,max_tokens=MAX_TOKENS)
            out.append(make_features(ranks,row.text))
            totals["seen"]+=1
            totals["truncated"]+=stats["truncated"]
            totals["tie_count"]+=stats["ties"]
            if totals["seen"]%25==0:
                print(f"SHORT_TRACE_8B_PROGRESS {totals['seen']}/{len(sampled_train)+len(sampled_test)}",flush=True)
    metrics=run_readout(sampled_train,sampled_test,ftr,fte,labels)
    result={"experiment":"short-trace-8b-v1","model":"Qwen3-8B-Base",
            "pool_sha256":POOL_SHA,
            "train_count":len(sampled_train),"heldout_count":len(sampled_test),
            "eligible_labels":len(labels),"heldout_sources":list(HOLDOUT),
            "max_tokens":MAX_TOKENS,"data_scope":"public_external_only",
            "evaluator_stats":totals,"result":metrics}
    Path(output_path).write_text(json.dumps(result,indent=2,sort_keys=True))
    print("SHORT_TRACE_8B_RESULT "+json.dumps(result,sort_keys=True),flush=True)
    return result


if __name__=="__main__":
    import argparse
    cli=argparse.ArgumentParser()
    cli.add_argument("--pool",type=Path,required=True)
    cli.add_argument("--model",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    a=cli.parse_args()
    run(a.pool,a.model,a.output)
