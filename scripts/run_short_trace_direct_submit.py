"""Exploratory one-shot Kaggle code submission: frozen 8B Short-TRACE treatment.

No model tuning, target-model API generation or external training rows.
Official train text remains inside Kaggle and is never printed/exported.
The 3-row fixture writes schema-only placeholder; the hidden rerun runs real 8B.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.short_trace_core import make_features, evaluator_ranks
from scripts.run_short_trace_8b_pilot import load_model

LABEL_COUNT=18
EXPECTED_TRAIN=2699
MAX_TOKENS=384
SEED=20261107


def find_competition_input(name):
    for base in (Path("/kaggle/input/competitions/name-that-llm"),
                 Path("/kaggle/input/name-that-llm")):
        path=base/name
        if path.is_file():
            return path
    raise RuntimeError(f"Missing competition input {name}")


def assert_inputs(train,test):
    if len(train)!=EXPECTED_TRAIN or not {"id","prompt_id","text","model"}.issubset(train.columns):
        raise RuntimeError("Official train schema/row count drift")
    if not {"id","text"}.issubset(test.columns):
        raise RuntimeError("Official test schema drift")
    if train["id"].isna().any() or train["id"].duplicated().any():
        raise RuntimeError("Train IDs invalid")
    if test["id"].isna().any() or test["id"].duplicated().any():
        raise RuntimeError("Test IDs invalid")
    if train["text"].isna().any() or test["text"].isna().any():
        raise RuntimeError("Null texts")
    labels=sorted(train.model.astype(str).unique().tolist())
    if len(labels)!=LABEL_COUNT:
        raise RuntimeError("Label count mismatch")
    return labels


def fit_readout(x,y):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    pipe=make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0,class_weight="balanced",
                           max_iter=2000,random_state=SEED))
    pipe.fit(x,y)
    return pipe


def features_for(frame,model,tokenizer,prefix):
    out={"H":[],"S":[],"T":[]}
    token_truncated=0
    tied_logits=0
    for ix,text in enumerate(frame["text"].astype(str)):
        rank,stats=evaluator_ranks(model,tokenizer,text,max_tokens=MAX_TOKENS)
        feat=make_features(rank,text)
        for arm in out:
            out[arm].append(feat[arm])
        token_truncated+=stats["truncated"]
        tied_logits+=stats["ties"]
        if (ix+1)%100==0:
            print(f"SHORT_TRACE_{prefix}_PROGRESS {ix+1}/{len(frame)}",flush=True)
    return {arm:np.stack(feats) for arm,feats in out.items()}, {
        "count":len(frame),"max_tokens":MAX_TOKENS,
        "truncated_count":token_truncated,"tied_logit_count":tied_logits}


def exploratory_oof(train,features):
    """Historical grouped official-OOF diagnostic; NOT fresh transfer evidence."""
    from sklearn.model_selection import GroupKFold
    y=train.model.astype(str).to_numpy()
    group=train.prompt_id.astype(str).to_numpy()
    folds=GroupKFold(n_splits=5)
    scores={}
    for arm in ("H","S","T"):
        pred=np.empty(len(y),dtype=object)
        x=features[arm]
        for train_idx,test_idx in folds.split(x,y,groups=group):
            clf=fit_readout(x[train_idx],y[train_idx])
            pred[test_idx]=clf.predict(x[test_idx])
        labels=sorted(set(y))
        scores[arm]=float(np.mean([np.mean(pred[y==lab]==lab) for lab in labels]))
    return scores


def run():
    train=pd.read_csv(find_competition_input("train.csv"))
    test=pd.read_csv(find_competition_input("test.csv"))
    labels=assert_inputs(train,test)
    out=Path("/kaggle/working/submission.parquet")
    out.parent.mkdir(parents=True,exist_ok=True)

    # Standard Kaggle Code-Competition 1/3-row fixture needs a well-formed file,
    # but it is NOT a scientific proof that hidden 4000-row inference succeeds.
    # On the official rerun, the hidden test is large and this branch is not used.
    if len(test)<=3 and os.getenv("KAGGLE_IS_COMPETITION_RERUN","").lower() not in ("1","true"):
        pd.DataFrame({"id":test["id"].to_numpy(),"model":[labels[0]]*len(test)}).to_parquet(out,index=False)
        print("SHORT_TRACE_PREVIEW_SCHEMA_ONLY_NO_8B_MODEL_INFERENCE",flush=True)
        return {"preview_only":True,"rows":len(test)}

    roots=[]
    for cp in sorted(Path("/kaggle/input/models").rglob("config.json")):
        try:
            cfg=json.loads(cp.read_text(encoding="utf8"))
        except Exception:
            continue
        if (cfg.get("model_type")=="qwen3" and
            int(cfg.get("hidden_size",-1))==4096 and
            int(cfg.get("num_hidden_layers",-1))==36 and
            (cp.parent/"tokenizer_config.json").is_file() and
            any(cp.parent.glob("*.safetensors"))):
            roots.append(cp.parent)
    if len(roots)!=1:
        raise RuntimeError(f"Expected exactly one frozen Qwen3 8B source, found {len(roots)}")

    model,tok=load_model(roots[0])
    xtrain,train_stats=features_for(train,model,tok,"TRAIN")
    xtest,test_stats=features_for(test,model,tok,"HIDDEN")
    scores=exploratory_oof(train,xtrain)
    print("SHORT_TRACE_18_CLASS_GROUPED_OOF_DESCRIPTIVE "+
          json.dumps(scores,sort_keys=True),flush=True)
    fitted=fit_readout(xtrain["T"],train.model.astype(str).to_numpy())
    pred=fitted.predict(xtest["T"])
    if len(pred)!=len(test) or not set(pred).issubset(set(labels)):
        raise RuntimeError("Unexpected prediction length/label")
    submission=pd.DataFrame({"id":test["id"].to_numpy(),"model":pred})
    if submission["id"].duplicated().any() or submission.isna().any().any():
        raise RuntimeError("Invalid submission")
    submission.to_parquet(out,index=False)
    return {
        "output_exists":out.is_file(),"rows":len(test),
        "labels":len(labels),"OOF_repeated_official_only_not_fresh":scores,
        "train_token_stats":train_stats,"test_token_stats":test_stats,
        "treatment":"T_hist_plus_real_transition","shuffled_used_only_as_diagnostic":True
    }


if __name__=="__main__":
    result=run()
    print("SHORT_TRACE_DIRECT_SUBMISSION_RESULT "+
          json.dumps(result,sort_keys=True),flush=True)
