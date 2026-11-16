"""Fixed no-fit typographic CONTRADICTION detector on two frozen champion scores.

This is NOT a rescorer; no substitute author label exists for a veto. It tests
whether the already cross-source corroborated features contradict the frozen
0.691 champion's predicted provider on known labeled diagnostic cohorts.
The Oleg 216 rows and official train 1200 are repeatedly inspected/spent.
Only safe aggregates are emitted; source text and prediction IDs remain private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.ntl_oleg18_frozen_selector_20261116 import (
    NTL_OLEG18_SHA, ntl_oleg18_select
)

TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
PRED1200_SHA = "926dffc6ab0909f63767e8fcec7ca9488f7d1dc7c49656b0eaa8cf50064e0826"


def contradiction(pred: str, text: str)->tuple[bool,str]:
    p=str(pred)
    t=str(text)
    if (p.startswith("claude-") or p.startswith("glm-")) and "\u2019" in t:
        return True,"curly_apostrophe_violates_predicted_claude_or_glm"
    if p.startswith("gemini-") and "\u2014" in t:
        return True,"em_dash_violates_predicted_gemini"
    return False,"none"


def score(truth:pd.DataFrame, predictions:pd.DataFrame, name:str, wrong_expected:int)->dict:
    if list(predictions.columns)!=["id","model"]:
        raise ValueError("prediction columns unexpected")
    if len(truth)!=len(predictions) or predictions.id.duplicated().any():
        raise ValueError("prediction coverage problem")
    frame=truth[["id","model","text"]].merge(predictions,on="id",
        how="left",validate="one_to_one",suffixes=("_truth","_pred"))
    if frame.model_pred.isna().any():
        raise ValueError("missing model predictions")
    err=frame.model_truth.ne(frame.model_pred)
    if int(err.sum())!=wrong_expected:
        raise ValueError("authentic champion error total drift")
    flags=[contradiction(p,t) for p,t in zip(frame.model_pred,frame.text)]
    mask=pd.Series([f[0] for f in flags],dtype=bool)
    reasons=pd.Series([f[1] for f in flags],dtype=str)
    wrong=int((mask&err).sum())
    right=int((mask&(~err)).sum())
    by_rule={}
    for reason in sorted(set(reasons)):
        if reason=="none":continue
        flag=reasons.eq(reason)
        by_rule[reason]={
            "flagged":int(flag.sum()),"wrong":int((flag&err).sum()),
            "right":int((flag&(~err)).sum())
        }
    return {
        "source":name,"n":len(frame),"wrong_total":int(err.sum()),
        "flagged":int(mask.sum()),
        "flagged_wrong":wrong,"flagged_right":right,
        "veto_precision":round(wrong/int(mask.sum()),6) if mask.sum() else None,
        "wrong_recall":round(wrong/wrong_expected,6),
        "by_rule":by_rule
    }


def run(train:Path,pred1200:Path,external:Path,external_preds:Path)->dict:
    if hashlib.sha256(train.read_bytes()).hexdigest()!=TRAIN_SHA:
        raise ValueError("official train content changed")
    if hashlib.sha256(pred1200.read_bytes()).hexdigest()!=PRED1200_SHA:
        raise ValueError("not frozen authentic 1200 predictions")
    official=pd.read_csv(train,usecols=["id","model","text"],dtype=str).iloc[:1200].copy()
    official["id"]=["scale1200_%04d"%i for i in range(1200)]
    preds=pd.read_parquet(pred1200)
    if preds.id.astype(str).tolist()!=official.id.tolist():
        raise ValueError("frozen text/prediction order drift")
    a=score(official,preds,"historical_spent_official_train1200",51)
    for name,digest in NTL_OLEG18_SHA.items():
        if hashlib.sha256((external/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("frozen external source SHA drift")
    btruth=ntl_oleg18_select(
        pd.read_parquet(external/"external_short_wide.parquet"),
        pd.read_parquet(external/"external_cell_metadata.parquet"))
    if (len(btruth)!=216 or btruth.model.nunique()!=18
        or btruth.prompt_group_id.nunique()!=216):
        raise ValueError("216 source eligibility drift")
    b=score(btruth,pd.read_parquet(external_preds),
            "historical_spent_Oleg18_sensitive216",127)
    # Precommitted gate: each source >=3 actual wrong flags with <=1 false
    # flag, and >=80% combined error precision. A pass can only justify
    # a separate independent ablation, NOT a score-changing use of this rule.
    passed=(a["flagged_wrong"]>=3 and b["flagged_wrong"]>=3 and
            a["flagged_right"]<=1 and b["flagged_right"]<=1 and
            (a["flagged_wrong"]+b["flagged_wrong"])>=8 and
            (a["flagged_wrong"]+b["flagged_wrong"]) /
            max(1,a["flagged"]+b["flagged"])>=0.80)
    return {
      "protocol":"ntl-two-source-hard-typography-contradiction-v0-20261116",
      "hypotheses":{
        "H_stable":"model-prediction/typographic incompatibility reliably identifies wrong provider on both spent sources",
        "H_shift":"provider syntax signal is insufficient and generates substantial false vetoes or few errors"
      },
      "predeclared_gate":"at_least_3_wrong_flags_each_source,_<=1_correct_flag_each_source,_>=8_total_wrong_flags,_>=0.8_pooled_precision",
      "cohorts":[a,b],
      "gate_pass":bool(passed),
      "decision":"ONLY_NEW_DISJOINT_ABLATION_PERMITTED" if passed else "STOP_HARD_TYPOGRAPHY_VETO",
      "not_a_corrected_label":True,"balanced_accuracy_delta":None,
      "not_independent_transfer":True,"model_training":False,
      "GPU":False,"official_submission":False
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--pred1200",type=Path,required=True)
    p.add_argument("--external",type=Path,required=True)
    p.add_argument("--external-predictions",type=Path,required=True)
    a=p.parse_args()
    print("NTL_PRO_TWO_SOURCE_TYPO_VETO="+json.dumps(run(a.train,a.pred1200,a.external,a.external_predictions),sort_keys=True),flush=True)
