"""Pair a frozen independent LoRA24 A1 private inference with scored champion.

Only public-source row IDs, label names, prediction names; never competition
rows, raw prompts/text, private model weights or per-row JSON in public logs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_SOURCES=("toan","nimitha")
OBSERVED_12=frozenset((
    "claude-opus-4-6","claude-opus-4-8","claude-opus-5",
    "claude-sonnet-5","gemini-3.1-pro-preview","gemini-3.6-flash",
    "gemini-3.8-flash","gpt-5.5","gpt-5.6-luna","gpt-5.6-sol",
    "gpt-5.6-terra","gpt-6-astra",
))

def _family(label:str)->str:
    return str(label).split("-")[0]

def _counts(frame:pd.DataFrame)->dict:
    incumbent=frame["model_predicted"]
    candidate=frame["prediction"]
    truth=frame["model_expected"]
    base_ok=(incumbent==truth)
    cand_ok=(candidate==truth)
    rescues=int((~base_ok & cand_ok).sum())
    regressions=int((base_ok & ~cand_ok).sum())
    top_err=frame[(~base_ok)&cand_ok]
    labels=sorted(OBSERVED_12)
    per_class={k:round(float((candidate[truth==k]==k).mean()),6)
               for k in labels}
    return {
        "rows":int(len(frame)),
        "champion_correct":int(base_ok.sum()),
        "candidate_correct":int(cand_ok.sum()),
        "champion_wrong":int((~base_ok).sum()),
        "candidate_wrong":int((~cand_ok).sum()),
        "candidate_conditional_12_target_ba_18_predictions":
            round(float(np.mean(list(per_class.values()))),6),
        "candidate_per_label_recall":per_class,
        "paired_rescues":rescues,
        "paired_regressions":regressions,
        "paired_net":rescues-regressions,
        "both_wrong":int((~base_ok & ~cand_ok).sum()),
        "both_right":int((base_ok & cand_ok).sum()),
        "disagree":int((incumbent!=candidate).sum()),
        "rescue_from_same_family_incumbent_error":int(sum(
            _family(row["model_predicted"])==_family(row["model_expected"])
            for _,row in top_err.iterrows()
        )),
        "provider":{
            k:{
                "n":int((truth.map(_family)==k).sum()),
                "rescue":int((~base_ok & cand_ok & (truth.map(_family)==k)).sum()),
                "regression":int((base_ok & ~cand_ok & (truth.map(_family)==k)).sum())
            } for k in ("claude","gemini","gpt")
        },
    }


def compare(champion:pd.DataFrame,candidate:pd.DataFrame)->dict:
    old_cols={"id","source","prompt_id","model_expected","model_predicted"}
    new_cols={"id","source","prompt_id","model","prediction"}
    if set(champion.columns)!=old_cols or set(candidate.columns)!=new_cols:
        raise ValueError("frozen private diagnostic schema mismatch")
    for name,frame in (("champion",champion),("A1 candidate",candidate)):
        if len(frame)!=347 or frame.id.isna().any() or not frame.id.is_unique:
            raise ValueError(name+" must have exactly 347 unique IDs")
        if set(frame.source)!=set(CANONICAL_SOURCES):
            raise ValueError(name+" source coverage drift")
        if frame.source.value_counts().to_dict()!={"nimitha":240,"toan":107}:
            raise ValueError(name+" row count/source drift")
        if frame.isna().any().any():
            raise ValueError(name+" contains nulls")
    joined=champion.merge(candidate,on="id",how="left",
        suffixes=("_inc","_cand"),validate="one_to_one")
    if len(joined)!=347 or joined.prediction.isna().any():
        raise ValueError("candidate predictions missing or duplicated")
    if not ((joined.source_inc==joined.source_cand)&
            (joined.prompt_id_inc.astype(str)==joined.prompt_id_cand.astype(str))&
            (joined.model_expected==joined.model)).all():
        raise ValueError("same-row source/prompt/model identity mismatch")
    joined=joined.rename(columns={"source_inc":"source","prompt_id_inc":"prompt_id"})
    if set(joined.model_expected)!=OBSERVED_12:
        raise ValueError("frozen ground-truth labels drift")
    combined=_counts(joined)
    slices={source:_counts(joined[joined.source==source])
            for source in CANONICAL_SOURCES}
    result={
        "scope":"same-row already-inspected 12-label diagnostic, NOT full 18-class Kaggle BA",
        "original_kaggle_champion_public_ba":0.679,
        "champion_ref":"58210835",
        "candidate":"existing own Stage1 A1 Qwen3-8B LoRA24 checkpoint (NOT Stage2 0.520)",
        "supervised_training":False,
        "competition_submission":False,
        "paired":combined,
        "by_source":slices,
        "pass_sourcewise_rescue_gate":all(
            slices[s]["paired_rescues"]>slices[s]["paired_regressions"]
            for s in CANONICAL_SOURCES),
        "novel_model_claim":False,
        "fresh_holdout_claim":False,
    }
    return result


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--champion",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    args=p.parse_args()
    old=pd.read_parquet(args.champion)
    new=pd.read_parquet(args.candidate)
    report=compare(old,new)
    print("NTL_A1_PAIRED_AGGREGATES",json.dumps(report,sort_keys=True),
          flush=True)


if __name__=="__main__":
    main()
