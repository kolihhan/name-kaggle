"""Fixed three-feature typography transfer falsifier on existing 18-label sources.

This tests cross-source *feature direction*, not classification. No fitting,
test-time labels, hidden Kaggle inference, source weighting, or threshold sweep.
Original official training rows and Oleg short-source samples are both spent.
Only source-level aggregates leave the authenticated Action.
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

EXPECTED_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FEATURES = (
    # feature, target family, polarity. These come from external public
    # hypotheses, not from looking at Oleg feature rates in this run.
    ("curly_apostrophe", "claude_or_glm", "absent"),
    ("em_dash", "gemini", "absent"),
    ("curly_double_quote", "gpt", "present"),
)
MIN_GAP = 0.15
MAX_TARGET_ABSENT_RATE = 0.05


def fam(model: str) -> str:
    for prefix, value in (
        ("claude-", "claude"), ("glm-", "glm"), ("gemini-", "gemini"),
        ("gpt-", "gpt"), ("qwen", "qwen"), ("kimi-", "kimi")
    ):
        if str(model).startswith(prefix):
            return value
    raise ValueError("UNKNOWN_NTL_FAMILY")


def flagged(text: str, name: str) -> bool:
    if name == "curly_apostrophe":
        return "\u2019" in text
    if name == "em_dash":
        return "\u2014" in text
    if name == "curly_double_quote":
        return "\u201c" in text or "\u201d" in text
    raise ValueError("UNLISTED_FEATURE")


def summary(rows: pd.DataFrame, label: str) -> dict:
    rows = rows.copy()
    rows["family"] = rows["model"].astype(str).map(fam)
    if rows.model.nunique() != 18:
        raise ValueError("SCOPE_LOST_18_LABELS")
    counts = rows.groupby("family").size().to_dict()
    output = {}
    for feat, target, polarity in FEATURES:
        p = rows.text.astype(str).map(lambda x: flagged(x, feat))
        group = rows.family.isin(("claude","glm")) if target == "claude_or_glm" else rows.family.eq(target)
        inside, outside = p[group], p[~group]
        a = float(inside.mean())
        b = float(outside.mean())
        gap = b - a if polarity == "absent" else a - b
        output[feat] = {
            "target": target,
            "polarity": polarity,
            "target_n": int(len(inside)),
            "other_n": int(len(outside)),
            "target_feature_prevalence": round(a, 6),
            "other_feature_prevalence": round(b, 6),
            "signed_target_gap": round(gap, 6),
            "discriminative_direction_positive": bool(gap >= MIN_GAP),
            "target_absence_confirmed": bool(polarity!="absent" or a<=MAX_TARGET_ABSENT_RATE),
        }
    return {
        "cohort":label,
        "n":int(len(rows)),
        "model_count":int(rows.model.nunique()),
        "family_counts":counts,
        "features":output,
    }


def audit(train:Path,external_root:Path)->dict:
    if hashlib.sha256(train.read_bytes()).hexdigest()!=EXPECTED_SHA:
        raise ValueError("OFFICIAL_TRAIN_SHA_DRIFT")
    original=pd.read_csv(train,usecols=["id","text","model","prompt_id"],dtype=str)
    if len(original)!=2699 or original.prompt_id.nunique()!=150:
        raise ValueError("OFFICIAL_SCOPE_DRIFT")
    for name, digest in NTL_OLEG18_SHA.items():
        src=external_root/name
        if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest()!=digest:
            raise ValueError("EXTERNAL_DATA_SOURCE_DRIFT")
    wide=pd.read_parquet(external_root/"external_short_wide.parquet")
    meta=pd.read_parquet(external_root/"external_cell_metadata.parquet")
    held=ntl_oleg18_select(wide,meta)
    if len(held)!=216 or held.model.nunique()!=18 or held.prompt_group_id.nunique()!=216:
        raise ValueError("FROZEN_EXTERNAL_SELECTOR_DRIFT")
    official=summary(original,"historical_official_train_spent")
    external=summary(held,"historical_oleg_short_spent_sensitive_domain")
    decisions={}
    for name,_,polarity in FEATURES:
        x=official["features"][name]
        y=external["features"][name]
        stable=(
            x["discriminative_direction_positive"]
            and y["discriminative_direction_positive"]
            and x["target_absence_confirmed"]
            and y["target_absence_confirmed"]
        )
        decisions[name]={
            "crosssource_direction_supported":bool(stable),
            "official_signed_gap":x["signed_target_gap"],
            "external_signed_gap":y["signed_target_gap"],
        }
    passes=sum(x["crosssource_direction_supported"] for x in decisions.values())
    return {
      "protocol":"ntl-3-typography-contrast-fixed-prevalence-v0-20261116",
      "hypotheses":["generator_specific_typography_stable_across_source","typography_is_generation_protocol_or_domain_confound"],
      "metric":"three prechosen family-conditioned binary-character prevalence gap >=0.15 in each source; absent-family rate <=0.05",
      "cohorts":[official,external],
      "source_level_results":decisions,
      "passed_feature_count":passes,
      "predeclared_gate":"at_least_two_of_three_crosssource_consistent",
      "gate_pass":bool(passes>=2),
      "decision":"PROBE_NEW_SOURCE_SEPARATELY" if passes>=2 else "NO_TYPOGRAPHY_RESCORING",
      "caveats":"Oleg single-domain preinspected source, official train repeatedly inspected; not clean holdout; no same-row paired champion BA comparison",
      "new_model_fitting":False,"new_gpu":False,"official_submission":False,
      "balanced_accuracy_measured":False
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--external",type=Path,required=True)
    a=p.parse_args()
    print("NTL_PRO_TYPOGRAPHY_CROSSSOURCE="+json.dumps(audit(a.train,a.external),sort_keys=True),flush=True)
