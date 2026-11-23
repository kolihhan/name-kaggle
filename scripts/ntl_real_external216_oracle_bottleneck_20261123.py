"""No-training bottleneck falsifier on frozen *real* outside-source NTL 216.

Truth: publisher-audited Oleg216; predictions: exact historical 0.691 full
37-module strong pipeline, NOT present 0.715. Selection predetermined Nov16.
Outputs aggregate only; never use oracle labels for prediction changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.ntl_oleg18_frozen_selector_20261116 import (
    NTL_OLEG18_SHA, NTL_OLEG18_MODELS, ntl_oleg18_select,
)
LABELS=tuple(NTL_OLEG18_MODELS)
SEED=20261123
BOOT=4000

def family(label):
    if label.startswith("claude-"):return "anthropic"
    if label.startswith("gemini-"):return "google"
    if label.startswith("gpt-"):return "openai"
    if label.startswith("glm-"):return "zai"
    if label.startswith("kimi-"):return "moonshot"
    if label.startswith("qwen"):return "alibaba"
    raise ValueError("unknown class family")

def evaluate(external_dir,pred_path):
    for file,expected in NTL_OLEG18_SHA.items():
        actual=hashlib.sha256((external_dir/file).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("PUBLISHER_FILE_SHA_DRIFT:"+file)
    wide=pd.read_parquet(external_dir/"external_short_wide.parquet")
    meta=pd.read_parquet(external_dir/"external_cell_metadata.parquet")
    truth=ntl_oleg18_select(wide,meta)
    assert (len(truth)==216 and truth.model.nunique()==18
            and truth.prompt_group_id.nunique()==216
            and truth.model.value_counts().eq(12).all())
    pred=pd.read_parquet(pred_path)
    assert set(pred.columns)=={"id","model"} and pred.id.is_unique
    assert len(pred)==216 and set(pred.id.astype(str))==set(truth.id.astype(str))
    assert set(pred.model).issubset(LABELS)
    d=truth[["id","model","text","prompt_group_id"]].merge(
        pred.rename(columns={"model":"pred"}),on="id",validate="one_to_one")
    assert len(d)==216 and set(d.pred).issubset(LABELS)
    y=d.model.to_numpy(dtype=str);p=d.pred.to_numpy(dtype=str)
    yf=np.array([family(v) for v in y]);pf=np.array([family(v) for v in p])
    exact=p==y;cross=yf!=pf;within=(~exact)&(~cross)
    assert int(exact.sum())==89 and int(cross.sum())==63 and int(within.sum())==64
    # Actual model "family top-1 accuracy" is label family agreement,
    # NOT a hypothetical family classifier that predicts a different family.
    fam_correct=int((yf==pf).sum())
    def byclass(a):
        return round(float(np.mean([np.mean(a[y==label]) for label in LABELS])),8)
    class_counts={
        label:{
            "correct":int(np.sum(exact&(y==label))),
            "within_family_wrong":int(np.sum(within&(y==label))),
            "across_family_wrong":int(np.sum(cross&(y==label))),
        } for label in LABELS
    }
    families=sorted(set(yf))
    byfam={
        f:{
            "rows":int((yf==f).sum()),
            "exact_correct":int((exact&(yf==f)).sum()),
            "family_correct":int(((yf==pf)&(yf==f)).sum()),
            "sibling_errors":int((within&(yf==f)).sum()),
            "cross_family_errors":int((cross&(yf==f)).sum())
        } for f in families
    }
    pred_hist={label:int((p==label).sum()) for label in LABELS}
    # Bootstrap heldout 216 observed *balanced* labels (12 per class), sample
    # conditionally on class only; old/domain-selected cohort, descriptive only.
    rng=np.random.default_rng(SEED)
    boot=np.zeros((BOOT,3))
    for k in range(BOOT):
        ix=np.concatenate([
            rng.choice(np.where(y==label)[0],size=12,replace=True)
            for label in LABELS
        ])
        boot[k,0]=np.mean(exact[ix])
        boot[k,1]=np.mean((~cross)[ix])
        boot[k,2]=np.mean(within[ix])
    ci=np.quantile(boot,[0.025,0.975],axis=0)
    vals={
        "test_cohort":"preselected Oleg216 2026-11-16 all sensitive_topics",
        "model":"historical 0.691 Lightning V2 fully trained inference, NOT own 0.715",
        "total":216,"num_classes":18,"class_support_exact":12,
        "true_external_source_and_model_versions_not_fully_authenticated":True,
        "exact_correct":int(exact.sum()),
        "exact_BA":byclass(exact),
        "true_family_correct":fam_correct,
        "family_top1_BA":byclass(~cross),
        "wrong_sibling_given_correct_family":int(within.sum()),
        "wrong_across_family":int(cross.sum()),
        "sibling_only_ORACLE_final_correct":int(exact.sum()+within.sum()),
        "sibling_only_ORACLE_BA":byclass(exact|within),
        "family_only_ORACLE_final_correct":int(exact.sum()+cross.sum()),
        "family_only_ORACLE_BA":byclass(exact|cross),
        "both_oracles_using_truth_are_not_deployable":True,
        "class_counts":class_counts,"family_counts":byfam,
        "predicted_class_histogram":pred_hist,
        "exact_BA_conditional_BOOTCI95":ci[:,0].round(6).tolist(),
        "oracle_sibling_BA_conditional_BOOTCI95":ci[:,1].round(6).tolist(),
        "sibling_error_frac_BOOTCI95":ci[:,2].round(6).tolist(),
        "not_an_independent_validation_for_model_selection":True,
        "same_narrow_spent_cohort":True,
        "no_0715_predictions_available":True,
        "no_model_training_or_inference_or_submission":True
    }
    assert vals["sibling_only_ORACLE_final_correct"]==153
    assert vals["family_only_ORACLE_final_correct"]==152
    print("NTL_REAL_216_ORACLE_BOTTLENECK="+json.dumps(vals,sort_keys=True),flush=True)
    return vals

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--external",type=Path,required=True)
    ap.add_argument("--pred",type=Path,required=True)
    args=ap.parse_args()
    evaluate(args.external,args.pred)
