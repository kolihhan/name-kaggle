"""Read-only official 5-fold vs external Phase 15 contamination audit.

This checks an actual historical validation failure mechanism, not model quality.
No fitting, model inference, Public-LB score, or competition submission.

Pre-registered observable:
  N=2699 official rows; 150 prompt groups, five validation groups of 30.
  26,025 external Phase 15 fit rows enter EVERY fold, but whose source/prompt
  lineage needs empirical overlap checks. An observed exact repeated response
  in an external fit row vs heldout official response is direct contamination.
  Identical string prompt IDs without verified namespaces are *possible* prompt
  collision, not proof the exact requests match.
  Detectable change in model BA: N/A; no model scoring. The historical
  +0.0353 OOF vs -0.012 Public motivates the audit but is NOT its measured effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_phase14_public_sparse import (
    EXPECTED_RAW_SHA256, build_conservative_subset, sha256_file,
)
from scripts.phase15_terms_clean import build_terms_clean_subset
from scripts.run_strong_sparse_anchor import load_train


def normalized_hash(v: object) -> str:
    text = unicodedata.normalize("NFKC", str(v)).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode()).hexdigest()


def audit(train: pd.DataFrame, external: pd.DataFrame) -> dict:
    if len(train) != 2699 or train.id.nunique() != 2699:
        raise RuntimeError("OFFICIAL_TRAIN_DENOMINATOR_MISMATCH")
    if train.prompt_id.nunique() != 150 or train.model.nunique() != 18:
        raise RuntimeError("OFFICIAL_GROUP_CLASS_DENOMINATOR_MISMATCH")
    if len(external) != 26025 or external.prompt_id.nunique() != 2801:
        raise RuntimeError("EXTERNAL_PHASE15_DENOMINATOR_MISMATCH")
    if not set(train.fold_id.astype(int)) == set(range(5)):
        raise RuntimeError("FOLD_LABEL_MISMATCH")
    prompt_map=train.groupby("prompt_id").fold_id.nunique()
    if prompt_map.max()!=1:
        raise RuntimeError("OFFICIAL_PROMPT_SPLIT_ACROSS_FOLDS")

    t=train.copy()
    x=external.copy()
    t["h"]=t.text.map(normalized_hash)
    x["h"]=x.text.map(normalized_hash)
    x["p"]=x.prompt_id.astype(str)
    x["key"]=x.prompt_key.astype(str)
    t["p"]=t.prompt_id.astype(str)
    xhash_counts=x.h.value_counts()
    xtotaldup=int((xhash_counts>1).sum())
    overall_true={k:int(v) for k,v in t.model.astype(str).value_counts().items()}
    per_fold=[]
    for fold in range(5):
        hold=t.loc[t.fold_id.eq(fold)].copy()
        fit=t.loc[~t.fold_id.eq(fold)].copy()
        va_p=set(hold.p)
        tr_p=set(fit.p)
        if va_p & tr_p:
            raise RuntimeError("OFFICIAL_GROUP_SPLIT_DRIFT_FOLD_"+str(fold))
        if len(va_p)!=30:
            raise RuntimeError("OFFICIAL_PROMPT_COUNT_DRIFT_FOLD_"+str(fold))
        fit_hash=set(fit.h)
        held_hash=set(hold.h)
        raw_prompt_collisions=x.p.isin(va_p)
        key_collisions=x.key.isin(va_p)
        exact_ext=x.h.isin(held_hash)
        exact_official=hold.h.isin(fit_hash)
        same_model_exact=0
        if exact_ext.any():
            gold_pairs=set(zip(hold.h,hold.model.astype(str)))
            same_model_exact=sum(
                (h,m) in gold_pairs for h,m in
                zip(x.loc[exact_ext,"h"],x.loc[exact_ext,"model"].astype(str))
            )
        per_fold.append({
            "fold":fold,
            "official_train_rows":len(fit),
            "official_validation_rows":len(hold),
            "official_train_prompt_groups":len(tr_p),
            "official_validation_prompt_groups":len(va_p),
            "official_train_to_validation_prompt_id_overlap":len(tr_p&va_p),
            "external_fit_rows_in_every_fold":len(x),
            "external_prompt_id_equal_to_official_validation_prompt_id_rows":int(raw_prompt_collisions.sum()),
            "external_prompt_id_equal_to_official_validation_prompt_id_unique":int(x.loc[raw_prompt_collisions,"p"].nunique()),
            "external_prompt_key_equal_to_official_validation_prompt_id_rows":int(key_collisions.sum()),
            "external_normalized_response_exact_overlap_rows":int(exact_ext.sum()),
            "external_normalized_response_exact_overlap_same_class_rows":int(same_model_exact),
            "external_normalized_response_exact_overlap_distinct_validation_responses":int(len(set(x.loc[exact_ext,"h"]))),
            "official_train_to_validation_exact_response_overlap_rows":int(exact_official.sum()),
        })
    # Optimistic paired-MDE sensitivity (not a trained model result):
    # detectable d approximately (z_1-a/2+z_1-b) sqrt(q/N) for q discordance,
    # assuming IID and balanced classes. Source/prompt correlation and class
    # imbalance invalidate using these as genuine power guarantees.
    zsum=1.959963984540054+0.8416212335729143
    sensitivity=[
        {"q_assumed":q,
         "iid_n2699_optimistic_mde":round(zsum*math.sqrt(q/2699),5),
         "all_150_prompt_clusters_as_independent_mde":round(zsum*math.sqrt(q/150),5)}
        for q in (0.10,0.20,0.30)
    ]
    result={
        "protocol":"ntl-oof-leak-audit-20261118-v1",
        "metric":"NONE_NO_MODEL_EVALUATION",
        "historical_phase7_oof":0.3700521998508576,
        "historical_phase15_oof":0.4053218990802883,
        "historical_phase7_public":0.369,
        "historical_phase15_public":0.357,
        "official_rows":len(t),"official_prompt_groups":int(t.p.nunique()),
        "official_model_labels":int(t.model.nunique()),
        "external_phase15_training_rows":len(x),
        "external_phase15_labels":int(x.model.nunique()),
        "external_phase15_prompt_ids":int(x.p.nunique()),
        "official_prompt_never_split":bool(prompt_map.max()==1),
        "official_cross_prompt_exact_response_duplicates":int(
            t.groupby("h").p.nunique().gt(1).sum()
        ),
        "external_duplicate_normalized_response_hash_groups":xtotaldup,
        "per_fold":per_fold,
        "total_fold_exposure_to_external_response_exact_duplicates":
            sum(z["external_normalized_response_exact_overlap_rows"] for z in per_fold),
        "total_fold_exposure_to_external_prompt_id_string_collisions":
            sum(z["external_prompt_id_equal_to_official_validation_prompt_id_rows"] for z in per_fold),
        "total_official_train_validation_exact_response_overlap":
            sum(z["official_train_to_validation_exact_response_overlap_rows"] for z in per_fold),
        "mde_sensitivity_not_proof":sensitivity,
        "expected_effect_under_test":"identity/provenance contamination could explain historical OOF-Public reversal",
        "minimum_detectable_impact_for_this_no_model_audit":"not applicable; detection of exact overlap is exact counting",
        "source_namespace_collision_is_not_proven_same_prompt":True,
        "external_prompt_bank_lineage_semantic_overlap":'UNKNOWN',
        "current_lightning_out_of_fold_status":"UNSUPPORTED_PREFIT_EACH_FOLD_NOT_RETRAINED",
        "official_train_public_ooof_not_independent_of_adaptive_historical_research":True,
        "no_model_fit":True,"no_model_inference":True,"no_gpu":True,
        "no_official_submission":True,"raw_training_data_exported":False,
    }
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw-pool",required=True,type=Path)
    ap.add_argument("--official",required=True,type=Path)
    ap.add_argument("--folds",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    if sha256_file(a.raw_pool)!=EXPECTED_RAW_SHA256:
        raise RuntimeError("SOURCE_SNAPSHOT_CHANGED")
    train=load_train(a.official,a.folds)
    raw=pd.read_parquet(a.raw_pool)
    phase14=build_conservative_subset(raw,validate_full_contract=True)
    phase15=build_terms_clean_subset(phase14,validate_full_contract=True)
    report=audit(train,phase15)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("NTL_PHASE15_REAL_GROUP_AND_EXTERNAL_LEAK_AUDIT="+json.dumps({
        k:report[k] for k in
        ("official_rows","official_prompt_groups","official_prompt_never_split",
         "official_cross_prompt_exact_response_duplicates",
         "external_phase15_training_rows","external_phase15_prompt_ids",
         "total_fold_exposure_to_external_response_exact_duplicates",
         "total_fold_exposure_to_external_prompt_id_string_collisions",
         "total_official_train_validation_exact_response_overlap",
         "mde_sensitivity_not_proof")
    },sort_keys=True),flush=True)
    print("NTL_PHASE15_PER_FOLD_EXTERNAL_EXPOSURE="+json.dumps(report["per_fold"],sort_keys=True),flush=True)


if __name__=="__main__":
    main()
