"""One-shot original-source manifest for the already frozen PC-CL 8347-row train cohort.
No new training, no hidden-test input, no text export, no inferred rights clearance.
Outputs public source names and publisher-claimed license/protocol metadata only.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.ntl_pccl_18source_prepare_20261120 import (
    POOL_SHA, RISK, SALT, normalize, sha, split_sources, sample_validation, SEED
)
import hashlib

def truthy_col(v):
    return v.fillna(False).astype(str).str.lower().isin(("true","1","yes"))

def main(root):
    pool=root/"ntl_open_pool.parquet"
    if sha(pool)!=POOL_SHA:raise RuntimeError("POOL_SHA_DRIFT")
    sc=pd.read_csv(root/"sources.csv").fillna("")
    pr=pd.read_csv(root/"source_protocol.csv").fillna("")
    labs=sorted(pd.read_csv(root/"per_model.csv").model.astype(str).unique())
    if len(labs)!=18:raise RuntimeError("LABEL_DRIFT")
    if sc.source_dataset.duplicated().any() or pr.source_dataset.duplicated().any():
        raise RuntimeError("SOURCE_METADATA_NOT_UNIQUE")
    sc=sc.set_index("source_dataset")
    pr=pr.set_index("source_dataset")
    cols=["text","model","source_dataset","prompt_group","prompt_canon","n_words",
          "gen_served_model",*RISK]
    df=pd.read_parquet(pool,columns=cols)
    src=df.source_dataset.astype(str)
    ok=src.isin(sc.index)&src.isin(pr.index)&df.model.astype(str).isin(labs)
    ok &= src.map(pr["served_model_check"]).fillna("").str.lower().str.strip().eq("yes")
    license_meta=src.map(sc["source_license"]).fillna("").astype(str)
    ok &= license_meta.str.contains("cc-by|attribution|apache|mit|open data",case=False,regex=True)
    ok &= df.text.notna() & df.prompt_group.notna() & df.prompt_canon.notna()
    ok &= df.n_words.between(40,250,inclusive="both")
    for col in RISK:ok &= ~truthy_col(df[col])
    c=df.loc[ok,["text","model","source_dataset","prompt_group","prompt_canon",
                  "gen_served_model"]].copy()
    c.rename(columns={"source_dataset":"source","prompt_group":"prompt_id"},inplace=True)
    c["source"]=c.source.astype(str)
    c["prompt_id"]=c.prompt_id.astype(str)
    c["model"]=c.model.astype(str)
    c["prompt_key"]=c.prompt_canon.map(normalize)
    c=c[c.prompt_key.str.len().ge(8)].copy()
    c=c.drop_duplicates(["source","prompt_id","model"],keep="first")
    vsrc=split_sources(sorted(c.source.unique()))
    train=c[~c.source.isin(vsrc)].copy()
    val=c[c.source.isin(vsrc)].copy()
    train=train[~train.prompt_key.isin(set(val.prompt_key))
                &~train.prompt_id.isin(set(val.prompt_id))].copy()
    hashes={hashlib.sha256(normalize(x).encode()).digest() for x in val.text}
    train=train[[hashlib.sha256(normalize(x).encode()).digest() not in hashes
                 for x in train.text]].copy()
    if len(train)!=8347:raise RuntimeError("EXPECTED_8347_PREPARED_TRAIN")
    val=sample_validation(val,labs)
    if len(val)!=1800:raise RuntimeError("EXPECTED_1800_PREPARED_VALID")
    if train.source.nunique()!=8 or val.source.nunique()!=3:
        raise RuntimeError("EXPECTED_8_PLUS_3_SOURCE_COUNT")
    summary=[]
    for name,g in train.groupby("source"):
        license_claim=str(sc.loc[name,"source_license"])
        protocol_claim=str(pr.loc[name,"served_model_check"])
        present=(g.gen_served_model.astype(str).str.strip().str.lower()
                 .isin(("","unknown","nan","none","null"))==False).sum()
        summary.append({
            "public_source_dataset":name,
            "train_rows":int(len(g)),
            "label_count":int(g.model.nunique()),
            "per_label_counts":{str(k):int(v) for k,v in g.model.value_counts().items()},
            "publisher_claimed_source_license":license_claim[:180],
            "publisher_claimed_served_model_check":protocol_claim,
            "rows_with_nonempty_publisher_model_route":int(present),
            "original_provider_output_rights_independently_verified":False,
            "provider_served_model_independently_authenticated":False
        })
    return {
        "status":"ORIGINAL_SOURCE_IDENTITY_REQUIRED_NOT_LICENSE_CLEARED",
        "dataset_sha256":POOL_SHA,
        "frozen_train_rows":len(train),
        "frozen_validation_rows":len(val),
        "train_source_count":len(summary),
        "train_public_source_metadata":summary,
        "needs_original_producer_terms_review":True,
        "publisher_license_claim_is_not_a_provider_permission":True,
        "no_raw_text_prompt_id_prediction_or_weights_exported":True,
        "training":False,"gpu":False,"submission":False
    }

def entry():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    obj=main(args.root)
    args.out.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    print("NTL_PCCL_ORIGINAL_SOURCE_MANIFEST="+json.dumps(obj,sort_keys=True))

if __name__=="__main__":entry()
