"""Recover published 800+1899 GLM token-cache versions and merge losslessly.

Only numeric NPZ channels and public fixed feature CSVs are materialized.
Object ID arrays are decoded statically with pickletools (NO unpickling of
third-party code). The 2699-row merged data remain ephemeral on CI.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import os
from pathlib import Path
import tempfile
import zipfile
import json

import numpy as np
import pandas as pd
import requests

from scripts.ntl_glm_distributed_evidence_20261122 import safe_pickle_object_string_ids

URL="https://www.kaggle.com/api/v1/kernels/output/download/estopadilla/ntl-glm-prefix-ab"
FOUR=("flash_features.csv","glm53_features.csv","flash_scores.npz","glm53_scores.npz")
EXPECTED={1:800,2:1899}

def recover(version:int,target:Path,token:str):
    target.mkdir(parents=True,exist_ok=False)
    headers={"Authorization":"Bearer "+token} if token else {}
    with requests.get(URL,params={"version_number":version},headers=headers,
            stream=True,timeout=(15,180)) as response:
        if response.status_code!=200:
            raise ValueError("public_cache_version_HTTP_"+str(response.status_code))
        with tempfile.TemporaryFile(mode="w+b") as store:
            for chunk in response.iter_content(1<<20):
                if chunk:store.write(chunk)
            store.seek(0)
            with zipfile.ZipFile(store) as archive:
                for name in FOUR:
                    names=[p for p in archive.namelist() if Path(p).name==name]
                    if len(names)!=1:raise ValueError("public_cache_missing_or_duplicate_"+name)
                    with archive.open(names[0]) as source:
                        data=source.read()
                    if len(data)>25_000_000:raise ValueError("public_cache_unexpectedly_large_"+name)
                    (target/name).write_bytes(data)
    return {k:hashlib.sha256((target/k).read_bytes()).hexdigest() for k in FOUR}

def load_chunk(directory:Path,model:str,valid_ids:set[str],n:int):
    zpath=directory/(model+"_scores.npz")
    ids=safe_pickle_object_string_ids(zpath,valid_ids,n)
    with np.load(zpath,allow_pickle=False) as npz:
        values={name:np.asarray(npz[name]) for name in ("scores","offsets","n_prefix","tokens")}
    if (len(ids)!=n or values["offsets"].shape!=(n+1,)
        or values["scores"].ndim!=2 or values["scores"].shape[1]!=4
        or values["n_prefix"].shape!=(n,)
        or len(values["tokens"])!=len(values["scores"])
        or values["offsets"][0]!=0 or values["offsets"][-1]!=len(values["scores"])
        or np.any(np.diff(values["offsets"])<0)):
        raise ValueError("bad_chunk_numeric_contract")
    frame=pd.read_csv(directory/(model+"_features.csv"),dtype={"id":str})
    if len(frame)!=n or set(frame["id"])!=set(ids):
        raise ValueError("feature_table_and_npz_ID_mismatch")
    return ids,values,frame

def merge(train:Path,root:Path,out:Path,token:str):
    labels=pd.read_csv(train,dtype={"id":str})
    official_ids=set(labels["id"])
    if len(labels)!=2699 or len(official_ids)!=2699:
        raise ValueError("official_train_cohort_changed")
    chunks={}
    digests={}
    for version,n in EXPECTED.items():
        folder=root/("version_"+str(version))
        digests[version]=recover(version,folder,token)
        chunks[version]={}
        for model in ("flash","glm53"):
            chunks[version][model]=load_chunk(folder,model,official_ids,n)
        # Both models must have exactly identical tokenization and ordering.
        a,pa,_=chunks[version]["glm53"]
        b,pb,_=chunks[version]["flash"]
        if not np.array_equal(a,b):
            raise ValueError("cross_model_ID_alignment_failed")
        for k in ("offsets","n_prefix","tokens"):
            if not np.array_equal(pa[k],pb[k]):
                raise ValueError("cross_model_token_alignment_failed_"+k)
    both=np.concatenate([chunks[v]["glm53"][0] for v in EXPECTED])
    if len(both)!=2699 or len(set(both.tolist()))!=2699 or set(both)!=official_ids:
        raise ValueError("merged_versions_overlap_or_incomplete")
    out.mkdir(parents=True,exist_ok=False)
    for model in ("flash","glm53"):
        entries=[chunks[v][model] for v in EXPECTED]
        arrays=[entry[1] for entry in entries]
        offsets=[0]
        for entry in arrays:
            shifted=entry["offsets"][1:]+offsets[-1]
            offsets.extend(shifted.tolist())
        merged={
            "ids":np.asarray(both,dtype=object),
            "scores":np.concatenate([a["scores"] for a in arrays]),
            "tokens":np.concatenate([a["tokens"] for a in arrays]),
            "offsets":np.asarray(offsets,dtype=np.int64),
            "n_prefix":np.concatenate([a["n_prefix"] for a in arrays])
        }
        np.savez_compressed(out/(model+"_scores.npz"),**merged)
        frame=pd.concat([entry[2] for entry in entries],ignore_index=True)
        frame.to_csv(out/(model+"_features.csv"),index=False)
    print("NTL_RECOVERED_FULL_PUBLIC_GLM_CACHE="+json.dumps({
        "versions":EXPECTED,"merged_rows":len(both),"four_files_per_version":True,
        "complete_official_ID_coverage":True,
        "token_alignment_verified":True,
        "version_source_sha256":digests
    },sort_keys=True))
    print("NO_COMPETITION_SUBMISSION_NO_MODEL_NO_RAW_TEXT_EXPORTED=TRUE")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",type=Path,required=True)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    merge(args.train,args.root,args.out,os.getenv("KAGGLE_API_TOKEN",""))
