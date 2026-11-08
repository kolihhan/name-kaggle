"""One-off independent label/source eligibility for a newly published public pool.

This is a DATA provenance feasibility study. No model fitting, inference,
score, predictions, raw passages, IDs or per-row features are exported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import pandas as pd

MIN_SOURCE_ROWS = 300
MIN_LABELS = 8
MIN_PER_LABEL = 20
SALT = "ntl-next-coherent-architecture-unused-source-20261108-v1"

def normalized(s: object) -> str:
    s=unicodedata.normalize("NFKC",str(s)).lower()
    return re.sub(r"\s+"," ",s).strip()

def sha(s: object) -> str:
    return hashlib.sha256(normalized(s).encode("utf-8")).hexdigest()

def load_candidate(directory: Path):
    # Exact official filename of the public publisher's main text collection,
    # learned from first read-only manifest. Never silently mix in
    # gemini37_unverified.parquet, removed_rows.csv, or diagnostic tables.
    files=list(directory.rglob("ntl_open_pool.parquet"))
    if len(files)!=1:
        return None,{"status":"CANONICAL_MAIN_FILE_MISSING",
                     "canonical_file_count":len(files)}
    p=files[0]
    frame=pd.read_parquet(p)
    return frame,{"status":"LOADED","canonical_main_filename":p.name,
                  "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                  "bytes":p.stat().st_size}

def source_eligibility(new: pd.DataFrame, known_train: pd.DataFrame,
                       official: pd.DataFrame) -> dict:
    # Publisher schema, verified against immutable main-file metadata. Keep
    # source_dataset as the collection source, not the provider lab.
    if "source" not in new and "source_dataset" in new:
        new=new.rename(columns={"source_dataset":"source"})
    if "prompt_id" not in new and "prompt_group" in new:
        new=new.rename(columns={"prompt_group":"prompt_id"})
    if "prompt_key" not in new and "prompt_canon" in new:
        new=new.rename(columns={"prompt_canon":"prompt_key"})
    required={"text","model","source"}
    if not required.issubset(new.columns):
        return {"status":"INVALID_MISSING_COLUMNS",
                "missing":sorted(required-set(new.columns)),
                "columns":sorted(map(str,new.columns))}
    if new[list(required)].isna().any().any():
        return {"status":"INVALID_NULL_REQUIRED"}
    # Self-report provider or dataset name is NOT source provenance proof;
    # source IDs alone do not validate licenses or generation protocols.
    if not {"text","source","model"}.issubset(known_train):
        raise ValueError("Known train baseline lacks required columns")
    models=set(official.model.astype(str))
    official_hashes=set(official.text.map(sha))
    old_hashes=set(known_train.text.map(sha))
    old_sources=set(known_train.source.astype(str))
    old_keys=(set(known_train.prompt_key.astype(str))
              if "prompt_key" in known_train else set())
    official_prompts=set(official.prompt_id.astype(str))
    new=new.loc[new.model.astype(str).isin(models)].copy()
    new["origin_source"]=new.source.astype(str)
    new["text_hash"]=new.text.map(sha)
    new["overlap_text"]=new.text_hash.isin(official_hashes | old_hashes)
    risk_columns=["flag_near_train_prompt","flag_train_prompt_regen",
                  "flag_kb_proxy","flag_gemini37_routing_risk"]
    observed_flags=[col for col in risk_columns if col in new]
    # Missing risk flags may be unassessed; don't misrepresent them as safe.
    new["risky_upstream_flags"]=False
    for col in observed_flags:
        x=new[col]
        bad=x.isna() | x.astype(str).str.lower().isin(
            ["1","true","yes","y","risk","unverified"])
        new["risky_upstream_flags"] |= bad
    if "source_license" in new:
        l=new["source_license"]
        new["unknown_license"]=l.isna() | l.astype(str).str.lower().str.strip().isin(
            ["","unknown","none","nan","null","unverified","no license"])
        license_counts={str(k)[:75]:int(v) for k,v in l.astype(str).value_counts().items()}
    else:
        new["unknown_license"]=True
        license_counts={"ABSENT":int(len(new))}
    new["overlap_prompt"]=(new.prompt_id.astype(str).isin(official_prompts)
                          if "prompt_id" in new else False)
    if "prompt_key" in new:
        new["overlap_prompt"] |= new.prompt_key.astype(str).isin(old_keys | official_prompts)
    # Crucial: even if one specific text differs, the whole previously-used
    # source is NOT an independent held-out source.
    output={"status":"SCANNED",
      "rows":int(len(new)),
      "label_count":len(models & set(new.model.astype(str))),
      "origin_source_count":int(new.origin_source.nunique()),
      "already_trained_source_count":int(new.origin_source.isin(old_sources).groupby(new.origin_source).max().sum()),
      "new_pool_schema_columns":sorted(str(x) for x in new.columns if x not in ("text","overlap_text","overlap_prompt")),
      "source_record_fields_present":{col:col in new for col in ["prompt_id","prompt_key","source_file","origin_hint","source_license","generation_model","served_model","model_version","temperature","created_at"]},
      "license_values":license_counts,
      "upstream_risk_flags_checked":observed_flags,
      "upstream_flagged_rows":int(new.risky_upstream_flags.sum()),
      "unknown_license_rows":int(new.unknown_license.sum()),
      "source_candidates":[]}
    for source,g in new.groupby("origin_source",sort=True):
        if source in old_sources:continue
        clean=g.loc[~g.overlap_text & ~g.overlap_prompt &
                    ~g.risky_upstream_flags & ~g.unknown_license].copy()
        # No duplicate normalized text or stable prompt reused from known fit.
        clean=clean.drop_duplicates(subset=["text_hash"],keep="first")
        supports=clean.model.astype(str).value_counts()
        supported=supports[supports>=MIN_PER_LABEL]
        record={"source":str(source)[:100],
          "original_rows":int(len(g)),
          "eligible_no_overlaps":int(len(clean)),
          "eligible_labels_with_20":int(len(supported)),
          "max_label_rows":int(supports.max()) if len(supports) else 0,
          "distinct_prompt_ids":int(clean.prompt_id.astype(str).nunique()) if "prompt_id" in clean else None,
          "exact_train_or_official_text_overlap_count":int(g.overlap_text.sum()),
          "prompt_overlap_count":int(g.overlap_prompt.sum()),
          "publisher_flagged_rows":int(g.risky_upstream_flags.sum()),
          "unknown_license_rows":int(g.unknown_license.sum()),
          "name_not_in_own_stage1_sources":True,
          "status":"CAPACITY_ELIGIBLE_NOT_PROVEN_INDEPENDENT" if (
             len(clean)>=MIN_SOURCE_ROWS and len(supported)>=MIN_LABELS)
             else "INSUFFICIENT_COVERAGE"}
        output["source_candidates"].append(record)
    good=[d for d in output["source_candidates"] if d["status"]=="CAPACITY_ELIGIBLE_NOT_PROVEN_INDEPENDENT"]
    output["count_capacity_eligible"]=len(good)
    output["selected_by_frozen_source_hash"]=(min(good,key=lambda d:hashlib.sha256(
         (SALT+"|"+d["source"]).encode()).hexdigest())["source"] if good else None)
    output["source_integrity_verdict"]="UNPROVEN"  # publisher's license field is a claim, not independently verified original terms
    output["independent_scientific_transfer_verified"]=False
    return output

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--new-pool-dir",type=Path,required=True)
    ap.add_argument("--known-pool",type=Path,required=True)
    ap.add_argument("--official",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    candidate,manifest=load_candidate(args.new_pool_dir)
    if candidate is None:
        result={"dataset":"crtproduction/name-that-llm-open-pool",
                "manifest":manifest,"status":"BLOCKED","no_scientific_fit_performed":True,
                "no_competition_submission":True,"independent_scientific_transfer_verified":False}
    else:
        from scripts.run_phase14_public_sparse import build_conservative_subset
        from scripts.run_two_stage_lora24_gate_a import build_stage1_split
        old=build_conservative_subset(pd.read_parquet(args.known_pool))
        split=build_stage1_split(old)
        old_fit=old.iloc[split["train_idx"]].copy()
        official=pd.read_csv(args.official)
        audit=source_eligibility(candidate,old_fit,official)
        result={"dataset":"crtproduction/name-that-llm-open-pool",
                "manifest":manifest,
                "known_fit_rows":len(old_fit),
                "known_fit_sources":int(old_fit.source.nunique()),
                "no_scientific_fit_performed":True,
                "no_competition_submission":True,**audit}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("ONE_SOURCE_FEASIBILITY_RESULT",json.dumps({
      k:result.get(k) for k in ("status","rows","label_count","origin_source_count",
      "count_capacity_eligible","selected_by_frozen_source_hash","source_integrity_verdict")
    },sort_keys=True))

if __name__=="__main__":
    main()
