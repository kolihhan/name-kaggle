"""CPU only, original published NTL generator-route metadata integrity check.

Read public upstream generations.parquet, but load ONLY generator metadata
columns (never generated text or original prompts). Do not infer provider
authentication or training rights from publisher-reported returned_model.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
from collections import Counter
import pandas as pd
import pyarrow.parquet as pq

def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()

def clean(v):
    if v is None:return ""
    s=str(v).strip()
    return "" if s.lower() in ("","nan","null","none") else s

def protocol_flags(setting,settings_json):
    text=(clean(setting)+" "+clean(settings_json)).lower()
    if re.search(r"""enable_thinking["'\s:=]+(?:false|0)""",text):
        thinking="explicit_off"
    elif re.search(r"""enable_thinking["'\s:=]+(?:true|1)""",text):
        thinking="explicit_on"
    elif "thinking" in text:thinking="mentioned_unknown"
    else:thinking="not_stated"
    return thinking

def audit(p:Path):
    meta=pq.ParquetFile(p)
    cols=meta.schema_arrow.names
    required={"label","provider","requested_model","returned_model"}
    if not required.issubset(cols):raise RuntimeError("ORIGINAL_PROVENANCE_COLUMNS_MISSING:"+",".join(sorted(required-set(cols))))
    wanted=[x for x in ("label","provider","requested_model","returned_model",
                       "setting","settings","run","pid","target_words") if x in cols]
    frame=meta.read(columns=wanted).to_pandas()
    if len(frame)<4000 or frame.label.nunique()!=13:
        raise RuntimeError(f"UNEXPECTED_UPSTREAM_ROWS_OR_13_CLASSES_{len(frame)}_{frame.label.nunique()}")
    frame=frame.fillna("")
    report={}
    for label,g in frame.groupby("label",sort=True):
        req=g.requested_model.map(clean)
        ret=g.returned_model.map(clean)
        same=(req==ret)&req.ne("")
        rec={
          "rows":int(len(g)),
          "original_requested_nonempty":int(req.ne("").sum()),
          "returned_nonempty":int(ret.ne("").sum()),
          "requested_returned_exact_equal_rows":int(same.sum()),
          "requested_returned_mismatch_or_missing":int((~same).sum()),
          "publisher_route_names":dict(sorted(Counter(g.provider.astype(str)).items())),
        }
        if "pid" in g.columns:rec["distinct_original_prompt_ids"]=int(g.pid.astype(str).nunique())
        if "run" in g.columns:rec["generator_runs"]=dict(sorted(Counter(g.run.astype(str)).items()))
        if "setting" in g.columns:
            ss=g.setting.map(clean)
            rec["setting_nonempty"]=int(ss.ne("").sum())
            rec["setting_distinct"]=int(ss.nunique())
        rec["thinking_protocol_flags"]=dict(sorted(Counter(
           protocol_flags(row.get("setting",""),row.get("settings",""))
           for row in g.to_dict("records")).items()))
        report[str(label)]=rec
    total_equal=sum(x["requested_returned_exact_equal_rows"] for x in report.values())
    return {
      "audit":"upstream_4270_generator_metadata_real_rows_readonly_20261120",
      "upstream":"lxdlxdlxd/ntl-synth-generations-v1",
      "original_parquet_sha256":digest(p),
      "original_parquet_rows":len(frame),
      "original_13_labels":len(report),
      "nonempty_identical_requested_returned_rows":total_equal,
      "not_exact_or_missing_count":len(frame)-total_equal,
      "family_details":report,
      "publisher_claim_is_not_provider_authenticated":True,
      "upstream_CC_BY_does_not_override_provider_output_terms":True,
      "no_generated_text_or_prompt_loaded":True,
      "no_generated_text_or_prompt_exported":True,
      "no_training_or_inference_or_scored_submission":True,
    }

def smoke():
    assert protocol_flags("temperature=0.3",'{"enable_thinking":false}')=="explicit_off"
    assert protocol_flags("on",'{"enable_thinking":true}')=="explicit_on"
    assert protocol_flags("default","")=="not_stated"
    assert clean(None)=="" and clean("null")==""
    print("NTL_ORIGINAL_GENERATION_METADATA_UNIT=PASS",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--parquet",type=Path)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:smoke()
    else:
        if not a.parquet:ap.error("--parquet needed")
        print("NTL_ORIGINAL_GENERATOR_ROUTE_AUDIT="+json.dumps(audit(a.parquet),sort_keys=True),flush=True)
