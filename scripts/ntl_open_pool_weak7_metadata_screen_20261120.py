"""Public Kaggle Open Pool metadata-only risk census for 7 Phase15 missing labels.

No model training, no official TRAIN text, no Kaggle submission.
Source-reported license never proves downstream API-output permissions.
A row with provenance metadata is only a candidate for manual original-source
authorization, not independently verified lawful training data.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow.parquet as pq

WEAK_LABELS=(
    "gemini-3.1-pro-preview","gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash",
    "glm-5.3","glm-5.3-flash","kimi-k3",
)
FLAGS=("flag_kb_proxy","flag_gemini37_routing_risk","flag_near_train_prompt",
       "flag_train_prompt_regen","flag_kb_proxy_after_ruling")
NEED=("model","source_dataset","source_license","gen_route","gen_served_model",
      "gen_date","match_weight","max_sim_train_prompt",*FLAGS)
def truthy(s):
    # False/None/NaN are not counted as positive flags.
    return str(s).strip().lower() in ("true","1","yes","y")
def nonempty(x):
    return not (pd.isna(x) or str(x).strip().lower() in ("","none","nan","null","unknown","n/a"))
def roundrate(x):
    return round(float(np.mean(x)),5) if len(x) else None
def count_top(d,col,n=6):
    return dict(Counter(str(x) for x in d[col].fillna("<missing>")).most_common(n))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pool",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    f=pq.ParquetFile(a.pool)
    keys=set(f.schema_arrow.names)
    if not {"model","source_dataset","source_license","gen_served_model","match_weight"}.issubset(keys):
        raise RuntimeError("CRITICAL_METADATA_FIELDS_MISSING")
    names=[c for c in NEED if c in keys]
    data=pq.read_table(a.pool,columns=names).to_pandas()
    if int(f.metadata.num_rows)!=len(data):raise RuntimeError("PARQUET_ROW_MISMATCH")
    sha=hashlib.file_digest(a.pool.open("rb"),"sha256").hexdigest()
    labeled={k:len(x) for k,x in data.groupby("model")}
    if set(WEAK_LABELS)-set(labeled):raise RuntimeError("WEAK_LABEL_MISSING")
    report={}
    for model in WEAK_LABELS:
        d=data.loc[data.model.astype(str).eq(model)].copy()
        flags={flag:d[flag].map(truthy) if flag in d else pd.Series(False,index=d.index)
               for flag in FLAGS}
        any_flag=np.logical_or.reduce([x.to_numpy(dtype=bool) for x in flags.values()])
        license_avail=d.source_license.map(nonempty).to_numpy()
        served_avail=d.gen_served_model.map(nonempty).to_numpy()
        route_avail=d.gen_route.map(nonempty).to_numpy() if "gen_route" in d else np.zeros(len(d),dtype=bool)
        date_avail=d.gen_date.map(nonempty).to_numpy() if "gen_date" in d else np.zeros(len(d),dtype=bool)
        # this has NO automatic legal admission criterion, just a metadata screen
        observed_candidate=(~any_flag)&license_avail&served_avail
        weights=pd.to_numeric(d.match_weight,errors="coerce")
        validweights=weights.dropna()
        src_stats=[]
        for source,g in d.groupby("source_dataset",dropna=False):
            src_stats.append({
                "source":str(source),
                "rows":len(g),
                "metadata_source_license_values":dict(Counter(g.source_license.fillna("<missing>").astype(str)).most_common(3)),
                "rows_with_served_model_field":int(g.gen_served_model.map(nonempty).sum()),
            })
        src_stats.sort(key=lambda x:-x["rows"])
        report[model]={
            "rows":len(d),
            "source_names":int(d.source_dataset.nunique()),
            "publisher_license_metadata_present":int(license_avail.sum()),
            "generation_served_model_metadata_present":int(served_avail.sum()),
            "generation_route_metadata_present":int(route_avail.sum()),
            "generation_date_metadata_present":int(date_avail.sum()),
            "flags_positive":{flag:int(mask.sum()) for flag,mask in flags.items()},
            "any_flag_positive":int(any_flag.sum()),
            "preliminary_unflagged_with_license_and_served_model_fields":int(observed_candidate.sum()),
            "match_weight_nonnull":int(validweights.size),
            "match_weight_median":round(float(validweights.median()),5) if len(validweights) else None,
            "match_weight_p90":round(float(validweights.quantile(.9)),5) if len(validweights) else None,
            "reported_source_license_top":count_top(d,"source_license",6),
            "top5_public_source_names":src_stats[:5],
        }
    result={
        "study":"NTL_OPEN_POOL_WEAK7_CURRENT_METADATA_ONLY_NOT_LEGAL_CLEARANCE",
        "public_dataset":"crtproduction/name-that-llm-open-pool",
        "source_sha256":sha,
        "current_file_rows":len(data),
        "current_source_names":int(data.source_dataset.nunique()),
        "current_labels":int(data.model.nunique()),
        "weak7":report,
        "not_checked":"Original API output rights, publisher source provenance, verified served model identity, true host prompt independence; cannot infer legal admission from license metadata",
        "all_candidate_counts_are_METADATA_ONLY_NOT_AUTHORIZED":True,
        "no_training":True,"no_gpu":True,"no_competition_submission":True,
        "no_text_exported":True,
    }
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NTL_OPEN_POOL_WEAK7_METADATA_SUMMARY="+json.dumps({
        "current_rows":len(data),"current_labels":int(data.model.nunique()),
        "weak7_summary":{key:{
            "rows":val["rows"],"sources":val["source_names"],
            "served_model_meta":val["generation_served_model_metadata_present"],
            "license_meta":val["publisher_license_metadata_present"],
            "flagged_any":val["any_flag_positive"],
            "metadata_complete_unflagged_candidate":val["preliminary_unflagged_with_license_and_served_model_fields"],
            "reported_license_values":val["reported_source_license_top"],
        } for key,val in report.items()}
    },sort_keys=True),flush=True)
if __name__=="__main__":main()
