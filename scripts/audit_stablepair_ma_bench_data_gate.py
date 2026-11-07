"""Metadata-only audit of a *public external* MA-Bench data source.

No official competition data, no model calls, no training, no user content.
Does not persist or print raw response text.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request

SOURCE_URL="https://huggingface.co/datasets/ScaleAI/MA-Bench/resolve/main/runs/briefs.jsonl?download=true"
MODEL_PAIR=("openai_gpt_5_6_luna","openai_gpt_5_6_sol")
CONDITIONS=("benchmark_capped","uncapped")
EXPECTED_KEYS={"brief_id","condition","case_id","scenario","model","model_slug","seed","text"}
RUN_MAX_BYTES=200_000_000
RUN_MAX_ROWS=3000
MIN_DISTINCT_MATCHED_PROMPTS=100
META_FIELDS=("served_model","endpoint_version","temperature","request_max_words",
             "system_prompt_hash","timestamp")

def metadata_audit(records, source_sha256, read_bytes):
    counts=Counter()
    prompts={c:defaultdict(set) for c in CONDITIONS}
    repeated={c:defaultdict(set) for c in CONDITIONS}
    cross={m:defaultdict(set) for m in MODEL_PAIR}
    lengths=defaultdict(list)
    keys_seen=set()
    total=0
    bad=0
    unique_ids=set()
    for row in records:
        total+=1
        if total>RUN_MAX_ROWS: raise RuntimeError("too many rows")
        if not isinstance(row,dict) or not EXPECTED_KEYS.issubset(row):
            raise RuntimeError("schema mismatch")
        if row["brief_id"] in unique_ids:raise RuntimeError("duplicate brief_id")
        unique_ids.add(row["brief_id"])
        model=str(row["model_slug"])
        cond=str(row["condition"])
        case=str(row["case_id"])
        counts[(cond,model)]+=1
        keys_seen.update(row)
        if model in MODEL_PAIR and cond in CONDITIONS:
            prompts[cond][model].add(case)
            repeated[cond][model].add((case,str(row["seed"])))
            cross[model][case].add(cond)
            lengths[(cond,model)].append(len(str(row["text"]).split()))
    pair_cases={c:prompts[c][MODEL_PAIR[0]]&prompts[c][MODEL_PAIR[1]] for c in CONDITIONS}
    cross_cases=pair_cases[CONDITIONS[0]]&pair_cases[CONDITIONS[1]]
    pair_case_seed={
        c:repeated[c][MODEL_PAIR[0]]&repeated[c][MODEL_PAIR[1]] for c in CONDITIONS}
    cross_case_seed=pair_case_seed[CONDITIONS[0]]&pair_case_seed[CONDITIONS[1]]
    # Source data *can* have global settings documented on the card; individual response
    # keys must not be mistaken for documented version/endpoint trace.
    explicit={key:key in keys_seen for key in META_FIELDS}
    def summaries(vals):
        if not vals:return {"n":0}
        a=sorted(vals)
        return {"n":len(a),"min":a[0],"median":a[len(a)//2],
                "max":a[-1]}
    result={
        "dataset":"ScaleAI/MA-Bench",
        "source_path":"runs/briefs.jsonl",
        "source_sha256":source_sha256,
        "bytes_read":read_bytes,
        "records":total,
        "distinct_brief_ids":len(unique_ids),
        "unique_scenarios_global":len({
            str(k) for c in CONDITIONS for model in MODEL_PAIR for k in prompts[c][model]}),
        "model_pair":list(MODEL_PAIR),
        "conditions":list(CONDITIONS),
        "matched_distinct_cases_by_condition":{c:len(pair_cases[c]) for c in CONDITIONS},
        "matched_distinct_cases_both_conditions":len(cross_cases),
        "matched_case_seed_by_condition":{c:len(pair_case_seed[c]) for c in CONDITIONS},
        "matched_case_seed_both_conditions":len(cross_case_seed),
        "model_condition_rows":{
          c:{m:counts[(c,m)] for m in MODEL_PAIR} for c in CONDITIONS},
        "word_lengths":{
          c:{m:summaries(lengths[(c,m)]) for m in MODEL_PAIR} for c in CONDITIONS},
        "explicit_per_response_provenance_fields":explicit,
        "minimum_required_distinct_prompts":MIN_DISTINCT_MATCHED_PROMPTS,
        "scientific_model_inference_performed":False,
        "kaggle_competition_data_used":False,
        "eligible_for_pilot": bool(
            len(cross_cases)>=MIN_DISTINCT_MATCHED_PROMPTS
            and all(explicit.values())),
        "gate":"PASS_FOR_PILOT_CONTRACT_ONLY" if (
            len(cross_cases)>=MIN_DISTINCT_MATCHED_PROMPTS and all(explicit.values()))
            else "NOT_ELIGIBLE_FOR_PAIRED_PROTOCOL_PILOT",
        "full_stablepair_source_gate":"NOT_MET_ONE_SOURCE_ONLY",
    }
    return result

def read_public_metadata(url=SOURCE_URL, *, max_bytes=RUN_MAX_BYTES):
    # URL passed only for unit tests using fixture mock source; command line does not
    # accept URL to avoid arbitrary remote content injection via workflow parameters.
    sha=hashlib.sha256()
    count=0
    def rows(response):
        nonlocal count
        for line in response:
            count+=len(line)
            if count>max_bytes:raise RuntimeError("Source size limit")
            sha.update(line)
            # Parse the line and discard response text immediately after numeric length
            data=json.loads(line)
            yield data
    with urllib.request.urlopen(url,timeout=80) as response:
        return metadata_audit(rows(response),sha.hexdigest(),count)
    
# NOTE: sha and count are evaluated before generator exhaustion, so return
# must be constructed after rows are consumed. Implement below in run() instead.

def run(source_url=SOURCE_URL):
    sha=hashlib.sha256()
    bytecount=0
    def stream_rows(response):
        nonlocal bytecount
        for raw in response:
            bytecount+=len(raw)
            if bytecount>RUN_MAX_BYTES: raise RuntimeError("Source exceeded 200MB cap")
            sha.update(raw)
            yield json.loads(raw)
    with urllib.request.urlopen(source_url,timeout=80) as response:
        # Source sha needs to be finalized before output; metadata_audit can safely
        # receive placeholders which are replaced after the stream is fully consumed.
        res=metadata_audit(stream_rows(response),"pending",0)
    res["source_sha256"]=sha.hexdigest()
    res["bytes_read"]=bytecount
    return res

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("ma_bench_stablepair_gate.json"))
    a=p.parse_args()
    try:
        result=run()
    except Exception as e:
        # No raw response printing (error exception may contain url but not text).
        result={"dataset":"ScaleAI/MA-Bench","gate":"INACCESSIBLE_OR_SCHEMA_ERROR",
                "error_type":type(e).__name__,
                "error_summary":str(e)[:300],
                "scientific_model_inference_performed":False}
        a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        print("STABLEPAIR_MA_GATE_RESULT "+json.dumps(result,sort_keys=True))
        raise
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("STABLEPAIR_MA_GATE_RESULT "+json.dumps(result,sort_keys=True))
