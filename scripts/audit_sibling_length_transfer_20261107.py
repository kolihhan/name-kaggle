#!/usr/bin/env python3
"""Frozen no-training external transfer test of an observed length-ordering shortcut.

Original (already observed): SWEeper Oleg short text Sol > Luna in 195/196, 
Sol > Terra 186/199. Test these two signs, without fitting/tuning, against
the independent public proxy-generation corpus. No official competition data.
Only aggregate counts are printed: no text, prompt, request, or row-level output.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256="a8f61b647b5f2b2f99b73f083b7358445d93c3ebbff8607b44423850e9fb5583"
PAIRS={
    "Sol_over_Luna":("gpt-5.6-sol","gpt-5.6-luna"),
    "Sol_over_Terra":("gpt-5.6-sol","gpt-5.6-terra"),
}
VALID_SETTINGS={"rlow_t1","rlow_t0","default","t1","t1_rlow"}

def sign(a,b):
    return (a>b)-(a<b)

def audit(csv_path):
    p=Path(csv_path)
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA256:
        raise ValueError("External corpus snapshot drift. Refuse unregistered source")
    with p.open(encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        req={"bank","prompt_id","model","setting","target_words","n_words","run","text"}
        if not req.issubset(reader.fieldnames or []):
            raise ValueError("Missing columns: "+str(sorted(req-set(reader.fieldnames or []))))
        rows=list(reader)
    if len(rows)!=834:
        raise ValueError("Source row count mismatch")
    buckets=collections.defaultdict(list)
    for row in rows:
        key=(row["bank"],row["prompt_id"],row["setting"],row["model"])
        buckets[key].append(row)
    out={"source":"zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations",
         "sha256":sha,"rows":len(rows),
         "frozen_rule":"SWEeper observation: Sol answer longer than Luna and Terra",
         "threshold":"at least 20 disjoint prompt pairs for descriptive cross-source check; not causal attribution",
         "results":{}}
    for name,(higher,lower) in PAIRS.items():
        strata=collections.defaultdict(lambda:collections.Counter())
        distinct_pair_prompts=set()
        for (bank,prompt,setting,model),_ in buckets.items():
            if model!=higher or setting not in VALID_SETTINGS:continue
            k1=(bank,prompt,setting,higher);k2=(bank,prompt,setting,lower)
            if k2 not in buckets: continue
            if len(buckets[k1])!=1 or len(buckets[k2])!=1:
                strata[setting]["excluded_duplicate_model_prompt_setting"]+=1
                continue
            a,b=buckets[k1][0],buckets[k2][0]
            count=strata[setting]
            try:
                actual_a=len(a["text"].split());actual_b=len(b["text"].split())
                target_a=int(a["target_words"]);target_b=int(b["target_words"])
                stored_a=int(a["n_words"]);stored_b=int(b["n_words"])
            except ValueError:
                count["excluded_invalid_word_metadata"]+=1
                continue
            count["paired_prompt_keys"]+=1
            distinct_pair_prompts.add((bank,prompt))
            if stored_a!=actual_a or stored_b!=actual_b:count["stored_word_count_mismatch"]+=1
            out_sign=sign(actual_a,actual_b)
            target_sign=sign(target_a,target_b)
            count[("sol_longer" if out_sign>0 else "sol_shorter" if out_sign<0 else "actual_tie")]+=1
            count[("sol_target_longer" if target_sign>0 else "sol_target_shorter" if target_sign<0 else "target_tie")]+=1
            if target_sign==0:
                count["equal_target_count"]+=1
                count[("equal_target_sol_longer" if out_sign>0 else "equal_target_sol_shorter" if out_sign<0 else "equal_target_actual_tie")]+=1
            if out_sign!=0 and target_sign!=0:
                count["both_non_tied"]+=1
                count["actual_matches_target_direction"]+=int(out_sign==target_sign)
        out["results"][name]={
            "strata":{k:dict(sorted(v.items())) for k,v in sorted(strata.items())},
            "total_paired_prompt_keys":sum(v["paired_prompt_keys"] for v in strata.values()),
            "distinct_bank_prompt_keys":len(distinct_pair_prompts),
            "prompt_keys_repeated_across_settings":sum(v["paired_prompt_keys"] for v in strata.values())-len(distinct_pair_prompts),
            "primary_rlow_t1":dict(sorted(strata["rlow_t1"].items())),
            "total_sol_longer":sum(v["sol_longer"] for v in strata.values()),
            "total_sol_shorter":sum(v["sol_shorter"] for v in strata.values()),
            "total_actual_ties":sum(v["actual_tie"] for v in strata.values()),
            "total_equal_target":sum(v["equal_target_count"] for v in strata.values()),
            "total_both_non_tied":sum(v["both_non_tied"] for v in strata.values()),
            "total_actual_matches_target_direction":sum(v["actual_matches_target_direction"] for v in strata.values()),
        }
        item=out["results"][name]
        item["passes_sample_20_threshold"]=item["distinct_bank_prompt_keys"]>=20
    out["scientific_limits"]=[
        "Fixed paired output ordering is not actionable on hidden unpaired test passages.",
        "Target lengths may be class-conditioned; source transfer cannot prove model-intrinsic identity.",
        "All passages are external; no official Kaggle test or training data used.",
        "No model fitted, checkpoint loaded, GPU used or submission made.",
    ]
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("csv",type=Path)
    args=ap.parse_args()
    print("LENGTH_RULE_TRANSFER_RESULT",json.dumps(audit(args.csv),sort_keys=True))
