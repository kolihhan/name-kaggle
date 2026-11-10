#!/usr/bin/env python3
"""Single source-only, no-fit MPCF pilot. Prints aggregate results, never text."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from scripts.matched_panel_evidence import matched_pair_evidence

PAIRS = (
    ("gpt_luna_sol", "gpt-5.6-luna", "gpt-5.6-sol"),
    ("gpt_sol_terra", "gpt-5.6-sol", "gpt-5.6-terra"),
    ("glm", "glm-5.3", "glm-5.3-flash"),
)
SOURCE = "zlab-princeton/SWEeper-Bench-traces"

def extract_pairs(wide, cells, provenance, response, *, model_a, model_b,
                  source, min_words=40, max_words=250):
    """Enforce exact-prompt same-source, audited legally usable SHORT cells."""
    if not {"id", "prompt", model_a, model_b}.issubset(wide.columns):
        raise ValueError("wide corpus missing prompt/model columns")
    if not {"view","id","model","response_id"}.issubset(cells.columns):
        raise ValueError("cell metadata missing audit keys")
    if not {"response_id","source","cell_id_short"}.issubset(provenance.columns):
        raise ValueError("provenance missing audit keys")
    if not {"response_id","prompt_exact"}.issubset(response.columns):
        raise ValueError("response metadata missing exact-prompt evidence")
    w=wide.copy()
    w["id"]=w["id"].astype(str)
    if w["id"].duplicated().any():
        raise ValueError("wide corpus has duplicate id")
    eligible=cells[cells["view"].eq("short")].copy()
    for flag in ("mechanically_eligible","text_only_suitable"):
        if flag in eligible:
            eligible=eligible[eligible[flag].fillna(False).astype(bool)]
    for flag in ("no_train","possible_truncation","possible_reasoning_in_answer",
                 "text_shared_across_labels","license_unknown","output_terms_review"):
        if flag in eligible:
            eligible=eligible[~eligible[flag].fillna(True).astype(bool)]
    exact=response.loc[response["prompt_exact"].fillna(False).astype(bool),
                       ["response_id"]].drop_duplicates()
    eligible=eligible.merge(exact,on="response_id",how="inner")
    source_rows=provenance.loc[
        provenance["cell_id_short"].notna() & provenance["source"].eq(source),
        ["response_id","source"]].drop_duplicates()
    eligible=eligible.merge(source_rows,on="response_id",how="inner")
    eligible["id"]=eligible["id"].astype(str)
    allowed=set(zip(eligible["id"],eligible["model"]))
    rows=[]
    for key,prompt,a,b in w[["id","prompt",model_a,model_b]].itertuples(
            index=False,name=None):
        if (str(key),model_a) not in allowed or (str(key),model_b) not in allowed:
            continue
        if not isinstance(prompt,str) or not prompt.strip():
            continue
        if not isinstance(a,str) or not isinstance(b,str) or not a.strip() or not b.strip():
            continue
        if not (min_words<=len(a.split())<=max_words and
                min_words<=len(b.split())<=max_words):
            continue
        if a.strip()==b.strip():
            continue
        rows.append({"prompt":prompt,"a":a,"b":b})
    return rows

def prompt_key(prompt):
    return " ".join(prompt.casefold().split())

def split_by_prompt(pairs, train_fraction=.7):
    if not 0<train_fraction<1:
        raise ValueError("train_fraction outside (0,1)")
    keys={prompt_key(p["prompt"]) for p in pairs}
    if len(keys)<2:
        raise ValueError("at least 2 distinct prompts required")
    ranked=sorted(keys,key=lambda k:(hashlib.sha256(k.encode()).hexdigest(),k))
    n=min(len(keys)-1,max(1,int(len(keys)*train_fraction)))
    kept=set(ranked[:n])
    return ([p for p in pairs if prompt_key(p["prompt"]) in kept],
            [p for p in pairs if prompt_key(p["prompt"]) not in kept])

def _ba(scores_a,scores_b):
    a=np.asarray(scores_a,dtype=float)
    b=np.asarray(scores_b,dtype=float)
    return float(.5*((np.mean(a>0)+.5*np.mean(a==0))+
                      (np.mean(b<0)+.5*np.mean(b==0))))

def _unit(x):
    n=np.linalg.norm(x)
    return x/n if n>0 else x

def evaluate_pair(rows, *, seed=20261110):
    ref,held=split_by_prompt(rows,train_fraction=.7)
    # Exclude exact answer reoccurrence across reference vs heldout.
    original={prompt_key(p[k]) for p in ref for k in ("a","b")}
    held=[p for p in held if not any(prompt_key(p[k]) in original
                                    for k in ("a","b"))]
    if len(ref)<16 or len(held)<10:
        return {"status":"INSUFFICIENT_REFERENCE_OR_HELDOUT",
                "reference_prompts":len({prompt_key(x["prompt"]) for x in ref}),
                "heldout_prompts":len({prompt_key(x["prompt"]) for x in held})}
    # HashingVectorizer is a deterministic transformation: NO .fit or weights.
    vec=HashingVectorizer(analyzer="char",ngram_range=(3,5),n_features=4096,
                          alternate_sign=False,norm="l2",lowercase=True,
                          dtype=np.float64)
    embedded=vec.transform([p[k] for p in ref+held for k in ("a","b")]).toarray()
    n=len(ref)
    a=embedded[:2*n:2]
    b=embedded[1:2*n:2]
    ah=embedded[2*n::2]
    bh=embedded[2*n+1::2]
    top_k=16
    def score(left,right,queries):
        return [matched_pair_evidence(q,left,right,top_k=top_k).margin
                for q in queries]
    raw_a=score(a,b,ah)
    raw_b=score(a,b,bh)
    swap=np.random.default_rng(seed).permutation(n)
    shuffled_a=score(a,b[swap],ah)
    shuffled_b=score(a,b[swap],bh)
    proto=_unit(a.mean(axis=0))-_unit(b.mean(axis=0))
    proto_a=ah@proto
    proto_b=bh@proto
    median_a=np.median([len(p["a"].split()) for p in ref])
    median_b=np.median([len(p["b"].split()) for p in ref])
    def length_scores(k):
        return [abs(len(p[k].split())-median_b)-abs(len(p[k].split())-median_a)
                for p in held]
    matched=_ba(raw_a,raw_b)
    shuffled=_ba(shuffled_a,shuffled_b)
    prototype=_ba(proto_a,proto_b)
    length=_ba(length_scores("a"),length_scores("b"))
    return {
        "status":"DIAGNOSTIC_ONLY_SINGLE_SOURCE_NO_EXTERNAL_TRANSFER",
        "reference_prompts":len({prompt_key(p["prompt"]) for p in ref}),
        "heldout_prompts":len({prompt_key(p["prompt"]) for p in held}),
        "reference_rows":len(ref),
        "heldout_pair_rows":len(held),
        "hash_dimensions":4096,"top_k":16,"shuffle_seed":seed,
        "balanced_accuracy_binary":{
            "matched_pair":matched,
            "prompt_shuffled_pair":shuffled,
            "unpaired_class_prototype":prototype,
            "length_only_median":length,
        },
        "paired_contrast_vs_shuffle":matched-shuffled,
        "paired_contrast_vs_prototype":matched-prototype,
        "local_feasibility_pass":(
            matched>=.55 and matched-shuffled>=.05
            and matched-prototype>=.05 and matched>length),
    }

def main(root):
    root=Path(root)
    names=("external_short_wide.parquet","external_cell_metadata.parquet",
           "external_cell_provenance.parquet","external_response_metadata.parquet")
    paths=[root/name for name in names]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("required public Oleg files missing")
    shas={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    wide,cells,provenance,response=[pd.read_parquet(p) for p in paths]
    report={
        "source":"olegzholobov/name-that-llm-response-corpus",
        "specific_origin":SOURCE,"files_sha256":shas,
        "method":"MPCF matched vs shuffled vs prototype vs length",
        "no_training":True,"new_kaggle_submission":False,
        "independent_source_transfer":False,
        "official_18_label_BA":False,"pairs":{},
    }
    for name,a,b in PAIRS:
        candidate=extract_pairs(wide,cells,provenance,response,model_a=a,
                                model_b=b,source=SOURCE,min_words=40,max_words=250)
        # First occurrence only, no sorting by expected outcome or label.
        seen=set()
        unique=[]
        for item in candidate:
            key=prompt_key(item["prompt"])
            if key not in seen:
                unique.append(item)
                seen.add(key)
        stats={"eligible_rows":len(candidate),"unique_prompt_pairs":len(unique)}
        stats.update(evaluate_pair(unique) if len(unique)>=30 else
                     {"status":"INSUFFICIENT_QUALIFIED_PAIRS"})
        report["pairs"][name]=stats
        print("MPCF_PAIR_RESULT",name,json.dumps(stats,sort_keys=True),flush=True)
    print("MPCF_DIAGNOSTIC_RESULT",json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root")
    main(p.parse_args().root)
