#!/usr/bin/env python3
"""Audit Oleg's public 18-label corpus for genuine same-source, same-prompt siblings.
No competition data or outputs; only aggregate metadata is printed.
"""
import argparse, collections, hashlib, itertools, json
from pathlib import Path
import pandas as pd

PAIRS = {
  "gemini_36_38": ("gemini-3.6-flash", "gemini-3.8-flash"),
  "gemini_37_38": ("gemini-3.7-flash", "gemini-3.8-flash"),
  "gpt_luna_sol": ("gpt-5.6-luna", "gpt-5.6-sol"),
  "gpt_sol_terra": ("gpt-5.6-sol", "gpt-5.6-terra"),
  "glm": ("glm-5.3", "glm-5.3-flash"),
  "claude_opus_46_48": ("claude-opus-4-6", "claude-opus-4-8"),
  "claude_opus_48_5": ("claude-opus-4-8", "claude-opus-5"),
}

def require(path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    return path

def audit(root):
    root=Path(root)
    paths = {
      "wide": require(root/"external_short_wide.parquet"),
      "cells": require(root/"external_cell_metadata.parquet"),
      "prov": require(root/"external_cell_provenance.parquet"),
      "response": require(root/"external_response_metadata.parquet"),
    }
    file_shas={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in paths.items()}
    wide=pd.read_parquet(paths["wide"])
    cells=pd.read_parquet(paths["cells"])
    prov=pd.read_parquet(paths["prov"])
    response=pd.read_parquet(paths["response"])
    assert {"id","prompt"}.issubset(wide.columns)
    assert {"view","id","model","response_id"}.issubset(cells.columns)
    assert {"response_id","source","cell_id_short"}.issubset(prov.columns)
    if "prompt_exact" not in response.columns:raise ValueError("Missing prompt provenance")
    pcols=[x for x in set(wide.columns) - {"id","prompt"}]
    pairs_by_prompt={}
    for name,(a,b) in PAIRS.items():
        if a not in wide.columns or b not in wide.columns:
            pairs_by_prompt[name] = 0
            continue
        mask=wide[[a,b]].notna().all(axis=1)
        pairs_by_prompt[name]=int(mask.sum())
    filtered=cells[cells["view"].eq("short")].copy()
    for positive in ["mechanically_eligible","text_only_suitable"]:
        if positive in filtered.columns:
            filtered=filtered[filtered[positive].fillna(False).astype(bool)]
    for negative in ["no_train","possible_truncation","possible_reasoning_in_answer","text_shared_across_labels",
                     "license_unknown","output_terms_review"]:
        if negative in filtered.columns:
            filtered=filtered[~filtered[negative].fillna(True).astype(bool)]
    exact_resp=response[response["prompt_exact"].fillna(False).astype(bool)]
    filtered=filtered.merge(exact_resp[["response_id"]].drop_duplicates(),on="response_id",how="inner")
    pro=prov[prov["cell_id_short"].notna() & prov["source"].notna()]
    joined=filtered[["id","model","response_id"]].merge(
       pro[["response_id","source"]].drop_duplicates(),on="response_id",how="inner")
    # Provenance source membership itself is not verified as a shared generation protocol.
    same_source=collections.defaultdict(set)
    for row in joined.itertuples(index=False):
        same_source[(str(row.source),str(row.id))].add(str(row.model))
    source_pair_counts=collections.Counter()
    per_source=collections.Counter()
    # Precommitted diagnostic: conditional on the same source+prompt, count whether
    # one model habitually outputs more words. Actual length is NOT a requested
    # length and cannot prove identical generation parameters.
    length_signs=collections.defaultdict(lambda: collections.Counter())
    lengths=wide.set_index("id")
    lengths.index=lengths.index.astype(str)
    assert lengths.index.is_unique
    for (origin, row_id), models in same_source.items():
        for a,b in itertools.combinations(sorted(models),2):
            source_pair_counts[(a,b)] += 1
            per_source[(a,b,origin)] += 1
            if a in lengths.columns and b in lengths.columns:
                t1=lengths.at[row_id,a];t2=lengths.at[row_id,b]
                if isinstance(t1,str) and isinstance(t2,str):
                    w1=len(t1.split());w2=len(t2.split())
                    length_signs[(a,b)][("a_longer" if w1>w2 else
                                         "b_longer" if w2>w1 else "tied")] += 1
    result={
      "source":"olegzholobov/name-that-llm-response-corpus",
      "file_sha256":file_shas,
      "rows_wide":len(wide),
      "pair_prompt_overlap":pairs_by_prompt,
      "filtered_licensed_exact_prompt_cells":len(filtered),
      "same_source_prompt_keys":len(same_source),
      "paired_sources_by_class":{
        name: [{"source":source,"pair_prompt_keys":count}
               for (ma,mb,source),count in per_source.most_common()
               if (ma,mb) == tuple(sorted(pair))]
        for name,pair in PAIRS.items()
      },
      "paired_actual_word_length_sign":{
        name:dict(length_signs[tuple(sorted(pair))])
        for name,pair in PAIRS.items()
      },
      "same_source_pair_prompt_overlap":{name:source_pair_counts[tuple(sorted(pair))] for name,pair in PAIRS.items()},
      "metadata_columns":{k:sorted(set(x.columns)&{
        "prompt_exact","request","target_words","temperature","reasoning","version","served_model","provider","source",
        "model_evidence_class","sampling","system_prompt","generator","seed"
      }) for k,x in [("wide",wide),("cells",cells),("provenance",prov),("response",response)]},
      "protocol_control":"UNKNOWN_without_complete_generation_request_and_setting_evidence",
      "no_classifier_or_competition_data_used":True,
    }
    return result

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("root")
    args=ap.parse_args()
    print("PUBLIC_CORPUS_PAIR_RESULT",json.dumps(audit(args.root),sort_keys=True))
