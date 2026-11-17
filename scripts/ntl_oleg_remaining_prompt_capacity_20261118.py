"""Read-only, frozen-cohort-excluded Oleg corpus capacity audit.

No inference or fitting, and the source remains the same Oleg publisher;
finding unused prompts does NOT establish source/protocol independence.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.ntl_oleg18_frozen_selector_20261116 import (
  NTL_OLEG18_SHA,NTL_OLEG18_MODELS,NTL_OLEG18_STRONG_EVIDENCE,ntl_oleg18_select
)
SELECTION_SALT="ntl-unused-oleg18-capacity-20261118-nontuning"

def audit(root):
  for name,h in NTL_OLEG18_SHA.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
  wide=pd.read_parquet(root/"external_short_wide.parquet")
  meta=pd.read_parquet(root/"external_cell_metadata.parquet")
  used=ntl_oleg18_select(wide,meta)
  assert len(used)==216 and used.prompt_group_id.nunique()==216
  allowed=meta.loc[
    meta.view.eq("short")&
    meta.short_answer_candidate.eq(True)&
    meta.mechanically_eligible.eq(True)&
    meta.text_only_suitable.eq(True)&
    ~meta.text_shared_across_labels.eq(True)&
    ~meta.license_unknown.eq(True)&
    meta.model_evidence_class.isin(NTL_OLEG18_STRONG_EVIDENCE),
    ["id","model","prompt_group_id"]
  ]
  long=wide.melt(id_vars=["id"],value_vars=list(NTL_OLEG18_MODELS),
                 var_name="model",value_name="text").dropna(subset=["text"])
  joined=long.merge(allowed,on=["id","model"],how="inner",validate="one_to_one")
  joined=joined.loc[joined.text.map(
    lambda x:isinstance(x,str) and 40<=len(x.split())<=250)].copy()
  joined["prompt_group_id"]=joined.prompt_group_id.astype(str)
  joined=joined.loc[joined.prompt_group_id.str.len().gt(0)&joined.prompt_group_id.ne("nan")]
  used_group=set(used.prompt_group_id)
  used_hash={hashlib.sha256(str(t).strip().encode()).hexdigest() for t in used.text}
  unused=joined.loc[~joined.prompt_group_id.isin(used_group)].copy()
  unused=unused.loc[~unused.text.map(lambda x:hashlib.sha256(str(x).strip().encode()).hexdigest() in used_hash)]
  counts=unused.groupby("model").agg(
     rows=("id","size"),groups=("prompt_group_id","nunique")
  )
  labels=list(NTL_OLEG18_MODELS)
  per={l:{"remaining_qualified_rows":int(counts.loc[l,"rows"]) if l in counts.index else 0,
           "distinct_prompt_groups":int(counts.loc[l,"groups"]) if l in counts.index else 0}
       for l in labels}
  usable=set()
  # Conservative sequential greedy whole-pool prompt-group exclusion,
  # deterministic selection independent of model predictions or labels' score.
  max_available={}
  for per_class in (12,18,30):
      local_taken=set()
      reached={}
      for lab in labels:
          g=unused.loc[unused.model.eq(lab)].copy()
          g["stable_hash"]=[
            hashlib.sha256((SELECTION_SALT+"|"+str(x)).encode()).hexdigest()
            for x in g.id]
          g=g.sort_values("stable_hash",kind="stable")
          count=0
          for row in g.itertuples(index=False):
            if str(row.prompt_group_id) in local_taken:continue
            local_taken.add(str(row.prompt_group_id))
            count+=1
            if count==per_class:break
          reached[lab]=count
      max_available[str(per_class)]={
          "full_18_label_panel_feasible":all(n==per_class for n in reached.values()),
          "worst_class_selected":min(reached.values()),
          "total_prompt_groups_selected":len(local_taken),
          "per_class_frozen_selection_count":reached,
      }
  # Max-flow is a necessary-and-sufficient test for distinct prompt groups
  # per true label, ignoring further unique-response-text conflicts (an upper
  # bound on fully admissible rows). Greedy label order is not proof of
  # infeasibility, so report this exact matching capacity separately.
  import networkx as nx
  edges=unused[["model","prompt_group_id"]].drop_duplicates()
  exact={}
  for per_class in (12,18,30):
      g=nx.DiGraph()
      source,sink="__SRC__","__SINK__"
      for label in labels:g.add_edge(source,"L:"+label,capacity=per_class)
      for lab,pid in edges.itertuples(index=False,name=None):
          g.add_edge("L:"+str(lab),"G:"+str(pid),capacity=1)
      for pid in edges.prompt_group_id.unique():
          g.add_edge("G:"+str(pid),sink,capacity=1)
      value=nx.maximum_flow_value(g,source,sink)
      exact[str(per_class)]={
         "maxflow_distinct_prompt_assignments":int(value),
         "requested":int(per_class*18),
         "full_18_labels_feasible_ignoring_response_duplication":bool(value==per_class*18),
         "upper_bound_only":True,
      }
  out={
    "used_old_cohort_rows":216,"old_prompt_groups":216,
    "source_name":"olegzholobov/name-that-llm-response-corpus",
    "same_source_as_old":True,"new_inference":False,"competition_submit":False,
    "no_raw_text_or_prompt_export":True,
    "qualified_unused_rows":len(unused),
    "qualified_unused_unique_groups":int(unused.prompt_group_id.nunique()),
    "model_count_with_eligible_responses":int(unused.model.nunique()),
    "per_class_capacity":per,
    "frozen_new_group_capacity":max_available,
    "exact_prompt_flow_upper_bound":exact,
    "all_model_serving_ids_independently_verified":False,
    "source_disjoint_from_spent_Oleg":False,
    "retrospective_method_selection_from_Oleg_not_allowed":True,
  }
  return out

if __name__=="__main__":
  p=argparse.ArgumentParser()
  p.add_argument("--root",type=Path,required=True)
  p.add_argument("--out",type=Path,required=True)
  args=p.parse_args()
  d=audit(args.root)
  args.out.write_text(json.dumps(d,sort_keys=True,indent=2))
  print("NTL_UNUSED_OLEG18_PROMPT_CAPACITY="+json.dumps({
       k:v for k,v in d.items() if k!="per_class_capacity"
  },sort_keys=True),flush=True)
