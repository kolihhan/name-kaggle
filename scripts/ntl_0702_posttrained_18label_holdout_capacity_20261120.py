"""One-shot chronological and route metadata eligibility for honest 0.702 evaluation.

This is metadata screening ONLY. Publisher-provided generation timestamps and
served-model fields are not cryptographic proofs of provenance. No raw text,
prompts, row IDs, prediction vectors, or model weights are exported.
"""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow.parquet as pq
from collections import Counter
SHA='7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492'
WEAKFLAGS=('flag_near_train_prompt','flag_train_prompt_regen','flag_gemini37_routing_risk')
def ok(s):return (s.notna())&(~s.astype(str).str.strip().str.lower().isin(['','nan','null','none','unknown','n/a']))
def norm(x):return re.sub(r'[^a-z0-9]+','',str(x).lower())
def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--pool',required=True,type=Path);pa.add_argument('--out',required=True,type=Path)
    a=pa.parse_args()
    if hashlib.file_digest(a.pool.open('rb'),'sha256').hexdigest()!=SHA:raise RuntimeError('CHANGED_POOL_SHA')
    file=pq.ParquetFile(a.pool)
    fields=file.schema_arrow.names
    needed=['model','source_dataset','gen_date','gen_served_model','gen_route','match_weight','prompt_canon',*WEAKFLAGS]
    if any(k not in fields for k in needed):raise RuntimeError('MISSING_REQUIRED_PUBLISHER_FIELDS')
    df=pq.read_table(a.pool,columns=needed).to_pandas()
    if len(df)!=231036 or len(set(df.model))!=18:raise RuntimeError('POOL_DRIFT')
    ts=pd.to_datetime(df.gen_date,errors='coerce',utc=True,format='mixed')
    # Some publisher gen_date strings can be inexact. Only the validated parse participates.
    bad_flag=np.zeros(len(df),dtype=bool)
    for f in WEAKFLAGS:
        bad_flag|=df[f].astype(str).str.lower().str.strip().isin(('true','1','yes')).to_numpy()
    route=ok(df.gen_served_model).to_numpy()
    # Syntax-match (not source authentication); served raw ID can include family
    # qualifiers and revision suffix. Strip punctuation, then compare.
    route_match=np.asarray([
        (norm(m)==norm(s) or norm(m) in norm(s))
        if valid else False
        for m,s,valid in zip(df.model.to_numpy(),df.gen_served_model.to_numpy(),route)
    ],dtype=bool)
    nonempty_prompt=ok(df.prompt_canon).to_numpy()
    res={'source_sha':SHA,'n':len(df),'timestamp_nonnull':int(ts.notna().sum()),
         'served_model_nonempty':int(route.sum()),'served_model_alias_syntax_match':int(route_match.sum()),
         'publisher_prompt_key_nonempty':int(nonempty_prompt.sum()),
         'weakflag_positive_any':int(bad_flag.sum()),
         'caveats':'Publisher-controlled gen_date, labels and route strings are NOT original provider authentication; a source cut cannot certify not-in-pretraining, original prompt-independent or competition-host-matched',
         'cutoffs':{}}
    for d in ('2026-11-16','2026-11-18','2026-11-19'):
        mask=(ts>=pd.Timestamp(d,tz='UTC')).to_numpy()
        strict=mask & route_match & nonempty_prompt & ~bad_flag
        counts=dict(sorted(Counter(df.loc[strict,'model'].astype(str)).items()))
        prompts=df.loc[strict,'prompt_canon'].astype(str).nunique()
        src=df.loc[strict,'source_dataset'].astype(str).nunique()
        enough=all(counts.get(x,0)>=12 for x in sorted(set(df.model)))
        res['cutoffs'][d]={
         'publisher_dated_rows':int(mask.sum()),
         'dated_and_alias_route_match_unflagged_with_prompt_key':int(strict.sum()),
         'covered_labels':len(counts),'all18_have_at_least_12_records':bool(enough),
         'per_label_counts':counts,
         'distinct_publisher_prompt_canon':int(prompts),
         'distinct_publisher_source_names':int(src),
         'candidate_use':'descriptive only, not source authenticated',
        }
    a.out.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
    print('NTL_0702_CHRONOLOGICAL_HOLDOUT_CAPACITY='+json.dumps(res,sort_keys=True),flush=True)
if __name__=='__main__':main()
