"""One-shot NTL host-match proxy feasibility audit (CPU, metadata/report only).

Answer: Can match_weight select a meaningful, class-complete, source-controlled
external training cohort? No training, no inference, no Kaggle submission.
Publisher metadata is NOT authenticated model provenance or an output-use license.
No raw text/IDs or train/valid arrays are written to GitHub or artifacts.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

POOL_SHA = "7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492"
SALT = "ntl-heldout-20261117"
SEED = 20261120
FLAGS = ("flag_near_train_prompt", "flag_train_prompt_regen",
         "flag_kb_proxy", "flag_kb_proxy_after_ruling",
         "flag_gemini37_routing_risk")
EXPECTED_TRAIN = 8347
EXPECTED_VALID_SAMPLED = 1800
MIN_STRATUM = 12
TOP_FRACTION = 0.25

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def norm(s):
    return " ".join(str(s).casefold().split())

def finite_values(x):
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)

def stats(x):
    v = finite_values(x).dropna()
    return {"n": int(len(v)), "min": round(float(v.min()),6) if len(v) else None,
            "q25": round(float(v.quantile(.25)),6) if len(v) else None,
            "median": round(float(v.median()),6) if len(v) else None,
            "q75": round(float(v.quantile(.75)),6) if len(v) else None,
            "max": round(float(v.max()),6) if len(v) else None,
            "distinct": int(v.nunique())}

def select_source_locked(frame, min_stratum=MIN_STRATUM, fraction=TOP_FRACTION):
    """Within each fixed (publisher, model), choose equal-N top-weight and random.

    Controls class and publisher counts exactly, not the model generation route.
    Strata with too few *numeric-weight* rows are omitted in BOTH arms.
    """
    high, random_rows, skipped = [], [], []
    for (src, label), g in frame.groupby(["source","model"],sort=True):
        eligible = g.loc[g.match_weight.notna()].copy()
        if len(eligible) < min_stratum:
            skipped.append({"label":str(label),"rows":int(len(g)),
                            "numeric_weight_rows":int(len(eligible))})
            continue
        k = max(1, math.ceil(len(eligible)*fraction))
        high.append(eligible.sort_values(
            ["match_weight","stable_key"],ascending=[False,True],
            kind="stable").iloc[:k])
        random_rows.append(eligible.sample(n=k,random_state=SEED))
    if not high:
        return None,None,skipped
    return pd.concat(high,ignore_index=True),pd.concat(random_rows,ignore_index=True),skipped

def self_test():
    rows = []
    for lab in ("a","b"):
        for src in ("s1","s2"):
            for i in range(16):
                rows.append({"model":lab,"source":src,"prompt_id":f"{src}_{i}",
                             "stable_key":f"{src}_{lab}_{i}",
                             "match_weight":float(i)})
    h,r,skip = select_source_locked(pd.DataFrame(rows))
    assert not skip and len(h)==len(r)==16
    assert Counter(zip(h.source,h.model)) == Counter(zip(r.source,r.model))
    assert h.match_weight.mean() > r.match_weight.mean()
    print("NTL_HOSTMATCH_SOURCE_LOCKED_UNIT=PASS",flush=True)

def build(root):
    root = Path(root)
    pool = root/"ntl_open_pool.parquet"
    source_sha = sha(pool)
    if source_sha != POOL_SHA:
        raise RuntimeError("POOL_SHA_DRIFT: refusing a changed cohort")
    sc = pd.read_csv(root/"sources.csv").fillna("")
    pr = pd.read_csv(root/"source_protocol.csv").fillna("")
    labels = sorted(pd.read_csv(root/"per_model.csv").model.astype(str).unique())
    if len(labels)!=18:
        raise RuntimeError("EXPECTED_18_OFFICIAL_LABELS")
    if sc.source_dataset.duplicated().any() or pr.source_dataset.duplicated().any():
        raise RuntimeError("SOURCE_MANIFEST_DUPLICATES")
    sc=sc.set_index("source_dataset")
    pr=pr.set_index("source_dataset")
    cols=["text","model","source_dataset","prompt_group","prompt_canon",
          "n_words","gen_served_model","match_weight",*FLAGS]
    df=pd.read_parquet(pool,columns=cols)
    src=df.source_dataset.astype(str)
    ok=src.isin(sc.index)&src.isin(pr.index)&df.model.astype(str).isin(labels)
    ok &= src.map(pr["served_model_check"]).fillna("").str.lower().str.strip().eq("yes")
    licenses=src.map(sc["source_license"]).fillna("").astype(str)
    ok &= licenses.str.contains("cc-by|attribution|apache|mit|open data",case=False,regex=True)
    ok &= df.text.notna() & df.prompt_group.notna() & df.prompt_canon.notna()
    ok &= df.n_words.between(40,250,inclusive="both")
    for flag in FLAGS:
        ok &= ~df[flag].fillna(False).astype(str).str.lower().isin(("true","1","yes"))
    c=df.loc[ok,["text","model","source_dataset","prompt_group","prompt_canon",
                   "gen_served_model","match_weight"]].copy()
    c.rename(columns={"source_dataset":"source","prompt_group":"prompt_id"},inplace=True)
    c["source"]=c.source.astype(str)
    c["model"]=c.model.astype(str)
    c["prompt_id"]=c.prompt_id.astype(str)
    c["prompt_key"]=c.prompt_canon.map(norm)
    c=c.loc[c.prompt_key.str.len().ge(8)].copy()
    c=c.drop_duplicates(["source","prompt_id","model"],keep="first")
    held={s for s in c.source.unique()
          if int(hashlib.sha256((SALT+"|"+s).encode()).hexdigest(),16)%5==0}
    train=c.loc[~c.source.isin(held)].copy()
    valid=c.loc[c.source.isin(held)].copy()
    train=train.loc[~train.prompt_key.isin(set(valid.prompt_key))
                    &~train.prompt_id.isin(set(valid.prompt_id))].copy()
    seen={hashlib.sha256(norm(t).encode()).digest() for t in valid.text}
    train=train.loc[[hashlib.sha256(norm(t).encode()).digest() not in seen
                     for t in train.text]].copy()
    if len(train)!=EXPECTED_TRAIN:
        raise RuntimeError(f"PC_CL_COHORT_DRIFT expected {EXPECTED_TRAIN} actual {len(train)}")
    if set(train.model)!=set(labels) or set(valid.model)!=set(labels):
        raise RuntimeError("NOT_FULL_18_LABELS")
    valid=pd.concat([valid.loc[valid.model.eq(l)].sample(
        n=min(100,int(valid.model.eq(l).sum())),random_state=SEED+i)
        for i,l in enumerate(labels)],ignore_index=True)
    if len(valid)!=EXPECTED_VALID_SAMPLED:
        raise RuntimeError("PC_CL_VALID_COHORT_DRIFT")
    train["match_weight"]=finite_values(train.match_weight)
    train["stable_key"]=train.source.astype(str)+"|"+train.prompt_id+"|"+train.model
    h,r,skip=select_source_locked(train)
    if h is None:raise RuntimeError("NO_NUMERIC_STRATA")
    assert len(h)==len(r)
    assert Counter(zip(h.source,h.model))==Counter(zip(r.source,r.model))
    def counts(g):
        return {lab:int(g.model.eq(lab).sum()) for lab in labels}
    def source_support(g):
        return {lab:int(g.loc[g.model.eq(lab),"source"].nunique()) for lab in labels}
    def prompt_support(g):
        return {lab:int(g.loc[g.model.eq(lab),"prompt_key"].nunique()) for lab in labels}
    served=train.gen_served_model.astype(str).str.strip()
    row_claims=served.str.lower().isin(("", "nan", "none", "unknown"))
    high_counts=counts(h)
    low_labels=[lab for lab,n in high_counts.items() if n<100]
    under_sources=[lab for lab,n in source_support(h).items() if n<2]
    # R2 from source means: match-weight differences may mostly encode publisher.
    weighted=train.loc[train.match_weight.notna(),["source","match_weight"]]
    x=weighted.match_weight.to_numpy(dtype=float)
    overall_var=float(np.var(x)) if len(x) else 0.0
    between=float(sum(len(g)*(float(g.match_weight.mean())-float(np.mean(x)))**2
                  for _,g in weighted.groupby("source"))/len(x)) if len(x) else 0.0
    source_eta2=between/overall_var if overall_var>0 else None
    report={
      "experiment":"NTL_HOSTMATCH_SOURCE_LOCKED_METADATA_FEASIBILITY_V0",
      "dataset_sha256":source_sha,"dataset_total_rows":len(df),
      "frozen_source_disjoint_train_rows":len(train),
      "frozen_valid_sampled_rows":len(valid),
      "train_source_count":int(train.source.nunique()),
      "validation_source_count":int(valid.source.nunique()),
      "train_match_weight":stats(train.match_weight),
      "per_label_train_numeric_weight_rows":{l:int(train.loc[train.model.eq(l),"match_weight"].notna().sum()) for l in labels},
      "per_label_train_rows":counts(train),
      "publisher_reported_served_model_nonempty":int((~row_claims).sum()),
      "publisher_reported_served_model_exact_label_strings":int(
          (served.str.casefold()==train.model.str.casefold()).sum()),
      "source_explained_weight_variance_eta2":round(float(source_eta2),6) if source_eta2 is not None else None,
      "source_locked_top_fraction":TOP_FRACTION,
      "min_numeric_weight_stratum":MIN_STRATUM,
      "matched_train_count":len(h),
      "random_same_source_label_count":len(r),
      "exact_per_source_label_counts_matched":True,
      "high_match_weight_stats":stats(h.match_weight),
      "random_match_weight_stats":stats(r.match_weight),
      "high_per_label_rows":high_counts,
      "random_per_label_rows":counts(r),
      "high_per_label_source_counts":source_support(h),
      "high_per_label_distinct_prompt_counts":prompt_support(h),
      "labels_below_100_high_match_rows":low_labels,
      "labels_with_fewer_than_2_train_sources":under_sources,
      "skipped_small_or_non_numeric_strata":skip,
      "weak7_high_counts":{l:high_counts[l] for l in labels if l.startswith(("glm","kimi","gemini"))},
      "numeric_weights_not_all_identical":stats(h.match_weight)["distinct"]>1,
      "source_level_and_row_level_license_not_independently_verified":True,
      "served_model_claims_are_publisher_metadata_not_provider_proof":True,
      "host_match_weight_is_proxy_not_verified_competition_match":True,
      "candidate_is_metadata_feasible_full18":len(low_labels)==0 and not under_sources,
      "no_model_training_or_score_or_raw_text_export":True,
    }
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:return self_test()
    if not args.input or not args.out:p.error("--input and --out required")
    r=build(args.input)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print("NTL_HOSTMATCH_SOURCE_LOCKED_RESULT="+json.dumps({
      "source_sha":r["dataset_sha256"],"n_train":r["frozen_source_disjoint_train_rows"],
      "matched_n":r["matched_train_count"],"random_n":r["random_same_source_label_count"],
      "hi_weight":r["high_match_weight_stats"],"random_weight":r["random_match_weight_stats"],
      "source_eta2":r["source_explained_weight_variance_eta2"],
      "weak7_counts":r["weak7_high_counts"],
      "missing100":r["labels_below_100_high_match_rows"],
      "missing2sources":r["labels_with_fewer_than_2_train_sources"],
      "full18_feasible":r["candidate_is_metadata_feasible_full18"],
      "no_gpu":True,"no_submit":True,
    },sort_keys=True),flush=True)

if __name__=="__main__":main()
