"""NTL PC-CL v0: frozen external-only 18-label source/prompt split + 2x2 training batches.

No Kaggle official dataset, hidden labels, submission, or model weights.
The data remain in temporary job/Colab workspaces, NEVER GitHub artifacts.
A "serving match" asserted by a publisher is not provider authentication.
"""
from __future__ import annotations
import argparse, hashlib, json, itertools
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

POOL_SHA = "7b1e500ad6512caecc64f6e512d1189a6813448386ee2476398683f6a1ddf492"
SALT = "ntl-heldout-20261117"  # Reuse earlier frozen held-out source rule, no new seed hunt.
SEED = 20261120
OPT_STEPS = 160
RECTANGLE_STEPS = 120
MAX_VALID_PER_CLASS = 100
MIN_LABEL_VALID = 25
MIN_RECTANGLE_LABELS = 12
RISK = ("flag_near_train_prompt", "flag_train_prompt_regen",
        "flag_kb_proxy", "flag_kb_proxy_after_ruling", "flag_gemini37_routing_risk")


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def normalize(value):
    return " ".join(str(value).casefold().split())


def split_sources(sources):
    return sorted(z for z in sources
                  if int(hashlib.sha256((SALT + "|" + z).encode()).hexdigest(), 16) % 5 == 0)


def sample_validation(frame, labels, n=MAX_VALID_PER_CLASS):
    # Exactly one immutable seed. Do not choose heldout sources based on model results.
    sampled = []
    for lab in labels:
        x = frame.loc[frame.model == lab]
        sampled.append(x.sample(n=min(len(x), n), random_state=SEED + labels.index(lab)))
    return pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def make_schedule(frame, labels):
    """A rectangle has A(prompt1,class1), B(prompt1,class2),
       A(prompt2,class1), B(prompt2,class2). A/C positive, A/B negative.
       Prefer prompt1 and prompt2 from different source datasets when possible.
    """
    groups = []
    bygroup = frame.groupby(["source", "prompt_id"], sort=True)
    for (src, pid), g in bygroup:
        # A prompt ID can be ambiguous in some upstream corpora. Require ONE canon.
        if g.prompt_key.nunique() != 1:
            continue
        seen = g.drop_duplicates("model", keep="first")
        if seen.model.nunique() < 2:
            continue
        entries = {str(model): int(idx) for idx, model in zip(seen.index, seen.model)}
        groups.append((str(src), str(pid), str(seen.prompt_key.iloc[0]), entries))
    pair_to_groups = defaultdict(list)
    for src,pid,pkey,entries in groups:
        for a,b in itertools.combinations(sorted(entries), 2):
            pair_to_groups[(a,b)].append((src,pid,pkey,entries[a],entries[b]))
    eligible = {}
    for pair,items in pair_to_groups.items():
        if len(items) < 2:
            continue
        # At least two distinct canonical prompt requests required.
        if len({g[2] for g in items}) >= 2:
            eligible[pair] = items
    coverage = sorted(set(itertools.chain.from_iterable(eligible)))
    if len(coverage) < MIN_RECTANGLE_LABELS:
        raise RuntimeError("RECTANGLE_COVERAGE_BELOW_12_OF_18: " + str(len(coverage)))
    rng = np.random.default_rng(SEED)
    pairs = sorted(eligible)
    counts = Counter()
    batches = []
    cross_source_count = 0
    distinct_labels = set()
    for step in range(OPT_STEPS):
        if step < RECTANGLE_STEPS:
            weights = np.array([1.0 / (1 + counts[pair[0]] + counts[pair[1]]) for pair in pairs])
            weights /= weights.sum()
            a,b = pairs[int(rng.choice(len(pairs),p=weights))]
            items = eligible[(a,b)]
            i=int(rng.integers(0,len(items)))
            first=items[i]
            candidate=[g for g in items if g[2] != first[2]]
            diverse=[g for g in candidate if g[0] != first[0]]
            if diverse:
                candidate=diverse
            if not candidate:
                raise RuntimeError("IMPOSSIBLE_RECTANGLE")
            second=candidate[int(rng.integers(0,len(candidate)))]
            indices=[first[3],first[4],second[3],second[4]]
            # Grounded rather than assumed invariant checks.
            g=frame.iloc[indices]
            if list(g.model)!=[a,b,a,b] or g.prompt_key.iloc[0]!=g.prompt_key.iloc[1] \
                    or g.prompt_key.iloc[2]!=g.prompt_key.iloc[3] \
                    or g.prompt_key.iloc[0]==g.prompt_key.iloc[2]:
                raise RuntimeError("RECTANGLE_LABEL_OR_PROMPT_ORDER_INVALID")
            cross_source_count+=int(first[0]!=second[0])
            counts[a]+=1;counts[b]+=1
            distinct_labels.update((a,b))
            batches.append({"kind":"rectangle","indices":indices})
        else:
            # Same CE controls see identical batches. Random label-balanced fallback
            # ensures all 18 classes still get supervised CE.
            ys=rng.choice(len(labels),size=4,replace=True)
            ix=[]
            for y in ys:
                positions=np.flatnonzero(frame.model.to_numpy()==labels[int(y)])
                ix.append(int(rng.choice(positions)))
            batches.append({"kind":"ordinary","indices":ix})
    # Unbiased arm comparison: exact 4 ids/step shared by CE and CE+CL.
    return batches, {
        "distinct_eligible_label_pairs":len(eligible),
        "rectangle_eligible_labels":coverage,
        "rectangle_eligible_label_count":len(coverage),
        "coherent_multilabel_prompt_groups":len(groups),
        "train_prompt_groups":int(frame.prompt_id.nunique()),
        "real_rectangle_steps":RECTANGLE_STEPS,
        "all_ce_optimizer_steps":OPT_STEPS,
        "cross_source_positive_rectangle_count":cross_source_count,
        "different_source_fraction":round(cross_source_count/RECTANGLE_STEPS,6),
        "rectangle_step_seen_labels":sorted(distinct_labels),
        "ce_fallback_steps":OPT_STEPS-RECTANGLE_STEPS,
        "train_label_min":int(frame.model.value_counts().min()),
    }


def build(root, out=None):
    pool = root / "ntl_open_pool.parquet"
    if sha(pool) != POOL_SHA:
        raise RuntimeError("CURRENT_KAGGLE_OPEN_POOL_SHA_PIN_BROKEN")
    sources=pd.read_csv(root/"sources.csv").fillna("")
    protocol=pd.read_csv(root/"source_protocol.csv").fillna("")
    labels=sorted(pd.read_csv(root/"per_model.csv").model.astype(str).unique())
    if len(labels)!=18:
        raise RuntimeError("SOURCE_MANIFEST_NOT_18_LABELS")
    if sources.source_dataset.duplicated().any() or protocol.source_dataset.duplicated().any():
        raise RuntimeError("DUPLICATE_SOURCE_MANIFEST")
    schema=["text","model","source_dataset","prompt_group","prompt_canon",
            "n_words","gen_served_model",*RISK]
    df=pd.read_parquet(pool,columns=schema)
    sc=sources.set_index("source_dataset")
    pr=protocol.set_index("source_dataset")
    src=df.source_dataset.astype(str)
    ok=src.isin(sc.index)&src.isin(pr.index)&df.model.astype(str).isin(labels)
    ok &= src.map(pr["served_model_check"]).fillna("").str.lower().str.strip().eq("yes")
    lic=src.map(sc["source_license"]).fillna("").astype(str)
    ok &= lic.str.contains("cc-by|attribution|apache|mit|open data",case=False,regex=True)
    ok &= df.text.notna() & df.prompt_group.notna() & df.prompt_canon.notna()
    ok &= df.n_words.between(40,250,inclusive="both")
    for flag in RISK:
        ok &= ~df[flag].fillna(False).astype(str).str.lower().isin(("true","1","yes"))
    clean=df.loc[ok,["text","model","source_dataset","prompt_group","prompt_canon"]].copy()
    clean.rename(columns={"source_dataset":"source","prompt_group":"prompt_id"},inplace=True)
    clean["source"]=clean.source.astype(str)
    clean["model"]=clean.model.astype(str)
    clean["prompt_id"]=clean.prompt_id.astype(str)
    clean["prompt_key"]=clean.prompt_canon.map(normalize)
    clean=clean.loc[clean.prompt_key.str.len().ge(8)].copy()
    # Drop exact within-data duplicates; don't count duplicates as contrastive evidence.
    clean=clean.drop_duplicates(["source","prompt_id","model"],keep="first")
    valid_sources=split_sources(sorted(clean.source.unique()))
    train=clean.loc[~clean.source.isin(valid_sources)].copy()
    valid=clean.loc[clean.source.isin(valid_sources)].copy()
    valid_key=set(valid.prompt_key)
    valid_prompt_id=set(valid.prompt_id)
    train=train.loc[~train.prompt_key.isin(valid_key)&~train.prompt_id.isin(valid_prompt_id)].copy()
    seen_valid={hashlib.sha256(normalize(t).encode()).digest() for t in valid.text}
    train=train.loc[[hashlib.sha256(normalize(t).encode()).digest() not in seen_valid
                     for t in train.text]].copy()
    if set(train.model)!=set(labels) or set(valid.model)!=set(labels):
        raise RuntimeError("SOURCE_DISJOINT_TRAIN_VALID_NOT_BOTH_18_CLASSES")
    if len(train)<2000 or len(valid)<500 or valid.source.nunique()<2:
        raise RuntimeError("SOURCE_HOLDOUT_TOO_SMALL")
    # Keep all model-balanced train source evidence; validation capped before looking at scores.
    train=train.sort_values(["source","prompt_id","model"],kind="stable").reset_index(drop=True)
    valid=sample_validation(valid,labels)
    if int(valid.model.value_counts().min())<MIN_LABEL_VALID:
        raise RuntimeError("SOURCE_HOLDOUT_LABEL_SUPPORT_TOO_SMALL")
    schedule,info=make_schedule(train,labels)
    if any(max(b["indices"])>=len(train) or min(b["indices"])<0 for b in schedule):
        raise RuntimeError("BATCH_INDEX_OUT_OF_RANGE")
    manifest={
        "source_pool_sha256":POOL_SHA,"labels":labels,"train_rows":len(train),
        "valid_rows":len(valid),"train_sources":sorted(train.source.unique()),
        "valid_sources":sorted(valid.source.unique()),
        "seed":SEED,"max_length_tokens":192,
        "steps_each_arm":OPT_STEPS,"batch_size":4,"contrastive_weight":0.2,
        "contrastive_temperature":0.10,"model":"Qwen/Qwen3-1.7B-Base",
        "same_batch_indices_for_both_arms":True,"same_backbone_and_train_data":True,
        "external_only":True,"source_split":SALT,"prompt_content_disjoint":True,
        "exact_response_disjoint":True,"source_disjoint":True,
        "not_a_clean_incumbent_0702_oof":True,
        "reported_served_model_not_independently_authenticated":True,
        "train_label_counts":dict(Counter(train.model)),
        "valid_label_counts":dict(Counter(valid.model)),
        "batch_info":info,
    }
    if out:
        out.mkdir(parents=True,exist_ok=False)
        train[["text","model","source","prompt_id","prompt_key"]].to_parquet(out/"train.parquet",index=False)
        valid[["text","model","source","prompt_id","prompt_key"]].to_parquet(out/"valid.parquet",index=False)
        (out/"batches.json").write_text(json.dumps(schedule,separators=(",",":")))
        manifest["train_sha256"]=sha(out/"train.parquet")
        manifest["valid_sha256"]=sha(out/"valid.parquet")
        manifest["batches_sha256"]=sha(out/"batches.json")
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True))
    public={k:manifest[k] for k in
         ("source_pool_sha256","train_rows","valid_rows","model","steps_each_arm",
          "source_disjoint","prompt_content_disjoint","exact_response_disjoint",
          "train_label_counts","valid_label_counts","batch_info")}
    public["train_source_count"]=len(manifest["train_sources"])
    public["valid_source_count"]=len(manifest["valid_sources"])
    print("NTL_PCCL_SOURCE_PAIR_FEASIBILITY="+json.dumps(public,sort_keys=True),flush=True)
    return manifest


def self_test():
    labels=["a","b","c"]
    frame=pd.DataFrame([
        {"source":"s1","prompt_id":"p1","prompt_key":"first prompt","model":"a"},
        {"source":"s1","prompt_id":"p1","prompt_key":"first prompt","model":"b"},
        {"source":"s2","prompt_id":"p2","prompt_key":"second prompt","model":"a"},
        {"source":"s2","prompt_id":"p2","prompt_key":"second prompt","model":"b"},
        {"source":"s1","prompt_id":"p3","prompt_key":"third prompt","model":"c"},
    ])
    global MIN_RECTANGLE_LABELS,OPT_STEPS,RECTANGLE_STEPS
    old=(MIN_RECTANGLE_LABELS,OPT_STEPS,RECTANGLE_STEPS)
    try:
        MIN_RECTANGLE_LABELS=2;OPT_STEPS=8;RECTANGLE_STEPS=6
        b,info=make_schedule(frame,labels)
        assert len(b)==8 and sum(v["kind"]=="rectangle" for v in b)==6
        assert info["cross_source_positive_rectangle_count"]==6
        assert sorted(info["rectangle_eligible_labels"])==["a","b"]
        print("NTL_PCCL_RECTANGLE_SELFTEST=PASS")
    finally:
        MIN_RECTANGLE_LABELS,OPT_STEPS,RECTANGLE_STEPS=old


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:self_test()
    else:
        if not a.input:ap.error("--input required")
        build(a.input,a.output)
