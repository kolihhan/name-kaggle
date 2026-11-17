"""Prepare a frozen 18-label, source-disjoint external-only training pilot.

Not competition data. No model is trained by this script. The source/holdout
selection and label/source balancing are fixed before classifier evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

POOL_SHA = "5a3dc82288328abedf49b7dd5e866e5c69eb922f9d6008e6fd78cf5153c12332"
HOLDOUT_SALT = "ntl-heldout-20261117"
SEED = 20261117
MAX_PER_LABEL_TRAIN = 360
MAX_PER_LABEL_VALID = 150
RISK = (
    "flag_near_train_prompt", "flag_train_prompt_regen",
    "flag_kb_proxy", "flag_kb_proxy_after_ruling",
    "flag_gemini37_routing_risk",
)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pick_by_label_source(frame, max_count, seed):
    """Round-robin each source, then deterministic shuffle per true label."""
    rng = np.random.default_rng(seed)
    chosen = []
    for label in sorted(frame.model.unique()):
        group = frame.loc[frame.model.eq(label)]
        blocks = []
        for source in sorted(group.source.unique()):
            g = group.loc[group.source.eq(source)]
            order = rng.permutation(g.index.to_numpy())
            blocks.append([int(i) for i in order])
        output = []
        while len(output) < max_count and any(blocks):
            for block in blocks:
                if block and len(output) < max_count:
                    output.append(block.pop())
        chosen += output
    return frame.loc[chosen].sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.input, args.out
    pool = root / "ntl_open_pool.parquet"
    if sha(pool) != POOL_SHA:
        raise RuntimeError("PUBLIC_OPENPOOL_VERSION_HASH_DRIFT")
    sources = pd.read_csv(root / "sources.csv").fillna("")
    protocols = pd.read_csv(root / "source_protocol.csv").fillna("")
    labels = sorted(pd.read_csv(root / "per_model.csv").model.astype(str).unique())
    if len(labels) != 18 or len(sources) != 65 or len(protocols) != 65:
        raise RuntimeError("OPENPOOL_SOURCE_OR_LABEL_VERSION_DRIFT")
    required = ["text", "model", "source_dataset", "prompt_group", "prompt_canon",
                "n_words", "gen_served_model", *RISK]
    frame = pd.read_parquet(pool, columns=required)
    s = frame.source_dataset.astype(str)
    m = frame.model.astype(str)
    src = sources.set_index("source_dataset")
    pr = protocols.set_index("source_dataset")
    ok = s.isin(src.index) & s.isin(pr.index) & m.isin(labels)
    ok &= s.map(pr["served_model_check"]).fillna("").str.lower().str.strip().eq("yes")
    license_str = s.map(src["source_license"]).fillna("").astype(str)
    ok &= license_str.str.contains("cc-by|attribution|apache|mit|open data",case=False,regex=True)
    ok &= frame.text.notna() & frame.prompt_group.notna()
    ok &= frame.prompt_group.astype(str).str.len().gt(2)
    ok &= frame.n_words.between(40,250,inclusive="both")
    for flag in RISK:
        ok &= ~frame[flag].fillna(False).astype(str).str.lower().isin(("true","1","yes"))
    good = frame.loc[ok, ["text","model","source_dataset","prompt_group","prompt_canon","gen_served_model"]].copy()
    good.rename(columns={"source_dataset":"source", "prompt_group":"prompt_id"},inplace=True)
    good["source"] = good.source.astype(str)
    good["model"] = good.model.astype(str)
    sources_set = sorted(good.source.unique())
    heldout_sources = sorted(z for z in sources_set
            if int(hashlib.sha256((HOLDOUT_SALT+"|"+z).encode()).hexdigest(),16)%5==0)
    train = good.loc[~good.source.isin(heldout_sources)].copy()
    valid = good.loc[good.source.isin(heldout_sources)].copy()
    # Never allow identical prompt contents/IDs or response text to cross source split.
    def normalize(v):
        return " ".join(str(v).strip().lower().split()) if pd.notna(v) else ""
    train["prompt_key"] = [normalize(v) for v in train.prompt_canon]
    valid["prompt_key"] = [normalize(v) for v in valid.prompt_canon]
    valid_keys = set(valid.prompt_key) - {""}
    valid_ids = set(valid.prompt_id.astype(str))
    train = train.loc[
        ~train.prompt_key.isin(valid_keys) &
        ~train.prompt_id.astype(str).isin(valid_ids)
    ].copy()
    # Exact response hashes are a narrow but deterministic duplicate guard.
    valid_hashes = {
        hashlib.sha256(normalize(v).encode()).digest()
        for v in valid.text
    }
    train = train.loc[[
        hashlib.sha256(normalize(v).encode()).digest() not in valid_hashes
        for v in train.text
    ]].copy()
    if set(train.model) != set(labels) or set(valid.model) != set(labels):
        raise RuntimeError("SPLIT_NO_LONGER_HAS_ALL_18_LABELS")
    if len(heldout_sources) < 2 or len(train) < 1800 or len(valid) < 360:
        raise RuntimeError("INSUFFICIENT_INDEPENDENT_UNITS")
    exact_served = good.gen_served_model.fillna("").astype(str).str.strip().str.lower().eq(
        good.model.str.strip().str.lower())
    print("NTL_TRAIN_SOURCE_INCLUDED="+json.dumps({
        "all_quality_rows":int(len(good)),
        "train_source_count":int(train.source.nunique()),
        "valid_source_count":int(valid.source.nunique()),
        "train_pre_sample":int(len(train)),
        "valid_pre_sample":int(len(valid)),
        "source_id_exact_row_support":int(exact_served.sum()),
        "source_id_exact_class_support":int(good.loc[exact_served,"model"].nunique()),
        "prompt_content_disjoint":True,
        "exact_text_disjoint":True,
        "original_served_id_independently_verified":False,
        "published_source_assertion_only":True,
    },sort_keys=True),flush=True)
    train = pick_by_label_source(train,MAX_PER_LABEL_TRAIN,SEED)
    valid = pick_by_label_source(valid,MAX_PER_LABEL_VALID,SEED+1)
    for key, data in (("train",train),("valid",valid)):
        print("NTL_TRAIN_FROZEN_"+key.upper()+"_SUMMARY="+json.dumps({
            "rows":len(data),
            "model_labels":int(data.model.nunique()),
            "source_count":int(data.source.nunique()),
            "min_label_rows":int(data.model.value_counts().min()),
            "max_label_rows":int(data.model.value_counts().max()),
            "min_prompt_groups":int(data.prompt_id.nunique()),
        },sort_keys=True),flush=True)
        if data.model.nunique()!=18 or int(data.model.value_counts().min())<20:
            raise RuntimeError("INSUFFICIENT_PER_LABEL_SUPPORT_"+key.upper())
    fields=["text","model","source","prompt_id"]
    out.mkdir(parents=True,exist_ok=False)
    train[fields].to_parquet(out/"train.parquet",index=False)
    valid[fields].to_parquet(out/"valid.parquet",index=False)
    manifest={
        "data_sha256":POOL_SHA,
        "train_sha256":sha(out/"train.parquet"),
        "valid_sha256":sha(out/"valid.parquet"),
        "training_sources":sorted(train.source.unique()),
        "validation_sources":sorted(valid.source.unique()),
        "labels":labels,
        "sampling":"source-roundrobin within label, deterministic seed, max360 train/150 valid",
        "heldout_source_rule":HOLDOUT_SALT+" sha256 mod 5==0",
        "new_training_authorized":True,
        "model_snapshot_identity_proven":False,
    }
    (out/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2))
    print("NTL_TRAIN_PREPARED_FULL18=TRUE",flush=True)
    print("NTL_TRAIN_SHA256="+manifest["train_sha256"],flush=True)
    print("NTL_VALID_SHA256="+manifest["valid_sha256"],flush=True)


if __name__=="__main__":
    main()
