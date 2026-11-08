"""CDA-L24 v2: single preregistered original-source training experiment.

Two upstream, licensed original Kaggle datasets are used for Stage A:
  arxitekton/ntl-claude-gen-v1 and trnhip/ntl-api-gen.
The unseen kilupy/ntl-18-model-corpus is used ONLY for a one-shot external
transfer score, split into 80 SHA-selected prompts from each of its disjoint
Claude-5 and other-13 blocks. Never tune on original official OOF or Public LB.

This is a source-transfer study, NOT a claim of complete protocol or model
version equivalence. The upstream training authors may have sampled prompts
or model settings informed by competition training distributions; neither
origin is treated as an independent science holdout.

The retired CDA v1 script remains disabled; this is a different contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd

from scripts.conditional_domain_adversarial_lora24 import (
    ADV_LAMBDA,
    ConditionalDomainAdversary,
    crossed_source_support,
    joint_training_loss,
)
from scripts.run_two_stage_lora24_gate_a import (
    build_prompt_triplets,
    compute_class_weights,
    group_triplets_for_accumulation,
)
from scripts.run_two_stage_lora24_gate_b_gpu import (
    _make_encoding,
    _representation,
    load_exact_stage1_checkpoint,
)
from scripts.run_two_stage_lora24_gate_a import gate_a_training_spec

EXPECTED_POOL_SHA256 = "DIRECT_ORIGINAL_ARX_TRN_TRAIN_KIL_80x2_HELDOUT_V2"
RESERVED_SOURCE = "kilupy/ntl-18-model-corpus"
SEED = 20261101
MIN_CROSSED_CLASSES = 8
AUTHORITATIVE_LABELS = 18
CHECKPOINT_ORIGIN = "two-stage-lora24-v1-A1"
CANDIDATE = "cda-l24-direct-sources-v2"
MUST_HAVE_FLAGS = (
    "flag_near_train_prompt",
    "flag_train_prompt_regen",
    "flag_kb_proxy",
    "flag_gemini37_routing_risk",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalized(value: Any) -> str:
    import re
    import unicodedata
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).lower()).strip()


def _text_hash(value: Any) -> str:
    return hashlib.sha256(_normalized(value).encode("utf-8")).hexdigest()


def build_clean_partitions(
    pool: pd.DataFrame, *, labels: list[str],
    approved_sources: tuple[str, ...],
    old_fit_text_hashes: set[str], official_text_hashes: set[str],
    old_prompt_keys: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"text", "model", "source_dataset", "prompt_group",
                "prompt_canon", "source_license", *MUST_HAVE_FLAGS}
    if not required.issubset(pool.columns):
        raise ValueError(f"source schema missing {sorted(required - set(pool.columns))}")
    selected = pool.loc[pool["source_dataset"].astype(str).isin(
        (*approved_sources, RESERVED_SOURCE))].copy()
    if selected.empty or selected[["text", "model", "source_dataset",
                                   "prompt_canon", "prompt_group"]].isna().any().any():
        raise ValueError("missing selected source rows or essential values")
    if not set(selected["model"].astype(str)) <= set(labels):
        raise ValueError("selected source includes unknown official labels")
    selected["source"] = selected["source_dataset"].astype(str)
    selected["model"] = selected["model"].astype(str)
    selected["text_hash"] = selected["text"].map(_text_hash)
    selected["prompt_key"] = selected["prompt_canon"].map(_normalized)
    selected["prompt_id"] = selected["source"] + "::" + selected["prompt_group"].astype(str)

    flags = np.zeros(len(selected), dtype=bool)
    for name in MUST_HAVE_FLAGS:
        s = selected[name]
        # Missing publisher flags are unsafe, rather than assumed false.
        flags |= s.isna().to_numpy() | s.astype(str).str.lower().isin(
            ["true", "1", "yes", "y", "unverified", "risk"]).to_numpy()
    unknown_license = selected["source_license"].isna() | (
        selected["source_license"].astype(str).str.strip().str.lower().isin(
            ["", "nan", "none", "unknown", "unverified", "no license"]))
    leaked = selected["text_hash"].isin(
        old_fit_text_hashes | official_text_hashes)
    leaked_prompt = selected["prompt_key"].isin(old_prompt_keys)
    safe = selected.loc[~(flags | unknown_license.to_numpy() |
                          leaked.to_numpy() | leaked_prompt.to_numpy())].copy()
    # Exact same answer across source groups: remove all copies, not one.
    duplicated = safe["text_hash"].duplicated(keep=False)
    safe = safe.loc[~duplicated].copy()
    # A shared prompt across training and transfer is not a new prompt.
    transfer = safe.loc[safe["source"].eq(RESERVED_SOURCE)].copy()
    train = safe.loc[safe["source"].isin(approved_sources)].copy()
    transfer_prompt_keys = set(transfer["prompt_key"])
    shared_keys = set(train["prompt_key"]) & transfer_prompt_keys
    if shared_keys:
        train = train.loc[~train["prompt_key"].isin(shared_keys)].copy()
        transfer = transfer.loc[~transfer["prompt_key"].isin(shared_keys)].copy()
    if train.empty or transfer.empty:
        raise ValueError("no eligible training or transfer rows")
    if train["source"].nunique() < 2:
        raise ValueError("need >=2 training sources")
    # Unchanged official model labels at fit/infer. A class can exist in official
    # but be absent from source transfer; never change label order.
    train = train.reset_index(drop=True)
    transfer = transfer.reset_index(drop=True)
    return train, transfer, {
        "raw_selected_rows": len(selected),
        "flagged_rows": int(flags.sum()),
        "license_unknown_rows": int(unknown_license.sum()),
        "known_answer_overlap_rows": int(leaked.sum()),
        "known_prompt_overlap_rows": int(leaked_prompt.sum()),
        "duplicated_answer_rows_removed": int(duplicated.sum()),
        "shared_train_transfer_prompt_keys_removed": len(shared_keys),
        "train_rows": len(train),
        "transfer_rows": len(transfer),
        "train_sources": sorted(set(train.source)),
        "transfer_prompt_groups": int(transfer.prompt_id.nunique()),
        "train_labels": int(train.model.nunique()),
        "transfer_labels": int(transfer.model.nunique()),
    }


def _canonical_file(parent: Path, name: str) -> Path:
    files=list(parent.rglob(name))
    if len(files)!=1:
        raise ValueError(f"Expected one original Kaggle dataset file {name}, found {len(files)}")
    return files[0]


def _stable_prompts_by_block(source: pd.DataFrame, per_block: int = 80) -> pd.DataFrame:
    # Both kilupy prompt blocks are independently authored; a prompt group
    # contains responses from all members of its 5/13 model block. Never
    # mix groups or choose based on model prediction.
    selected=[]
    for block,rows in source.groupby("block",sort=True):
        groups=rows["prompt_id"].astype(str).unique().tolist()
        if len(groups)<per_block:
            raise ValueError(f"insufficient original prompts in {block}")
        chosen=sorted(groups,key=lambda x:hashlib.sha256(
            f"cda-l24-v2-original-unseen-20261108|{block}|{x}".encode()
        ).hexdigest())[:per_block]
        selected.append(rows.loc[rows["prompt_id"].astype(str).isin(chosen)])
    if len(selected)!=2:
        raise ValueError("kilupy must have two original source prompt blocks")
    return pd.concat(selected,ignore_index=True)


def _prepare_input(args: argparse.Namespace) -> tuple[pd.DataFrame,pd.DataFrame,list[str],dict]:
    # First-party original datasets attached in Kaggle; no competition-source
    # data are loaded. The old Stage-A pool is read for exact-text exclusion.
    a_path=_canonical_file(args.arx_directory,"passages.csv")
    b_path=_canonical_file(args.trn_directory,"passages.csv")
    c_path=_canonical_file(args.kil_directory,"ntl_18model_passages.parquet")
    pinned={
        "arxitekton/ntl-claude-gen-v1":_sha256(a_path),
        "trnhip/ntl-api-gen":_sha256(b_path),
        "kilupy/ntl-18-model-corpus":_sha256(c_path),
    }
    baseline=json.loads((args.original_a1/"meta.json").read_text())
    labels=[str(x) for x in baseline["labels"]]
    if len(labels)!=AUTHORITATIVE_LABELS or len(set(labels))!=18:
        raise RuntimeError("original locked 18-label checkpoint mismatch")
    a=pd.read_csv(a_path)
    b=pd.read_csv(b_path)
    c=pd.read_parquet(c_path)
    required={"text","model","prompt_id"}
    for name,data in (("arx",a),("trn",b),("kil",c)):
        if not required.issubset(data):
            raise ValueError(f"{name} missing essential original fields")
    if "block" not in c:
        raise ValueError("original 2-block heldout required")
    c=_stable_prompts_by_block(c)
    source_frames=[]
    for name,data,source in [
        ("arx",a,"arxitekton/ntl-claude-gen-v1"),
        ("trn",b,"trnhip/ntl-api-gen"),
        ("kil",c,RESERVED_SOURCE),
    ]:
        data=data.loc[
            data["text"].notna() & data["model"].notna() &
            data["prompt_id"].notna()].copy()
        data["model"]=data["model"].astype(str)
        if not set(data["model"])<=set(labels):
            raise ValueError(f"{name}: unrecognized official model labels")
        data["source"]=source
        data["prompt_id"]=name+"::"+(
            data["block"].astype(str)+"::" if name=="kil" else ""
        )+data["prompt_id"].astype(str)
        data["text_hash"]=data["text"].map(_text_hash)
        source_frames.append(data[["text","model","source","prompt_id","text_hash"]])
    merged=pd.concat(source_frames,ignore_index=True)
    # One exact source-text identity should not appear in either train+transfer
    # nor multiple sources. Fail closed across *all* copies of repeated text.
    before=len(merged)
    merged=merged.loc[~merged["text_hash"].duplicated(keep=False)].copy()
    duplicate_removed=before-len(merged)
    # Avoid prior Stage-A exact output memorization by excluding all prior
    # LoRA24 A1 training strings; the old fit is not copied into new model.
    from scripts.run_phase14_public_sparse import build_conservative_subset
    from scripts.run_two_stage_lora24_gate_a import build_stage1_split
    prior=build_conservative_subset(pd.read_parquet(args.prior_pool))
    fit_idx=build_stage1_split(prior)["train_idx"]
    known=set(prior.iloc[fit_idx]["text"].map(_text_hash))
    n_prior=int(merged["text_hash"].isin(known).sum())
    merged=merged.loc[~merged["text_hash"].isin(known)].copy()
    train=merged.loc[merged["source"].ne(RESERVED_SOURCE)].copy()
    transfer=merged.loc[merged["source"].eq(RESERVED_SOURCE)].copy()
    if set(train["prompt_id"])&set(transfer["prompt_id"]):
        raise RuntimeError("same prompt group in train and transfer")
    if set(train["text_hash"])&set(transfer["text_hash"]):
        raise RuntimeError("duplicate exact text in train and transfer")
    # A single source must never encode exactly one model label; the
    # source-conditional adversary needs >=8 labels with crossed support.
    if train.source.nunique()!=2 or train.model.nunique()<12:
        raise RuntimeError("source/label scope collapsed")
    if transfer.model.nunique()!=18 or transfer.prompt_id.nunique()<120:
        raise RuntimeError("expected 18-label paired unseen prompt groups")
    train=train[["text","model","source","prompt_id"]].reset_index(drop=True)
    transfer=transfer[["text","model","source","prompt_id"]].reset_index(drop=True)
    stats={
        "dataset_sha256":pinned,
        "source_origin_note":"training authors host-distribution-conditioned; eval has original prompts",
        "prior_A1_exact_train_duplicates_removed":n_prior,
        "cross_source_exact_duplicates_removed":duplicate_removed,
        "train_source_rows":{str(k):int(v) for k,v in train.source.value_counts().items()},
        "train_labels":int(train.model.nunique()),
        "train_rows":int(len(train)),
        "transfer_rows":int(len(transfer)),
        "transfer_labels":int(transfer.model.nunique()),
        "transfer_prompt_groups":int(transfer.prompt_id.nunique()),
        "heldout_source":RESERVED_SOURCE,
        "official_rows_used_at_stage_A":0,
    }
    return train,transfer,labels,stats


def _gpu_run(args: argparse.Namespace):
    import torch
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        raise RuntimeError("CDA-L24 Stage A requires real CUDA; not a CPU proxy")
    if args.arm not in ("C0", "C1"):
        raise ValueError("fixed arms C0 or C1 only")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    train, transfer, labels, source_stats = _prepare_input(args)
    source_ids = sorted(train["source"].unique())
    label_to_id = {name:i for i,name in enumerate(labels)}
    source_to_id = {name:i for i,name in enumerate(source_ids)}
    train_y = torch.tensor([label_to_id[x] for x in train["model"]], dtype=torch.long)
    train_s = torch.tensor([source_to_id[x] for x in train["source"]], dtype=torch.long)
    support = crossed_source_support(train_y, train_s,
             label_count=len(labels), source_count=len(source_ids), minimum=20)
    if int(support.eligible_labels.sum()) < MIN_CROSSED_CLASSES:
        raise RuntimeError("uncrossed model-by-source provenance: adversarial CE unidentifiable")
    strength = 0.0 if args.arm == "C0" else ADV_LAMBDA
    device = torch.device("cuda:0")
    loaded = load_exact_stage1_checkpoint(
        checkpoint_dir=args.original_a1,
        snapshot_path=args.base_snapshot,
        official_labels=labels, device=device)
    model=loaded["model"]; norms=loaded["norms"]; plain=loaded["plain"]
    centered=loaded["centered"]; dropout=loaded["dropout"]
    tokenizer=loaded["tokenizer"]; layers=loaded["layers"]
    source_head = ConditionalDomainAdversary(
        feature_dim=4096*len(layers),label_count=len(labels),
        source_count=len(source_ids)).to(device)
    # Fresh control/treatment runs use identical seed and source-head init.
    spec = gate_a_training_spec()
    if spec["epochs"] != 1 or spec["seed"] != SEED:
        raise RuntimeError("original train schedule changed")
    model.train(); norms.train(); plain.train(); centered.train()
    source_head.train(); dropout.train()
    class_weights = compute_class_weights(train.model.values, classes=labels)
    weight = torch.tensor([class_weights[label] for label in labels],
                          device=device,dtype=torch.float32)
    train_tokens = tokenizer(train["text"].astype(str).tolist(),
        add_special_tokens=False, truncation=True,
        max_length=int(spec["max_length_before_eos"]))["input_ids"]
    eos = int(tokenizer.eos_token_id)
    train_tokens = [list(t)+[eos] for t in train_tokens]
    triplets = build_prompt_triplets(train, np.arange(len(train)), seed=SEED)
    steps=group_triplets_for_accumulation(
        triplets,accumulation=int(spec["gradient_accumulation_triplets"]))
    optimizer = torch.optim.AdamW([
         {"params": [p for p in model.parameters() if p.requires_grad],
          "lr": float(spec["adapter_lr"])},
         {"params": list(norms.parameters())+list(plain.parameters())+
                    list(centered.parameters())+list(source_head.parameters()),
          "lr": float(spec["head_lr"])},
       ],weight_decay=float(spec["weight_decay"]))
    total_steps=len(steps)
    warmup=max(1,int(total_steps*float(spec["warmup_fraction"])))
    def schedule(t):
        if t<warmup:return (t+1)/warmup
        progress=(t-warmup)/max(1,total_steps-warmup)
        return 0.5*(1.0+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    scaler=torch.amp.GradScaler("cuda",init_scale=256)
    eligible_total=0
    stats={"arm":args.arm,"strength":strength,"epochs":1,
           "train_rows":len(train),"transfer_rows":len(transfer),
           "crossed_labels":int(support.eligible_labels.sum()),
           "train_sources":len(source_ids),"optimizer_steps_expected":total_steps}
    for step,microbatches in enumerate(steps):
        optimizer.zero_grad(set_to_none=True)
        for ix in microbatches:
            encoded=_make_encoding(tokenizer,train_tokens,ix,device)
            z=_representation(model=model,norms=norms,layers=layers,
                              encoding=encoded,torch=torch)
            targets=train_y[ix].to(device)
            sources=train_s[ix].to(device)
            logits=plain(dropout(z))
            loss,diag=joint_training_loss(
                logits,targets,z,sources,source_head,support,
                gradient_strength=strength,model_class_weights=weight)
            # Train the original centered head as in the original reference,
            # with the original objective scaling for both arms.
            from scripts.run_two_stage_lora24_gate_a import center_triplet
            centered_logits=centered(dropout(center_triplet(z)))
            import torch.nn.functional as F
            loss=loss+F.cross_entropy(centered_logits,targets,weight=weight)
            if not bool(torch.isfinite(loss)):raise RuntimeError("non-finite CDA loss")
            scaler.scale(loss / len(microbatches)).backward()
            eligible_total+=diag["eligible_source_rows"]
        scaler.unscale_(optimizer)
        all_params=[p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(all_params,float(spec["gradient_clip"]))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    if eligible_total < 1:
        raise RuntimeError("No crossed training cells contributed domain CE")
    stats["eligible_source_rows_processed"]=eligible_total
    stats["optimizer_steps"]=total_steps
    stats["source_pool_sha256"]=EXPECTED_POOL_SHA256
    stats["original_stage1_manifest_sha256"]=loaded["manifest_sha256"]
    stats["stage1_metrics_are_train_only"]=True

    # Frozen, transfer-only scientific prediction. Never fit/test-gate on
    # these rows. Persist row predictions privately for prompt-level paired
    # C0/C1 bootstrap AFTER both independent arms finish.
    from sklearn.metrics import balanced_accuracy_score
    heldout_tokens = tokenizer(transfer["text"].astype(str).tolist(),
        add_special_tokens=False, truncation=True,
        max_length=int(spec["max_length_before_eos"]))["input_ids"]
    heldout_tokens = [list(t)+[eos] for t in heldout_tokens]
    pred_id = []
    model.eval(); norms.eval(); plain.eval(); dropout.eval()
    with torch.inference_mode():
        for start in range(0,len(heldout_tokens),3):
            ix=np.arange(start,min(start+3,len(heldout_tokens)))
            encoded=_make_encoding(tokenizer,heldout_tokens,ix,device)
            z=_representation(model=model,norms=norms,layers=layers,
                              encoding=encoded,torch=torch)
            pred_id.extend(plain(z).argmax(1).detach().cpu().tolist())
    if len(pred_id)!=len(transfer):
        raise RuntimeError("incomplete heldout predictions")
    original=list(transfer["model"].astype(str))
    predicted=[labels[i] for i in pred_id]
    from collections import Counter
    class_totals=Counter(original)
    class_hit=Counter(y for y,p in zip(original,predicted) if y==p)
    recalls={label:class_hit[label]/class_totals[label] for label in
             sorted(class_totals)}
    transfer_score=float(balanced_accuracy_score(original,predicted))
    if len(recalls)<8: raise RuntimeError("too few transfer labels")
    if args.private_eval_output.exists():
        raise FileExistsError("refusing overwrite of private heldout predictions")
    args.private_eval_output.parent.mkdir(parents=True,exist_ok=True)
    private_rows=pd.DataFrame({
        "source":list(transfer["source"]),
        "prompt_group":list(transfer["prompt_id"]),
        "row_index":list(range(len(transfer))),
        "label":original,
        "prediction":predicted,
    })
    private_rows.to_csv(args.private_eval_output,index=False)
    # Aggregate score and per-class recalls may be published; no passage or
    # original IDs or prompt data are included in summary metadata.
    stats["transfer_source"]=RESERVED_SOURCE
    stats["transfer_labels_present"]=len(recalls)
    stats["transfer_balanced_accuracy"]=transfer_score
    stats["transfer_recall_by_label"]=recalls
    stats["transfer_prompt_groups"]=int(transfer.prompt_id.nunique())
    stats["source_transfer_only_not_public_18class_score"]=True

    # Save NEW checkpoint in private runtime only; training branch never
    # uploads weights/answers to public repo or public Actions artifacts.
    out=args.private_checkpoint
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing overwrite: {out}")
    out.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(out/"adapter")
    tokenizer.save_pretrained(out/"tokenizer")
    state={}
    for module_name,module in (
        ("norms",norms),("plain",plain),("centered",centered)):
        for k,v in module.state_dict().items():
            state[f"{module_name}.{k}"]=v.cpu().contiguous()
    save_file(state,str(out/"heads.safetensors"))
    (out/"meta.json").write_text(json.dumps({
       "candidate":CANDIDATE,"arm":args.arm,"labels":labels,
       "source_ids":source_ids,"source_pool_sha256":EXPECTED_POOL_SHA256,
       "source_transfer_reserved":RESERVED_SOURCE,
       "source_head_persisted":False,
       "base_stage1_sha256":loaded["manifest_sha256"],
       "provenance_manifest_sha256":hashlib.sha256(json.dumps(source_stats["dataset_sha256"],sort_keys=True).encode()).hexdigest(),
       "train_stats":source_stats,
       "run_stats":stats
    },indent=2,sort_keys=True)+"\n")
    # Source head deliberately discarded after Stage A checkpoint export.
    (out/"checkpoint.sha256").write_text(_sha256(out/"heads.safetensors")+"\n")
    from scripts.cda_l24_checkpoint import directory_manifest_digest
    stats["new_private_checkpoint_manifest_sha256"]=directory_manifest_digest(out)
    print("CDA_STAGE_A_NEW_PRIVATE_CHECKPOINT",json.dumps(stats,sort_keys=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--arm",choices=["C0","C1"],required=True)
    parser.add_argument("--arx-directory",type=Path,required=True)
    parser.add_argument("--trn-directory",type=Path,required=True)
    parser.add_argument("--kil-directory",type=Path,required=True)
    parser.add_argument("--prior-pool",type=Path,required=True)
    parser.add_argument("--original-a1",type=Path,required=True)
    parser.add_argument("--base-snapshot",type=Path,required=True)
    parser.add_argument("--private-checkpoint",type=Path,required=True)
    parser.add_argument("--private-eval-output",type=Path,required=True)
    args=parser.parse_args()
    _gpu_run(args)

if __name__=="__main__":
    main()
