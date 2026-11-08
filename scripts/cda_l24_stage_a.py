"""CDA-L24: real LoRA24 Stage-A continuation trainer, frozen C0/C1.

This runner deliberately CANNOT train on a publisher CSV alone. A manually
approved and versioned source-clearance manifest must be supplied, checked
against the exact source-pool bytes and a source-disjoint heldout. It is a
science runner, NOT a Kaggle submission and does not use official labels
for parameter updates.

No original v1 train/inference code or checkpoint is modified.
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

EXPECTED_POOL_SHA256 = "1e4200aed80f52d655d4900a301cb427ca1341e06be31a8b7d5fde9ea3ef425f"
RESERVED_SOURCE = "arxitekton/ntl-claude-gen-v1"
SEED = 20261101
MIN_CROSSED_CLASSES = 8
AUTHORITATIVE_LABELS = 18
CHECKPOINT_ORIGIN = "two-stage-lora24-v1-A1"
CANDIDATE = "conditional-domain-adversarial-lora24-v1"
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


def validate_clearance(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Local claim check; the approval is NOT itself evidence of true origin.

    The independent reviewer must inspect the original upstream sources and
    fill this manifest before this runner may be used for science.
    """
    expected = {
        "candidate": CANDIDATE,
        "source_pool_sha256": EXPECTED_POOL_SHA256,
        "reserved_transfer_source": RESERVED_SOURCE,
        "source_lineage_reviewed": True,
        "original_terms_reviewed": True,
        "model_identity_reviewed": True,
        "overlap_reviewed": True,
        "provenance_status": "APPROVED",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise PermissionError(f"source clearance blocked: {key}")
    sources = manifest.get("approved_train_source_ids")
    if not isinstance(sources, list) or len(sources) < 2:
        raise PermissionError("at least two distinct approved train sources required")
    if len(sources) != len(set(sources)) or RESERVED_SOURCE in sources:
        raise PermissionError("train sources must be unique and exclude transfer source")
    entries = manifest.get("original_source_evidence")
    if not isinstance(entries, dict):
        raise PermissionError("original source evidence missing")
    for source in sources + [RESERVED_SOURCE]:
        evidence = entries.get(source)
        if not isinstance(evidence, dict):
            raise PermissionError(f"missing independent evidence: {source}")
        for key in ("origin_ref", "original_license", "license_evidence_url",
                    "model_versions_evidence_url", "generation_protocol_evidence_url",
                    "independent_source_lineage_evidence_url"):
            if not isinstance(evidence.get(key), str) or not evidence[key].strip():
                raise PermissionError(f"missing {source} evidence field {key}")
    return tuple(sorted(sources))


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


def _prepare_input(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    manifest = json.loads(args.clearance_manifest.read_text())
    allowed = validate_clearance(manifest)
    if _sha256(args.pool) != EXPECTED_POOL_SHA256:
        raise RuntimeError("public source pool SHA changed")
    from scripts.run_phase14_public_sparse import build_conservative_subset
    from scripts.run_two_stage_lora24_gate_a import build_stage1_split
    official = pd.read_csv(args.official_train)
    labels = sorted(official["model"].astype(str).unique())
    if len(labels) != AUTHORITATIVE_LABELS:
        raise RuntimeError("official label vocabulary changed")
    old = build_conservative_subset(pd.read_parquet(args.prior_pool))
    idx = build_stage1_split(old)["train_idx"]
    old_fit = old.iloc[idx]
    old_hashes = set(old_fit["text"].map(_text_hash))
    official_hashes = set(official["text"].map(_text_hash))
    old_keys = set(old_fit["prompt_key"].map(_normalized))
    pool = pd.read_parquet(args.pool)
    train, heldout, stats = build_clean_partitions(
        pool, labels=labels, approved_sources=allowed,
        old_fit_text_hashes=old_hashes,
        official_text_hashes=official_hashes,
        old_prompt_keys=old_keys,
    )
    if heldout["model"].nunique() < 8 or len(heldout) < 300:
        raise RuntimeError("frozen transfer source has inadequate qualified coverage")
    return train, heldout, labels, stats


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
       "provenance_manifest_sha256":_sha256(args.clearance_manifest),
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
    parser.add_argument("--pool",type=Path,required=True)
    parser.add_argument("--prior-pool",type=Path,required=True)
    parser.add_argument("--official-train",type=Path,required=True)
    parser.add_argument("--original-a1",type=Path,required=True)
    parser.add_argument("--base-snapshot",type=Path,required=True)
    parser.add_argument("--clearance-manifest",type=Path,required=True)
    parser.add_argument("--private-checkpoint",type=Path,required=True)
    parser.add_argument("--private-eval-output",type=Path,required=True)
    args=parser.parse_args()
    # Evaluate the EXACT clearance and source disjoint gates before loading
    # expensive model weights. No quick-run override.
    _prepare_input(args)
    _gpu_run(args)

if __name__=="__main__":
    main()
