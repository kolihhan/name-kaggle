"""CDA-L24 Stage-B: fold-local official adaptation of a promoted frozen C1.

No fit on competition test labels; never evaluates Stage-B until source
transfer gate has passed and private Stage-A checkpoint hash is locked.
Each of the five folds runs in its own process to avoid GPU memory reuse.
This does NOT submit anything to Kaggle.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from scripts.cda_l24_checkpoint import load_cda_stage1_checkpoint
from scripts.run_strong_sparse_anchor import load_train
from scripts.run_two_stage_lora24_gate_a import (
    build_prompt_triplets,compute_class_weights
)
from scripts.run_two_stage_lora24_gate_b import (
    gate_b_training_spec,split_official_fold,
)
from scripts.run_two_stage_lora24_gate_b_gpu import (
    _seed_all,_tokenize_official,train_two_stage2_epochs,
    _infer_validation,validate_fold_predictions,
)

HISTORIC_OWN_OOF=0.5467611235
MIN_GROUPED_OOF_GAIN=0.010
LABELS_COUNT=18
ROWS_COUNT=2699
FOLDS=(0,1,2,3,4)


def assert_promoted_stage_a(path:Path)->dict:
    r=json.loads(path.read_text())
    if r.get("decision")!="PROMOTE_TO_GROUPED_OFFICIAL_OOF":
        raise PermissionError("Stage A scientific promotion gate not passed")
    if (r.get("C1_minus_C0",-100)<.02 or
        r.get("prompt_bootstrap_CI95",[0])[0]<=0 or
        r.get("provider_macro_delta",-100)<-.01 or
        r.get("eligible_true_labels",0)<8):
        raise PermissionError("Stage A aggregate metrics inconsistent with promotion")
    if r.get("official_competition_OOF") is not False:
        raise PermissionError("not genuine independent Stage A source evidence")
    return r


def fold_worker(args):
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CDA Stage B requires CUDA")
    assert_promoted_stage_a(args.stage_a_evidence)
    if args.fold not in FOLDS:
        raise ValueError("five frozen folds only")
    _seed_all(int(gate_b_training_spec()["seed"]),torch)
    device=torch.device("cuda:0")
    frame=load_train(args.official_train,args.folds_file).reset_index(drop=True)
    labels=sorted(frame.model.astype(str).unique().tolist())
    if len(frame)!=ROWS_COUNT or len(labels)!=LABELS_COUNT:
        raise RuntimeError("official dataset identity changed")
    fit_idx,valid_idx=split_official_fold(frame,args.fold)
    fit=frame.iloc[fit_idx]
    valid=frame.iloc[valid_idx]
    assert not set(fit.prompt_id.astype(str))&set(valid.prompt_id.astype(str))
    checkpoint=load_cda_stage1_checkpoint(
        checkpoint_dir=args.stage_a_checkpoint,
        snapshot_path=args.base_snapshot,
        official_labels=labels,device=device,arm="C1",
        expected_sha256=args.stage_a_checkpoint_sha256,
        clearance_sha256=args.source_clearance_sha256)
    tokenizer=checkpoint["tokenizer"]
    spec=gate_b_training_spec()
    tokens=_tokenize_official(frame,tokenizer,
        max_length_before_eos=int(spec["max_length_before_eos"]))
    target_ids={l:i for i,l in enumerate(labels)}
    y=frame.model.astype(str).map(target_ids).to_numpy(dtype=np.int64)
    weights=compute_class_weights(fit.model.astype(str).values,classes=labels)
    w=torch.tensor([weights[l] for l in labels],device=device,dtype=torch.float32)
    triplets=build_prompt_triplets(frame,fit_idx,seed=int(spec["seed"]))
    training=train_two_stage2_epochs(
       frame=frame,fit_idx=fit_idx,triplets=triplets,tokens=tokens,
       y=y,class_weights=w,model=checkpoint["model"],
       tokenizer=tokenizer,norms=checkpoint["norms"],plain=checkpoint["plain"],
       centered=checkpoint["centered"],dropout=checkpoint["dropout"],
       layers=checkpoint["layers"],device=device,torch=torch,F=F)
    predictions=_infer_validation(
       frame=frame,valid_idx=valid_idx,tokens=tokens,
       model=checkpoint["model"],tokenizer=tokenizer,
       norms=checkpoint["norms"],plain=checkpoint["plain"],
       centered=checkpoint["centered"],dropout=checkpoint["dropout"],
       layers=checkpoint["layers"],labels=labels,device=device,torch=torch,
       seed=int(spec["seed"]))
    validate_fold_predictions(valid,predictions,labels=labels,fold=args.fold)
    ba=float(balanced_accuracy_score(
        predictions["model"],predictions["plain_prediction"]))
    root=args.private_oof_directory
    root.mkdir(parents=True,exist_ok=True)
    pred_path=root/f"fold_{args.fold}.csv"
    rep_path=root/f"fold_{args.fold}.json"
    if pred_path.exists() or rep_path.exists():
        raise FileExistsError("refusing to overwrite locked fold output")
    predictions.to_csv(pred_path,index=False)
    rep_path.write_text(json.dumps({
       "candidate":"conditional-domain-adversarial-lora24-v1",
       "checkpoint_arm":"C1",
       "fold":args.fold,
       "trained_on_official_validation":False,
       "source_transfer_precondition":True,
       "official_rows":len(frame),"validation_rows":len(valid),
       "fit_prompts":fit.prompt_id.astype(str).nunique(),
       "validation_prompts":valid.prompt_id.astype(str).nunique(),
       "plain_balanced_accuracy":ba,
       "checkpoint_digest":args.stage_a_checkpoint_sha256,
       "training_metadata":training,
    },sort_keys=True,indent=2)+"\n")
    print("CDA_STAGE_B_FOLD_FINISHED",json.dumps({"fold":args.fold,"ba":ba}))


def orchestrate(args):
    assert_promoted_stage_a(args.stage_a_evidence)
    args.private_oof_directory.mkdir(parents=True,exist_ok=True)
    # Persistent check of exact Stage-A checkpoint bytes BEFORE GPU start.
    from scripts.cda_l24_checkpoint import validate_cda_checkpoint
    frame=load_train(args.official_train,args.folds_file).reset_index(drop=True)
    labels=sorted(frame.model.astype(str).unique().tolist())
    validate_cda_checkpoint(args.stage_a_checkpoint,labels=labels,
        expected_sha256=args.stage_a_checkpoint_sha256,arm="C1",
        expected_clearance_sha256=args.source_clearance_sha256)
    for fold in FOLDS:
        cmd=[sys.executable,"-m","scripts.cda_l24_stage_b",
             "--fold",str(fold),
             "--official-train",str(args.official_train),
             "--folds-file",str(args.folds_file),
             "--stage-a-checkpoint",str(args.stage_a_checkpoint),
             "--stage-a-checkpoint-sha256",args.stage_a_checkpoint_sha256,
             "--source-clearance-sha256",args.source_clearance_sha256,
             "--stage-a-evidence",str(args.stage_a_evidence),
             "--base-snapshot",str(args.base_snapshot),
             "--private-oof-directory",str(args.private_oof_directory)]
        subprocess.run(cmd,check=True)
    files=[args.private_oof_directory/f"fold_{f}.csv" for f in FOLDS]
    if not all(p.is_file() for p in files):
        raise RuntimeError("one or more frozen fold outputs missing")
    combined=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    if len(combined)!=ROWS_COUNT or combined.id.duplicated().any():
        raise RuntimeError("fold outputs do not cover official rows exactly once")
    expect=load_train(args.official_train,args.folds_file).reset_index(drop=True)
    joined=combined.set_index("id").reindex(expect.id.values)
    if joined.isna().any().any() or joined.model.tolist()!=expect.model.tolist():
        raise RuntimeError("CDA Stage-B official label alignment failure")
    ba=float(balanced_accuracy_score(
        joined.model,joined.plain_prediction))
    if ba+1e-10<HISTORIC_OWN_OOF+MIN_GROUPED_OOF_GAIN:
        decision="REJECT_CDA_STAGE_B_OOF"
    else:
        # This is conditional; final authorization still requires the paired
        # prompt-bootstrap and no-provider/label-loss guardrails in CONTRACT.
        decision="PENDING_PAIRED_REGRESSION_AND_PROMPT_BOOTSTRAP"
    report={
      "decision":decision,
      "candidate":"conditional-domain-adversarial-lora24-v1",
      "stage_a_checkpoint_sha256":args.stage_a_checkpoint_sha256,
      "folds_complete":list(FOLDS),
      "official_rows":len(combined),
      "candidate_historical_grouped_OOF_BA":ba,
      "own_historical_LoRA24_grouped_OOF_BA":HISTORIC_OWN_OOF,
      "absolute_delta_vs_historical":ba-HISTORIC_OWN_OOF,
      "historical_OOF_reused_and_spent":True,
      "fully_promoted_to_Kaggle":False,
      "new_kaggle_submission":False,
    }
    (args.private_oof_directory/"aggregate.json").write_text(
       json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("CDA_STAGE_B_AGGREGATE",json.dumps(report,sort_keys=True))


def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--orchestrate",action="store_true")
    mode.add_argument("--fold",type=int)
    p.add_argument("--official-train",type=Path,required=True)
    p.add_argument("--folds-file",type=Path,required=True)
    p.add_argument("--stage-a-checkpoint",type=Path,required=True)
    p.add_argument("--stage-a-checkpoint-sha256",required=True)
    p.add_argument("--source-clearance-sha256",required=True)
    p.add_argument("--stage-a-evidence",type=Path,required=True)
    p.add_argument("--base-snapshot",type=Path,required=True)
    p.add_argument("--private-oof-directory",type=Path,required=True)
    args=p.parse_args()
    if args.orchestrate:orchestrate(args)
    else:fold_worker(args)

if __name__=="__main__":main()
