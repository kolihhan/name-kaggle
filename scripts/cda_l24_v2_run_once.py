"""One actual Kaggle GPU C0/C1 completion -> private C1 model -> one official submission.

For a single consented, fixed candidate. This is a GitHub Actions runner
orchestrator; no model weights/raw prompts/official data enter public Git.
Each of the two phases has a separate Actions job and failure boundary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

C1_REF="kolihhan/ntl-cda-l24-v2-original-source-training-c1/2"
C0_REF="kolihhan/ntl-cda-l24-v2-original-source-training-c0-r1/1"
DATASET_REF="kolihhan/ntl-cda-l24-v2-c1-weights"
KERNEL_REF="kolihhan/ntl-cda-l24-v2-final"
COMPETITION="name-that-llm"
DESCRIPTION="One-shot independent CDA-L24 v2 source-adversarial LoRA24 20261108 no LB tuning"


def command(parts:list[str],*,check:bool=True)->str:
    p=subprocess.run(parts,capture_output=True,text=True)
    if check and p.returncode!=0:
        message="\n".join(
            x for x in (p.stdout+"\n"+p.stderr).splitlines()
            if re.search(r"error|fail|invalid|quota|permission|not found",x,re.I)
        )
        raise RuntimeError(f"command failed {parts[:3]}: {message[:1600]}")
    return (p.stdout+"\n"+p.stderr).strip()


def state(ref:str)->str:
    value=command(["kaggle","kernels","status",ref])
    result=re.search(r"KernelWorkerStatus\.([A-Z_]+)",value)
    if not result:
        raise RuntimeError(f"Unknown real Kaggle status for {ref}: {value[:300]}")
    return result.group(1)


def require_completed(refs:list[str],*,maximum_checks:int=150,seconds:int=120):
    for i in range(maximum_checks):
        observed={ref:state(ref) for ref in refs}
        print("REAL_CDA_KAGGLE_GPU_STATUS",
              i+1,{k.split("/")[-1]:v for k,v in observed.items()},flush=True)
        for ref,current in observed.items():
            if current in ("ERROR","FAILED","CANCELLED","CANCEL_ACKNOWLEDGED"):
                errors=command(["kaggle","kernels","logs",ref],check=False)
                relevant=[line for line in errors.splitlines()
                          if re.search(r"Traceback|Error|Exception|out of memory|missing|Failure",line,re.I)]
                print("GPU_FAILURE_LOG",ref,"\n".join(relevant[-20:])[:3800],flush=True)
                raise RuntimeError("locked Kaggle GPU experiment failed: "+ref)
        if all(x=="COMPLETE" for x in observed.values()):
            print("EXACT_C0_AND_C1_KAGGLE_RUNS_SUCCESSFULLY_COMPLETED",flush=True)
            return
        if i<maximum_checks-1:
            time.sleep(seconds)
    raise RuntimeError("GPU runs did not complete within this verifier job window; no submission attempted")


def _write_outputs(values:dict[str,str]):
    target=os.environ.get("GITHUB_OUTPUT")
    if not target:raise RuntimeError("GitHub Actions job output path missing")
    with open(target,"a") as f:
        for k,v in values.items():
            if not re.fullmatch(r"[a-z0-9_]+",k) or "\n" in v:
                raise ValueError("unsafe Actions output")
            f.write(k+"="+v+"\n")


def _get_private_model_dataset_ready():
    for i in range(100):
        raw=command(["kaggle","datasets","status",DATASET_REF,"--format","json"],check=False)
        try:
            j=json.loads(raw)
        except ValueError:
            j={}
        if j.get("status")=="ready" and int(j.get("current_version_number",0))==1:
            print("PRIVATE_C1_WEIGHT_DATASET_READY_V1",flush=True)
            return
        if i<99:time.sleep(10)
    raise RuntimeError("private C1 dataset not ready v1; final Notebook not authorized")


def stage_a():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Kaggle API authentication unavailable")
    require_completed([C1_REF,C0_REF])
    root=Path("/tmp/ntl-cda-once")
    root.mkdir(parents=True,exist_ok=True)
    for arm,ref in (("C0",C0_REF),("C1",C1_REF)):
        path=root/arm
        path.mkdir(parents=True,exist_ok=True)
        command(["kaggle","kernels","output",ref,"-p",str(path),"-o","-q"])
        print("PRIVATE_SOURCE_TRAIN_OUTPUT_RETRIEVED",arm,flush=True)
    from scripts.cda_l24_v2_finish_package import prepare
    summary=prepare(root/"C0",root/"C1",
                    root/"private-c1-dataset",root/"aggregate.json")
    if len(summary["new_checkpoint_directory_manifest_sha256"])!=64 or len(summary["source_manifest_sha256"])!=64:
        raise RuntimeError("model or source checkpoint not content-addressed")
    # A negative source-transfer finding remains a scientifically NEGATIVE
    # one-shot diagnostic. We still submit frozen C1 once as the user asked;
    # no selection of C0 or other training hyperparameter by public score.
    print("FROZEN_ONE_SHOT_C1_SUBMISSION_DECISION",
          summary["decision"],"no adaptive leaderboard tuning",flush=True)
    prior=command(["kaggle","datasets","status",DATASET_REF,"--format","json"],check=False)
    try:already=json.loads(prior).get("status")=="ready"
    except ValueError:already=False
    if already:raise RuntimeError("private C1 dataset already exists; refusing overwrite")
    response=command(["kaggle","datasets","create","-p",str(root/"private-c1-dataset")])
    print("PRIVATE_SIGNED_C1_MODEL_DATASET_CREATE_ACCEPTED",response[-250:],flush=True)
    _get_private_model_dataset_ready()
    _write_outputs({
        "checkpoint_sha":summary["new_checkpoint_directory_manifest_sha256"],
        "source_sha":summary["source_manifest_sha256"],
        "transfer_decision":summary["decision"],
    })
    print("CDA_V2_AGGREGATE_REPORT",root/"aggregate.json",flush=True)


def official_submissions()->list[dict]:
    csv_text=command(["kaggle","competitions","submissions",COMPETITION,"-v","-q"])
    return list(csv.DictReader(csv_text.splitlines()))


def verify_not_previous_submit():
    existing=[r for r in official_submissions() if r.get("description")==DESCRIPTION]
    if existing:raise RuntimeError("exact same official submission already exists: "+existing[0].get("ref",""))


def final(stage_a_checkpoint_sha:str,source_manifest_sha:str):
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("Kaggle API authentication unavailable")
    for value in (stage_a_checkpoint_sha,source_manifest_sha):
        if not re.fullmatch(r"[0-9a-f]{64}",value):
            raise ValueError("final model hash must be a full immutable SHA256")
    verify_not_previous_submit()
    prior=command(["kaggle","kernels","status",KERNEL_REF],check=False)
    if "KernelWorkerStatus." in prior:
        raise RuntimeError("final Kaggle Kernel slug already exists; refusing silent overwrite")
    from scripts.build_cda_l24_v2_kaggle_submission import write_kernel_bundle
    folder=Path("/tmp/ntl-cda-final-kernel")
    meta=write_kernel_bundle(folder,
         stage_a_sha256=stage_a_checkpoint_sha,
         source_sha256=source_manifest_sha,
         repo_root=Path("."))
    if (meta["competition_sources"]!=[COMPETITION] or
        meta["dataset_sources"]!=[
            DATASET_REF,"kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2"] or
        meta["enable_internet"] is not False or
        meta["model_sources"]!=["qwen-lm/qwen-3/transformers/8b-base/1"]):
        raise RuntimeError("final Code Competition source manifest drift")
    code=(folder/"kernel_main.py").read_text()
    compile(code,str(folder/"kernel_main.py"),"exec")
    print("CDA_V2_FINAL_INFERENCE_KERNEL_CHECKED",KERNEL_REF,flush=True)
    push=command(["kaggle","kernels","push","-p",str(folder),
                  "--accelerator","NvidiaTeslaT4","--timeout","32400"],check=False)
    if "Kernel version 1 successfully pushed" not in push:
        raise RuntimeError("No accepted final code notebook version 1: "+push[-700:])
    print("CDA_V2_FINAL_KAGGLE_CODE_COMPETITION_NOTEBOOK_PUSHED_V1",flush=True)
    require_completed([KERNEL_REF],maximum_checks=150,seconds=120)
    files=command(["kaggle","kernels","files",KERNEL_REF,"-v"])
    if "submission.parquet" not in files:
        raise RuntimeError("finished Kaggle kernel has no submission.parquet")
    # Freeze exactly completed notebook V1; never select based on LB.
    submit=command(["kaggle","competitions","submit",COMPETITION,
                    "-k",KERNEL_REF,"-v","1",
                    "-f","submission.parquet","-m",DESCRIPTION])
    matches=re.findall(r"Submission ref:\s*(\d+)",submit,re.I)
    if len(matches)!=1:
        raise RuntimeError("submission endpoint did not return unique accepted ref")
    official_ref=matches[0]
    print("CDA_V2_OFFICIAL_KAGGLE_SUBMISSION_ACCEPTED",official_ref,flush=True)
    rows=official_submissions()
    match=[x for x in rows if x.get("ref")==official_ref]
    if len(match)!=1 or match[0].get("description")!=DESCRIPTION:
        raise RuntimeError("official competition ledger not yet consistent with accepted ref")
    score=match[0].get("publicScore") or ""
    report={k:match[0].get(k) for k in
        ("ref","status","publicScore","privateScore","date","description")}
    report["reproduced_historical_best"]=0.668
    if score:
        report["delta_vs_0668"]=float(score)-.668
    print("CDA_V2_ONE_SHOT_VERIFIED_OFFICIAL_MY_SUBMISSIONS",json.dumps(report,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["stage-a","final"])
    p.add_argument("--stage-a-checkpoint-sha256")
    p.add_argument("--source-manifest-sha256")
    args=p.parse_args()
    if args.phase=="stage-a":
        stage_a()
    else:
        final(args.stage_a_checkpoint_sha256 or "",
              args.source_manifest_sha256 or "")


if __name__=="__main__":
    main()
