"""One official NTL code-submission test: the EXISTING, COMPLETE private
0.715 Estopadilla-v5 source + final joint-assignment cell. No kernel repush.

Research variable: use weak official-train char SVC score vectors + blind
lexical prompt grouping to reconcile a strong model's final hard labels.
Hypothesis: hidden Public BA >=0.725 versus frozen 0.715. Kill if <=0.715,
incomplete, or no actual prediction changes. No adaptive parameter tuning.

The Kaggle host re-runs the *same* existing kernel on ~4000 hidden rows;
the 3-row interactive preview is not the hidden scorer. This script checks
source provenance, completion, account-specific quota and duplicate markers
BEFORE a single official code-submission action. It never starts a new private
development GPU kernel. It never edits baseline or winner.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

COMP="name-that-llm"
KERNEL="kolihhan/ntl-0-715-label-only-joint-decoder-research"
MARKER="ntl-0715-joint-label-hard-anchor-once-20261122"
CHAMPION_REF="58680531"
CHAMPION_SCORE="0.715"
EXPECTED_BASE_SHA="202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8"
EXPECTED_ADDED_RUNTIME_SHA="70d4b95eef7900c99927967ab63d60b6f20c21537b7090515a1de78893ba08d6"
EXPECTED_COUNTS={"model_sources":9,"dataset_sources":26,"kernel_sources":10}
EXPECTED_CELLS=33


def run_cli(*argv: str,timeout:int=180)->str:
    p=subprocess.run(["kaggle",*argv],capture_output=True,
                     text=True,timeout=timeout)
    if p.returncode:
        # Never expose the raw CLI error which may contain credential-bearing URLs.
        kind=("PERMISSION_DENIED" if ("denied" in p.stderr.lower() or "access" in p.stderr.lower())
              else "RESOURCE_UNAVAILABLE_OR_KAGGLE_CLI_ERROR")
        sanitized=re.sub(r"(?:KGAT_[A-Za-z0-9_-]+|Bearer\\s+\\S+|https?://\\S+)", "[REDACTED]", (p.stderr or p.stdout)[-400:]).replace("\\n"," ")
        raise RuntimeError(f"KAGGLE_{argv[0]}_{argv[1] if len(argv)>1 else ''}:{kind}:EXIT{p.returncode}:{sanitized[:280]}")
    return p.stdout


def source_verified()->None:
    stat=run_cli("kernels","status",KERNEL,timeout=80)
    if "KernelWorkerStatus.COMPLETE" not in stat:
        raise RuntimeError("PRIVATE_KERNEL_NOT_COMPLETE")
    with tempfile.TemporaryDirectory(prefix="ntl-0715-provenance-") as d:
        dest=Path(d)
        run_cli("kernels","pull",KERNEL,"-p",str(dest),"-m",timeout=120)
        p=dest/"kernel-metadata.json"
        if not p.exists():raise RuntimeError("PRIVATE_KERNEL_METADATA_MISSING")
        meta=json.loads(p.read_text(encoding="utf8"))
        if meta.get("id")!=KERNEL:
            raise RuntimeError("KAGGLE_NORMALIZED_KERNEL_SLUG_CHANGED")
        name=Path(str(meta.get("code_file",""))).name
        if not name.endswith(".ipynb"):raise RuntimeError("NOT_IPYNB")
        nb=json.loads((dest/name).read_text(encoding="utf8"))
        cells=nb.get("cells",[])
        if len(cells)!=EXPECTED_CELLS:raise RuntimeError("KAGGLE_CELL_COUNT_DRIFT")
        original="\n".join("".join(c.get("source",[])) for c in cells[:-1]
                           if c.get("cell_type")=="code")
        digest=hashlib.sha256(original.encode()).hexdigest()
        if digest!=EXPECTED_BASE_SHA:
            raise RuntimeError("SCORED_0715_BASE_SOURCE_MISMATCH")
        final="".join(cells[-1].get("source",[]))
        expr=ast.parse(final)
        constants=[node.value.value for node in expr.body
                   if isinstance(node,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=="runtime_code"
                           for t in node.targets)
                   and isinstance(node.value,ast.Constant)
                   and isinstance(node.value.value,str)]
        if len(constants)!=1:
            raise RuntimeError("APPENDED_RUNTIME_NOT_EXACTLY_ONE_STRING")
        runtime_hash=hashlib.sha256(constants[0].encode()).hexdigest()
        if runtime_hash!=EXPECTED_ADDED_RUNTIME_SHA:
            raise RuntimeError("APPENDED_JOINT_RUNTIME_SHA_MISMATCH")
        if 'NTL_SCORER_DATA_DIR' not in final or 'FAIL_OPEN_KEEP_SCORED_OUTPUT' not in final:
            raise RuntimeError("NO_SAFE_RUNTIME_DATA_ORIGINAL_FALLBACK")
        for k,v in EXPECTED_COUNTS.items():
            if len(meta.get(k) or [])!=v:
                raise RuntimeError(f"RESOURCE_MOUNTS_CHANGED_{k}")
        if str(meta.get("is_private")).lower() not in ("true","1"):
            raise RuntimeError("RESEARCH_NOTEBOOK_IS_NOT_PRIVATE")
        print("NTL_0715_JOINT_PINNED_SOURCE="+json.dumps({
            "scored_original_32_cells_sha256":digest,
            "added_jointer_runtime_sha256":runtime_hash,
            "cells":len(cells),"private":True,
            "kernel_ref":KERNEL,"mounts":EXPECTED_COUNTS
        },sort_keys=True),flush=True)


def latest_submission_rows():
    csvtext=run_cli("competitions","submissions",COMP,"-v",timeout=90)
    return list(csv.DictReader(io.StringIO(csvtext.lstrip("\ufeff"))))


def check_once()->bool:
    from kaggle.api.kaggle_api_extended import KaggleApi
    rows=latest_submission_rows()
    prior=[r for r in rows if MARKER in (r.get("description") or "")]
    if len(prior)>1:raise RuntimeError("DUPLICATE_MARKER_ALREADY_IN_LEDGER")
    if prior:
        print("NTL_0715_JOINT_EXISTING_OFFICIAL_RECEIPT="+json.dumps({
            k:prior[0].get(k) for k in ("ref","status","publicScore","date")
        },sort_keys=True),flush=True)
        return False
    score=[r for r in rows if r.get("ref")==CHAMPION_REF]
    if len(score)!=1 or score[0].get("publicScore")!=CHAMPION_SCORE or (
            "COMPLETE" not in (score[0].get("status") or "")):
        raise RuntimeError("EXACT_0715_WINNER_MISSING_OR_INCOMPLETE")
    active=[
      r.get("ref") for r in rows if
      (r.get("status") or "").upper().split(".")[-1] in
       {"PENDING","QUEUED","RUNNING","EVALUATING","SCORING"}]
    api=KaggleApi()
    api.authenticate()
    quota=api.competition_get_submission_limits(COMP)
    stats={
       "allowed_now":int(quota.num_allowed_now),
       "limited_by_total":bool(quota.limited_by_total),
       "num_today":getattr(quota,"num_today",None),
       "active_refs":active,
       "reserve_one_slot":True
    }
    print("NTL_0715_JOINT_OFFICIAL_SLOT="+json.dumps(stats,sort_keys=True),flush=True)
    if int(quota.num_allowed_now)<2 or bool(quota.limited_by_total) or active:
        print("NTL_0715_JOINT_OFFICIAL_DECISION=NO_SAFE_SLOT_NO_SUBMISSION",
              flush=True)
        return False
    return True


def submit_exactly_once()->None:
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN_REQUIRED")
    source_verified()
    if not check_once():return
    # No download/repush: Kaggle code-submission API references the actual
    # owner-private COMPLETE notebook and reexecutes against hidden rerun.
    response=run_cli("competitions","submit",COMP,
       "-f","submission.parquet","-k",KERNEL,"-v","1",
       "-m",MARKER,timeout=240)
    print("NTL_0715_JOINT_SUBMISSION_CLI_ACCEPTED="+str(
       "Submission ref:" in response),flush=True)
    refmatch=re.findall(r"Submission ref:\s*(\d+)",response)
    if refmatch:
        print("NTL_0715_JOINT_OFFICIAL_REF="+refmatch[-1],flush=True)
    for i in range(5):
        latest=latest_submission_rows()
        records=[r for r in latest if MARKER in (r.get("description") or "")]
        if records:
            print("NTL_0715_JOINT_AUTHENTICATED_RECEIPT="+json.dumps({
                k:records[0].get(k)
                for k in ("ref","date","status","publicScore","description")
            },sort_keys=True),flush=True)
            return
        if i<4:time.sleep(12)
    print("NTL_0715_JOINT_OFFICIAL_RECEIPT_UNCONFIRMED_DO_NOT_AUTORETRY=TRUE",
          flush=True)


if __name__=="__main__":
    submit_exactly_once()
