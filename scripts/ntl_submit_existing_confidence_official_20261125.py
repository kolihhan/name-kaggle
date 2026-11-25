#!/usr/bin/env python3
"""One submission attempt of already-pushed immutable official confidence kernel."""
from __future__ import annotations
import csv,io,json,re,tempfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from ntl_confidence_fusion_official_once_20261125 import (
    COMP,MARKER,REFERENCE_ID,FROZEN_REF_SCORE,cli,rows,historical_source
)
KERNEL="kolihhan/ntl-frozen-confidence-gsq-fusion-20261125"

def submit():
    with tempfile.TemporaryDirectory(prefix="ntl-fusion-submission-inspect-") as d:
        cli("kernels","pull",KERNEL,"-p",d,"-m")
        meta=json.loads((Path(d)/"kernel-metadata.json").read_text())
        nb=json.loads(next(Path(d).glob("*.ipynb")).read_text())
    orig=historical_source()
    a=["".join(x.get("source",[])) for x in nb["cells"]]
    b=["".join(x.get("source",[])) for x in orig["cells"]]
    assert len(a)==len(b)==58
    changed=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    assert len(changed)==1 and a[changed[0]].startswith("%%writefile infer.py\n")
    src=a[changed[0]]
    assert src.count("_ntl_fuse(P, Q27)")==1
    assert src.count("P = _new")==1
    assert src.count('if a.gsq_share > 0 and a.gsq_dir:')==1
    assert meta.get("is_private") is True
    assert meta.get("enable_gpu") is True and meta.get("enable_internet") is False
    assert meta.get("competition_sources")==[COMP]
    assert [len(meta[k]) for k in ("model_sources","dataset_sources","kernel_sources")]==[9,26,10]
    assert "nischaydnk/ntl-topic-synthetic-18llms" not in meta["dataset_sources"]
    print("NTL_FUSION_OFFICIAL_REMOTE_SOURCE_VERIFIED=TRUE",flush=True)
    api=KaggleApi();api.authenticate()
    ledger=rows()
    matches=[r for r in ledger if MARKER in (r.get("description") or "")]
    if matches:
        print("NTL_FUSION_OFFICIAL_EXISTING_SUBMISSION="+json.dumps(
          [{k:r.get(k) for k in ("ref","date","status","publicScore")} for r in matches]),flush=True)
        return
    controls=[r for r in ledger if r.get("ref")==REFERENCE_ID]
    assert len(controls)==1 and controls[0].get("publicScore")==FROZEN_REF_SCORE
    busy=[r.get("ref") for r in ledger if r.get("status","").upper().split(".")[-1]
          in ("PENDING","QUEUED","RUNNING","SCORING","EVALUATING")]
    limit=api.competition_get_submission_limits(COMP)
    print("NTL_FUSION_OFFICIAL_SUBMIT_GATE="+json.dumps({
        "allowed_now":int(limit.num_allowed_now),"busy":busy,
        "limited_by_total":bool(limit.limited_by_total)}),flush=True)
    if int(limit.num_allowed_now)<2 or limit.limited_by_total or busy:
        print("NTL_FUSION_OFFICIAL_SUBMISSION_NOT_SAFE=TRUE",flush=True);return
    status=cli("kernels","status",KERNEL,check=False)
    if status.returncode:raise RuntimeError("OFFICIAL_KERNEL_NOT_FOUND")
    states=re.findall(r"KernelWorkerStatus\.[A-Z_]+",status.stdout+status.stderr)
    assert states
    print("NTL_FUSION_OFFICIAL_KERNEL_STATUS="+states[-1],flush=True)
    if states[-1].endswith("ERROR"):
        print("NTL_FUSION_OFFICIAL_KERNEL_ERROR_NO_SUBMISSION=TRUE",flush=True);return
    p=cli("competitions","submit",COMP,"-f","submission.parquet",
          "-k",KERNEL,"-v","1","-m",MARKER,timeout=240,check=False)
    if p.returncode:
        # Sanitize source error, but print selected innocuous categories.
        msg=(p.stdout+" "+p.stderr).lower()
        category=("KERNEL_NOT_READY" if any(z in msg for z in
            ("kernel must","not complete","not finished","running","not ready","queued"))
            else "KAGGLE_SUBMIT_REJECTED")
        print("NTL_FUSION_OFFICIAL_SUBMIT_BLOCKED="+category,flush=True)
        return
    print("NTL_FUSION_OFFICIAL_SUBMIT_CLI_ACCEPTED=TRUE",flush=True)
    m=re.findall(r"Submission ref:\s*(\d+)",p.stdout)
    if m:print("NTL_FUSION_OFFICIAL_SUBMISSION_ID="+m[-1],flush=True)
    matches=[r for r in rows() if MARKER in (r.get("description") or "")]
    if matches:
        print("NTL_FUSION_OFFICIAL_LEDGER_RECEIPT="+json.dumps({
           k:matches[0].get(k) for k in ("ref","date","status","publicScore")
        }),flush=True)
    else:
        print("NTL_FUSION_OFFICIAL_RECEIPT_NOT_YET_VISIBLE_DO_NOT_AUTORETRY=TRUE",flush=True)

if __name__=="__main__":submit()
