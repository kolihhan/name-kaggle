#!/usr/bin/env python3
"""Owner-authorized, one-shot NTL official Kaggle code submission candidate.

Single change from authenticated Evgen v36 (the owner's 0.718 lineage):
confidence-conditioned fusion of existing main predictor and GSQ expert.
No train/test material committed or altered, no retraining, no LB tuning.
The 252-publisher-panel proof is spent; baseline same-run GSQ stage -2,
full pipeline historical control +1. Expected official Public BA 0.720;
kill if <=0.718 or scoring incomplete. One attempt only.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import httpx
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

KERNEL="kolihhan/ntl-confidence-fusion-official-once-20261125"
COMP="name-that-llm"
MARKER="ntl-confidence-fusion-v36-gsq-20261125-once"
SOURCE_SHA="5988d64922650bf94490810effb7278907b301c07f581efbc6d4865d84e63d52"
REFERENCE_ID="58755304"
FROZEN_REF_SCORE="0.718"
HELPER=Path("scripts/ntl_confidence_expert_fusion_20261125.py")

def cli(*cmd, timeout=150, check=True):
    p=subprocess.run(["kaggle",*cmd],capture_output=True,text=True,timeout=timeout)
    if check and p.returncode:
        # Deliberately avoid printing raw error text or credential-bearing URLs.
        raise RuntimeError("KAGGLE_CLI_FAILED:"+":".join(cmd[:2])+":EXIT"+str(p.returncode))
    return p

def rows():
    p=cli("competitions","submissions",COMP,"-v")
    return list(csv.DictReader(io.StringIO(p.stdout.lstrip("\ufeff"))))

def gate(api):
    items=rows()
    already=[r for r in items if MARKER in (r.get("description") or "")]
    if already:
        print("NTL_CONFIDENCE_OFFICIAL_ALREADY_SUBMITTED="+json.dumps(
            [{k:r.get(k) for k in ("ref","status","publicScore")} for r in already]),flush=True)
        return False
    champions=[r for r in items if r.get("ref")==REFERENCE_ID]
    assert len(champions)==1 and champions[0].get("publicScore")==FROZEN_REF_SCORE
    assert "COMPLETE" in champions[0].get("status","")
    running=[r.get("ref") for r in items if r.get("status","").upper().split(".")[-1] in
             ("PENDING","RUNNING","QUEUED","EVALUATING","SCORING")]
    lim=api.competition_get_submission_limits(COMP)
    p=cli("quota","--format","json")
    q=json.loads(p.stdout)
    rem=[float(str(x["remaining"]).rstrip("h")) for x in q if x.get("resource")=="GPU"]
    assert len(rem)==1
    out={"allowed_now":int(lim.num_allowed_now),"limited_by_total":bool(lim.limited_by_total),
         "running":running,"gpu_hours":rem[0],"champion":"0.718","reference_id":REFERENCE_ID}
    print("NTL_CONFIDENCE_OFFICIAL_GATE="+json.dumps(out,sort_keys=True),flush=True)
    if int(lim.num_allowed_now)<2 or bool(lim.limited_by_total) or running or rem[0]<9.5:
        print("NTL_CONFIDENCE_OFFICIAL_NOT_SAFE_TO_LAUNCH=TRUE",flush=True)
        return False
    return True

def historical_source():
    with httpx.Client(follow_redirects=True,timeout=60,
                      headers={"Authorization":"Bearer "+os.environ["KAGGLE_API_TOKEN"]}) as s:
        s.get("https://www.kaggle.com")
        token=s.cookies.get("XSRF-TOKEN")
        assert token
        h={"Authorization":"Bearer "+os.environ["KAGGLE_API_TOKEN"],
           "Content-Type":"application/json","X-XSRF-TOKEN":token}
        def post(route,body):
            r=s.post("https://www.kaggle.com/api/i/"+route,json=body,headers=h)
            r.raise_for_status()
            return r
        v=post("kernels.LegacyKernelsService/GetKernelViewModel",
               {"authorUserName":"evgendvorkin",
                "kernelSlug":"name-that-llm-deep-forensic-eda-with-analogie","tab":"output"}).json()
        es=post("kernels.KernelsService/ListKernelVersions",
                {"kernelId":int(v["kernel"]["id"]),"sortOption":"VERSION_ID","pageSize":200}).json()["items"]
        found=[e for e in es if (e.get("version") or {}).get("versionNumber")==36]
        assert len(found)==1
        raw=post("kernels.KernelsService/GetKernelSessionSource",
                 {"kernelSessionId":int(found[0]["run"]["id"]),
                  "includeOutputIfAvailable":False}).text
        assert hashlib.sha256(raw.encode()).hexdigest()==SOURCE_SHA
        nb=json.loads(raw)
        assert len(nb["cells"])==58
    return nb

def build(folder):
    nb=historical_source()
    originals=["".join(c.get("source",[])) for c in nb["cells"]]
    found=[i for i,s in enumerate(originals) if s.startswith("%%writefile infer.py\n")]
    assert len(found)==1
    i=found[0]
    header,body=originals[i].split("\n",1)
    anchor="            P = (1 - a.gsq_share) * P + a.gsq_share * Q27"
    point="if a.gsq_share > 0 and a.gsq_dir:"
    assert body.count(anchor)==body.count(point)==1
    helper=HELPER.read_text()
    nodes=ast.parse(helper).body
    funcs=[ast.get_source_segment(helper,n) for n in nodes
           if isinstance(n,ast.FunctionDef) and n.name in ("validate","fuse")]
    assert len(funcs)==2
    code="\n\n".join(funcs)
    code=code.replace("def validate(", "def _ntl_validate(")
    code=code.replace("p,q=validate(p,q)", "p,q=_ntl_validate(p,q)")
    code=code.replace("def fuse(", "def _ntl_fuse(")
    # Same tested, label-blind fusion from the real252 run. The diagnostic
    # sidecar used for the spent 252 panel is NOT included in official code.
    injected="\n".join(" "*12+s for s in [
        "if abs(a.gsq_share - 0.03) > 1e-12: raise ValueError('GSQ share drift')",
        "_fixed, _adaptive, _gate = _ntl_fuse(P, Q27)",
        "_original = (1 - a.gsq_share) * P + a.gsq_share * Q27",
        "assert np.allclose(_original, _fixed, atol=1e-7, rtol=1e-6)",
        "_new = _original.copy()",
        "_new[_gate] = _adaptive[_gate].astype(_original.dtype, copy=False)",
        "P = _new",
    ])
    changed=body.replace(point,code+"\n\n"+point,1).replace(anchor,injected,1)
    compile(changed,"infer.py","exec")
    # A local two-row identical numerical sanity check, not an accuracy test.
    fixture={"np":np}
    exec(code,fixture)
    p=np.full((2,18),.01,dtype=np.float32);p[:,0]=.44;p[:,1]=.40
    q=p.copy();q[0,0]=.20;q[0,1]=.64
    a,b,m=fixture["_ntl_fuse"](p,q)
    assert m.tolist()==[True,False] and np.argmax(b,axis=1).tolist()==[1,0]
    assert np.allclose(a,.97*p+.03*q)
    nb["cells"][i]["source"]=(header+"\n"+changed).splitlines(keepends=True)
    assert sum("".join(c.get("source",[]))!=s for c,s in zip(nb["cells"],originals))==1

    with tempfile.TemporaryDirectory(prefix="ntl-fusion-donor-") as d:
        cli("kernels","pull","evgendvorkin/name-that-llm-deep-forensic-eda-with-analogie",
            "-p",d,"-m")
        path=Path(d)/"kernel-metadata.json"
        meta=json.loads(path.read_text())
    assert meta["competition_sources"]==[COMP]
    assert [len(meta[k]) for k in ("model_sources","dataset_sources","kernel_sources")]==[9,26,10]
    assert meta.get("enable_gpu") is True
    meta["id"]=KERNEL
    meta["title"]="NTL frozen confidence GSQ fusion 20261125"
    meta["is_private"]=True
    meta["enable_gpu"]=True
    meta["enable_internet"]=False
    meta["code_file"]="ntl_confidence_fusion_official_20261125.ipynb"
    folder.mkdir(parents=True,exist_ok=False)
    data=json.dumps(nb,ensure_ascii=False,indent=1).encode("utf-8")
    (folder/meta["code_file"]).write_bytes(data)
    (folder/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
    print("NTL_CONFIDENCE_OFFICIAL_CANDIDATE_PIN="+json.dumps({
        "notebook_sha256":hashlib.sha256(data).hexdigest(),
        "source_sha256":SOURCE_SHA,"changed_source_cells":1,
        "original_cells":len(originals),"competition_source":COMP,
        "model_mounts":9,"dataset_mounts":26,"kernel_mounts":10,
        "no_publisher_252_panel":True,"official_output":"submission.parquet",
        "expected_public_ba":0.720,"kill_at_or_below":0.718
    },sort_keys=True),flush=True)

def execute():
    assert os.environ.get("KAGGLE_API_TOKEN")
    api=KaggleApi();api.authenticate()
    if not gate(api):return
    status=cli("kernels","status",KERNEL,check=False)
    if status.returncode==0:
        # Never overwrite or repush a previously created candidate.
        print("NTL_CONFIDENCE_OFFICIAL_CANDIDATE_ALREADY_EXISTS_NO_REPUSH=TRUE",flush=True)
    else:
        with tempfile.TemporaryDirectory(prefix="ntl-fusion-official-") as d:
            build(Path(d)/"kernel")
            push=cli("kernels","push","-p",str(Path(d)/"kernel"),
                     "--accelerator","NvidiaTeslaT4","--timeout","32400",timeout=420,check=False)
            if push.returncode:
                print("NTL_CONFIDENCE_OFFICIAL_PUSH_FAILED=EXIT"+str(push.returncode),flush=True)
                raise RuntimeError("new scored-path kernel push rejected")
            print("NTL_CONFIDENCE_OFFICIAL_KERNEL_PUSH_ACCEPTED=TRUE",flush=True)
    p=cli("kernels","status",KERNEL,check=False)
    if p.returncode:
        raise RuntimeError("new candidate kernel inaccessible")
    states=re.findall(r"KernelWorkerStatus\.[A-Z_]+",p.stdout+p.stderr)
    state=states[-1] if states else "UNKNOWN"
    print("NTL_CONFIDENCE_OFFICIAL_PRE_SUBMIT_WORKER="+state,flush=True)
    if state.endswith("ERROR"):
        raise RuntimeError("pre-submission notebook failed")
    # Attempt a direct code competition submission on the immutable v1.
    # If the Kaggle host requires notebook COMPLETE first, this will be a
    # separate blocked state, not misreported as an accepted submission.
    # GPU quota was checked before the private push; here require an official
    # slot, not an additional 9.5h of unreserved development GPU.
    remaining=api.competition_get_submission_limits(COMP)
    ledger=rows()
    pending=[q.get("ref") for q in ledger if (q.get("status") or "").upper().split(".")[-1] in
             ("PENDING","RUNNING","QUEUED","EVALUATING","SCORING")]
    if int(remaining.num_allowed_now)<2 or bool(remaining.limited_by_total) or pending:
        print("NTL_CONFIDENCE_OFFICIAL_SUBMIT_BLOCKED_QUOTA_OR_ACTIVE=TRUE",flush=True)
        return
    p=cli("competitions","submit",COMP,"-f","submission.parquet","-k",KERNEL,"-v","1",
          "-m",MARKER,timeout=240,check=False)
    if p.returncode:
        category=("KERNEL_NOT_COMPLETE" if any(w in (p.stderr+p.stdout).lower()
                   for w in ("complete","running","ready","finished")) else "KAGGLE_REJECTED")
        print("NTL_CONFIDENCE_OFFICIAL_SUBMIT_ATTEMPT_BLOCKED="+category,flush=True)
        # Do not retry blindly; the frozen source remains a private kernel.
        return
    print("NTL_CONFIDENCE_OFFICIAL_SUBMIT_ACCEPTED=TRUE",flush=True)
    ms=re.findall(r"Submission ref:\s*(\d+)",p.stdout)
    if ms:print("NTL_CONFIDENCE_OFFICIAL_SUBMISSION_REF="+ms[-1],flush=True)
    matches=[r for r in rows() if MARKER in (r.get("description") or "")]
    if matches:
        print("NTL_CONFIDENCE_OFFICIAL_RECEIPT="+json.dumps({
            k:matches[0].get(k) for k in ("ref","date","status","publicScore")
        }),flush=True)
    else:
        print("NTL_CONFIDENCE_OFFICIAL_RECEIPT_PENDING_DONT_AUTORETRY=TRUE",flush=True)

if __name__=="__main__":
    execute()
