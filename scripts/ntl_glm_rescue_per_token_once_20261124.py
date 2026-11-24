"""Build and run ONE private, SHA-pinned NTL candidate, then submit once if ready.

No scheduling, new training, public competition data or paid compute. The only
scientific change to the public parent is the existing GLM stage's ordering.
"""
from __future__ import annotations

import argparse
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

import numpy as np

COMP = "name-that-llm"
SOURCE = "dennis6886/glm-user-prefix-guarded-fable-return"
SOURCE_SHA = "f36faa76126819e799b9a70b79b76c4c21f3fc7498e9e60d5106a8c654eceaa9"
DEST = "kolihhan/ntl-glm-rescue-per-token-20261124"
MARKER = "ntl-glm-rescue-per-token-once-20261124"
CHAMPION = "58680531"
COUNTS = {"model_sources": 9, "dataset_sources": 26, "kernel_sources": 10}
LABELS = {
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max",
}
HELPER = """def ntl_rescue_per_token(zl, sel_idx, rows_i):
    # Idealized rescue opportunity: a perfect GLM specialist fixes the base
    # errors whose truth is GLM; this does NOT predict actual specialist gain.
    probs = np.exp(np.asarray(zl, dtype=np.float64))
    sel_idx = np.asarray(sel_idx, dtype=np.int64)
    rows_i = np.asarray(rows_i, dtype=np.float64)
    assert probs.ndim == 2 and rows_i.shape == (len(probs),)
    assert np.isfinite(probs).all() and np.isfinite(rows_i).all()
    assert np.all(rows_i > 0) and len(sel_idx) > 0
    assert np.allclose(probs.sum(1), 1.0, atol=1e-6)
    mass = probs[:, sel_idx].sum(1)
    top = probs.argmax(1)
    existing = np.where(np.isin(top, sel_idx), probs[np.arange(len(probs)), top], 0.0)
    gain = np.maximum(mass - existing, 0.0)
    utility = gain / np.maximum(rows_i, 1.0)
    # Last key is primary; tie-break by original mass and then row order.
    order = np.lexsort((np.arange(len(probs)), -mass, -utility))
    return order, gain, utility
"""
STAGE_HEADER = "%%writefile /kaggle/working/prefix_variant_stage.py"
OLD_SORT = "    order = np.argsort(-mass, kind='stable')\n"
CUM = "    cum = np.cumsum(rows_i[order])"
MAIN = "\ndef main():"
NEW_SORT = """    order, ntl_ideal_rescue, ntl_rescue_utility = ntl_rescue_per_token(
        zl, [labels.index(c) for c in sel_cls], rows_i)
    # The inherited cumulative-cost cap, max-cover and hard deadline are unchanged.
    print('NTL_GLM_RESCUE_PER_TOKEN_ACTIVE', flush=True)
"""


def emit(name, obj):
    print(name+"="+json.dumps(obj, sort_keys=True, default=str), flush=True)


def cli(*args, timeout=150):
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        # Error type only, no token-bearing URLs or server bodies.
        raise RuntimeError(f"KAGGLE_{args[0]}_{args[1]}_EXIT_{p.returncode}")
    return p.stdout


def source_text(cell):
    v = cell.get("source", "")
    return "".join(v) if isinstance(v, list) else str(v)


def digest(nb):
    return hashlib.sha256("\n".join(source_text(c) for c in nb["cells"]
                             if c.get("cell_type") == "code").encode()).hexdigest()


def pull(ref, path):
    path.mkdir(parents=True, exist_ok=True)
    cli("kernels", "pull", ref, "-p", str(path), "-m", timeout=180)
    md = json.loads((path/"kernel-metadata.json").read_text())
    name = Path(md["code_file"]).name
    if not name.endswith(".ipynb"):
        raise ValueError("expected notebook")
    return md, json.loads((path/name).read_text()), name


def patch_stage(body):
    if body.count(OLD_SORT) != 1 or body.count(CUM) != 1 or body.count(MAIN) != 1:
        raise ValueError("rank/rows/main source anchors changed")
    if body.index(OLD_SORT) >= body.index(CUM):
        raise ValueError("unexpected source order")
    changed = body.replace(OLD_SORT, "", 1)
    changed = changed.replace(MAIN, "\n"+HELPER+"\n"+MAIN, 1)
    changed = changed.replace(CUM, NEW_SORT+CUM, 1)
    ast.parse(changed)
    compile(changed, "prefix_variant_stage.py", "exec")
    # Reverse the three narrow edits and demand exact original source.
    restored = changed.replace(NEW_SORT+CUM, CUM, 1)
    restored = restored.replace("\n"+HELPER+"\n"+MAIN, MAIN, 1)
    anchor = "    mass = np.exp(zl[:, [labels.index(c) for c in sel_cls]]).sum(1)\n"
    if restored.count(anchor) != 1:
        raise ValueError("mass expression drift")
    restored = restored.replace(anchor, anchor+OLD_SORT, 1)
    if restored != body:
        raise AssertionError("other stage bytes changed")
    return changed


def selftest():
    scope = {"np": np}
    exec(compile(HELPER, "rescue_per_token", "exec"), scope)
    fn = scope["ntl_rescue_per_token"]
    p = np.array([[.90,.05,.05], [.34,.33,.33], [.20,.25,.55], [.40,.30,.30]])
    rows = np.array([100,100,200,50], dtype=float)
    order, gain, utility = fn(np.log(p), [0,1], rows)
    assert np.allclose(gain, [.05,.33,.45,.30])
    assert np.array_equal(order, [3,1,2,0])
    assert np.all(np.diff(np.cumsum(rows[order]))>0)
    assert np.array_equal(fn(np.log(p), [0,1], rows)[0], order)
    # Equal utility and mass preserve original row order.
    assert np.array_equal(fn(np.log(p[[0,0]]), [0,1], [50,50])[0], [0,1])
    print("NTL_GLM_RESCUE_PER_TOKEN_SELFTEST=PASS", flush=True)


def build(root):
    md, nb, name = pull(SOURCE, root/"parent")
    actual = digest(nb)
    if actual != SOURCE_SHA:
        raise ValueError("PARENT_SHA_CHANGED:"+actual)
    if md.get("id") != SOURCE:
        raise ValueError("unexpected source kernel")
    for k,n in COUNTS.items():
        assert len(md.get(k) or []) == n, "parent mount drift "+k
    orig = [source_text(c) for c in nb["cells"]]
    indices = [i for i,c in enumerate(nb["cells"]) if c.get("cell_type") == "code"
               and source_text(c).startswith(STAGE_HEADER+"\n")]
    assert len(indices)==1, "stage cell not unique"
    i=indices[0]
    body=orig[i].split("\n",1)[1]
    nb["cells"][i]["source"]=(STAGE_HEADER+"\n"+patch_stage(body)).splitlines(keepends=True)
    changes=[j for j,c in enumerate(nb["cells"]) if source_text(c)!=orig[j]]
    assert changes==[i]
    # Leave the existing fast-commit path, 8.8 h deadline, asset/memory policy,
    # likelihood model, calibration, prefix, Fable guard and postprocessing intact.
    original_code="\n".join(orig)
    assert "V1_DEADLINE_H = 8.8" in original_code
    assert "KAGGLE_IS_COMPETITION_RERUN" in original_code
    for cell in nb["cells"]:
        if cell.get("cell_type")=="code":
            cell["outputs"]=[]
            cell["execution_count"]=None
    meta=dict(md)
    meta.pop("id_no",None)
    meta.update(id=DEST,title=DEST.split("/")[-1],is_private=True)
    for k in ("model_sources","dataset_sources","kernel_sources","competition_sources",
              "enable_gpu","enable_internet","machine_shape","docker_image","kernel_type"):
        assert meta.get(k)==md.get(k), "runtime drift"
    out=root/"candidate"
    out.mkdir()
    (out/name).write_text(json.dumps(nb, ensure_ascii=False, indent=1)+"\n")
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    sha=digest(nb)
    emit("NTL_GLM_RESCUE_CANDIDATE_PREFLIGHT", {
        "parent_sha":actual,"candidate_sha":sha,"modified_stage_cell":i,
        "changed_cells":1,"assets":COUNTS,"private":True,"deadline_hours":8.8,
        "no_training":True,"preview_is_not_hidden_inference":True})
    return out, sha, meta


def ledger():
    v=cli("competitions","submissions",COMP,"-v",timeout=100)
    return list(csv.DictReader(io.StringIO(v.lstrip("\ufeff"))))


def receipt(rows):
    found=[r for r in rows if MARKER in (r.get("description") or "")]
    assert len(found)<=1, "duplicate marker"
    if found:
        emit("NTL_GLM_RESCUE_OFFICIAL_RECEIPT", {k:found[0].get(k)
            for k in ("ref","status","publicScore","date","description")})
        return True
    return False


def quota_ok():
    from kaggle.api.kaggle_api_extended import KaggleApi
    rows=ledger()
    if receipt(rows):
        return False
    champion=[r for r in rows if r.get("ref")==CHAMPION]
    assert len(champion)==1 and champion[0].get("publicScore")=="0.715" and "COMPLETE" in champion[0].get("status",""), "champion receipt drift"
    active=[r.get("ref") for r in rows if (r.get("status") or "").upper().split(".")[-1] in
            {"PENDING","QUEUED","RUNNING","EVALUATING","SCORING"}]
    api=KaggleApi()
    api.authenticate()
    q=api.competition_get_submission_limits(COMP)
    allowed=int(q.num_allowed_now)
    limit=bool(q.limited_by_total)
    emit("NTL_GLM_RESCUE_SUBMISSION_QUOTA", {"allowed_now":allowed,
         "limited_by_total":limit,"active_refs":active,"reserve_one_slot":True})
    return allowed>=2 and not limit and not active


def launch_or_reuse(out):
    slug=DEST.split("/")[-1]
    v=cli("kernels","list","--mine","--search",slug,"-v",timeout=90)
    if DEST in v:
        print("NTL_GLM_RESCUE_PRIVATE_ALREADY_EXISTS_NO_REPUSH=TRUE",flush=True)
        return
    print("NTL_GLM_RESCUE_PRIVATE_NOT_IN_EXACT_SEARCH=TRUE",flush=True)
    cli("kernels","push","-p",str(out),timeout=600)
    print("NTL_GLM_RESCUE_PRIVATE_PUSH_ACCEPTED=TRUE",flush=True)


def worker_status():
    raw=cli("kernels","status",DEST,timeout=90)
    values=re.findall(r"KernelWorkerStatus[.][A-Z_]+",raw)
    if not values: raise RuntimeError("KAGGLE_WORKER_STATUS_UNKNOWN")
    return values[0].split(".")[-1]


def wait_complete(max_minutes):
    deadline=time.monotonic()+60*max_minutes
    previous=None
    while True:
        state=worker_status()
        if state!=previous:
            emit("NTL_GLM_RESCUE_PRIVATE_WORKER", {"state":state})
            previous=state
        if state=="COMPLETE":
            return True
        if state in {"ERROR","FAILED","CANCELLED","CANCELED"}:
            print("NTL_GLM_RESCUE_PRIVATE_FAILED_NO_SUBMIT=TRUE",flush=True)
            return False
        if time.monotonic()>=deadline:
            print("NTL_GLM_RESCUE_PRIVATE_PENDING_AT_BOUND_NO_SUBMIT=TRUE",flush=True)
            return False
        time.sleep(60)


def verify_private(root, expected_sha, expected_meta):
    for key,ref in (("current",DEST),("version1",DEST+"/1")):
        md,nb,_=pull(ref,root/key)
        assert md.get("id")==DEST and str(md.get("is_private")).lower() in ("true","1")
        assert digest(nb)==expected_sha, "private source/version mismatch"
        for k in COUNTS:
            assert md.get(k)==expected_meta.get(k), "private mount mismatch "+k
    # Pull only the notebook outputs, never print raw rows.
    import pandas as pd
    output=root/"outputs"
    output.mkdir()
    cli("kernels","output",DEST,"-p",str(output),timeout=240)
    fs=list(output.rglob("submission.parquet"))
    assert len(fs)==1,"missing or ambiguous preview parquet"
    df=pd.read_parquet(fs[0])
    assert set(df.columns)=={"id","model"} and len(df)>0
    assert not df.id.isna().any() and df.id.astype(str).is_unique
    assert df.model.astype(str).isin(LABELS).all()
    emit("NTL_GLM_RESCUE_PRIVATE_PREVIEW_VERIFIED", {"rows":len(df),
         "candidate_sha":expected_sha,"preview_only_not_full_hidden_validation":True,
         "version_1_pulled_source_matches":True})


def submit_once():
    # Never retry an uncertain POST. Later manual status check must read ledger first.
    if not quota_ok():
        print("NTL_GLM_RESCUE_NO_SAFE_SLOT_NO_SUBMIT=TRUE",flush=True)
        return
    result=cli("competitions","submit",COMP,"-f","submission.parquet","-k",DEST,
               "-v","1","-m",MARKER,timeout=240)
    ref=re.findall(r"Submission ref:\s*(\d+)",result)
    if ref:
        emit("NTL_GLM_RESCUE_OFFICIAL_REF",{"ref":ref[-1]})
    else:
        print("NTL_GLM_RESCUE_POST_RETURNED_NO_REF_CHECK_LEDGER_NO_RETRY=TRUE",flush=True)
    for _ in range(5):
        if receipt(ledger()):
            return
        time.sleep(12)
    print("NTL_GLM_RESCUE_OFFICIAL_RECEIPT_UNCONFIRMED_NO_RETRY=TRUE",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--selftest",action="store_true")
    p.add_argument("--max-wait-minutes",type=int,default=30)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    selftest()
    if a.selftest: return
    assert a.execute and os.environ.get("KAGGLE_API_TOKEN")
    assert 0<=a.max_wait_minutes<=45
    if not quota_ok():
        print("NTL_GLM_RESCUE_NO_SAFE_SLOT_NO_LAUNCH=TRUE",flush=True)
        return
    with tempfile.TemporaryDirectory(prefix="ntl-glm-rescue-once-") as temp:
        root=Path(temp)
        out,sha,md=build(root)
        launch_or_reuse(out)
        if not wait_complete(a.max_wait_minutes):
            return
        verify_private(root,sha,md)
        submit_once()


if __name__=="__main__":
    main()
