"""Reproduce the exact public Sanpier V3 source once. Not original research.
No new training, no changed prediction code, no scheduler, no duplicate push.
Use the owner's authenticated Kaggle API; never export code/data or credentials.
"""
from __future__ import annotations
import json
import re
import tempfile
import time
from collections import Counter
from pathlib import Path

from scripts import ntl_glm_rescue_per_token_once_20261124 as h

SOURCE = "sanpier/0-709-320-token-glm-likelihood-qwen-ensemble/3"
PARENT_ID = SOURCE.rsplit("/",1)[0]
EXPECTED_SHA = "d526209d0eda5138867324fbb35e11345f1b2bb8dc74d772fcdfb929753a7434"
DEST = "kolihhan/ntl-sanpier-v3-0717-repro-20261124"
MARKER = "ntl-sanpier-v3-0717-exact-reproduction-20261124"
PREVIOUS = "58808007"
CHAMPION = "58680531"
COUNTS = {"model_sources":9,"dataset_sources":26,"kernel_sources":10}
MOUNTS = (*COUNTS,"competition_sources")
RUNTIME = ("enable_gpu","enable_internet","machine_shape","docker_image","kernel_type")


def emit(name, payload):
    h.emit("NTL_SANPIER_"+name, payload)


def quota():
    from kaggle.api.kaggle_api_extended import KaggleApi
    rows = h.ledger()
    existing = [r for r in rows if MARKER in (r.get("description") or "")]
    assert len(existing)<=1, "duplicate official marker"
    if existing:
        emit("ALREADY_SUBMITTED", {k:existing[0].get(k) for k in ("ref","status","publicScore","date")})
        return False
    # Prevent submitting another Sanpier reproduction under a changed marker.
    similar = [r for r in rows if "sanpier" in (r.get("description") or "").lower()]
    if similar:
        emit("POSSIBLE_DUPLICATE_NO_SUBMIT", [{k:r.get(k) for k in ("ref","status","publicScore")} for r in similar])
        return False
    numeric=[]
    for row in rows:
        try:
            value=float(row.get("publicScore") or "")
            numeric.append((value,row.get("ref")))
        except (TypeError,ValueError):
            pass
    prev=[r for r in rows if r.get("ref")==PREVIOUS]
    emit("OWNER_SCORE_SNAPSHOT", {"best_public":max(numeric) if numeric else None,
         "prior_rescue":{k:prev[0].get(k) for k in ("ref","status","publicScore","date")} if len(prev)==1 else None})
    champ=[r for r in rows if r.get("ref")==CHAMPION]
    assert len(champ)==1 and float(champ[0].get("publicScore") or -1)==.715 and "COMPLETE" in champ[0].get("status",""), "frozen champion receipt unavailable"
    active=[r.get("ref") for r in rows if (r.get("status") or "").upper().split(".")[-1] in
            {"PENDING","QUEUED","RUNNING","EVALUATING","SCORING"}]
    api=KaggleApi()
    api.authenticate()
    q=api.competition_get_submission_limits(h.COMP)
    available=int(q.num_allowed_now)
    limited=bool(q.limited_by_total)
    emit("QUOTA",{"allowed_now":available,"limited_by_total":limited,"active_refs":active,
        "one_existing_rescue_allowed":True,"reserve_one_slot":True})
    # Owner authorized additional distinct submission. Only the known prior
    # experiment may coexist; reserve at least one official slot.
    return available>=2 and not limited and len(active)<=1 and all(x==PREVIOUS for x in active)


def exact_assets(actual, expected, kind):
    a,b=actual.get(kind) or [],expected.get(kind) or []
    return Counter(map(str,a))==Counter(map(str,b))


def build(root):
    md,nb,name=h.pull(SOURCE,root/"source")
    assert md.get("id")==PARENT_ID,"source ref drift"
    assert str(md.get("is_private")).lower() in ("false","0"),"source not public"
    sha=h.digest(nb)
    assert sha==EXPECTED_SHA,"source SHA drift: "+sha
    assert len([c for c in nb["cells"] if c.get("cell_type")=="code"])==39,"code cell count drift"
    for k,n in COUNTS.items():
        assert len(md.get(k) or [])==n,"asset count drift: "+k
    assert len(md.get("competition_sources") or [])==1
    code="\n".join(h.source_text(c) for c in nb["cells"] if c.get("cell_type")=="code")
    assert "apex_postprocess.py" in code and "--deadline-mono" in code and "--shrink" in code,"expected scoring/runtime contract missing"
    assert "8.8" in code and "submission.parquet" in code
    assert str(md.get("enable_internet")).lower() in ("false","0"),"internet enabled"
    assert str(md.get("enable_gpu")).lower() in ("true","1"),"GPU setting missing"
    for c in nb["cells"]:
        if c.get("cell_type")=="code":
            c["outputs"]=[]
            c["execution_count"]=None
    assert h.digest(nb)==EXPECTED_SHA,"executable code changed"
    meta=dict(md)
    meta.pop("id_no",None)
    meta.update(id=DEST,title=DEST.split("/")[-1],is_private=True)
    for k in (*MOUNTS,*RUNTIME):
        assert meta.get(k)==md.get(k), "changed runtime/asset metadata "+k
    out=root/"candidate"
    out.mkdir()
    (out/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n")
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    emit("EXACT_REPRO_PREFLIGHT",{"source":SOURCE,"public_page_claim":.717,"owner_score_not_verified":True,
        "sha":sha,"changed_executable_cells":0,"mount_counts":COUNTS,"private":True,
        "attribution":"Sanpier and upstream public notebook authors","hidden_runtime_not_measured":True})
    return out,meta


def launch(out):
    raw=h.cli("kernels","list","--mine","--search",DEST.split("/")[-1],"-v",timeout=100)
    if DEST in raw:
        print("NTL_SANPIER_EXISTING_KERNEL_NO_REPUSH=TRUE",flush=True)
        return
    h.cli("kernels","push","-p",str(out),timeout=600)
    print("NTL_SANPIER_PRIVATE_PUSH_ACCEPTED=TRUE",flush=True)


def wait_complete(minutes=25):
    end=time.monotonic()+60*minutes
    last=None
    while True:
        raw=h.cli("kernels","status",DEST,timeout=90)
        m=re.search(r"KernelWorkerStatus[.]([A-Z_]+)",raw)
        assert m,"unknown private worker status"
        state=m.group(1)
        if state!=last:
            emit("PRIVATE_WORKER",{"state":state})
            last=state
        if state=="COMPLETE":
            return True
        if state in {"ERROR","FAILED","CANCELLED","CANCELED"} or time.monotonic()>=end:
            emit("STOP_NO_SUBMIT",{"state":state,"no_retry":True})
            return False
        time.sleep(60)


def verify(root,expected):
    import pandas as pd
    for key,ref in (("current",DEST),("v1",DEST+"/1")):
        md,nb,_=h.pull(ref,root/key)
        assert md.get("id")==DEST and str(md.get("is_private")).lower() in ("true","1")
        assert h.digest(nb)==EXPECTED_SHA,"candidate source drift"
        for k in MOUNTS:
            assert exact_assets(md,expected,k),"asset identity drift: "+k
        for k in RUNTIME:
            assert md.get(k)==expected.get(k),"runtime drift: "+k
    dest=root/"outputs"
    dest.mkdir()
    h.cli("kernels","output",DEST,"-p",str(dest),timeout=240)
    fs=list(dest.rglob("submission.parquet"))
    assert len(fs)==1,"preview output unavailable"
    df=pd.read_parquet(fs[0])
    assert set(df.columns)=={"id","model"} and len(df)>0
    assert not df.id.isna().any() and df.id.astype(str).is_unique
    assert df.model.astype(str).isin(h.LABELS).all()
    emit("PREVIEW_VERIFIED",{"rows":len(df),"code_sha":EXPECTED_SHA,
        "exact_versioned_assets":True,"preview_not_hidden_validation":True})


def submit():
    if not quota():
        print("NTL_SANPIER_QUOTA_OR_DUPLICATE_STOP=TRUE",flush=True)
        return
    raw=h.cli("competitions","submit",h.COMP,"-f","submission.parquet","-k",DEST,
              "-v","1","-m",MARKER,timeout=240)
    m=re.search(r"Submission ref:\s*(\d+)",raw)
    if m:
        emit("OFFICIAL_REF",{"ref":m.group(1)})
    else:
        print("NTL_SANPIER_POST_RETURNED_NO_REF_CHECK_LEDGER_NO_RETRY=TRUE",flush=True)
    for i in range(5):
        found=[r for r in h.ledger() if MARKER in (r.get("description") or "")]
        assert len(found)<=1
        if found:
            emit("OFFICIAL_RECEIPT",{k:found[0].get(k) for k in ("ref","date","status","publicScore","description")})
            return
        if i<4:time.sleep(12)
    print("NTL_SANPIER_UNCONFIRMED_RECEIPT_NO_RETRY=TRUE",flush=True)


def main():
    if not quota():
        print("NTL_SANPIER_STOP_BEFORE_GPU=TRUE",flush=True)
        return
    with tempfile.TemporaryDirectory(prefix="ntl-sanpier-v3-") as d:
        root=Path(d)
        out,meta=build(root)
        launch(out)
        if not wait_complete():
            return
        verify(root,meta)
        submit()


if __name__=="__main__":
    main()
