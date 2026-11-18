"""One official NTL score attempt using an attributed, pinned public 0.702 source.

Owner approved 2026-11-18 one private T4 production run after Quick Save code-submit
HTTP400. One immutable source, exactly one private GPU version 2, then official
code scoring if execution finishes. No retry or raw competition data in Git.
This is a reproduction, NOT an original medal-level method.
"""
from __future__ import annotations
import csv,hashlib,io,json,re,subprocess
from pathlib import Path

SOURCE="lightningv08/0-702-apex-style-15-glm-qwen3-8-27b-lora"
SRC_SHA="f9948b633b22bd5e63b8253e4d8027c073b44f57eb85bc60b775857e396b38c2"
CODE_SHA="2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc"
DEST="kolihhan/ntl-0702-apex-pinned-nov18"
DESCRIPTION="Attributed Lightning 0.702 Apex public source, frozen inference, no dev GPU 20261118"
ROOT=Path("/tmp/ntl-apex702-official-once")

def cli(*args,timeout=150,check=True):
    p=subprocess.run(["kaggle",*args],text=True,capture_output=True,timeout=timeout)
    if p.returncode and check:
        err=(p.stderr+"\n"+p.stdout)[-1000:]
        raise RuntimeError(f"Kaggle {args[:2]} failed with {p.returncode}: {err}")
    return p

def sha(content):
    return hashlib.sha256(content).hexdigest()

def quota():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi()
    api.authenticate()
    limit=api.competition_get_submission_limits("name-that-llm")
    now=int(limit.num_allowed_now or 0)
    rows=list(csv.DictReader(io.StringIO(cli("competitions","submissions","name-that-llm","-v","-q").stdout.lstrip("\ufeff"))))
    if any(x.get("description")==DESCRIPTION for x in rows):
        raise RuntimeError("ONE_SHOT_ALREADY_SUBMITTED")
    active=[x.get("ref") for x in rows if any(t in str(x.get("status") or "").upper()
            for t in ("PENDING","RUNNING","QUEUED"))]
    if active:raise RuntimeError("EXISTING_SCORER_ACTIVE:"+",".join(map(str,active[:5])))
    # Owner's frozen rule reserves at least one daily official entry.
    if now<2:raise RuntimeError("QUOTA_INSUFFICIENT_TO_RESERVE_ONE_SLOT:"+str(now))
    print("NTL_702_QUOTA_VERIFIED="+json.dumps({"now":now,"active":0,"reserve":1}),flush=True)

def prepare():
    if ROOT.exists() and list(ROOT.iterdir()):
        raise FileExistsError("refuse to reuse existing source/download destination")
    source=ROOT/"original";out=ROOT/"private"
    source.mkdir(parents=True);out.mkdir()
    cli("kernels","pull",SOURCE,"-p",str(source),"-m")
    notebooks=list(source.glob("*.ipynb"))
    if len(notebooks)!=1:raise RuntimeError("expected exactly one public notebook")
    blob=notebooks[0].read_bytes()
    if sha(blob)!=SRC_SHA:raise RuntimeError("PUBLIC_SOURCE_HASH_DRIFT")
    src=json.loads(blob)
    meta=json.loads((source/"kernel-metadata.json").read_text())
    if (meta.get("id")!=SOURCE or meta.get("is_private") is not False
        or meta.get("enable_internet") is not False or meta.get("enable_gpu") is not True
        or meta.get("competition_sources")!=["name-that-llm"]
        or len(meta.get("dataset_sources") or [])!=26
        or len(meta.get("kernel_sources") or [])!=9
        or len(meta.get("model_sources") or [])!=9):
        raise RuntimeError("PUBLIC_SOURCE_CONTRACT_DRIFT")
    cells=[c for c in src.get("cells",[]) if c.get("cell_type")=="code"]
    code="\n".join("".join(c.get("source",[])) if isinstance(c.get("source"),list) else str(c.get("source","")) for c in cells)
    if sha(code.encode())!=CODE_SHA or len(cells)!=29:
        raise RuntimeError("PUBLIC_INFERENCE_CODE_HASH_DRIFT")
    if not ("submission.parquet" in code and "test.csv" in code):
        raise RuntimeError("NO_HIDDEN_TEST_SUBMISSION_PATH")
    # Fail closed on known old-preview shortcuts or direct supervised fit loops.
    forbidden=(r"emit_known_preview\s*\(",r"preview\s*=",r"gauss\.fit\s*\(",
                r"optimizer\.step\s*\(",r"\.backward\s*\(",
                r"fit\s*\(\s*(?:test|test_df|test_data)\s*\[")
    if any(re.search(p,code) for p in forbidden):
        raise RuntimeError("PUBLIC_SOURCE_HAS_UNAPPROVED_PREVIEW_OR_TRAIN_PATH")
    # Preserve EVERY original executable cell and all pretrained resource mounts.
    src["cells"].insert(0,{"cell_type":"markdown","metadata":{},
        "source":["Attributed inference-only reproduction of public notebook by lightningv08: ",
                  "https://www.kaggle.com/code/"+SOURCE+" .\n",
                  "No original authorship claim. All dataset, model and notebook inputs retain their own licenses.\n",
                  "0.702 is a public author's notebook title, NOT this account's scored result.\n"]})
    meta.pop("id_no",None);meta.pop("docker_image",None)
    meta.update({"id":DEST,"title":"ntl-0702-apex-pinned-nov18","is_private":True,
                 "enable_gpu":True,"enable_tpu":False,"enable_internet":False,
                 "machine_shape":"NvidiaTeslaT4"})
    name=meta.get("code_file")
    if not name or name!=Path(name).name or not name.endswith(".ipynb"):
        raise RuntimeError("INVALID_UPSTREAM_CODE_FILENAME")
    (out/name).write_text(json.dumps(src,ensure_ascii=False),encoding="utf8")
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True))
    # Source hash, model, data mount identities and no executable code changes.
    check=json.loads((out/name).read_text())
    code2="\n".join("".join(c.get("source",[])) if isinstance(c.get("source"),list) else str(c.get("source","")) for c in check["cells"] if c.get("cell_type")=="code")
    if code!=code2:raise RuntimeError("EXECUTABLE_CELLS_CHANGED")
    p=ROOT/"prepared.json"
    p.write_text(json.dumps({"source":SOURCE,"source_sha":SRC_SHA,"code_sha":CODE_SHA,"destination":DEST,
                             "kernel_sha":sha((out/name).read_bytes()),"status":"READY_NO_GPU"},sort_keys=True))
    print("NTL_APEX702_ATTRIBUTED_SOURCE_PINNED="+p.read_text(),flush=True)

def submit():
    # A true one-shot: never perform a dev GPU preview. Kaggle may reject the
    # Quick Save version for scoring; if so, STOP without workaround.
    quota()
    if not (ROOT/"prepared.json").is_file():
        raise RuntimeError("NO_PREPARED_FROZEN_PRIVATE_SOURCE")
    status=cli("kernels","status",DEST,check=False)
    if status.returncode==0:
        raise RuntimeError("DESTINATION_ALREADY_EXISTS_REFUSE_OVERWRITE")
    help_text=cli("kernels","push","--help").stdout
    if "--no-run" not in help_text:
        raise RuntimeError("NO_RUN_NOT_SUPPORTED_NO_GPU_PREVIEW_PERMITTED")
    quota()
    result=cli("kernels","push","-p",str(ROOT/"private"),"--no-run")
    print("NTL_APEX702_QUICK_SAVE_RESPONSE="+(result.stdout+result.stderr)[-1000:],flush=True)
    print("NTL_APEX702_NO_DEVELOPMENT_GPU_RUN=TRUE",flush=True)
    quota()
    response=cli("competitions","submit","name-that-llm","-k",DEST,"-v","1",
                 "-f","submission.parquet","-m",DESCRIPTION,timeout=200)
    receipt=re.findall(r"Submission ref:\s*(\d+)",response.stdout+response.stderr,re.I)
    if len(receipt)!=1:
        raise RuntimeError("KAGGLE_SUBMIT_RETURNED_NO_SINGLE_NUMERIC_RECEIPT_NO_RESUBMIT")
    print("NTL_APEX702_AUTHENTICATED_OFFICIAL_SUBMISSION_REF="+receipt[0],flush=True)
    matches=[row for row in csv.DictReader(io.StringIO(
        cli("competitions","submissions","name-that-llm","-v","-q").stdout.lstrip("\ufeff")))
        if row.get("ref")==receipt[0]]
    if len(matches)==1:
        print("NTL_APEX702_OFFICIAL_READBACK="+json.dumps({
          k:matches[0].get(k) for k in ("ref","description","status","publicScore")
        },sort_keys=True),flush=True)
    else:
        print("NTL_APEX702_OFFICIAL_ID_ACCEPTED_LEDGER_DELAYED_NO_RETRY=TRUE",flush=True)

def start_gpu_once():
    """One explicitly owner-authorized private T4 production run; no training."""
    quota()
    if not (ROOT/"prepared.json").is_file():
        raise RuntimeError("MISSING_FRESH_PINNED_SOURCE_PREPARE")
    # Quick Save version 1 already exists. Verify it belongs to this pinned source.
    previous=ROOT/"saved_v1"
    previous.mkdir()
    cli("kernels","pull",DEST,"-p",str(previous),"-m")
    old=list(previous.glob("*.ipynb"))
    if len(old)!=1:
        raise RuntimeError("PRIVATE_SAVED_V1_NOT_FOUND")
    oldnb=json.loads(old[0].read_text(encoding="utf-8"))
    def code(nb):
        return "\n".join("".join(c.get("source",[])) if isinstance(c.get("source"),list)
            else str(c.get("source","")) for c in nb["cells"] if c.get("cell_type")=="code")
    if sha(code(oldnb).encode())!=CODE_SHA or \
            "Attributed inference-only reproduction" not in json.dumps(oldnb.get("cells",[])[:1]):
        raise RuntimeError("PRIVATE_V1_SOURCE_MISMATCH")
    oldmeta=json.loads((previous/"kernel-metadata.json").read_text())
    freshmeta=json.loads((ROOT/"private"/"kernel-metadata.json").read_text())
    for key in ("dataset_sources","model_sources","kernel_sources","competition_sources"):
        if oldmeta.get(key)!=freshmeta.get(key):
            raise RuntimeError("PRIVATE_V1_SOURCE_MOUNT_DRIFT:"+key)
    old_status=cli("kernels","status",DEST,check=False)
    status_text=(old_status.stdout+"\n"+old_status.stderr).upper()
    if any(w in status_text for w in ("RUNNING","QUEUED","PENDING")):
        raise RuntimeError("OWN_PRIVATE_KERNEL_ALREADY_ACTIVE")
    if "COMPLETE" in status_text:
        raise RuntimeError("OWN_PRIVATE_KERNEL_ALREADY_RAN; DO_NOT_PUSH_VERSION3")
    # Do not overwrite a working version or silently move the original score.
    help_text=cli("kernels","push","--help").stdout
    if "--accelerator" not in help_text:
        raise RuntimeError("OFFICIAL_KAGGLE_GPU_PUSH_FLAG_UNAVAILABLE")
    quota()
    pushed=cli("kernels","push","-p",str(ROOT/"private"),
               "--accelerator","NvidiaTeslaT4","--timeout","32400",timeout=180)
    output=(pushed.stdout+"\n"+pushed.stderr)
    if not re.search(r"Kernel version 2 successfully pushed",output,re.I):
        print("NTL_APEX702_PUSH_UNKNOWN_RESULT_NO_RETRY="+output[-850:],flush=True)
        raise RuntimeError("KAGGLE_GPU_PUSH_NOT_VERIFIED_AS_EXACT_VERSION_2")
    print("NTL_APEX702_ONE_AUTHORIZED_PRIVATE_GPU_VERSION2_LAUNCHED=TRUE",flush=True)
    print("NTL_APEX702_PRIVATE_NOTEBOOK=https://www.kaggle.com/code/"+DEST,flush=True)
    print("NTL_APEX702_REEXECUTION_ONLY_SOURCE_CODE_SHA="+CODE_SHA,flush=True)
    print("NTL_APEX702_NO_TRAINING_NO_SOURCE_MODIFICATION=TRUE",flush=True)


def finish_gpu_once():
    """Read once: only score completed private version 2 with real output."""
    quota()
    status=cli("kernels","status",DEST,check=False)
    state=(status.stdout+"\n"+status.stderr).upper()
    if "KERNELWORKERSTATUS.COMPLETE" not in state:
        if any(x in state for x in ("ERROR","FAILED","CANCELLED")):
            raise RuntimeError("NTL_APEX702_PRIVATE_GPU_FAILED_DO_NOT_RETRY")
        print("NTL_APEX702_PRIVATE_GPU_NOT_COMPLETE_YET="+state[-350:],flush=True)
        return False
    listings=cli("kernels","files",DEST,"--page-size","200","-v").stdout
    if "submission.parquet" not in listings or "Next Page Token =" in listings:
        raise RuntimeError("NTL_APEX702_COMPLETE_WITHOUT_SINGLE_OUTPUT_PAGE")
    print("NTL_APEX702_PRIVATE_VERSION2_COMPLETE_WITH_SUBMISSION_PARQUET=TRUE",flush=True)
    quota()
    response=cli("competitions","submit","name-that-llm","-k",DEST,"-v","2",
                 "-f","submission.parquet","-m",DESCRIPTION,timeout=180)
    receipt=re.findall(r"Submission ref:\s*(\d+)",response.stdout+response.stderr,re.I)
    if len(receipt)!=1:
        print("NTL_APEX702_OFFICIAL_AMBIGUOUS_RESULT_NO_RETRY="+
              (response.stdout+"\n"+response.stderr)[-700:],flush=True)
        raise RuntimeError("OFFICIAL_ENTRY_ID_UNCONFIRMED_NO_RETRY")
    print("NTL_APEX702_OFFICIAL_SUBMISSION_ID="+receipt[0],flush=True)
    rows=list(csv.DictReader(io.StringIO(cli("competitions","submissions","name-that-llm","-v","-q").stdout.lstrip("\ufeff"))))
    matched=[x for x in rows if x.get("ref")==receipt[0]]
    if len(matched)==1:
        print("NTL_APEX702_AUTHENTICATED_RECEIPT="+json.dumps({
          k:matched[0].get(k) for k in ("ref","status","publicScore","description")
        },sort_keys=True),flush=True)
    return True


def wait_finish_once(limit_minutes=310):
    import time
    deadline=time.monotonic()+limit_minutes*60
    while True:
        if finish_gpu_once():
            return
        if time.monotonic()+180 > deadline:
            print("NTL_APEX702_GPU_STILL_RUNNING_AT_CI_WAIT_LIMIT=TRUE",flush=True)
            print("NTL_APEX702_MANUAL_FINISH_REQUIRED_NO_RESTART=TRUE",flush=True)
            return
        time.sleep(180)


if __name__=="__main__":
    import sys
    if len(sys.argv)!=2 or sys.argv[1] not in {"quota","prepare","submit","start_gpu_once","finish_gpu_once","wait_finish_once"}:
        raise SystemExit("usage: python -m scripts.ntl_apex702_official_once quota|prepare|start_gpu_once|wait_finish_once")
    globals()[sys.argv[1]]()
