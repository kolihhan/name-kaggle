"""One NTL native-pretrained, no-new-training Kaggle Code Competition entry.

Attrib: public shapshamkarkiii native ensemble. Original Notebook and 21
source datasets / 8 public kernels / 6 pretrained models remain intact.
Guard all known supervised/training fallbacks. No source data enters GitHub.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

SOURCE = "shakshamkarkiii/ntl-native-ensemble"
DEST = "kolihhan/ntl-native-frozen-nov12-one-shot"
CONTEST = "name-that-llm"
DESCRIPTION = "NTL frozen native pretrained ensemble guarded no supervised fitting 20261112"
SOURCE_SHA = "2b04b4d7ccdeeaef6ed62a52e24ee6f6f1702a50143c740a6d0a3bbce90e6920"
SOURCE_CODE_SHA = "5c8c2b63b3ae61b00bc50e74761723b81f7c23b0fc2808ada3310408f3a59a16"
RULES = Path("artifacts/candidates/native-frozen-nov12-guarded/CONTRACT.md")
BLOCK_GAUSS = "NO_NEW_TRAINING: supervised Gaussian fitting forbidden"
BLOCK_ATLAS = "NO_NEW_TRAINING: Atlas training mode forbidden"
BLOCK_L24 = "NO_NEW_TRAINING: core_worker train-final forbidden"


def run(args: list[str], *, timeout: int = 135, checked: bool=True) -> subprocess.CompletedProcess:
    result = subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if checked and result.returncode:
        # stdout/stderr from Kaggle can contain signed tokens. Never print them.
        raise RuntimeError(f"CLI command {args[:3]} failed rc={result.returncode}; no automatic retry")
    return result


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_text(cell: dict) -> str:
    src=cell.get("source",[])
    return "".join(src) if isinstance(src,list) else str(src)


def fetch_and_guard(work: Path) -> dict:
    if not RULES.is_file():
        raise RuntimeError("frozen contract missing")
    source = work / "source"
    target = work / "kernel"
    source.mkdir(parents=True,exist_ok=False)
    target.mkdir(parents=True,exist_ok=False)
    run(["kaggle","kernels","pull",SOURCE,"-p",str(source),"-m"])
    meta=json.loads((source/"kernel-metadata.json").read_text())
    files=list(source.glob("*.ipynb"))
    if len(files)!=1:
        raise ValueError("upstream Notebook not unique")
    original=files[0]
    content=original.read_bytes()
    if sha(content)!=SOURCE_SHA:
        raise RuntimeError("public native source content changed: do not substitute unknown code")
    notebook=json.loads(content)
    raw_code="\n".join(code_text(c) for c in notebook["cells"] if c.get("cell_type")=="code")
    if sha(raw_code.encode())!=SOURCE_CODE_SHA:
        raise RuntimeError("upstream code changed independently of source provenance")
    if (meta.get("id")!=SOURCE or meta.get("is_private") is not False or
        meta.get("enable_internet") is not False or meta.get("kernel_type")!="notebook" or
        meta.get("competition_sources")!=[CONTEST] or
        len(meta.get("dataset_sources") or [])!=21 or
        len(meta.get("kernel_sources") or [])!=8 or
        len(meta.get("model_sources") or [])!=6):
        raise RuntimeError("published source / mounted model topology changed")
    if len(re.findall(r"\.(?:fit|fit_transform|partial_fit)\s*\(",raw_code))!=4:
        raise RuntimeError("upstream fit call inventory changed")
    if re.search(r"\.(?:backward|zero_grad|step)\s*\(",raw_code):
        raise RuntimeError("unexpected new neural training operations")
    if raw_code.count("gauss.fit(")!=1:
        raise RuntimeError("expected single guarded Gaussian fallback")
    if raw_code.count("if training:\n        base.gradient_checkpointing_enable")!=1:
        raise RuntimeError("Atlas training branch drift")
    if raw_code.count("--train-final")!=1:
        raise RuntimeError("core worker training flag drift")
    if raw_code.count("'RELEASE_READY': True")!=1:
        raise RuntimeError("frozen inference release configuration missing")

    counts={"gauss":0,"atlas":0,"core":0}
    for cell in notebook["cells"]:
        if cell.get("cell_type")!="code":
            continue
        code=code_text(cell)
        if "def member_scores" in code and "import gauss\n" in code:
            if counts["gauss"]:
                raise RuntimeError("nonunique Gaussian guard insertion")
            code=code.replace("import gauss\n",
               "import gauss\n"
               "def _forbid_supervised_gaussian(*args, **kwargs):\n"
               f"    raise RuntimeError('{BLOCK_GAUSS}')\n"
               "gauss.fit = _forbid_supervised_gaussian\n",1)
            counts["gauss"]+=1
        if "model = AtlasModel(base, labels, config)" in code and "if training:" in code:
            if counts["atlas"]:
                raise RuntimeError("nonunique Atlas guard insertion")
            anchor="if training:\n        base.gradient_checkpointing_enable"
            if anchor not in code:
                raise RuntimeError("Atlas exact training guard no longer applies")
            code=code.replace(anchor,
                "if training:\n"
                f"        raise RuntimeError('{BLOCK_ATLAS}')\n"
                "        base.gradient_checkpointing_enable",1)
            counts["atlas"]+=1
        if "def l24_features" in code and "--train-final" in code:
            if counts["core"]:
                raise RuntimeError("nonunique L24 worker guard")
            anchor="args = parser.parse_args()\n    config ="
            if anchor not in code:
                raise RuntimeError("L24 exact worker guard missing")
            code=code.replace(anchor,
                "args = parser.parse_args()\n"
                f"    if args.train_final: raise RuntimeError('{BLOCK_L24}')\n"
                "    config =",1)
            counts["core"]+=1
        cell["source"]=code.splitlines(keepends=True)
    if counts!={"gauss":1,"atlas":1,"core":1}:
        raise RuntimeError("not all supervised-train guards injected: "+str(counts))
    finalized="\n".join(code_text(c) for c in notebook["cells"] if c.get("cell_type")=="code")
    for guard in (BLOCK_GAUSS,BLOCK_ATLAS,BLOCK_L24):
        if finalized.count(guard)!=1:
            raise RuntimeError("required no-train guard missing or duplicated")
    if not all("NO_NEW_TRAINING" in finalized for _ in (0,)):
        raise RuntimeError("guard postcondition missing")
    notebook["cells"].insert(0,{
        "cell_type":"markdown",
        "metadata":{},
        "source":[
           "Attributed, inference-only reproduction of [Shaksham Karkiii's public NTL Native Ensemble]"
           "(https://www.kaggle.com/code/shakshamkarkiii/ntl-native-ensemble). "
           "Original source and asset references preserved. This fork forbids new "
           "supervised Gaussian fitting, Atlas training, and core-worker --train-final. "
           "Public source claims are not our own official score.\n"
        ],
    })
    meta.pop("id_no",None)
    meta.update({
        "id":DEST,"title":"Frozen native NTL ensemble | no new training",
        "is_private":True,"enable_internet":False,"enable_gpu":True,
        "machine_shape":"NvidiaTeslaT4",
    })
    if meta.get("id_no") is not None:
        raise RuntimeError("inherited upstream numeric ID")
    outfile=meta.get("code_file")
    if not isinstance(outfile,str) or Path(outfile).name!=outfile or not outfile.endswith(".ipynb"):
        raise RuntimeError("invalid Notebook output filename")
    path=target/outfile
    path.write_text(json.dumps(notebook,ensure_ascii=False),encoding="utf-8")
    (target/"kernel-metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
    # Validate in-memory and on-disk content match without execution.
    saved=json.loads(path.read_text(encoding="utf-8"))
    if sha(json.dumps(saved,ensure_ascii=False).encode())!=sha(path.read_bytes()):
        raise RuntimeError("generated source re-read failed")
    report={
      "source":SOURCE,"source_sha":SOURCE_SHA,"guarded_sha":sha(path.read_bytes()),
      "source_code_sha":SOURCE_CODE_SHA,"destination":DEST,
      "source_data_mounts":21,"source_kernel_mounts":8,"source_model_mounts":6,
      "guarded_training_paths":counts,
      "no_new_training":True,
      "new_official_submission":False,
    }
    print("NTL_NATIVE_FROZEN_SOURCE_GUARDS="+json.dumps(report,sort_keys=True),flush=True)
    return {"target":target,"report":report}


def submissions() -> list[dict]:
    raw=run(["kaggle","competitions","submissions",CONTEST,"-v","-q"],timeout=100).stdout
    return list(csv.DictReader(raw.lstrip("\ufeff").splitlines()))


def quota() -> None:
    api=KaggleApi()
    api.authenticate()
    lim=api.competition_get_submission_limits(CONTEST)
    if int(lim.num_allowed_now or 0)<1:
        raise RuntimeError("NO_AVAILABLE_NTL_OFFICIAL_SUBMISSION_SLOT")
    rows=submissions()
    if any(x.get("description")==DESCRIPTION for x in rows):
        raise RuntimeError("NTL_NATIVE_ALREADY_SUBMITTED_DO_NOT_REPEAT")
    print("NTL_NATIVE_FROZEN_OFFICIAL_SLOT="+json.dumps({
      "allowed_now":lim.num_allowed_now,
      "already_submitted":False,
      "active_count":sum("PENDING" in (x.get("status") or "").upper() for x in rows),
    },sort_keys=True),flush=True)


def destination_absent() -> None:
    p=run(["kaggle","kernels","status",DEST],checked=False,timeout=80)
    if p.returncode==0 or "KernelWorkerStatus." in (p.stdout+p.stderr):
        raise RuntimeError("own frozen kernel already exists; refuse overwrite")
    # Prove this exact slug is absent from owned kernel list (status may be access denied).
    owner=DEST.split("/")[0]
    seen=0
    for page in range(1,16):
        p=run(["kaggle","kernels","list","--user",owner,"--page-size","100","--page",str(page),"-v"],timeout=100)
        rows=list(csv.DictReader(p.stdout.lstrip("\ufeff").splitlines()))
        if any(x.get("ref","").lower()==DEST.lower() for x in rows):
            raise RuntimeError("exact Kaggle owned kernel already exists")
        seen+=len(rows)
        if len(rows)<100:
            break
    else:
        raise RuntimeError("Kaggle owned kernel pagination overflow")
    print("NTL_NATIVE_FROZEN_UNIQUE_PRIVATE_KERNEL_AVAILABLE="+str(seen),flush=True)


def preview() -> None:
    for n in range(150):
        p=run(["kaggle","kernels","status",DEST],checked=False,timeout=90)
        state=(p.stdout+p.stderr).upper()
        if "KERNELWORKERSTATUS.COMPLETE" in state:
            break
        if any(x in state for x in ("KERNELWORKERSTATUS.ERROR","KERNELWORKERSTATUS.FAILED",
                                     "KERNELWORKERSTATUS.CANCELLED")):
            raise RuntimeError("NATIVE_SOURCE_PREVIEW_FAILED_NO_OFFICIAL_SUBMISSION")
        if n%10==0:
            print("NTL_NATIVE_PRIVATE_PREVIEW_POLL="+str(n+1),flush=True)
        time.sleep(100)
    else:
        raise TimeoutError("NATIVE_SOURCE_PREVIEW_NOT_COMPLETE_NO_RETRY")
    listing=run(["kaggle","kernels","files",DEST,"-v","--page-size","200"],timeout=130).stdout
    if "submission.parquet" not in listing or "Next Page Token =" in listing:
        raise RuntimeError("preview did not produce unambiguous submission.parquet")
    print("NTL_NATIVE_OWN_PREVIEW_COMPLETE_WITH_SUBMISSION_PARQUET=TRUE",flush=True)


def execute() -> None:
    quota()
    destination_absent()
    status=run(["kaggle","kernels","status",SOURCE],timeout=85)
    if "KernelWorkerStatus.COMPLETE" not in (status.stdout+status.stderr):
        raise RuntimeError("upstream exact source no longer complete")
    built=fetch_and_guard(Path("/tmp/ntl-native-frozen-20261112"))
    quota()
    p=run(["kaggle","kernels","push","-p",str(built["target"]),
        "--accelerator","NvidiaTeslaT4","--timeout","32400"],timeout=150)
    if "Kernel version 1 successfully pushed" not in (p.stdout+p.stderr):
        raise RuntimeError("expected new single frozen private Notebook version 1")
    print("NTL_NATIVE_OWN_PRIVATE_KERNEL_VERSION1_PUSHED=TRUE",flush=True)
    preview()
    quota()
    r=run(["kaggle","competitions","submit",CONTEST,
          "-k",DEST,"-v","1","-f","submission.parquet","-m",DESCRIPTION],timeout=160)
    match=re.findall(r"Submission ref:\s*(\d+)",r.stdout+r.stderr,re.I)
    if len(match)!=1:
        raise RuntimeError("Kaggle sent entry but no confirmed numeric receipt; NEVER auto-resubmit")
    ref=match[0]
    print("NTL_NATIVE_ACTUAL_OFFICIAL_SUBMISSION_REF="+ref,flush=True)
    for i in range(16):
        rows=[x for x in submissions() if x.get("ref")==ref]
        if len(rows)==1 and rows[0].get("description")==DESCRIPTION:
            record={k:rows[0].get(k) for k in ("ref","date","status","publicScore","privateScore")}
            print("NTL_NATIVE_OWN_OFFICIAL_LEDGER="+json.dumps(record,sort_keys=True),flush=True)
            return
        time.sleep(10)
    raise RuntimeError("official numeric ref accepted but ledger not yet consistent; no retry")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-preflight",action="store_true")
    args=ap.parse_args()
    if args.source_preflight:
        fetch_and_guard(Path("/tmp/ntl-native-source-preflight-20261112"))
        print("NTL_NATIVE_SOURCE_PREFLIGHT=PASS_NO_GPU_NO_SUBMISSION",flush=True)
    else:
        execute()


if __name__=="__main__":
    main()
