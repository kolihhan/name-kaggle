"""CPU-only candidate: exact original Apex 0.702, change one final caller penalty.

Only source-preparation and validation. No training, GPU or Kaggle submission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

SOURCE="lightningv08/0-702-apex-style-15-glm-qwen3-8-27b-lora"
SOURCE_FILE_SHA="f9948b633b22bd5e63b8253e4d8027c073b44f57eb85bc60b775857e396b38c2"
SCORED_CODE_SHA="2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc"
DEST="kolihhan/ntl-0702-typo-off-onevar-20261119"
PATTERN=r"\bpenalty\s*=\s*0\.15\b"

def sha(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()

def contents(cell):
    s=cell.get("source",[])
    return "".join(s) if isinstance(s,list) else str(s)

def code_cells(nb):
    return [contents(c) for c in nb["cells"] if c.get("cell_type")=="code"]

def prepare(source_dir:Path,dest_dir:Path):
    if dest_dir.exists():
        raise FileExistsError("DESTINATION_ALREADY_EXISTS")
    metadata=source_dir/"kernel-metadata.json"
    books=list(source_dir.glob("*.ipynb"))
    if not metadata.is_file() or len(books)!=1:
        raise ValueError("MISSING_EXACT_PUBLIC_NOTEBOOK_OR_METADATA")
    if sha(books[0].read_bytes())!=SOURCE_FILE_SHA:
        raise ValueError("PUBLIC_SOURCE_FILE_SHA_DRIFT")
    nb=json.loads(books[0].read_text())
    meta=json.loads(metadata.read_text())
    original=code_cells(nb)
    if len(original)!=29 or sha("\n".join(original).encode())!=SCORED_CODE_SHA:
        raise ValueError("PUBLIC_SCORED_EXECUTABLE_SOURCE_DRIFT")
    if (meta.get("id")!=SOURCE or meta.get("is_private") is not False
        or meta.get("enable_internet") is not False
        or meta.get("enable_gpu") is not True
        or meta.get("competition_sources")!=["name-that-llm"]
        or len(meta.get("dataset_sources") or [])!=26
        or len(meta.get("kernel_sources") or [])!=9
        or len(meta.get("model_sources") or [])!=9):
        raise ValueError("SCORING_MOUNT_OR_RUNTIME_DRIFT")
    matches=[]
    for cell_i,cell in enumerate(nb["cells"]):
        if cell.get("cell_type")!="code":
            continue
        src=contents(cell)
        for m in re.finditer(PATTERN,src):
            matches.append((cell_i,src[:m.start()].count("\n")+1))
    if len(matches)!=1:
        raise ValueError("STOP_AMBIGUOUS_PENALTY_015_MATCHES_COUNT="+str(len(matches)))
    index,line=matches[0]
    cell=nb["cells"][index]
    before=contents(cell)
    if before.lstrip().startswith("%%writefile"):
        raise ValueError("STOP_MODULE_DEFAULT_EDIT_NOT_FINAL_CALL")
    after,n=re.subn(PATTERN,"penalty=1.0",before,count=1)
    if n!=1 or after==before:
        raise RuntimeError("NOT_EXACTLY_ONE_CALLSITE_REPLACEMENT")
    cell["source"]=after.splitlines(keepends=True)
    updated=code_cells(nb)
    differences=[i for i,(a,b) in enumerate(zip(original,updated)) if a!=b]
    if len(updated)!=29 or len(differences)!=1:
        raise ValueError("CHANGED_OTHER_EXECUTABLE_CODE")
    at=differences[0]
    if updated[at]!=re.sub(PATTERN,"penalty=1.0",original[at],count=1):
        raise ValueError("CANDIDATE_NOT_ONE_VARIABLE")
    dest_dir.mkdir(parents=True,exist_ok=False)
    meta.pop("id_no",None)
    meta.pop("docker_image",None)
    meta.update({"id":DEST, "title":DEST.split("/",1)[1], "is_private":True,
                 "enable_gpu":True, "enable_tpu":False,"enable_internet":False,
                 "machine_shape":"NvidiaTeslaT4"})
    nb["cells"].insert(0,{"cell_type":"markdown","metadata":{},
        "source":["Attributed one-variable Apex formatting rule test. ",
          "Original: https://www.kaggle.com/code/"+SOURCE+"\n",
          "Only final call penalty=0.15 to penalty=1.0. No efficacy claim."]})
    fname=meta.get("code_file")
    if not fname or Path(fname).name!=fname or not fname.endswith(".ipynb"):
        raise ValueError("INVALID_SOURCE_NOTEBOOK_FILENAME")
    (dest_dir/fname).write_text(json.dumps(nb,ensure_ascii=False))
    (dest_dir/"kernel-metadata.json").write_text(json.dumps(meta,sort_keys=True,indent=2))
    if code_cells(json.loads((dest_dir/fname).read_text()))!=updated:
        raise RuntimeError("KAGGLE_SERIALIZATION_DRIFT")
    return {"candidate":DEST,"source":SOURCE,
        "original_code_sha256":SCORED_CODE_SHA,
        "candidate_code_sha256":sha("\n".join(updated).encode()),
        "changed_cell_index":index,"changed_call_line":line,
        "executable_cells_changed":1,
        "one_difference":"final call penalty=0.15 to penalty=1.0",
        "models_and_input_mounts_unchanged":True,
        "no_gpu_no_training_no_submission":True,
        "official_score":None}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--dest",type=Path,required=True)
    a=p.parse_args()
    print("NTL_APEX_ONE_VARIABLE_CPU="+json.dumps(
        prepare(a.source,a.dest),sort_keys=True),flush=True)
