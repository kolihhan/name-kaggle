"""Build a private *unsubmitted* NTL 0.715 Champion + label-only joint-decode
candidate. No new training, GPU inference or Kaggle competition submission.

Source: owner's scored Estopadilla reproduction kernel. Reproduce source
code/metadata exactly except new private kernel ID + ONE appended final cell.
This creates an exact candidate to test later if source/runtime gate permits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

BASE="kolihhan/ntl-estopadilla-glm53-0716-repro-20261122"
DEST="kolihhan/ntl-0715-label-only-joint-20261122-v1"
RUNTIME=Path("kaggle/ntl_0715_label_only_joint_final_20261122.py")
NOTEBOOK_TITLE="NTL 0.715 Label-Only Joint Decoder Research"
COMP="name-that-llm"

def code_cell(source):
    return {
        "cell_type":"code",
        "execution_count":None,
        "metadata":{},
        "outputs":[],
        "source":source.splitlines(keepends=True),
    }

def build(download:Path,output:Path)->dict:
    if output.exists():
        raise FileExistsError("candidate dir must not exist")
    md=download/"kernel-metadata.json"
    if not md.is_file():raise FileNotFoundError("scored source metadata missing")
    data=json.loads(md.read_text())
    if data.get("id")!=BASE:
        raise ValueError("source notebook must be exact owned scored kernel")
    filename=Path(str(data.get("code_file",""))).name
    if not filename.endswith(".ipynb"):
        raise ValueError("source requires real Jupyter notebook")
    source_path=download/filename
    if not source_path.is_file():
        raise FileNotFoundError("source notebook bytes unavailable")
    notebook=json.loads(source_path.read_text(encoding="utf-8"))
    cells=notebook.get("cells")
    if not isinstance(cells,list) or len(cells)<3:
        raise ValueError("source did not have a recognizable cell structure")
    original_cell_count=len(cells)
    original_code="\n".join("".join(c.get("source",[]))
        for c in cells if c.get("cell_type")=="code")
    if "submission.parquet" not in original_code:
        raise RuntimeError("scored champion is not known to write a final submission")
    if "KAGGLE_API_TOKEN" in original_code:
        raise RuntimeError("source asks for competition credentials")
    script=RUNTIME.read_text(encoding="utf-8")
    compile(script,str(RUNTIME),"exec")
    if "competition_submit" in script or "get_secret(" in script:
        raise RuntimeError("unapproved submit or secrets access")
    append=(
        "# Last cell: inference-only group adjustment. Preserve scored source on failure.\n"
        "import os\n"
        "os.environ['NTL_SCORER_DATA_DIR']=str(globals().get('DATA',''))\n"
        f"runtime_code = {script!r}\n"
        "try:\n"
        "    exec(compile(runtime_code,'ntl_0715_label_only_joint_runtime.py','exec'),"
        " {'__name__':'__main__'})\n"
        "except Exception as exc:\n"
        "    print('NTL_0715_LABEL_ONLY_JOINT_FAIL_OPEN_KEEP_SCORED_OUTPUT='+type(exc).__name__)\n"
    )
    notebook["cells"].append(code_cell(append))
    revised=dict(data)
    revised.pop("id_no",None)
    revised["id"]=DEST
    revised["title"]=NOTEBOOK_TITLE
    revised["code_file"]=filename
    revised["is_private"]=True
    # NO change to image, model_sources, dataset_sources, kernel_sources,
    # kernel_type, GPU/machine shape or internet flags.
    for key in ("model_sources","dataset_sources","kernel_sources",
                "competition_sources","enable_gpu","machine_shape"):
        if revised.get(key)!=data.get(key):
            raise AssertionError("baseline Kaggle metadata unexpectedly changed: "+key)
    if COMP not in (data.get("competition_sources") or []):
        raise ValueError("private scored source not attached to exact competition")
    output.mkdir(parents=True)
    (output/filename).write_text(
        json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8"
    )
    (output/"kernel-metadata.json").write_text(
        json.dumps(revised,indent=2,ensure_ascii=False)+"\n"
    )
    if len(notebook["cells"])!=original_cell_count+1:
        raise AssertionError("unexpected cell delta")
    return {
        "source_kernel":BASE,
        "candidate_kernel":DEST,
        "original_cell_count":original_cell_count,
        "candidate_cell_count":len(notebook["cells"]),
        "source_code_sha256":hashlib.sha256(original_code.encode()).hexdigest(),
        "appended_runtime_sha256":hashlib.sha256(script.encode()).hexdigest(),
        "candidate_notebook_sha256":hashlib.sha256(
            (output/filename).read_bytes()).hexdigest(),
        "source_model_mount_count":len(data.get("model_sources") or []),
        "source_kernel_mount_count":len(data.get("kernel_sources") or []),
        "source_data_mount_count":len(data.get("dataset_sources") or []),
        "source_gpu_unchanged":True,
        "private_candidate_created":True,
        "new_GPU":False,"competition_submitted":False,
        "strong_champion_labels_only":True,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    info=build(a.source,a.out)
    print("NTL_0715_LABEL_ONLY_CANDIDATE_STATIC="+json.dumps(
        info,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
