"""Create a private trainable-encoder Kaggle notebook: one-shot, no submit.

No official texts/ids/labels committed or included in Notebook sources;
only hashed prompt IDs -> original fixed fold index, validated by SHA256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import pandas as pd

DEST = "kolihhan/ntl-tpc-roberta-trainable-ce-contrast-20261124"
METADATA_PARENT = "kolihhan/ntl-scal-source-contrastive-fold0-20261124"
FOLD_SHA = "94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"
FOLDS = Path("artifacts/phase1/folds.csv")
SOURCE = Path("scripts/ntl_trainable_prompt_contrastive_fold0_20261124.py")
DEPENDENCIES = (
    Path("scripts/run_style_baseline.py"),
    Path("scripts/run_strong_sparse_anchor.py")
)
NB_NAME = "ntl_tpc_roberta_trainable_ce_contrast_20261124.ipynb"
PUBLIC_PANELS = [
    "beroli/ntl-benchmark-c1-generated-panels",
    "beroli/ntl-benchmark-c2-generated-panels",
]

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def private_folds():
    assert digest(FOLDS)==FOLD_SHA
    f=pd.read_csv(FOLDS,dtype={"id":str,"prompt_id":str,"model":str})
    assert len(f)==2699 and f.id.is_unique and f.prompt_id.nunique()==150
    assert set(f.fold_id)==set(range(5))
    assert f.groupby("prompt_id").fold_id.nunique().max()==1
    raw=f.groupby("prompt_id").fold_id.first().to_dict()
    hashed={hashlib.sha256(k.encode()).hexdigest():int(v) for k,v in raw.items()}
    assert len(hashed)==150
    return hashed

def code_cell(code):
    compile(code,"<preflight executable cell>","exec")
    return {"cell_type":"code","execution_count":None,"metadata":{},
            "outputs":[],"source":code.splitlines(keepends=True)}

def make(out):
    if out.exists():raise FileExistsError("candidate exists: never replace")
    mapping=private_folds()
    source=SOURCE.read_text()
    assert source.count("FOLD_MAP = {}")==1
    patched=source.replace("FOLD_MAP = {}","FOLD_MAP = "+json.dumps(mapping,sort_keys=True),1)
    compile(patched, "ntl_tpc_roberta_trainable_fold0.py", "exec")
    depcode=(
        "import sys\nfrom pathlib import Path\n"
        "root=Path('/kaggle/working/scripts')\nroot.mkdir(parents=True,exist_ok=True)\n"
        "(root/'__init__.py').write_text('')\n"
        "sys.path.insert(0,'/kaggle/working')\n"
    )
    for dep in DEPENDENCIES:
        content=dep.read_text()
        compile(content,dep.name,"exec")
        depcode+=f"(root/{dep.name!r}).write_text({content!r},encoding='utf-8')\n"
    notebook={"cells":[
        code_cell("import subprocess\nsubprocess.run(['pip','-q','install','--disable-pip-version-check','transformers==4.48.3','tokenizers>=0.21,<0.22','safetensors>=0.4.5','scikit-learn>=1.4,<2','threadpoolctl>=3,<5'],check=True)\n"),
        code_cell(depcode),
        code_cell(patched),
    ],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    with tempfile.TemporaryDirectory(prefix="ntl-tpc-meta-") as tmp:
        subprocess.run(["kaggle","kernels","pull",METADATA_PARENT,"-p",tmp,"-m"],
                       timeout=180,capture_output=True,check=True)
        md=json.loads((Path(tmp)/"kernel-metadata.json").read_text())
    if not md.get("enable_gpu"):raise ValueError("GPU parent not enabled")
    runtime=dict(md)
    runtime.pop("id_no",None)
    runtime.update({"id":DEST,"title":DEST.split("/")[-1],"code_file":NB_NAME,
                    "is_private":True,"kernel_type":"notebook","language":"python",
                    "enable_gpu":True,"enable_internet":True,
                    "competition_sources":["name-that-llm"],
                    "dataset_sources":PUBLIC_PANELS,"kernel_sources":[],"model_sources":[]})
    out.mkdir(parents=True)
    (out/NB_NAME).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
    (out/"kernel-metadata.json").write_text(json.dumps(runtime,indent=1,ensure_ascii=False)+"\n")
    assert len(json.loads((out/NB_NAME).read_text())["cells"])==3
    print("NTL_TPC_NOTEBOOK_PREFLIGHT="+json.dumps({
        "candidate":DEST,"source_sha256":digest(SOURCE),"base_revision":"e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        "frozen_fold_sha256":FOLD_SHA,"num_hashed_folds":len(mapping),
        "scoring_dataset_sources":PUBLIC_PANELS,"private":True,
        "GPU":runtime["enable_gpu"],"internet_public_pinned_model":True,
        "official_training_rows_in_repo":0,
        "competition_submission":False,"model_cache_exported":False,
        "model_dependency_source_inline":True},sort_keys=True),flush=True)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    make(a.out)
