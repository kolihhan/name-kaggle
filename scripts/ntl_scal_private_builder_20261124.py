"""Build exact PRIVATE Kaggle GPU research notebook without exporting train data.

Folds use only SHA256(prompt_id) -> frozen fold ID, no raw IDs or passages
inside notebook code. No generated files are committed to GitHub.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import pandas as pd

DEST = "kolihhan/ntl-scal-source-contrastive-fold0-20261124"
SOURCE = "kolihhan/ntl-estopadilla-glm53-0716-repro-20261122"
FOLDS_SHA = "94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"
CODE = Path("scripts/ntl_scal_source_likelihood_fold0_20261124.py")
FOLDS = Path("artifacts/phase1/folds.csv")
NAME = "ntl_scal_source_contrastive_fold0_20261124.ipynb"

def digest(path):
    with open(path,"rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def safe_fold_mapping():
    assert digest(FOLDS)==FOLDS_SHA,"frozen fold bytes changed"
    x=pd.read_csv(FOLDS,dtype={"id":str,"prompt_id":str,"model":str})
    assert len(x)==2699 and x.id.is_unique and x.prompt_id.nunique()==150
    assert set(x.fold_id)==set(range(5)) and x.groupby("prompt_id").fold_id.nunique().max()==1
    prompts=x.groupby("prompt_id").fold_id.first().to_dict()
    h={hashlib.sha256(p.encode()).hexdigest():int(fold) for p,fold in prompts.items()}
    assert len(h)==150 and all(len(k)==64 for k in h)
    return h

def cell(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":src.splitlines(keepends=True)}

def build(where):
    where.mkdir(parents=True,exist_ok=False)
    folds=safe_fold_mapping()
    source=CODE.read_text()
    assert source.count("FOLD_MAP = {}")==1
    injected=source.replace("FOLD_MAP = {}", "FOLD_MAP = "+json.dumps(folds,sort_keys=True),1)
    compile(injected,"ntl_scal_source_likelihood_fold0_20261124.py","exec")
    notebook={"cells":[
        cell("!pip -q install --disable-pip-version-check 'transformers==4.48.3' 'tokenizers>=0.21,<0.22' 'safetensors>=0.4.5' 'scikit-learn>=1.4,<2'\n"),
        cell(injected)
      ],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
         "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    with tempfile.TemporaryDirectory(prefix="ntl-scal-source-meta-") as d:
        subprocess.run(["kaggle","kernels","pull",SOURCE,"-p",d,"-m"],
                       check=True,capture_output=True,timeout=150)
        pm=json.loads((Path(d)/"kernel-metadata.json").read_text())
    assert pm.get("enable_gpu") or pm.get("machine_shape"),"no verified GPU metadata parent"
    md=dict(pm)
    md.pop("id_no",None)
    md.update({"id":DEST,"title":DEST.split("/")[-1],"code_file":NAME,
         "language":"python","kernel_type":"notebook","is_private":True,
         "enable_gpu":True,"enable_internet":True,
         "dataset_sources":[],"kernel_sources":[],"model_sources":[],
         "competition_sources":["name-that-llm"]})
    (where/NAME).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
    (where/"kernel-metadata.json").write_text(json.dumps(md,ensure_ascii=False,indent=1)+"\n")
    assert json.loads((where/NAME).read_text())["cells"][1]["cell_type"]=="code"
    print("NTL_SCAL_PRIVATE_NOTEBOOK_BUNDLE="+json.dumps({
       "source_script_sha":digest(CODE),"fold_hash_verified":True,
       "fold_groups":len(folds),"code_cells":2,
       "masked_training_source":"official Kaggle TRAIN mounted only in private Kaggle",
       "gpu":md["enable_gpu"],"internet_for_pinned_pretrained_model":md["enable_internet"],
       "private":True,"no_raw_official_data_export":True,
       "no_competition_submission":True,"slug":DEST,
       "machine_shape":str(md.get("machine_shape"))},sort_keys=True),flush=True)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    build(a.out)
