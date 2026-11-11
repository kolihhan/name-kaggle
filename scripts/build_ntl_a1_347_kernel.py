"""Build exactly one private inference-only independent LoRA24 A1 kernel.

Reuses existing verified Stage-1 Kaggle packaging. Adds only a new frozen
external-cohort inference entrypoint; no fitting or competition submit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in ("", None):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_lora24_sibling_holdout_kernel import (
    CHECKPOINT_DATASET,
    CODE_FILE,
    MODEL_SOURCE,
    RUNTIME_DATASET,
    SOURCES as ORIGINAL_SOURCES,
    kernel_source,
)

KERNEL_REF = "kolihhan/ntl-a1-paired347-inference"
KERNEL_TITLE = KERNEL_REF.split("/",1)[1]
NEW_MODULE = "scripts/ntl_lora24_a1_paired347_inference.py"
PUBLIC_DATASETS = (
    "nimithashreea/ntl-proxy-candidate-generations-20261110",
    "toanplam/ntl-fresh-generation-pilot-nov08",
)

TAIL = '''
nimitha = unique("**/generated_proxy_candidates.csv")
toan = unique("**/passages.jsonl")
model = find_qwen()
from scripts.ntl_lora24_a1_paired347_inference import run
report = run(
    nimitha_file=nimitha,
    toan_file=toan,
    checkpoint_dir=checkpoint,
    model_root=model,
    output_dir=Path("/kaggle/working"),
)
if not report["no_training"] or report["competition_submission"]:
    raise RuntimeError("unexpected training or competition submission")
if report["n"] != 347 or report["true_labels_observed"] != 12:
    raise RuntimeError("incorrect independent test population")
print("NTL_A1_FROZEN_347_DONE", json.dumps({
    "rows":report["n"],
    "conditional_balanced_accuracy_12_label":
        report["combined_exploratory"][
            "conditional_balanced_accuracy_12_truth_18_predictions"],
    "training":False,
    "competition_submission":False,
},sort_keys=True),flush=True)
'''


def build(outdir: Path, *, root: Path=Path(".")) -> dict:
    outdir.mkdir(parents=True,exist_ok=True)
    sources = {
        name:(root/name).read_text(encoding="utf-8")
        for name in ORIGINAL_SOURCES
    }
    new_code=(root/NEW_MODULE).read_text(encoding="utf-8")
    code=kernel_source(sources)
    marker="PIP = "
    if code.count(marker)!=1:
        raise RuntimeError("original verified packaging changed: pip marker")
    code=code.replace(marker,
        "SOURCES["+repr(NEW_MODULE)+"] = "+repr(new_code)+"\n"+marker,1)
    target='\npool = unique("**/pool_v4.parquet")'
    if code.count(target)!=1:
        raise RuntimeError("original verified holdout entrypoint changed")
    code=code.split(target,1)[0]+"\n"+TAIL.lstrip()
    compile(code,CODE_FILE,"exec")
    if ("kaggle competitions submit" in code or
        "optimizer.step(" in TAIL or ".backward(" in TAIL):
        raise RuntimeError("forbidden competition submit or train call")
    metadata={
        "id":KERNEL_REF,
        "title":KERNEL_TITLE,
        "code_file":CODE_FILE,
        "language":"python",
        "kernel_type":"script",
        "is_private":True,
        "enable_gpu":True,
        "enable_internet":False,
        "machine_shape":"NvidiaTeslaT4",
        "competition_sources":[],
        "dataset_sources":[CHECKPOINT_DATASET,RUNTIME_DATASET,*PUBLIC_DATASETS],
        "kernel_sources":[],
        "model_sources":[MODEL_SOURCE],
    }
    assert len(metadata["dataset_sources"])==4
    (outdir/CODE_FILE).write_text(code,encoding="utf-8")
    (outdir/"kernel-metadata.json").write_text(
        json.dumps(metadata,indent=2,sort_keys=True),encoding="utf-8")
    return {
        "code_bytes":len(code.encode()),
        "sources_inlined":len(ORIGINAL_SOURCES)+1,
        "dataset_count":4,
        "model_count":1,
        "competition_sources":0,
        "private":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--repo-root",default=".",type=Path)
    a=p.parse_args()
    result=build(a.output_dir,root=a.repo_root)
    print("NTL_A1_FROZEN_KERNEL_BUILT",json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
