"""Private internet-disabled Kaggle GPU builder for CDA-L24 v2 source training.

No official competition data is mounted. This kernel trains one frozen arm,
outputs a privately scoped Stage-A checkpoint + transfer predictions, and
does NOT represent or initiate an official competition submission.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from scripts.build_two_stage_lora24_kaggle_model_kernel import (
    MODEL_SOURCE, RUNTIME_DATASET_REF, CHECKPOINT_DATASET_REF,
)

KERNEL_IDS = {
    "C0":"kolihhan/ntl-cda-l24-v2-original-source-training-c0-r1",
    "C1":"kolihhan/ntl-cda-l24-v2-original-source-training-c1-r1",
}
COMPETITION = "name-that-llm"
SOURCE_IDS = (
    "arxitekton/ntl-claude-gen-v1",
    "trnhip/ntl-api-gen",
    "kilupy/ntl-18-model-corpus",
    "nishantdahal/ntl-handwriting-blend",
)
REQUIRED_SOURCE_PATHS = (
    "scripts/__init__.py",
    "scripts/run_style_baseline.py",
    "scripts/run_strong_sparse_anchor.py",
    "scripts/run_two_stage_lora24_gate_a.py",
    "scripts/run_two_stage_lora24_gate_b.py",
    "scripts/run_two_stage_lora24_gate_b_gpu.py",
    "scripts/run_phase14_public_sparse.py",
    "scripts/cda_l24_stage_a.py",
    "scripts/cda_l24_checkpoint.py",
    "scripts/conditional_domain_adversarial_lora24.py",
    "scripts/cda_l24_v2_stage_a.py",
)
CODE_FILE="kernel_main.py"

def _kernel_source(*,arm:str, source_texts:Mapping[str,str])->str:
    if arm not in KERNEL_IDS:
        raise ValueError("only C0 and C1 are frozen")
    if set(source_texts)!=set(REQUIRED_SOURCE_PATHS):
        raise ValueError("frozen source set mismatch")
    sources_literal=repr(dict(source_texts))
    return f'''from pathlib import Path
import hashlib,json,shutil,subprocess,sys

ARM = {arm!r}
SOURCES = {sources_literal}
EXPECTED_RUNTIME_REF = {RUNTIME_DATASET_REF!r}
EXPECTED_CHECKPOINT_REF = {CHECKPOINT_DATASET_REF!r}
EXPECTED_MODEL_SOURCE = {MODEL_SOURCE!r}

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()

def find_one(pattern):
    found=sorted(Path("/kaggle/input").glob(pattern))
    if len(found)!=1:raise RuntimeError(f"expected ONE original data file for {{pattern}}, got {{len(found)}}")
    return found[0]

def find_qwen():
    found=[]
    for p in sorted(Path("/kaggle/input/models").rglob("config.json")):
        try:d=json.loads(p.read_text())
        except Exception:continue
        if d.get("model_type")!="qwen3":continue
        if int(d.get("hidden_size",-1))!=4096 or int(d.get("num_hidden_layers",-1))!=36:continue
        if not (p.parent/"tokenizer_config.json").is_file():continue
        if not any(p.parent.glob("*.safetensors")):continue
        found.append(p.parent)
    if len(found)!=1:raise RuntimeError(f"Qwen3-8B model mount ambiguous or missing: {{len(found)}}")
    return found[0]

def verify_runtime(root, lineage):
    observed=[]
    for d in lineage["files"]:
        p=root/d["path"]
        if not p.is_file():raise FileNotFoundError(p)
        h=sha256_file(p)
        if h!=d["sha256"] or p.stat().st_size!=int(d["size"]):
            raise RuntimeError(f"runtime wheelhouse drift {{d['path']}}")
        observed.append({{"path":d["path"],"sha256":h,"size":p.stat().st_size}})
    calc=hashlib.sha256(json.dumps(observed,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if calc!=lineage["file_manifest_sha256"]:
        raise RuntimeError("runtime manifest tampered")

runtime_file=find_one("**/runtime_lineage.json")
runtime=json.loads(runtime_file.read_text())
if runtime.get("dataset_ref")!=EXPECTED_RUNTIME_REF or runtime.get("dataset_version_expected")!=1:
    raise RuntimeError("private wheelhouse version drift")
if runtime.get("competition_train_rows_included")!=0 or runtime.get("competition_test_rows_included")!=0:
    raise RuntimeError("unsafe runtime dataset includes competition rows")
verify_runtime(runtime_file.parent,runtime)
subprocess.check_call([
    sys.executable,"-m","pip","install","-q","--no-index","--no-deps",
    "--find-links",str(runtime_file.parent),
    "transformers==4.57.1","peft==0.17.1","safetensors",
    "tokenizers==0.22.1","huggingface-hub==0.36.0"])
pack=Path("/kaggle/working/cda_l24_v2_pkg")
pack.mkdir(parents=True,exist_ok=True)
for rel,source in SOURCES.items():
    p=pack/rel
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(source,encoding="utf-8")
sys.path.insert(0,str(pack))

checkpoint_meta=find_one("**/a1-checkpoint/meta.json")
arx=find_one("**/ntl-claude-gen-v1/**/passages.csv")
trn=find_one("**/ntl-api-gen/**/passages.csv")
kil=find_one("**/ntl-18-model-corpus/**/ntl_18model_passages.parquet")
prior=find_one("**/pool_v4.parquet")
model_root=find_qwen()
print("PRIVATE_CDA_TRAINING_INPUTS_OK",json.dumps({{
    "arm":ARM,"model_source":EXPECTED_MODEL_SOURCE,
    "data_sha256":{{"arx":sha256_file(arx),"trn":sha256_file(trn),
                    "kil":sha256_file(kil),"prior":sha256_file(prior)}},
    "original_a1_directory":checkpoint_meta.parent.name,
    "official_competition_rows_mounted":0,
}},sort_keys=True),flush=True)

from scripts.cda_l24_v2_stage_a import main as run_stage_a
sys.argv=["cda_l24_v2_stage_a",
          "--arm",ARM,
          "--arx-directory",str(arx.parent),
          "--trn-directory",str(trn.parent),
          "--kil-directory",str(kil.parent),
          "--prior-pool",str(prior),
          "--original-a1",str(checkpoint_meta.parent),
          "--base-snapshot",str(model_root),
          "--private-checkpoint",f"/kaggle/working/cda_l24_v2_{{ARM}}_checkpoint",
          "--private-eval-output",f"/kaggle/working/cda_l24_v2_{{ARM}}_private_predictions.csv"]
run_stage_a()
output=Path(f"/kaggle/working/cda_l24_v2_{{ARM}}_checkpoint")
assert (output/"adapter"/"adapter_model.safetensors").is_file()
assert (output/"heads.safetensors").is_file()
assert (output/"meta.json").is_file()
assert Path(f"/kaggle/working/cda_l24_v2_{{ARM}}_private_predictions.csv").is_file()
print("PRIVATE_CDA_GPU_CANDIDATE_FINISHED",ARM,flush=True)
'''

def write_kernel_bundle(output_dir:Path,*,arm:str,repo_root:Path=Path("."),source_texts:Mapping[str,str]|None=None):
    if arm not in KERNEL_IDS:raise ValueError("invalid frozen arm")
    if source_texts is None:
        source_texts={p:(repo_root/p).read_text(encoding="utf-8") for p in REQUIRED_SOURCE_PATHS}
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/CODE_FILE).write_text(_kernel_source(arm=arm,source_texts=source_texts),encoding="utf-8")
    metadata={
        "id":KERNEL_IDS[arm],
        "title":f"NTL CDA L24 v2 original source training {arm} r1",
        "code_file":CODE_FILE,
        "language":"python",
        "kernel_type":"script",
        "is_private":True,
        "enable_internet":False,
        "machine_shape":"NvidiaTeslaT4",
        "competition_sources":[],
        "dataset_sources":[CHECKPOINT_DATASET_REF,RUNTIME_DATASET_REF,*SOURCE_IDS],
        "kernel_sources":[],
        "model_sources":[MODEL_SOURCE],
    }
    (output_dir/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    compile((output_dir/CODE_FILE).read_text(),str(output_dir/CODE_FILE),"exec")
    return metadata

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm",required=True,choices=list(KERNEL_IDS))
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--repo-root",type=Path,default=Path("."))
    a=p.parse_args()
    print(json.dumps(write_kernel_bundle(a.output_dir,arm=a.arm,repo_root=a.repo_root),sort_keys=True))

if __name__=="__main__":main()
