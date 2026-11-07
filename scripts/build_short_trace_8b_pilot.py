"""Build a private internet-disabled Kaggle research pilot from pinned repo source.

The generated script contains no Kaggle official data, model weights, or secrets.
"""
import argparse
import json
from pathlib import Path

KERNEL_REF="kolihhan/ntl-short-trace-8b-pilot"
DATASET_REF="nishantdahal/ntl-handwriting-blend"
RUNTIME_REF="kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2"
MODEL_REF="qwen-lm/qwen-3/transformers/8b-base/1"


def kernel_source(core_source,pilot_source):
    # Use source string literals to avoid network dependencies on repo at Kaggle.
    return f'''import sys
import types
import json
import hashlib
from pathlib import Path
import subprocess

RUNTIME_REF={RUNTIME_REF!r}
POOL_DATASET_REF={DATASET_REF!r}
MODEL_REF={MODEL_REF!r}


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def only_glob(pattern):
    paths=sorted(Path("/kaggle/input").glob(pattern))
    if len(paths)!=1:raise RuntimeError(f"Expected one source for {{pattern}}, found {{len(paths)}}")
    return paths[0]


def find_model():
    roots=[]
    for cp in Path("/kaggle/input/models").rglob("config.json"):
        try:
            config=json.loads(cp.read_text())
        except Exception:
            continue
        if (config.get("model_type")=="qwen3" and
            int(config.get("hidden_size",-1))==4096 and
            int(config.get("num_hidden_layers",-1))==36 and
            (cp.parent/"tokenizer_config.json").is_file() and
            any(cp.parent.glob("*.safetensors"))):
            roots.append(cp.parent)
    if len(roots)!=1:raise RuntimeError(f"Expected pinned Qwen3-8B root, found {{len(roots)}}")
    return roots[0]


runtime=only_glob("**/runtime_lineage.json")
lineage=json.loads(runtime.read_text())
if lineage.get("dataset_ref")!=RUNTIME_REF or int(lineage.get("competition_train_rows_included",-1))!=0 or int(lineage.get("competition_test_rows_included",-1))!=0:
    raise RuntimeError("Runtime lineage mismatch or official competition rows present")
for entry in lineage["files"]:
    path=runtime.parent/entry["path"]
    if not path.is_file() or sha256_file(path)!=entry["sha256"] or path.stat().st_size!=int(entry["size"]):
        raise RuntimeError("Runtime package byte drift")
wheels=runtime.parent
subprocess.check_call([sys.executable,"-m","pip","install","-q","--no-index","--find-links",str(wheels),
     "--no-deps","transformers==4.57.1","tokenizers==0.22.1","huggingface-hub==0.36.0","safetensors"])

# Only external data and frozen 8B evaluator, NO competition sources.
pool=only_glob("**/pool_v4.parquet")
model=find_model()

pkg=types.ModuleType("scripts")
pkg.__path__=[]
sys.modules["scripts"]=pkg
module=types.ModuleType("scripts.short_trace_core")
module.__package__="scripts"
exec({core_source!r},module.__dict__)
sys.modules["scripts.short_trace_core"]=module
setattr(pkg,"short_trace_core",module)

run_globals={{"__name__":"__main__","__package__":"scripts"}}
exec({pilot_source!r},run_globals)
run_globals["run"](pool,model,Path("/kaggle/working/short_trace_8b_result.json"))
'''


def build(out,repo_root):
    out.mkdir(parents=True,exist_ok=True)
    core=(repo_root/"scripts/short_trace_core.py").read_text(encoding="utf8")
    runner=(repo_root/"scripts/run_short_trace_8b_pilot.py").read_text(encoding="utf8")
    source=kernel_source(core,runner)
    compile(source,"kernel_main.py","exec")
    (out/"kernel_main.py").write_text(source)
    (out/"kernel-metadata.json").write_text(json.dumps({
        "id":KERNEL_REF, "title":"NTL Short-TRACE 8B frozen representation pilot",
        "code_file":"kernel_main.py","language":"python","kernel_type":"script",
        "is_private":True,"enable_internet":False,
        "machine_shape":"NvidiaTeslaT4",
        "competition_sources":[],
        "dataset_sources":[DATASET_REF,RUNTIME_REF],
        "kernel_sources":[],
        "model_sources":[MODEL_REF],
    },indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,default=Path("."))
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    build(a.output_dir,a.repo_root)
    print("SHORT_TRACE_8B_KERNEL_BUNDLE_BUILT",KERNEL_REF)
