"""Build frozen Short-TRACE 8B *competition* Notebook; code only, no data.

This is distinct from the external-only v1 pilot. It uses official train inside
Kaggle's private scoring container; does not publish official text/predictions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

KERNEL_REF="kolihhan/ntl-short-trace-8b-direct-18class"
KERNEL_TITLE="ntl-short-trace-8b-direct-18class"
MODEL_REF="qwen-lm/qwen-3/transformers/8b-base/1"
WHEEL_REF="kolihhan/ntl-short-trace-offline-bnb-0502"
RUNTIME_REF="kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2"
BNB_WHEEL_SHA="55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283"


def kernel_source(core,pilot,submission):
    return f'''import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

BNB_WHEEL_SHA={BNB_WHEEL_SHA!r}
RUNTIME_REF={RUNTIME_REF!r}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def only(pattern):
    files=sorted(Path('/kaggle/input').glob(pattern))
    if len(files)!=1:raise RuntimeError(f'Expected exactly one {{pattern}}, got {{len(files)}}')
    return files[0]


# The wheel is a byte-pinned publicly released PyPI package, not competition data.
wheel=only('**/bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl')
if sha(wheel)!=BNB_WHEEL_SHA:raise RuntimeError('Vendor bnb wheel SHA mismatch')
lineage_path=only('**/runtime_lineage.json')
lineage=json.loads(lineage_path.read_text())
if (lineage.get('dataset_ref')!=RUNTIME_REF or
    lineage.get('competition_train_rows_included')!=0 or
    lineage.get('competition_test_rows_included')!=0):
    raise RuntimeError('Runtime lineage invalid')
for entry in lineage['files']:
    p=lineage_path.parent/entry['path']
    if not p.is_file() or sha(p)!=entry['sha256'] or p.stat().st_size!=int(entry['size']):
        raise RuntimeError('Runtime file drift')
# No network access in either preview or competition scoring rerun.
subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-index','--no-deps',
                       '--find-links',str(lineage_path.parent),
                       'transformers==4.57.1','tokenizers==0.22.1',
                       'huggingface-hub==0.36.0','safetensors'])
subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-index',
                       '--no-deps',str(wheel)])
import importlib.metadata
if importlib.metadata.version('bitsandbytes')!='0.50.2':
    raise RuntimeError('Unexpected bitsandbytes version')

# Bundle all code into script; no external network or GitHub checkout on Kaggle.
pkg=types.ModuleType('scripts')
pkg.__path__=[]
sys.modules['scripts']=pkg
source_modules={{
  'short_trace_core':{core!r},
  'run_short_trace_8b_pilot':{pilot!r},
  'run_short_trace_direct_submit':{submission!r},
}}
for name,s in source_modules.items():
    mod=types.ModuleType('scripts.'+name)
    mod.__package__='scripts'
    mod.__name__='scripts.'+name
    sys.modules[mod.__name__]=mod
    setattr(pkg,name,mod)
    exec(compile(s,'scripts/'+name+'.py','exec'),mod.__dict__)

source_modules.clear()
from scripts.run_short_trace_direct_submit import run
result=run()
print('SHORT_TRACE_SUBMISSION_FINAL',json.dumps(result,sort_keys=True),flush=True)
'''


def build(root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    core=(root/"scripts/short_trace_core.py").read_text(encoding="utf8")
    pilot=(root/"scripts/run_short_trace_8b_pilot.py").read_text(encoding="utf8")
    submission=(root/"scripts/run_short_trace_direct_submit.py").read_text(encoding="utf8")
    source=kernel_source(core,pilot,submission)
    compile(source,"kernel_main.py","exec")
    (out/"kernel_main.py").write_text(source,encoding="utf8")
    meta={
      "id":KERNEL_REF,"title":KERNEL_TITLE,
      "code_file":"kernel_main.py","language":"python","kernel_type":"script",
      "is_private":True,"enable_internet":False,"machine_shape":"NvidiaTeslaT4",
      "competition_sources":["name-that-llm"],
      "dataset_sources":[RUNTIME_REF,WHEEL_REF],
      "model_sources":[MODEL_REF],"kernel_sources":[]
    }
    (out/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    return meta


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--repo-root",type=Path,default=Path("."))
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    meta=build(args.repo_root,args.output_dir)
    print("SHORT_TRACE_DIRECT_BUNDLE_READY",json.dumps(meta,sort_keys=True))
