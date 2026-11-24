"""Build a single private Kaggle GPT sibling ICL study using pinned 14B model.

No source passages, official train, prompts or model weights are stored in GitHub.
The Kaggle kernel does frozen forward-only Qwen3-14B inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

DONOR="kolihhan/ntl-evgen-forensic-eda-repro-20261123"
DONOR_SHA="74d86affa9f55464ab39f735f392b54e47e2d8a2b631a0bca45de9435c600c56"
DEST="kolihhan/ntl-qwen14b-gpt-triad-icl-private-20261124"
MODEL="qwen-lm/qwen-3/Transformers/14b/1"
C1="beroli/ntl-benchmark-c1-generated-panels"
OFFLINE_WHEELS="dhruvtibarewal/nlm-offline-wheels"
SCRIPT=Path("scripts/ntl_gpt14b_icl_inference_20261124.py")

def cell(code):
    compile(code,"<qwen14b-icl-build-cell>","exec")
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":code.splitlines(keepends=True)}

def build(source,out):
    if out.exists():raise FileExistsError("one-shot destination exists already")
    md=json.loads((source/"kernel-metadata.json").read_text())
    if md.get("id")!=DONOR or not md.get("is_private"):
        raise ValueError("donor private id mismatch")
    parent=source/Path(md["code_file"]).name
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=DONOR_SHA:
        raise ValueError("private 0.718 original exact source drift")
    if MODEL not in md.get("model_sources",[]):
        raise ValueError("Qwen3 14B missing from scored pipeline assets")
    if OFFLINE_WHEELS not in md.get("dataset_sources",[]):
        raise ValueError("pinned offline bitsandbytes wheel missing")
    code=SCRIPT.read_text()
    compile(code,SCRIPT.name,"exec")
    runner=cell(
      "from pathlib import Path\n"
      "import sys\n"
      "Path('/kaggle/working/ntl_gpt14b_icl_inference.py').write_text("+repr(code)+")\n"
      "print('NTL_14B_SOURCE_WRITTEN_EPHEMERALLY=TRUE',flush=True)\n"
    )
    setup=cell(
      "import glob,subprocess,sys\n"
      "w=glob.glob('/kaggle/input/**/bitsandbytes-*.whl',recursive=True)\n"
      "if w:\n"
      "    p=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','-q']+w,check=False)\n"
      "    if p.returncode:raise RuntimeError('offline bitsandbytes install failed')\n"
      "import torch,transformers,bitsandbytes,accelerate\n"
      "print('NTL_14B_OFFLINE_RUNTIME_READY='+str(torch.cuda.device_count()),flush=True)\n"
      "if torch.cuda.device_count()<2:raise RuntimeError('requires pinned 2xT4 GPU')\n"
    )
    main=cell(
      "import hashlib,subprocess,sys\n"
      "from pathlib import Path\n"
      "root=Path('/kaggle/input')\n"
      "def find(name,digest):\n"
      "    found=[]\n"
      "    for p in root.rglob(name):\n"
      "        if p.is_file():\n"
      "            with p.open('rb') as f:\n"
      "                if hashlib.file_digest(f,'sha256').hexdigest()==digest:found.append(p)\n"
      "    if len(found)!=1:raise RuntimeError('required source not uniquely found: '+name)\n"
      "    return found[0]\n"
      "official=find('train.csv','40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707')\n"
      "c1=find('passages.csv','8746c4b77c2b09b9abafaeecc381a39321b9919214c005a3657042051c51b177')\n"
      "p=subprocess.run([sys.executable,'/kaggle/working/ntl_gpt14b_icl_inference.py',"
      "'--official',str(official),'--c1',str(c1)],check=False)\n"
      "if p.returncode:raise RuntimeError('Qwen3-14B real three-arm frozen inference failed')\n"
    )
    nb={"cells":[runner,setup,main],"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"},
      "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    new=dict(md)
    new.update({
      "id":DEST,"title":DEST.split("/",1)[1],
      "code_file":"ntl-qwen14b-gpt-triad-icl-private-20261124.ipynb",
      "language":"python","kernel_type":"notebook","is_private":True,
      "enable_gpu":True,"enable_internet":False,
      "competition_sources":["name-that-llm"],
      "model_sources":[MODEL],
      "dataset_sources":[C1,OFFLINE_WHEELS],
      "kernel_sources":[],
    })
    out.mkdir(parents=True,exist_ok=False)
    (out/new["code_file"]).write_text(json.dumps(nb,indent=1,ensure_ascii=False)+"\n")
    (out/"kernel-metadata.json").write_text(json.dumps(new,indent=1)+"\n")
    print("NTL_GPT14B_PRIVATE_BUILD="+json.dumps({
       "ref":DEST,"source_donor_sha":DONOR_SHA,
       "model_sources":new["model_sources"],"dataset_sources":new["dataset_sources"],
       "competition":new["competition_sources"],"gpu":new["enable_gpu"],
       "code_cells":len(nb["cells"]),"single_pretrained_14b_frozen":True,
       "official_submission":False,"any_training":False
    },sort_keys=True),flush=True)

def smoke():
    assert DEST.split("/")[1].startswith("ntl-qwen14b")
    assert len(MODEL)>20 and MODEL in ("qwen-lm/qwen-3/Transformers/14b/1",)
    assert C1.startswith("beroli/")
    print("NTL_GPT14B_BUILDER_CPU_SELFTEST=PASS",flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--source",type=Path)
    a.add_argument("--out",type=Path)
    a.add_argument("--smoke",action="store_true")
    x=a.parse_args()
    smoke()
    if not x.smoke:
        if x.source is None or x.out is None:a.error("--source and --out are required")
        build(x.source,x.out)
