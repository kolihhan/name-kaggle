"""One model-free diagnostic reproducing Native Notebook CUDA process tree.

Build from the tested source/guarded mapfix builder; stop before loading any
model weights. Capture only CUDA counts and availability from early Notebook,
ensemble parent, and exact core_worker subprocess. No training, scoring, data
reads, or public leaderboard calls.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from scripts import ntl_native_gpu_mapfix_realpath_20261112 as parent

DEST="kolihhan/ntl-native-context-probe-nov12"
TITLE="ntl-native-context-probe-nov12"

# Replace exactly the pre-loader CUDA guard with a diagnostic and controlled
# termination (keeps the rest of original code literally unchanged).
OLD='''    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("NTL_CORE_EXPECTED_TWO_CUDA_DEVICES_BEFORE_MODEL_LOAD")
'''
NEW='''    import subprocess as _ntl_subprocess
    import os as _ntl_os
    _ntl_smi = _ntl_subprocess.run(["nvidia-smi","-L"],capture_output=True,text=True,timeout=25)
    print("NTL_CORE_CONTEXT_PROBE="+_ntl_json.dumps({
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": _ntl_os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_build": torch.version.cuda,
        "torch_version": str(torch.__version__),
        "nvidia_smi_gpu_count":sum(line.startswith("GPU ") for line in _ntl_smi.stdout.splitlines()),
        "nvidia_smi_exit": _ntl_smi.returncode,
    },sort_keys=True),flush=True)
    raise RuntimeError("NTL_CORE_CONTEXT_PROBE_CONTROLLED_STOP_NO_WEIGHTS")
'''

EARLY="""import os,subprocess,json
import torch
p=subprocess.run(["nvidia-smi","-L"],capture_output=True,text=True,timeout=25)
print("NTL_EARLY_NOTEBOOK_CUDA_CONTEXT="+json.dumps({
  "torch_version":torch.__version__,"torch_cuda_build":torch.version.cuda,
  "cuda_available":torch.cuda.is_available(),"device_count":torch.cuda.device_count(),
  "mask":os.environ.get("CUDA_VISIBLE_DEVICES"),
  "nvidia_smi_gpu_count":sum(l.startswith("GPU ") for l in p.stdout.splitlines()),
  "nvidia_smi_exit":p.returncode,
},sort_keys=True),flush=True)
"""

ENSEMBLE_OLD="    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',"
ENSEMBLE_NEW='''    import torch as _ntl_torch
    print("NTL_ENSEMBLE_PARENT_CUDA_CONTEXT="+json.dumps({
      "cuda_available":bool(_ntl_torch.cuda.is_available()),
      "device_count":int(_ntl_torch.cuda.device_count()),
      "parent_mask":os.environ.get("CUDA_VISIBLE_DEVICES"),
      "requested_child_mask":gpus,
      "torch_build":_ntl_torch.version.cuda,
    },sort_keys=True),flush=True)
''' + ENSEMBLE_OLD


def text(cell:dict)->str:
    s=cell.get("source",[])
    return "".join(s) if isinstance(s,list) else str(s)


def make(out:Path)->dict:
    original=parent.make(out)
    kernel=out/"kernel"
    files=list(kernel.glob("*.ipynb"))
    if len(files)!=1:raise ValueError("Expected exactly one Notebook")
    p=files[0]
    nb=json.loads(p.read_text(encoding="utf-8"))
    flags={"core":0,"ensemble":0}
    for cell in nb["cells"]:
        if cell.get("cell_type")!="code":continue
        src=text(cell)
        if OLD in src:
            src=src.replace(OLD,NEW,1)
            cell["source"]=src.splitlines(keepends=True)
            flags["core"]+=1
        if ENSEMBLE_OLD in src and "def run(root, logs, stage" in src:
            src=src.replace(ENSEMBLE_OLD,ENSEMBLE_NEW,1)
            cell["source"]=src.splitlines(keepends=True)
            flags["ensemble"]+=1
    if flags!={"core":1,"ensemble":1}:raise ValueError("Changed source unexpected")
    nb["cells"].insert(0,{"cell_type":"code","id":"ntl-early-gpu-probe-20261112",
      "metadata":{},"execution_count":None,"outputs":[],"source":EARLY.splitlines(keepends=True)})
    all_code="\n".join(text(c) for c in nb["cells"] if c.get("cell_type")=="code")
    if all_code.count("NTL_CORE_CONTEXT_PROBE_CONTROLLED_STOP_NO_WEIGHTS")!=1:
        raise ValueError("No controlled terminal")
    for guard in parent.base.GUARDS:
        if all_code.count(guard)!=1:raise ValueError("No-new-training guard lost")
    meta_path=kernel/"kernel-metadata.json"
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({"id":DEST,"title":TITLE,"is_private":True,
      "enable_gpu":True,"enable_internet":False})
    meta_path.write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    serialized=json.dumps(nb,ensure_ascii=False).encode("utf-8")
    p.write_bytes(serialized)
    return {
      "source":parent.base.SOURCE,"destination":DEST,
      "candidate_sha256":hashlib.sha256(serialized).hexdigest(),
      "exact_patched_sites":flags,"one_early_no_data_probe_cell":True,
      "no_new_training":True,"model_weights_loaded":False,
      "no_official_submission":True,
      "expected_controlled_end":"NTL_CORE_CONTEXT_PROBE_CONTROLLED_STOP_NO_WEIGHTS",
    }


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print("NTL_CONTEXT_PROBE_FROZEN="+json.dumps(make(a.out),sort_keys=True),flush=True)

if __name__=="__main__":
    main()
