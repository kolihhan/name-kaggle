"""One precise Transformers warmup GPU-ordinal compatibility experiment.

Build from the already-owned no-training Native Notebook. First remove the
known-fixture preview via the existing frozen builder, then modify exactly one
additional core_worker.py model-load call site. Do not modify classifier logic,
model weights, train flags, datasets, or existing official Kaggle entries.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import ntl_native_fullpath_smoke_20261112 as base

DEST = "kolihhan/ntl-native-gpu-mapfix-smoke-nov12"
TITLE = "ntl-native-gpu-mapfix-smoke-nov12"
MODEL_LINE = "    m = AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map='auto', **placement).eval()"
POLICY = "NTL_GPU_DEVICE_ORDINAL_TO_EXPLICIT_CUDA_V1"

def normalize_device_map(mapping: dict[str, Any], num_cuda_devices: int) -> dict[str, Any]:
    """Convert only valid integer GPU ordinals; preserve CPU/disk/string mappings."""
    if not isinstance(num_cuda_devices, int) or num_cuda_devices < 1:
        raise ValueError("No CUDA devices visible; do not silently run on CPU")
    mapped={}
    for key,value in mapping.items():
        if type(value) is int:
            if value < 0 or value >= num_cuda_devices:
                raise ValueError("accelerator device index outside visible CUDA devices")
            mapped[key]=f"cuda:{value}"
        else:
            mapped[key]=value
    return mapped

# Change only the warmup's expanded device-map ordinal interpretation.
# This addresses a PyTorch/Transformers compatibility fault where
# torch.device(0) raises despite devices being addressed explicitly as cuda:0.
# The original loader's device_map='auto' and max_memory are NOT changed.
INJECT = """
    # NTL_GPU_DEVICE_ORDINAL_TO_EXPLICIT_CUDA_V1 -- model loader compatibility
    import transformers.modeling_utils as _ntl_mutils
    import json as _ntl_json
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("NTL_CORE_EXPECTED_TWO_CUDA_DEVICES_BEFORE_MODEL_LOAD")
    torch.cuda.set_device(0)
    _ntl_prior_warmup = _ntl_mutils.caching_allocator_warmup
    def _ntl_safe_warmup(model, expanded_device_map, hf_quantizer):
        fixed = {}
        for param_name, raw_device in expanded_device_map.items():
            if type(raw_device) is int:
                if raw_device < 0 or raw_device >= torch.cuda.device_count():
                    raise RuntimeError("NTL_CORE_DEVICE_MAP_ORDINAL_INVALID")
                fixed[param_name] = f"cuda:{raw_device}"
            else:
                fixed[param_name] = raw_device
        return _ntl_prior_warmup(model, fixed, hf_quantizer)
    _ntl_mutils.caching_allocator_warmup = _ntl_safe_warmup
    print("NTL_CORE_GPU_LOADER_PROBE=" + _ntl_json.dumps({
        "policy": "NTL_GPU_DEVICE_ORDINAL_TO_EXPLICIT_CUDA_V1",
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "visible_mask": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_build": torch.version.cuda,
    }, sort_keys=True), flush=True)
"""


def code_text(cell: dict) -> str:
    s=cell.get("source", [])
    return "".join(s) if isinstance(s,list) else str(s)


def make(out: Path) -> dict:
    # This first creates the preview-disabled Notebook with all three frozen
    # original no-training guards intact, using known previously executed code.
    pre=base.build(out)
    kernel=out/"kernel"
    meta_path=kernel/"kernel-metadata.json"
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    notebooks=list(kernel.glob("*.ipynb"))
    if len(notebooks)!=1:
        raise RuntimeError("unexpected private smoke Notebook files")
    path=notebooks[0]
    nb=json.loads(path.read_text(encoding="utf-8"))
    before="\n".join(code_text(c) for c in nb["cells"] if c.get("cell_type")=="code")
    if before.count(MODEL_LINE)!=1 or POLICY in before:
        raise RuntimeError("unknown loader call-site or already patched source")
    if "preview = False  # diagnostic: force real inference path" not in before:
        raise RuntimeError("preview fixture not disabled")
    modified=0
    for c in nb["cells"]:
        if c.get("cell_type")!="code":
            continue
        source=code_text(c)
        if MODEL_LINE in source:
            if "def l24_features(" not in source:
                raise RuntimeError("unexpected model-loader location")
            source=source.replace(MODEL_LINE,INJECT+"\n"+MODEL_LINE,1)
            c["source"]=source.splitlines(keepends=True)
            modified+=1
    if modified!=1:
        raise RuntimeError("model loader not modified exactly once")
    after="\n".join(code_text(c) for c in nb["cells"] if c.get("cell_type")=="code")
    if after != before.replace(MODEL_LINE,INJECT+"\n"+MODEL_LINE,1):
        raise RuntimeError("unexpected additional model/policy changes")
    for flag in base.GUARDS:
        if before.count(flag)!=1 or after.count(flag)!=1:
            raise RuntimeError("frozen no-training guard lost")
    if after.count(POLICY)!=2:  # marker in comment and diagnostics JSON
        raise RuntimeError("unexpected compatibility policy multiplicity")
    meta.update({"id":DEST,"title":TITLE,"is_private":True,
                 "enable_gpu":True,"enable_internet":False})
    meta.pop("id_no",None)
    path.write_text(json.dumps(nb,ensure_ascii=False),encoding="utf-8")
    meta_path.write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    final=json.loads(path.read_text(encoding="utf-8"))
    if len(final["cells"])!=len(nb["cells"]):
        raise RuntimeError("Notebook reserialization modified cells")
    return {
        "source":base.SOURCE,"destination":DEST,
        "source_code_sha256":hashlib.sha256(before.encode()).hexdigest(),
        "candidate_code_sha256":hashlib.sha256(after.encode()).hexdigest(),
        "candidate_notebook_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_change_count":1,"preview_fixture_disabled":True,
        "no_new_training":True,"guard_count":len(base.GUARDS),
        "competition_submission":False,"model_weights_changed":False,
        "model_loader_api_changed":False,
        "policy":POLICY,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=make(a.out)
    print("NTL_GPU_MAPFIX_FROZEN_SOURCE="+json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
