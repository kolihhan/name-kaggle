"""Native GPU clean-metadata frozen prefit-only candidate, no new training.

Based only on the user's own already-submitted Native Notebook. Three changes:
1) disable the known-answer 3-row preview path (real model always runs);
2) set q14v5/q8v5 runtime supervised Gaussian flag to 0, keeping all
   model weights and published pretrained readouts unchanged;
3) REBUILD hardware metadata from a safe allowlist, removing legacy
   docker_image that made the original and all inherited clones CPU-only.

No speculative warmup monkeypatch. The model loader is literally unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from scripts import ntl_native_fullpath_smoke_20261112 as base
from scripts.ntl_native_prefit_only_gpu_mapfix_20261112 import disable_gaussian_only

DEST="kolihhan/ntl-native-clean-frozen-prefit-nov12"
TITLE="ntl-native-clean-frozen-prefit-nov12"
ASSETS=("competition_sources","dataset_sources","kernel_sources","model_sources")


def text(cell:dict)->str:
    src=cell.get("source",[])
    return "".join(src) if isinstance(src,list) else str(src)


def make(out:Path)->dict:
    report=base.build(out)
    folder=out/"kernel"
    nbpaths=list(folder.glob("*.ipynb"))
    if len(nbpaths)!=1:raise ValueError("expected one preserved Notebook")
    path=nbpaths[0]
    nb=json.loads(path.read_text(encoding="utf-8"))
    before="\n".join(text(c) for c in nb["cells"] if c.get("cell_type")=="code")
    changed=[]
    for cell in nb["cells"]:
        if cell.get("cell_type")!="code":continue
        code=text(cell)
        if code.startswith("CONFIG = {"):
            revised,policy=disable_gaussian_only(code)
            cell["source"]=revised.splitlines(keepends=True)
            changed.append(policy)
    if len(changed)!=1:raise ValueError("exact frozen CONFIG not unique")
    after="\n".join(text(c) for c in nb["cells"] if c.get("cell_type")=="code")
    if len(nb["cells"])<40:raise ValueError("Native full pipeline cells lost")
    if "preview = False  # diagnostic: force real inference path" not in after:
        raise ValueError("full real inference not selected")
    if "NTL_GPU_DEVICE_ORDINAL_TO_EXPLICIT_CUDA_V1" in after:
        raise ValueError("unproven Transformers loader monkeypatch present")
    if after.count("m = AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map='auto', **placement).eval()")!=1:
        raise ValueError("original AutoModel loader replaced")
    for guard in base.GUARDS:
        if before.count(guard)!=1 or after.count(guard)!=1:
            raise ValueError("supervised-training guard changed or removed")
    meta_path=folder/"kernel-metadata.json"
    original=json.loads(meta_path.read_text(encoding="utf-8"))
    counts=tuple(len(original.get(k) or []) for k in ASSETS)
    if counts!=(1,21,8,6):
        raise ValueError("frozen upstream source mount topology unexpectedly changed")
    clean={
        "id":DEST,"title":TITLE,"code_file":path.name,"language":"python",
        "kernel_type":"notebook","is_private":True,
        "enable_gpu":True,"enable_tpu":False,"enable_internet":False,
        "machine_shape":"NvidiaTeslaT4",
        **{k:original[k] for k in ASSETS},
    }
    if set(clean).intersection(("docker_image","keywords","id_no")):
        raise ValueError("legacy CPU Docker metadata not removed")
    data=json.dumps(nb,ensure_ascii=False).encode("utf-8")
    path.write_bytes(data)
    meta_path.write_text(json.dumps(clean,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    if not json.loads(path.read_text(encoding="utf-8"))["cells"]:
        raise ValueError("generated Notebook corrupt")
    return {
       "source":base.SOURCE,"candidate_slug":DEST,
       "notebook_sha256":hashlib.sha256(data).hexdigest(),
       "source_code_sha256":hashlib.sha256(before.encode()).hexdigest(),
       "candidate_code_sha256":hashlib.sha256(after.encode()).hexdigest(),
       "gaussian_flag_changes":changed[0],
       "original_model_loader_unchanged":True,
       "preview_fixture_disabled":True,
       "pretrained_assets_preserved":{"competition":1,"dataset":21,"kernel":8,"model":6},
       "hardware":"NvidiaTeslaT4",
       "legacy_cpu_docker_image_removed":True,
       "no_new_training":True,"new_official_submission":False,
       "score_known":False,
    }


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=make(args.out)
    print("NTL_CLEAN_FROZEN_GPU_CANDIDATE="+json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
