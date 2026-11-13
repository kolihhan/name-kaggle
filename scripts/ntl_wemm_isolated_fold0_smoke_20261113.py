"""Pinned public WeMM frozen fold0 ONLY private GPU smoke.

No competition submission, no training, no official text/labels in Github.
Distinct from the already-failed Ozertuu/Omkar full Q14+WeMM ensemble.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE = "omkarpattanaik123/name-that-llm-v28-sovereign-titan-apex"
SHA256 = "1702ea2b9c6101c78244ddbf169cc25c87fef61fcaca05937a24822e1b2e71a8"
DIGEST = "a65e0115aaa56e3ce64391e9911fed7d3521c7ace3a1e990d92cc4f1b8a03a82"
DEST = "kolihhan/ntl-wemm-only-fold0-smoke-nov13"
WEMM_SOURCE = "dangkhoa2016/tencent-wemm-embedding-4b/Transformers/default/1"
FROZEN_HEADS = "lightningv08/ntl-olmo2-native24-final-inference-assets"
RUNTIME = "/kaggle/temp/ntl-gsq-v27-runtime"


def cell_text(cell):
    v = cell.get("source", [])
    return "".join(v) if isinstance(v, list) else str(v)


def new_code(source: str, i: int):
    return {"cell_type": "code", "id": f"isolatedwemm{i:02}",
            "metadata": {}, "execution_count": None, "outputs": [],
            "source": source.splitlines(keepends=True)}


def build(source_dir: Path, out_dir: Path):
    nb_files = list(source_dir.glob("*.ipynb"))
    if len(nb_files) != 1:
        raise ValueError("must pull one pinned public notebook")
    raw = nb_files[0].read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("public source notebook changed, stop")
    source_nb = json.loads(raw)
    code = [cell_text(c) for c in source_nb["cells"] if c.get("cell_type") == "code"]
    actual_digest = hashlib.sha256(json.dumps(
        code, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if len(code) != 43 or actual_digest != DIGEST:
        raise ValueError("original runnable public source drift")
    meta = json.loads((source_dir / "kernel-metadata.json").read_text())
    if (meta.get("id") != SOURCE or meta.get("is_private") is not False
            or WEMM_SOURCE not in (meta.get("model_sources") or [])
            or FROZEN_HEADS not in (meta.get("dataset_sources") or [])
            or meta.get("competition_sources") != ["name-that-llm"]):
        raise ValueError("source assets and policy changed")

    src = source_nb["cells"]
    shim = cell_text(src[3])
    stage = cell_text(src[42])
    worker = cell_text(src[51])
    fusion = cell_text(src[53])
    config_text = cell_text(src[54])
    if not (shim.startswith("# Mount shim")
            and stage.startswith("%%writefile " + RUNTIME + "/stage_dataset_weights.py")
            and worker.startswith("%%writefile " + RUNTIME + "/wemm_embedding_worker.py")
            and fusion.startswith("%%writefile " + RUNTIME + "/wemm_fusion.py")):
        raise ValueError("isolated worker sources are not at pinned anchors")
    conf = [line for line in config_text.splitlines()
            if line.startswith("CONFIG['wemm_embedding'] = {")]
    if len(conf) != 1:
        raise ValueError("missing unique WeMM config")
    wemm = ast.literal_eval(conf[0].split(" = ", 1)[1])
    if (sorted(wemm) != ["assets", "base", "deadline_seconds", "share"]
            or wemm["share"] != .07 or wemm["deadline_seconds"] != 30240
            or not all(str(wemm[k]).startswith("/kaggle/input/") for k in ("assets", "base"))):
        raise ValueError("WeMM frozen config not intact")
    if any(t in worker for t in (".fit(", ".train(", "train_final", "torch.optim", ".backward(")):
        raise ValueError("worker has a fitting/training call")
    if any(t in fusion for t in (".fit(", ".train(", ".backward(", "trainable.update")):
        raise ValueError("calibration helper has fitting/training call")
    if "load_file(str(self.assets / f'fold{fold}/trainables.safetensors'))" not in worker:
        raise ValueError("missing frozen fold state load")
    if "def load_wemm(" not in fusion or "def calibrate_wemm(" not in fusion:
        raise ValueError("frozen 18-way calibration source missing")

    preflight = """
from pathlib import Path
import json, subprocess, sys, time
import numpy as np
import pandas as pd
from scipy.special import log_softmax
RUNTIME = Path("/kaggle/temp/ntl-gsq-v27-runtime")
RUNTIME.mkdir(parents=True,exist_ok=True)
WORK = Path("/kaggle/working/wemm-only-fold0")
WORK.mkdir(parents=True,exist_ok=True)
C = _ntl_fix_obj(%r)
BASE, ASSETS = Path(C["base"]), Path(C["assets"])
assert BASE.exists() and ASSETS.exists(), "frozen WeMM base/heads absent"
assert (ASSETS / "fold0" / "trainables.safetensors").is_file()
assert (ASSETS / "fold0" / "calibration.json").is_file()
synthetic = [
    "A quiet library holds thousands of stories in every language.",
    "The mountain path curves through cedar trees toward a clear lake.",
    "A small engineering team revised its test strategy after a review."
]
frame = pd.DataFrame({"id":["synthetic-a","synthetic-b","synthetic-c"], "text":synthetic})
query = WORK / "synthetic-only.csv"
frame.to_csv(query,index=False)
output = WORK / "fold0-raw.npz"
log_file = WORK / "worker.log"
cmd = [sys.executable,"-u",str(RUNTIME/"wemm_embedding_worker.py"),
       "--base",str(BASE),"--assets",str(ASSETS),
       "--query",str(query),"--out",str(output),
       "--work",str(WORK/"scratch"),"--folds","0",
       "--deadline",str(time.monotonic()+2400.0)]
start = time.monotonic()
with log_file.open("w") as log:
    result = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=2500)
print("WEMM_ONLY_FOLD0_EXIT="+str(result.returncode),flush=True)
print("WEMM_ONLY_FOLD0_WALL_S="+str(round(time.monotonic()-start,2)),flush=True)
if result.returncode:
    print("WEMM_ONLY_FOLD0_LOG_TAIL="+log_file.read_text(errors="replace")[-3500:],flush=True)
    raise RuntimeError("frozen WeMM worker did not complete")
assert output.is_file(), "no real frozen fold output"
with np.load(output, allow_pickle=False) as z:
    ids, labels = z["ids"].tolist(),z["labels"].tolist()
    assert ids == frame.id.tolist(), "ID order mismatch"
    assert len(labels)==18 and len(set(labels))==18
    assert z["folds"].tolist()==[0], "not isolated fold0"
    assert z["plain"].shape==(1,3,18)
    assert z["centered"].shape==(1,3,18)
    assert np.isfinite(z["plain"]).all() and np.isfinite(z["centered"]).all()
sys.path.insert(0,str(RUNTIME))
from wemm_fusion import load_wemm
lp,folds=load_wemm([output],ASSETS,ids,labels,np.arange(3,dtype=int))
assert lp.shape==(5,3,18) and folds==[0]
assert np.isfinite(lp).all()
assert np.allclose(np.exp(lp).sum(axis=2),1.0,atol=1e-6)
print("NTL_WEMM_ONLY_FROZEN_REAL_INFERENCE=TRUE",flush=True)
print("NTL_WEMM_FROZEN_FOLD0_SYNTHETIC_ROWS=3",flush=True)
print("NTL_WEMM_FROZEN_FOLD0_CLASS_COUNT=18",flush=True)
print("NTL_WEMM_FROZEN_FOLD0_ARGMAX="+json.dumps([labels[int(x)] for x in lp[0].argmax(axis=1)]),flush=True)
""" % (wemm,)
    new_nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                     "language_info":{"name":"python"}},
        "cells": [
            {"cell_type":"markdown","id":"sourcecredit","metadata":{},
             "source":[
                 "Frozen-head-only technical smoke derived from public Omkar / Ozertuu ",
                 "WeMM source; pretrained WeMM from CHEN Han. ",
                 "No competition source, no training, no official leaderboard submission."]},
            new_code("from pathlib import Path\nPath('" + RUNTIME + "').mkdir(parents=True,exist_ok=True)\n", 1),
            new_code(shim, 2),
            new_code(stage, 3),
            new_code(worker, 4),
            new_code(fusion, 5),
            new_code(preflight, 6),
        ]
    }
    if out_dir.exists():
        raise FileExistsError("one-shot pilot output may not be replaced")
    out_dir.mkdir(parents=True)
    meta = {
        "id":DEST, "title":"ntl-wemm-only-fold0-smoke-nov13",
        "code_file":"wemm-fold0-smoke.ipynb","language":"python",
        "kernel_type":"notebook","is_private":True,
        "enable_gpu":True,"enable_tpu":False,"enable_internet":False,
        "machine_shape":"NvidiaTeslaT4","competition_sources":[],
        "dataset_sources":[FROZEN_HEADS],"model_sources":[WEMM_SOURCE],
        "kernel_sources":[]
    }
    (out_dir/meta["code_file"]).write_text(json.dumps(new_nb, ensure_ascii=False))
    (out_dir/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    return {"source":SOURCE,"source_sha256":SHA256,"code_digest":DIGEST,
            "destination":DEST,"only_frozen_fold":0,
            "synthetic_only_rows":3,"new_training":False,
            "official_competition_submission":False,
            "notebook_sha256":hashlib.sha256((out_dir/meta["code_file"]).read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print("WEMM_ONLY_ISOLATED_BUILD="+json.dumps(build(a.source,a.out),sort_keys=True),flush=True)


if __name__=="__main__":
    main()
