"""Frozen NTL Lightning0694 1200-text throughput pilot (no labels, no training).

Reuses the exact successful 300-row builder, changes only amount of input.
Scientific model config, pretrained mounts, and no-fit guards are untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from scripts import ntl_lightning_0694_scale300_private as base

DEST = "kolihhan/ntl-lightning0694-scale1200-nov15"
BASE_300_SHA = "39ec970928147716ec6c0234e13cef8aea919632072ab9bfa6d752a6c0814ceb"
REPLACEMENTS = {
    "nrows=300": "nrows=1200",
    "assert len(_ntl_pilot)==300": "assert len(_ntl_pilot)==1200",
    "'scale300_%04d'%i for i in range(300)": "'scale1200_%04d'%i for i in range(1200)",
    "ntl0694_scale300_unlabelled.csv": "ntl0694_scale1200_unlabelled.csv",
    "NTL0694_SCALE300_INPUT_TEXT_ONLY": "NTL0694_SCALE1200_INPUT_TEXT_ONLY",
}

def make(source: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("never_overwrite_existing_pilot")
    with tempfile.TemporaryDirectory() as tmp:
        base_output = Path(tmp)/"source300"
        report=base.build(source, base_output)
        if report["pilot_ipynb_sha256"] != BASE_300_SHA:
            raise RuntimeError("300_row_frozen_source_drift")
        meta=json.loads((base_output/"kernel-metadata.json").read_text())
        name=meta["code_file"]
        nb=json.loads((base_output/name).read_text())
    updated=0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s=base.source(cell)
        if "NTL0694_SCALE300_INPUT_TEXT_ONLY" not in s:
            continue
        for before,after in REPLACEMENTS.items():
            if s.count(before)!=1:raise RuntimeError("unexpected_input_edit_inventory:"+before)
            s=s.replace(before,after,1)
        cell["source"]=s.splitlines(keepends=True)
        updated+=1
    if updated!=1:raise RuntimeError("one_real_input_cell_required")
    all_code="\n".join(base.source(c) for c in nb["cells"] if c.get("cell_type")=="code")
    for marker in ("NO_NEW_TRAINING: Gaussian fit blocked",
                   "NO_NEW_TRAINING: Atlas training blocked",
                   "preview = False  # always genuine model inference"):
        if all_code.count(marker)!=1:raise RuntimeError("training_or_realpath_guard_drift")
    if ("gauss.fit(" in all_code or
        "NTL0694_SCALE300_INPUT_TEXT_ONLY" in all_code or
        "NTL0694_SCALE1200_INPUT_TEXT_ONLY" not in all_code or
        "nrows=1200" not in all_code):
        raise RuntimeError("scale_input_patch_failed")
    meta["id"]=DEST
    meta["title"]="ntl-lightning0694-scale1200-nov15"
    meta.pop("id_no",None)
    meta.pop("docker_image",None)
    if (meta.get("competition_sources")!=["name-that-llm"] or
        [len(meta.get(k) or []) for k in ("dataset_sources","model_sources","kernel_sources")] != [23,6,9]):
        raise RuntimeError("frozen_pretrained_mounts_drift")
    output.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(nb,ensure_ascii=False)
    (output/name).write_text(payload)
    (output/"kernel-metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True))
    return {"source":base.FROM,"dest":DEST,"input_rows":1200,
            "input":"official_train_texts_only","labels_read":False,
            "source_300_builder_sha":BASE_300_SHA,
            "notebook_sha256":hashlib.sha256(payload.encode()).hexdigest(),
            "model_config_unchanged":True,"training":False,
            "official_submission":False,"balanced_accuracy_measured":False}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print("NTL0694_SCALE1200_FROZEN_BUILD="+json.dumps(make(a.source,a.out),sort_keys=True),flush=True)
