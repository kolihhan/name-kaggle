"""CDA-L24 v2: verify two frozen Kaggle GPU outputs, package ONE C1 checkpoint.

Called only after both private GPU kernels finish. GitHub Actions runner
receives private output files in /tmp, prints aggregate scores and digests,
then publishes only the new private Kaggle dataset. It never uploads raw
heldout predictions, checkpoint weights or official competition data to
the public repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

from scripts.cda_l24_transfer_gate import assess

CANDIDATE="cda-l24-direct-sources-v2"
C0="C0"
C1="C1"
RESERVED="kilupy/ntl-18-model-corpus"
FIRST_PARTY_SOURCES={
    "arxitekton/ntl-claude-gen-v1",
    "trnhip/ntl-api-gen",
    "kilupy/ntl-18-model-corpus",
}
HISTORIC_A1_SHA="c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0"
DATASET_REF="kolihhan/ntl-cda-l24-v2-c1-weights"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()


def freeze_checkpoint_directory(root:Path)->str:
    """Same exact manifest algorithm as cda_l24_v2_checkpoint.py."""
    if not root.is_dir():raise FileNotFoundError(root)
    entries=[]
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel=p.relative_to(root).as_posix()
        if rel.endswith("manifest_digest.json"):
            raise ValueError("self-referential manifest")
        if p.is_symlink():raise RuntimeError("checkpoint symlink forbidden")
        entries.append({"path":rel,"sha256":sha256(p),"size":p.stat().st_size})
    if not entries:raise ValueError("no checkpoint files")
    return hashlib.sha256(json.dumps(
         entries,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def find_prediction(root:Path,arm:str)->Path:
    name=f"cda_l24_v2_{arm}_private_predictions.csv"
    files=list(root.rglob(name))
    if len(files)!=1:raise RuntimeError(f"missing/ambiguous private {arm} predictions: {len(files)}")
    return files[0]


def find_checkpoint(root:Path,arm:str)->Path:
    files=[p.parent for p in root.rglob("meta.json")
           if p.parent.name==f"cda_l24_v2_{arm}_checkpoint"]
    if len(files)!=1:raise RuntimeError(f"missing/ambiguous private {arm} checkpoint: {len(files)}")
    return files[0]


def check_metadata(meta:dict,arm:str):
    for k,v in {
        "arm":arm,
        "candidate":CANDIDATE,
        "source_transfer_reserved":RESERVED,
        "base_stage1_sha256":HISTORIC_A1_SHA,
        "source_head_persisted":False,
    }.items():
        if meta.get(k)!=v:raise RuntimeError(f"Private {arm} origin integrity: {k}")
    labels=meta.get("labels",[])
    if not isinstance(labels,list) or len(set(labels))!=18:
        raise RuntimeError(f"{arm}: original 18-class head unavailable")
    stats=meta.get("run_stats",{})
    if stats.get("stage_A_source_missing_classes")!=["gemini-3.7-flash"]:
        raise RuntimeError(f"{arm}: unexpected source label coverage")
    if stats.get("stage_A_identity_CE_uses_only_observed_classes") is not True:
        raise RuntimeError(f"{arm}: absent model train class not protected")
    if stats.get("official_all_18_class_logits_retained_for_inference") is not True:
        raise RuntimeError(f"{arm}: full inference label head lost")
    if stats.get("transfer_labels_present")!=18:
        raise RuntimeError(f"{arm}: source-heldout model labels incomplete")
    if stats.get("transfer_source")!=RESERVED:
        raise RuntimeError(f"{arm}: unexpected heldout")
    if stats.get("crossed_labels",0)<8 or stats.get("train_sources")!=2:
        raise RuntimeError(f"{arm}: source×class domain head unsupported")
    if stats.get("optimizer_steps",0)<1:
        raise RuntimeError(f"{arm}: no actual GPU updates")
    dataset_shas=meta.get("train_stats",{}).get("dataset_sha256",{})
    if set(dataset_shas)!=FIRST_PARTY_SOURCES:
        raise RuntimeError(f"{arm}: unexpected original source inputs")
    for v in dataset_shas.values():
        if not isinstance(v,str) or len(v)!=64:
            raise RuntimeError(f"{arm}: invalid first-party source SHA")
    if meta.get("provenance_manifest_sha256")!=hashlib.sha256(
        json.dumps(dataset_shas,sort_keys=True).encode()).hexdigest():
        raise RuntimeError(f"{arm}: original dataset source SHA mismatch")
    return labels,dataset_shas


def prepare(c0_root:Path,c1_root:Path,package_dir:Path,summary_json:Path):
    c0_root=Path(c0_root);c1_root=Path(c1_root)
    c0_ckpt=find_checkpoint(c0_root,C0)
    c1_ckpt=find_checkpoint(c1_root,C1)
    m0=json.loads((c0_ckpt/"meta.json").read_text())
    m1=json.loads((c1_ckpt/"meta.json").read_text())
    labels0,shas0=check_metadata(m0,C0)
    labels1,shas1=check_metadata(m1,C1)
    if labels0!=labels1 or shas0!=shas1:
        raise RuntimeError("C0/C1 original model/source origins not identical")
    if m0["run_stats"].get("strength")!=0.0 or m1["run_stats"].get("strength")!=0.1:
        raise RuntimeError("Frozen C0/C1 lambda mismatch")
    for key in ("train_rows","transfer_rows","crossed_labels","optimizer_steps"):
        if m0["run_stats"].get(key)!=m1["run_stats"].get(key):
            raise RuntimeError(f"matched training schedule or data drift: {key}")
    pred0=pd.read_csv(find_prediction(c0_root,C0))
    pred1=pd.read_csv(find_prediction(c1_root,C1))
    report=assess(pred0,pred1,reps=5000,seed=20261108)
    # Fix unsupported provider string splitting / misaligned rows upstream
    # via assess() strict full 18-label pairing, not leaderboards.
    if report.get("eligible_true_labels")!=18:
        raise RuntimeError("exact original heldout is not all 18 models")
    ckpt_digest=freeze_checkpoint_directory(c1_ckpt)
    package_dir=Path(package_dir)
    if package_dir.exists() and any(package_dir.iterdir()):
        raise FileExistsError("private model dataset package already exists")
    package_dir.mkdir(parents=True,exist_ok=True)
    zipped=package_dir/"cda-v2-C1-checkpoint.zip"
    with zipfile.ZipFile(zipped,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(x for x in c1_ckpt.rglob("*") if x.is_file()):
            z.write(p,arcname=p.relative_to(c1_ckpt).as_posix())
    (package_dir/"dataset-metadata.json").write_text(json.dumps({
        "title":"NTL CDA L24 v2 Original Source C1 Private Stage A",
        "id":DATASET_REF,
        "licenses":[{"name":"other"}],
        "isPrivate":True,
    },indent=2,sort_keys=True)+"\n")
    report["source_manifest_sha256"]=m1["provenance_manifest_sha256"]
    report["new_checkpoint_directory_manifest_sha256"]=ckpt_digest
    report["checkpoint_archive_sha256"]=sha256(zipped)
    report["checkpoint_archive_bytes"]=zipped.stat().st_size
    report["candidate"]=CANDIDATE
    report["stage_A_original_A1_sha256"]=HISTORIC_A1_SHA
    report["one_shot_kaggle_submission_requested"]=True
    report["no_public_LB_tuning_or_model_selection"]=True
    report["stage_B_official_grouped_OOF_not_measured"]=True
    report["do_not_claim_scientific_promotion_if_transfer_gate_rejected"]=True
    summary_json=Path(summary_json)
    summary_json.parent.mkdir(parents=True,exist_ok=True)
    summary_json.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("CDA_C0_C1_FROZEN_EXTERNAL_BALANCED_ACCURACY",json.dumps({
      k:report.get(k) for k in (
        "decision","C0_balanced_accuracy","C1_balanced_accuracy",
        "C1_minus_C0","prompt_bootstrap_CI95","provider_macro_delta",
        "matched_rows","eligible_true_labels")},sort_keys=True),flush=True)
    print("PRIVATE_C1_FINAL_CHECKPOINT_MANIFEST",ckpt_digest,flush=True)
    print("PRIVATE_C1_SOURCE_MANIFEST",m1["provenance_manifest_sha256"],flush=True)
    return report


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--c0-output",type=Path,required=True)
    a.add_argument("--c1-output",type=Path,required=True)
    a.add_argument("--package-dir",type=Path,required=True)
    a.add_argument("--summary-json",type=Path,required=True)
    v=a.parse_args()
    prepare(v.c0_output,v.c1_output,v.package_dir,v.summary_json)


if __name__=="__main__":main()
