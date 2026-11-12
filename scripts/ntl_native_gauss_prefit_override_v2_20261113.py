"""Prospective Native Gaussian prefit-only test, no new supervised fit.

The previous recovered candidate was a proven scientific no-op:
source-test-conditioned -> prm=None, hence zero Gaussian scores.
This builder changes exactly that readout path to load existing
frozen metadata.json/parameters.npz, with the original semantic /
adapter-chain / 18-label assertions intact. Never calls gauss.fit.

This is an exploratory, potentially distribution-mismatched prefit
override (not a demonstrated 18-class accuracy gain). It may not be
promoted unless it actually emits nonzero scores on grouped inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts import ntl_native_restore_frozen_gauss_no_fit_20261112 as base

DEST="kolihhan/ntl-native-gauss-prefit-override-v2-nov13"
TITLE="ntl-native-gauss-prefit-override-v2-nov13"

OLD="prm = None  # no supervised refit; zero Gaussian contribution in redo cases"
NEW="""prm = gaussian_state(gaussian_folder, spec)  # immutable prefit, no supervised training
            print('NTL_GAUSS_PREFIT_RUNTIME_BOUND='+json.dumps({'member':spec['name'],'mode':'source-conditioned-test-features-with-frozen-train-only-head','test_redo_positive':True,'supervised_fit':False},sort_keys=True),flush=True)"""
ANCHOR="""value = np.zeros((len(groups), 18))"""
DIAG="""print('NTL_GAUSS_PREFIT_GROUP_ELIGIBILITY='+json.dumps({'member':spec['name'],'rows':int(len(groups)),'selected_rows':int(selected.sum()),'has_group':bool(selected.any())},sort_keys=True),flush=True)"""
ANCHOR_SCORE="""value[selected] = gauss.scores(final_features[selected], np.unique(groups[selected], return_inverse=True)[1], prm)"""
SCORE_DIAG="""print('NTL_GAUSS_PREFIT_SCORES='+json.dumps({'member':spec['name'],'scored_rows':int(selected.sum()),'nonzero':bool(np.any(value[selected]!=0)),'finite':bool(np.all(np.isfinite(value[selected]))),'max_abs':float(np.max(np.abs(value[selected])))},sort_keys=True),flush=True)"""


def text(cell:dict)->str:
    s=cell.get("source",[])
    return "".join(s) if isinstance(s,list) else str(s)


def transform_readout(code:str)->str:
    if not code.startswith("%%writefile") or "core_readout.py" not in code.splitlines()[0]:
        raise ValueError("expected exactly original frozen readout module")
    for token in (OLD,ANCHOR,ANCHOR_SCORE):
        if code.count(token)!=1:
            raise ValueError("precise prefit restoration anchor changed: "+token[:65])
    if "gauss.fit(training[" in code:
        raise ValueError("supervised Gaussian call reappeared")
    code=code.replace(OLD,NEW,1)
    # Instrument group eligibility and whether existing prefit Gaussian actually changed logits.
    code=code.replace(ANCHOR,ANCHOR+"\n        "+DIAG,1)
    code=code.replace(ANCHOR_SCORE,ANCHOR_SCORE+"\n            "+SCORE_DIAG,1)
    if code.count("prm = gaussian_state(gaussian_folder, spec)")!=2:
        raise ValueError("expected frozen loader in both prefit-compatible modes")
    if "gauss.fit(training[" in code or OLD in code:
        raise ValueError("no-op or retraining still present")
    return code


def make(out:Path)->dict:
    preflight=base.make(out)
    kernel=out/"kernel"
    notebooks=list(kernel.glob("*.ipynb"))
    if len(notebooks)!=1:raise ValueError("Notebook source ambiguity")
    path=notebooks[0]
    nb=json.loads(path.read_text(encoding="utf-8"))
    changed=0
    for cell in nb["cells"]:
        if cell.get("cell_type")!="code":continue
        code=text(cell)
        if code.startswith("%%writefile") and "core_readout.py" in code.splitlines()[0]:
            cell["source"]=transform_readout(code).splitlines(keepends=True)
            changed+=1
    if changed!=1:raise ValueError("not one exact inference-only Gaussian readout")
    code="\n".join(text(c) for c in nb["cells"] if c.get("cell_type")=="code")
    for guard in base.base.base.GUARDS:
        if code.count(guard)!=1:raise ValueError("no training guard must remain unchanged")
    if code.count("NTL_GAUSS_PREFIT_RUNTIME_BOUND=")!=1 or code.count("NTL_GAUSS_PREFIT_SCORES=")!=1:
        raise ValueError("no observable nonzero Gaussian evidence")
    if "gauss.fit(training[" in code or "train_frame = None" not in code or
       "if False and not train_final" not in code:
        raise ValueError("supervised Gaussian fitting no longer fail-closed")
    metadata_path=kernel/"kernel-metadata.json"
    meta=json.loads(metadata_path.read_text(encoding="utf-8"))
    if not meta["enable_gpu"] or "docker_image" in meta or len(meta["model_sources"])!=6:
        raise ValueError("GPU metadata or model input provenance changed")
    meta["id"]=DEST;meta["title"]=TITLE
    metadata_path.write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n")
    raw=json.dumps(nb,ensure_ascii=False).encode("utf-8")
    path.write_bytes(raw)
    return {"slug":DEST,"source":preflight["upstream_reference"],
            "pretrained_gaussian_weights":preflight["gaussian_prefit_original_weights"],
            "runtime_refit":False,"inference_only":True,
            "old_noop_branch_removed":True,"actual_gaussian_quality_unknown":True,
            "no_official_submission":True,"benchmark_required":True,
            "source_sha256":hashlib.sha256(raw).hexdigest()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    print("NTL_PREFIT_GAUSS_OVERRIDE_V2_BUILD="+json.dumps(make(args.out),sort_keys=True),flush=True)

if __name__=="__main__":
    main()
