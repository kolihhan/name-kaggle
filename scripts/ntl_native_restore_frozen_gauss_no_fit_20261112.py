"""Restore two ORIGINAL pretrained Gaussian heads, with no runtime fitting.

Evidence: private Kaggle CPU-only read proved BOTH old prefit Gaussian
metadata.json and parameters.npz exist, and both metadata contracts say
fixed-no-test-redo. This candidate restores the exact original weights
(q14v5 .7, q8v5 .44) only for compatible, no-redo model features.

If feature extraction takes the original source-test-conditioned redo branch,
this candidate bypasses the Gaussian contribution and never recomputes
training embeddings or calls Gaussian.fit. It NEVER trains a head.

Only source changes: restore two CONFIG numbers; mark runtime supervised
redo branch unreachable; remove unused 2699-row train CSV materialization;
replace supervised Gaussian.fit branch with a no-Gaussian fallback; guard
the scoring call against this fallback. All model weights, pretrained
Gaussian metadata/parameters, other ensemble settings remain unchanged.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts import ntl_native_clean_gpu_frozen_prefit_20261112 as base

DEST="kolihhan/ntl-native-frozen-gauss-readonly-nov12"
TITLE="ntl-native-frozen-gauss-readonly-nov12"
WANTED={"q14v5":0.7,"q8v5":0.44}

WORKER_SOURCE_FIT="""if not train_final and ai == len(adapters)-1 and spec.get('gauss',0)>0 and test_redo:"""
WORKER_NO_FIT="""if False and not train_final and ai == len(adapters)-1 and spec.get('gauss',0)>0 and test_redo:  # frozen-only: supervised feature reread forbidden"""
WORKER_TRAIN_READ="""train_frame = pd.read_csv(config['paths']['train_csv']) if not args.train_final and spec.get('gauss',0)>0 else None"""
WORKER_TRAIN_DISABLED="""train_frame = None  # frozen prefit only; official labels never loaded for a redo"""
SCORING_CONDITION="""if selected.any():"""
SCORING_FALLBACK="""if selected.any() and prm is not None:"""


def source(cell:dict)->str:
    s=cell.get("source",[])
    return "".join(s) if isinstance(s,list) else str(s)


def restore_config(s:str)->tuple[str,dict]:
    tree=ast.parse(s)
    configs=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and
             any(isinstance(x,ast.Name) and x.id=="CONFIG" for x in n.targets)]
    if len(configs)!=1 or not isinstance(configs[0],ast.Dict):
        raise ValueError("single frozen CONFIG dictionary missing")
    def fields(node:ast.AST):
        if not isinstance(node,ast.Dict):raise ValueError("expected literal dictionary")
        return {ast.literal_eval(k):v for k,v in zip(node.keys,node.values)}
    root=fields(configs[0])
    models=root.get("models")
    if not isinstance(models,ast.List) or len(models.elts)!=6:
        raise ValueError("expected six original models")
    found={}
    bytes_before=s.encode("utf-8")
    edits=[]
    src_lines=s.splitlines(keepends=True)
    for entry in models.elts:
        d=fields(entry)
        name=ast.literal_eval(d["name"])
        if name not in WANTED:continue
        if name in found:raise ValueError("duplicated Gaussian model")
        node=d.get("gauss")
        if node is None or ast.literal_eval(node)!=0.0:
            raise ValueError("no-fit frozen candidate flags not exactly 0")
        lo=sum(len(x.encode("utf-8")) for x in src_lines[:node.lineno-1])+node.col_offset
        hi=sum(len(x.encode("utf-8")) for x in src_lines[:node.end_lineno-1])+node.end_col_offset
        edits.append((lo,hi,str(WANTED[name]).encode("utf-8")))
        found[name]=WANTED[name]
    if found!=WANTED or len(edits)!=2:raise ValueError("not exactly two Gauss heads")
    data=bytes_before
    for lo,hi,replacement in sorted(edits,reverse=True):
        data=data[:lo]+replacement+data[hi:]
    revised=data.decode("utf-8")
    if len(revised)==len(s):raise ValueError("no change")
    final=ast.parse(revised)
    updated=[n.value for n in ast.walk(final) if isinstance(n,ast.Assign) and
          any(isinstance(x,ast.Name) and x.id=="CONFIG" for x in n.targets)]
    specs=fields(updated[0])["models"]
    for m in specs.elts:
        d=fields(m);key=ast.literal_eval(d["name"])
        if key in WANTED and ast.literal_eval(d["gauss"])!=WANTED[key]:
            raise ValueError("Gaussian restore postcondition failed")
    return revised,found


def guard_worker(s:str)->str:
    if s.count(WORKER_SOURCE_FIT)!=1 or s.count(WORKER_TRAIN_READ)!=1:
        raise ValueError("original supervised worker redos changed")
    return s.replace(WORKER_SOURCE_FIT,WORKER_NO_FIT,1).replace(
        WORKER_TRAIN_READ,WORKER_TRAIN_DISABLED,1)


def guard_readout(s:str)->str:
    lines=s.splitlines(keepends=True)
    candidates=[i for i,l in enumerate(lines) if l.strip()=="if branch['gaussian_branch'] == 'source-test-conditioned':"]
    if len(candidates)!=1 or s.count(SCORING_CONDITION)!=1:
        raise ValueError("original Gaussian source-readout branch drift")
    start=candidates[0]
    indent=lines[start][:len(lines[start])-len(lines[start].lstrip())]
    ends=[i for i in range(start+1,len(lines)) if lines[i]==indent+"else:\n"]
    if len(ends)!=1:raise ValueError("source-conditioned refit else not unique")
    end=ends[0]
    old="".join(lines[start:end])
    if "gauss.fit(" not in old or "gaussian-train-features.npz" not in old:
        raise ValueError("no original Gaussian refit branch to neutralize")
    replacement=(
      indent+"if branch['gaussian_branch'] == 'source-test-conditioned':\n"
      +indent+"    assert branch['test_redo'] > 0\n"
      +indent+"    prm = None  # no supervised refit; zero Gaussian contribution in redo cases\n"
    )
    result="".join(lines[:start])+replacement+"".join(lines[end:])
    result=result.replace(SCORING_CONDITION,SCORING_FALLBACK,1)
    if "gauss.fit(training[" in result or "gauss.fit(training[" not in old:
        raise ValueError("supervised Gaussian fit still reachable")
    if "prm = gaussian_state(gaussian_folder, spec)" not in result:
        raise ValueError("frozen Gaussian head read path removed")
    return result


def make(out:Path)->dict:
    record=base.make(out)
    folder=out/"kernel"
    nb_paths=list(folder.glob("*.ipynb"))
    if len(nb_paths)!=1:raise ValueError("candidate Notebook ambiguity")
    path=nb_paths[0]
    nb=json.loads(path.read_text(encoding="utf-8"))
    stats={"config":0,"worker":0,"readout":0}
    restored=None
    for cell in nb["cells"]:
        if cell.get("cell_type")!="code":continue
        code=source(cell)
        if code.startswith("CONFIG = {"):
            code,restored=restore_config(code)
            stats["config"]+=1
        elif code.startswith("%%writefile") and "/core_worker.py" in code.splitlines()[0]:
            code=guard_worker(code)
            stats["worker"]+=1
        elif code.startswith("%%writefile") and "/core_readout.py" in code.splitlines()[0]:
            code=guard_readout(code)
            stats["readout"]+=1
        else:
            continue
        cell["source"]=code.splitlines(keepends=True)
    if stats!={"config":1,"worker":1,"readout":1} or restored!=WANTED:
        raise ValueError("not exactly one safe frozen-head restoration")
    combined="\n".join(source(c) for c in nb["cells"] if c["cell_type"]=="code")
    for guard in base.base.GUARDS:
        if combined.count(guard)!=1:raise ValueError("frozen training guards lost")
    if combined.count("prm = gaussian_state(gaussian_folder, spec)")!=1:
        raise ValueError("no frozen head load")
    if combined.count(WORKER_NO_FIT)!=1 or combined.count(WORKER_TRAIN_DISABLED)!=1:
        raise ValueError("no supervised-fallback safeguard")
    if "preview = False  # diagnostic: force real inference path" not in combined:
        raise ValueError("fake fixture remains")
    meta_path=folder/"kernel-metadata.json"
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    if any(k in meta for k in ("docker_image","keywords","id_no")):
        raise ValueError("CPU-only legacy docker image metadata leaked")
    if [len(meta.get(k) or []) for k in ("dataset_sources","kernel_sources","model_sources")]!=[21,8,6]:
        raise ValueError("source assets changed")
    meta["id"]=DEST;meta["title"]=TITLE
    meta_path.write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    output=json.dumps(nb,ensure_ascii=False).encode("utf-8")
    path.write_bytes(output)
    return {
      "candidate":DEST,"gaussian_prefit_original_weights":restored,
      "runtime_retraining_disabled":True,"original_frozen_gaussian_state_path":True,
      "fallback_for_any_test_redo":"zero Gaussian contribution without training",
      "all_other_model_weights_unchanged":True,"clean_gpu_metadata":True,
      "no_official_submission":True,"actual_accuracy_unknown":True,
      "source_changes":stats,"sha256":hashlib.sha256(output).hexdigest(),
      "upstream_reference":record["source"],
    }


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print("NTL_FROZEN_GAUSS_RESTORE_CANDIDATE="+json.dumps(make(a.out),sort_keys=True),flush=True)

if __name__=="__main__":
    main()
