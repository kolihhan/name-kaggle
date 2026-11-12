"""Frozen-weight, no-Gaussian-fit Native compatibility candidate.

Strictly based on GPU-mapfix private candidate. Disable the two runtime-
supervised Gaussian options via AST targeting and preserve all source inputs,
no-training guards, model weights, LoRA branches, and other inference logic.
This is a NEW candidate, not the already submitted Notebook.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts import ntl_native_gpu_mapfix_realpath_20261112 as parent

DEST="kolihhan/ntl-native-prefit-only-gpufix-nov12"
TITLE="ntl-native-prefit-only-gpufix-nov12"
GAUSS_TARGETS={"q14v5":0.7, "q8v5":0.44}


def _simple_dict(node: ast.AST) -> dict:
    if not isinstance(node,ast.Dict):
        raise ValueError("Expected literal Python dictionary")
    out={}
    for k,v in zip(node.keys,node.values):
        try:key=ast.literal_eval(k)
        except Exception:continue
        out[key]=v
    return out


def disable_gaussian_only(source: str) -> tuple[str,dict]:
    """Return only a structural, two-numeric-value AST patch."""
    tree=ast.parse(source)
    nodes=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
       and any(isinstance(x,ast.Name) and x.id=="CONFIG" for x in n.targets)]
    if len(nodes)!=1:
        raise ValueError("CONFIG assignment not unique")
    m=_simple_dict(nodes[0])
    variants=m.get("models")
    if not isinstance(variants,ast.List) or len(variants.elts)!=6:
        raise ValueError("Expected exact six-model frozen source")
    expected=set(GAUSS_TARGETS)
    changes=[]
    for item in variants.elts:
        parts=_simple_dict(item)
        if "name" not in parts:
            raise ValueError("Missing model name")
        name=ast.literal_eval(parts["name"])
        if name not in GAUSS_TARGETS:
            continue
        if name not in expected:
            raise ValueError("duplicate targeted model name in frozen config")
        expected.remove(name)
        if "gauss" not in parts:
            raise ValueError("Target model has no Gaussian flag")
        node=parts["gauss"]
        previous=ast.literal_eval(node)
        if previous!=GAUSS_TARGETS[name]:
            raise ValueError("Original Gaussian weight not frozen expected value")
        lines=source.splitlines(keepends=True)
        first=sum(len(line.encode("utf-8")) for line in lines[:node.lineno-1])+node.col_offset
        last=sum(len(line.encode("utf-8")) for line in lines[:node.end_lineno-1])+node.end_col_offset
        changes.append((first,last,name))
    if expected or len(changes)!=2:
        raise ValueError("Gaussian targets missing or duplicated")
    output=source.encode("utf-8")
    for first,last,name in sorted(changes,reverse=True):
        output=output[:first]+b"0.0"+output[last:]
    new=output.decode("utf-8")
    after=ast.parse(new)
    model_lists=[_simple_dict(n.value).get("models") for n in ast.walk(after)
       if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="CONFIG" for x in n.targets)]
    updated={ast.literal_eval(_simple_dict(n)["name"]):_simple_dict(n)
             for n in model_lists[0].elts}
    if any(ast.literal_eval(updated[name]["gauss"])!=0.0 for name in GAUSS_TARGETS):
        raise ValueError("Gaussian flag postcondition failed")
    return new,{"models_disabled":sorted(GAUSS_TARGETS),"changes":2,"supervised_fit_unreachable_via_config":True}


def make(out: Path) -> dict:
    record=parent.make(out)
    kernel=out/"kernel"
    notebooks=list(kernel.glob("*.ipynb"))
    if len(notebooks)!=1:
        raise ValueError("Unexpected Notebook count")
    path=notebooks[0]
    nb=json.loads(path.read_text(encoding="utf-8"))
    modified=[]
    for cell in nb["cells"]:
        if cell.get("cell_type")!="code":continue
        s=cell.get("source",[])
        old="".join(s) if isinstance(s,list) else str(s)
        if old.startswith("CONFIG = {"):
            changed,stats=disable_gaussian_only(old)
            if changed==old:
                raise ValueError("No effect")
            cell["source"]=changed.splitlines(keepends=True)
            modified.append(stats)
    if len(modified)!=1:
        raise ValueError("Expected one frozen config cell")
    before=path.read_bytes()
    final=json.dumps(nb,ensure_ascii=False).encode("utf-8")
    path.write_bytes(final)
    data=json.loads(path.read_text(encoding="utf-8"))
    code="\n".join("".join(c.get("source",[])) if isinstance(c.get("source"),list)
             else str(c.get("source","")) for c in data["cells"] if c["cell_type"]=="code")
    for guard in parent.base.GUARDS:
        if code.count(guard)!=1:
            raise ValueError("No-new-training guard lost")
    if code.count(parent.POLICY)!=2 or "preview = False  # diagnostic: force real inference path" not in code:
        raise ValueError("Compatibility patch or real-path gate lost")
    meta_path=kernel/"kernel-metadata.json"
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    if (len(meta.get("model_sources") or [])!=6 or
        len(meta.get("dataset_sources") or [])!=21 or
        len(meta.get("kernel_sources") or [])!=8 or
        meta.get("competition_sources")!=["name-that-llm"]):
        raise ValueError("pretrained asset provenance missing")
    # The original pulled Notebook carries stale docker_image metadata.
    # Paired Kaggle probes showed old metadata => CPU-only, clean allowlist
    # with ALL source assets => two Tesla T4 and CUDA.
    clean={
        "id":DEST,"title":TITLE,
        "code_file":meta["code_file"],"language":"python",
        "kernel_type":"notebook","is_private":True,
        "enable_gpu":True,"enable_tpu":False,"enable_internet":False,
        "machine_shape":"NvidiaTeslaT4",
        **{k:meta[k] for k in (
             "competition_sources","dataset_sources","kernel_sources","model_sources")},
    }
    if any(k in clean for k in ("docker_image","keywords","id_no")):
        raise ValueError("legacy CPU runtime metadata leaked")
    meta_path.write_text(json.dumps(clean,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return {**record,
        "destination":DEST,
        "notebook_sha256":hashlib.sha256(final).hexdigest(),
        "previous_candidate_sha256":hashlib.sha256(before).hexdigest(),
        "gaussian":modified[0],
        "new_training":False,
        "competing_supervised_head_removed":True,
        "new_official_submission":False,
        "candidate_score_known":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=make(a.out)
    print("NTL_FROZEN_PREFIT_CANDIDATE="+json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
