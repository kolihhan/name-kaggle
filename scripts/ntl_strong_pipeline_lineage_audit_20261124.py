"""Read-only lineage audit of TWO already scored original 18-way inference notebooks.

No competition data access, inference, training, new Kernel or submission.
Emit module SHA, decision-stage provenance and missing trace checks, not code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

EXPECTED = {
    "older_0715": {
        "source_sha": "202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8",
        "owned": "kolihhan/ntl-estopadilla-glm53-0716-repro-20261122",
        "score": 0.715,
        "submission": "58680531",
    },
    "newer_0718": {
        "public_parent_file_sha": "74d86affa9f55464ab39f735f392b54e47e2d8a2b631a0bca45de9435c600c56",
        "owned": "kolihhan/ntl-evgen-forensic-eda-repro-20261123",
        "score": 0.718,
        "submission": "58755304",
    },
}
MARKERS = {
    "pre_glm_logits": ("save_final_logits", "final-logits.npz"),
    "glm_user_prefix": ("v1_stage.py", "prefix_variant_stage.py", "glm-5.3-flash"),
    "sibling_refiner": ("pngy_residual.py", "refine(", "strength="),
    "style_readout": ("ntl660_readout.py", "style_groups"),
    "group_covariance": ("gauss.py", "group", "soft"),
    "fable_return": ("fable_return", "fable", "applied"),
    "preview_fast_path": ("KAGGLE_IS_COMPETITION_RERUN", "len(", "preview"),
}
MAX_MODULES = 100

def sha(raw:bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def text_of(c):
    s=c.get("source") or ""
    return "".join(s) if isinstance(s,list) else str(s)

def read(root,name):
    m=json.loads((root/"kernel-metadata.json").read_text())
    p=root/Path(m["code_file"]).name
    nb=json.loads(p.read_text())
    if m.get("id")!=EXPECTED[name]["owned"] or not m.get("is_private",False):
        raise ValueError("owner or private notebook scope mismatch")
    cells=[text_of(c) for c in nb["cells"] if c.get("cell_type")=="code"]
    code="\n".join(cells)
    if name=="older_0715" and sha(code.encode())!=EXPECTED[name]["source_sha"]:
        raise ValueError("scored older code lineage sha changed")
    items={}
    for i,source in enumerate(cells):
        m2=re.search(r"^%%writefile\s+(\S+)",source.strip(),re.M)
        if not m2:continue
        path=m2.group(1)
        filename=Path(path).name
        # Report the *last* write of a module in notebook order.
        contents=source[source.find("\n")+1:]
        items[filename]={"sha":sha(contents.encode()),"chars":len(contents),"cell":i,
                         "lines":contents.count("\n")+1}
    if len(items)>MAX_MODULES:raise ValueError("unusually many notebook file writes")
    counts={k:len(m.get(k) or []) for k in ("model_sources","dataset_sources","kernel_sources")}
    return {
      "code_sha":sha(code.encode()),
      "notebook_file_sha":sha(p.read_bytes()),
      "num_code_cells":len(cells),"num_all_cells":len(nb["cells"]),
      "owned":m["id"],"private":m.get("is_private"),
      "source_mounts":counts,"attached_models":m.get("model_sources") or [],
      "attached_datasets":m.get("dataset_sources") or [],
      "attached_kernels":m.get("kernel_sources") or [],
      "modules":items,
      "signals":{key:{"all_present":all(x in code for x in needles),
                        "component_presence":{x:int(x in code) for x in needles},
                        "first_cell":next((i for i,v in enumerate(cells)
                                if any(x in v for x in needles)),None)}
                 for key,needles in MARKERS.items()},
      "code":code,"cell_texts":cells
    }


def reachability(a,b):
    """A narrow static proof of direct uses; does not assert dynamic impossibility."""
    cells=b["cell_texts"]
    def hits(needle):
        return [{"code_cell":i,"occurrences":c.count(needle)}
                for i,c in enumerate(cells) if needle in c]
    variant=hits("prefix_variant_stage.py")
    sibling=hits("sibling_agreement(")
    fable_call=hits("from prefix_decisions import fable_return")
    fable_exec=hits("fable_return(base_lp,_post,labels)")
    v1_explicit=hits("V1_DIR / 'v1_stage.py'")
    fast_gate=hits("if FAST_COMMIT:")
    fast_guard=hits("if V1_ENABLED and not FAST_COMMIT:")
    # All checks are limited to original notebook CODE CELLS, not third-party
    # extracted GGUF binaries or dynamically evaluated strings.
    return {
        "new_prefix_variant_file_write_references":variant,
        "variant_direct_reference_outside_write_cell":len(variant)>1,
        "variant_setup_use_context":[{"code_cell":i,"snippet":source[max(0,source.index("prefix_variant_stage.py")-150):source.index("prefix_variant_stage.py")+210]}
             for i,source in enumerate(cells) if "prefix_variant_stage.py" in source and not source.lstrip().startswith("%%writefile /kaggle/working/prefix_variant_stage.py")],
        "sibling_agreement_occurrences":sibling,
        "sibling_agreement_direct_invocation_outside_definition":len(sibling)>1,
        "fable_return_import_sites":fable_call,
        "fable_return_actual_call_sites":fable_exec,
        "v1_stage_invoked_explicitly":v1_explicit,
        "fast_commit_placeholder_branch_present":bool(fast_gate),
        "v1_skipped_when_fast_commit_true":bool(fast_guard),
        "pre_glm_logits_output_filename_in_source":bool(hits("final-logits.npz")),
        "old_source_has_fast_commit": "FAST_COMMIT" in a["code"],
        "new_source_has_rehearsal_enabled":bool(re.search(r"V1_REHEARSAL_N\s*=\s*[1-9]\d*",b["code"])),
        "deduction":"In 3-row Kaggle preview, FAST_COMMIT is true and the main inference/GLM path is skipped. FileNotFoundError on missing final-logits.npz is a predictable sidecar observation, not evidence of full-hidden inference failure.",
        "frozen_helper_is_not_proven_applied_just_because_written":True,
        "source_only_no_real_validation_accuracy":True
    }


def embedded_stage_diff(b):
    """Compare actual payload V1 base with the installed prefix_variant_stage source."""
    import ast,base64,difflib,io,tarfile
    candidate=[c for c in b["cell_texts"] if c.startswith("%%writefile /kaggle/working/prefix_variant_stage.py\\n")]
    setup=[c for c in b["cell_texts"] if "V1_PAYLOAD =" in c]
    if len(candidate)!=1 or len(setup)!=1:
        raise ValueError("stage source or compressed base ambiguous")
    variant=candidate[0].split("\\n",1)[1]
    # Find a literal payload without running the notebook's install statements.
    payload=None
    for node in ast.parse(setup[0]).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="V1_PAYLOAD" for t in node.targets):
            payload=ast.literal_eval(node.value)
    if not isinstance(payload,str) or len(payload)<200:
        raise ValueError("base stage payload missing or not a pinned literal")
    archive=base64.b64decode(payload,validate=True)
    with tarfile.open(fileobj=io.BytesIO(archive),mode="r:gz") as tf:
        matches=[m for m in tf.getmembers() if m.isfile() and Path(m.name).name=="v1_stage.py"]
        if len(matches)!=1:raise ValueError("expected exactly one original v1_stage.py")
        original=tf.extractfile(matches[0]).read().decode("utf-8")
    ast.parse(original);ast.parse(variant)
    old=original.splitlines();new=variant.splitlines()
    differ=list(difflib.unified_diff(old,new,n=0))
    added=[x[1:] for x in differ if x.startswith("+") and not x.startswith("+++")]
    removed=[x[1:] for x in differ if x.startswith("-") and not x.startswith("---")]
    def defs(s):
        return {n.name:sha(ast.dump(n,include_attributes=False).encode())
                for n in ast.walk(ast.parse(s)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    aa,bb=defs(original),defs(variant)
    keywords=["order","mass","priority","token","prefix","fable","guard","fallback","shrink","select","blend","budget","deadline","entropy"]
    changed_lines=[{"side":side,"line":line.strip()[:190]} for side,source in (("old",removed),("new",added))
       for line in source if any(k in line.lower() for k in keywords) and not len(line)>350]
    return {
        "base_embedded_stage_sha":sha(original.encode()),
        "variant_installed_stage_sha":sha(variant.encode()),
        "stage_replaced_during_notebook_setup":True,
        "old_line_count":len(old),"new_line_count":len(new),
        "diff_lines_added":len(added),"diff_lines_removed":len(removed),
        "functions_added":sorted(set(bb)-set(aa)),
        "functions_removed":sorted(set(aa)-set(bb)),
        "functions_changed":sorted(x for x in set(aa)&set(bb) if aa[x]!=bb[x]),
        "changed_decision_lines_sample":changed_lines[:15],
        "training_model_weights_changed_by_this_stage":False,
        "evidence_is_static_code_only_not_a_paired_accuracy_test":True
    }

def summary(a,b):
    ma,mb=a["modules"],b["modules"]
    commons=sorted(set(ma)&set(mb))
    changed=[{"filename":k,"old_sha":ma[k]["sha"],"new_sha":mb[k]["sha"],
              "old_size":ma[k]["chars"],"new_size":mb[k]["chars"]}
             for k in commons if ma[k]["sha"]!=mb[k]["sha"]]
    same=[k for k in commons if ma[k]["sha"]==mb[k]["sha"]]
    novel=sorted(set(mb)-set(ma))
    removed=sorted(set(ma)-set(mb))
    # Readable provenance markers, not the raw script/score or any original text.
    facts={}
    for k,v in (("older_0715",a),("newer_0718",b)):
        facts[k]={
            "already_scored_submission":EXPECTED[k]["submission"],
            "already_scored_public_BA":EXPECTED[k]["score"],
            "code_sha":v["code_sha"],
            "owner_notebook":v["owned"],
            "num_code_cells":v["num_code_cells"],
            "num_all_cells":v["num_all_cells"],
            "module_count":len(v["modules"]),
            "source_mount_count":v["source_mounts"],
            "signals":v["signals"],
            "runtime_full_hidden_not_measured_here":True,
        }
    return {
       "scope":"read_only_source_lineage_no_accuracy_attribution",
       "direct_execution_reachability":reachability(a,b),
       "embedded_stage_mechanism_diff":embedded_stage_diff(b),
       "scored_notebooks":facts,
       "older_scored_code_sha_verified":True,
       "newer_notebook_sha_matches_original_public_code_file":
         b["notebook_file_sha"]==EXPECTED["newer_0718"]["public_parent_file_sha"],
       "newer_only_modules":novel,
       "older_only_modules":removed,
       "changed_modules":changed,
       "identical_modules":same,
       "mount_drift":{"model":sorted(set(a["attached_models"])^set(b["attached_models"])),
          "dataset_count_change":len(b["attached_datasets"])-len(a["attached_datasets"]),
          "kernel_count_change":len(b["attached_kernels"])-len(a["attached_kernels"])},
       "source_notebook_similarity":{"identical_code_cells":len(set(a["cell_texts"])&set(b["cell_texts"])),
           "older_code_cells":len(a["cell_texts"]),"newer_code_cells":len(b["cell_texts"])},
       "cannot_claim_why_Public_0718_beats_0715_without_same_row_truth":True,
       "cannot_claim_why_previous_GLMSIDE_TRACE_FAILED_from_source_presence_alone":True,
       "private_or_competition_text_emitted":False,
       "new_model_training":False,"new_GPU":False,"new_competition_submission":False,
    }

def smoke():
    x={"cell_type":"code","source":["%%writefile /tmp/x.py\n","print('hi')\n"]}
    assert text_of(x).startswith("%%writefile")
    assert sha(b"abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert not text_of({"source":[]})
    print("NTL_LARGE_LINEAGE_SELFTEST=PASS",flush=True)

def main():
    args=argparse.ArgumentParser()
    args.add_argument("--old",type=Path)
    args.add_argument("--new",type=Path)
    args.add_argument("--smoke",action="store_true")
    v=args.parse_args()
    smoke()
    if v.smoke:return
    if not v.old or not v.new:raise ValueError("two owned Kaggle notebooks required")
    a=read(v.old,"older_0715")
    b=read(v.new,"newer_0718")
    print("NTL_LARGE_LINEAGE_REAL="+json.dumps(summary(a,b),sort_keys=True,default=str),flush=True)

if __name__=="__main__":main()
