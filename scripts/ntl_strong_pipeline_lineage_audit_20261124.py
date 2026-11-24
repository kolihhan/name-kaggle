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
