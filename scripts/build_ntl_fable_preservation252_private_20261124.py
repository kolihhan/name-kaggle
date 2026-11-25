"""Build one private 0.718 scored-path 252-row external Fable counterfactual.

Keep all original donor code cells in original order, patch only the exact
DATA substitution and GLM runtime deadline, add read-only sidecars. Any
unexpected scorer source drift fails closed. No output data committed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

DONOR="kolihhan/ntl-evgen-forensic-eda-repro-20261123"
PUBLIC_RECOVERY="evgendvorkin/name-that-llm-deep-forensic-eda-with-analogie"
DONOR_FILE_SHA="74d86affa9f55464ab39f735f392b54e47e2d8a2b631a0bca45de9435c600c56"
DEST="kolihhan/ntl-fable-preservation-252-v2-20261125"
PANEL="nischaydnk/ntl-topic-synthetic-18llms"
RUNNER=Path("scripts/ntl_fable_preservation252_runtime_20261124.py")
DATA_PATTERN=re.compile(r"(?m)^DATA = next\(p for p in .*os\.path\.exists\(p \+ '/train\.csv'\)\)$")
DEADLINE_PATTERN=re.compile(r"(?m)^V1_DEADLINE_H\s*=\s*8\.8\s*(?:#.*)?$")
APEX_MARKER="Starting Apex Grandmaster V2 Post-Processing Stage..."
V1_MARKER="V1_DEADLINE_H"
PROBE_MARKER="NTL_FABLE_RESEARCH_SIDECAR"

def text(c):
    t=c.get("source",[])
    return "".join(t) if isinstance(t,list) else str(t)

def code(src):
    compile(src,"<validated notebook research sidecar>","exec")
    return {"cell_type":"code","execution_count":None,"metadata":{},
            "outputs":[],"source":src.splitlines(keepends=True)}

def patch_data_assignment(source):
    """Keep original DATA discovery before using DATA in the diagnostic.

    Previous broken release replaced its FIRST assignment with
    DATA = prepare(DATA), causing NameError before any diagnostic markers.
    """
    def replace(m):
        return (m.group(0)
            + "\nDATA = str(ntl_fable_probe.prepare(DATA))"
            + "\nos.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'")
    return DATA_PATTERN.subn(replace,source)


def build(inp,out):
    if out.exists():raise FileExistsError("one-shot destination already exists")
    md=json.loads((inp/"kernel-metadata.json").read_text())
    # Kaggle returns 404 for the private scored copy. The publicly published
    # author's notebook was previously SHA-verified as the byte-identical
    # code source of that 0.718 reproduction. Keep the SHA guard below.
    if md.get("id") not in (DONOR,PUBLIC_RECOVERY) or not md.get("enable_gpu",False):
        raise ValueError("not the frozen 0.718 source or verified public recovery")
    if not md.get("competition_sources")==["name-that-llm"]:
        raise ValueError("unexpected original competition contract")
    if PANEL in (md.get("dataset_sources") or []):
        raise ValueError("external panel already mounted in original donor")
    notebook_path=inp/Path(md["code_file"]).name
    if hashlib.sha256(notebook_path.read_bytes()).hexdigest()!=DONOR_FILE_SHA:
        raise ValueError("0.718 scored public reproduction source file drift")
    nb=json.loads(notebook_path.read_text())
    original=nb["cells"]
    if len(original)!=58:raise ValueError("original notebook cell count drift")
    data_indices=[i for i,c in enumerate(original) if c.get("cell_type")=="code" and
                  DATA_PATTERN.search(text(c))]
    stage_indices=[i for i,c in enumerate(original) if c.get("cell_type")=="code" and
                  DEADLINE_PATTERN.search(text(c))]
    apex_indices=[i for i,c in enumerate(original) if c.get("cell_type")=="code" and
                  APEX_MARKER in text(c)]
    if any(len(x)!=1 for x in (data_indices,stage_indices,apex_indices)):
        raise ValueError("scorer stage anchors missing or ambiguous")
    d,v,a=data_indices[0],stage_indices[0],apex_indices[0]
    if not d<v<a:raise ValueError("wrong native GLM Apex stage order")
    runner=RUNNER.read_text()
    compile(runner,RUNNER.name,"exec")
    if hashlib.sha256(runner.encode()).hexdigest()!=hashlib.sha256(RUNNER.read_text().encode()).hexdigest():
        raise ValueError("runner changed mid-build")
    helper=code(
       "# "+PROBE_MARKER+" setup: does not read labels or run models.\n"
       "from pathlib import Path\n"
       "import sys\n"
       "Path('/kaggle/temp').mkdir(parents=True,exist_ok=True)\n"
       "Path('/kaggle/temp/ntl_fable_probe.py').write_text("+repr(runner)+")\n"
       "sys.path.insert(0,'/kaggle/temp')\n"
       "import ntl_fable_probe\n"
    )
    before=code(
       "# "+PROBE_MARKER+" after native/GLM, before ORIGINAL Apex.\n"
       "if not FAST_COMMIT:\n"
       "    ntl_fable_probe.snapshot_pre_apex(DATA,OUT)\n"
    )
    after=code(
       "# "+PROBE_MARKER+" post ORIGINAL Apex, aggregate labelled replay only.\n"
       "if not FAST_COMMIT:\n"
       "    ntl_fable_probe.evaluate(DATA,OUT,refined_lp,refined_preds,labels)\n"
    )
    out_cells=[]
    for i,c in enumerate(original):
        if i==d:out_cells.append(helper)
        if i==a:out_cells.append(before)
        if i==d:
            originalcode=text(c)
            changed,count=patch_data_assignment(originalcode)
            if count!=1:raise ValueError("failed deterministic fake input injection")
            # Source-scoped 18-label truth is stored only in temp and not passed to the scorer.
            c=dict(c);c["source"]=changed.splitlines(keepends=True)
        if i==v:
            originalcode=text(c)
            changed,count=DEADLINE_PATTERN.subn("V1_DEADLINE_H = 5.8  # bounded research-only",originalcode)
            if count!=1:raise ValueError("5.8h V1 research cap failed")
            c=dict(c);c["source"]=changed.splitlines(keepends=True)
        out_cells.append(c)
    out_cells.append(after)
    if len(out_cells)!=len(original)+3:raise AssertionError("exact three sidecars")
    for i,c in enumerate(original):
        if i not in (d,v) and c not in out_cells:raise AssertionError("original cell missing")
    if any(PROBE_MARKER in text(c) for c in original):
        raise ValueError("starting source already instrumented")
    candidate=dict(nb);candidate["cells"]=out_cells
    newmd=dict(md)
    newmd.pop("id_no",None)
    newmd.update({"id":DEST,"title":DEST.split("/",1)[1],
                  "is_private":True,"enable_gpu":True,"enable_internet":False,
                  "dataset_sources":list(md["dataset_sources"])+[PANEL]})
    for k in ("model_sources","kernel_sources","competition_sources","machine_shape"):
        if newmd.get(k)!=md.get(k):raise ValueError("scorer dependency modified: "+k)
    out.mkdir(parents=True,exist_ok=False)
    (out/notebook_path.name).write_text(json.dumps(candidate,ensure_ascii=False,indent=1)+"\n")
    newmd["code_file"]=notebook_path.name
    (out/"kernel-metadata.json").write_text(json.dumps(newmd,indent=1,ensure_ascii=False)+"\n")
    report={
      "one_private_kernel":DEST,"donor_source_exact_sha":DONOR_FILE_SHA,
      "source_ref_used":md["id"],"public_recovery_ref_used":md["id"]==PUBLIC_RECOVERY,
      "original_cells":len(original),"sidecars":3,"mutated_original_cells":2,
      "original_data_cell_index":d,"original_glm_config_cell_index":v,
      "original_apex_cell_index":a,
      "source_mounts_original":{"models":len(md["model_sources"]),
        "datasets":len(md["dataset_sources"]),"kernels":len(md["kernel_sources"])},
      "added_pinned_publisher_panel":PANEL,"rows":252,"classes":18,
      "real_training_started":False,"reused_existing_pretrained_pipeline":True,
      "official_submit":False,"use_gpu":True,
      "max_v1_h":5.8,"no_raw_dataset_in_github":True
    }
    print("NTL_FABLE_252_ONE_SHOT_BUILD="+json.dumps(report,sort_keys=True),flush=True)
    return report

def smoke():
    assert DATA_PATTERN.search("DATA = next(p for p in ['/x','/y'] if os.path.exists(p + '/train.csv'))")
    assert DEADLINE_PATTERN.search("V1_DEADLINE_H = 8.8    # fixed 9h Kaggle cap")
    assert text(code("x=1\n")).strip()=="x=1"
    # Compile AND EXECUTE the first-use DATA assignment with a safe fake path.
    # This detects the exact 2026-11-24 NameError that the old smoke missed.
    import os, tempfile
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as root:
        Path(root,"train.csv").write_text("id,model,text\n")
        original="DATA = next(p for p in ["+repr(root)+"] if os.path.exists(p + '/train.csv'))"
        patched,n=patch_data_assignment(original)
        assert n==1 and patched.startswith(original+"\n"),patched
        compile(patched,"<first-DATA-assignment-regression>","exec")
        ns={"os":os,"ntl_fable_probe":SimpleNamespace(prepare=lambda x:x+"/test-data")}
        previous=os.environ.get("KAGGLE_IS_COMPETITION_RERUN")
        try:
            exec(patched,ns)
            assert ns["DATA"]==root+"/test-data"
            assert os.environ["KAGGLE_IS_COMPETITION_RERUN"]=="1"
        finally:
            if previous is None:os.environ.pop("KAGGLE_IS_COMPETITION_RERUN",None)
            else:os.environ["KAGGLE_IS_COMPETITION_RERUN"]=previous
    print("NTL_FABLE_252_FIRST_ASSIGNMENT_NAMEERROR_REGRESSION=PASS",flush=True)
    print("NTL_FABLE_252_BUILDER_CPU_SELFTEST=PASS",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path)
    ap.add_argument("--out",type=Path)
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    if a.smoke:smoke()
    else:
        if a.source is None or a.out is None:ap.error("--source and --out required")
        build(a.source,a.out)
