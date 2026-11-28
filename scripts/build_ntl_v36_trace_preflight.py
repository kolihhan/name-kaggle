"""CPU-only source-pinned v36 trace instrumentation builder.
Does not launch Kaggle, read competition data, or modify classification logic.
An actual private run must supply an admitted cohort and measured runtime.
"""
import hashlib
import json
import re
from pathlib import Path

PIN="5988d64922650bf94490810effb7278907b301c07f581efbc6d4865d84e63d52"
APEX_SHA="15c3ff944e098a0fd5e5e8c6cd530f051d46f62fe915a6cc31f6a3d1148d0b35"
SELECTION="frame.iloc[sel][['id', 'text']].to_csv(query, index=False)"
FINAL="current_sub.to_parquet(OUT, index=False)"

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def source(c):return "".join(c.get("source",[]))
def code(s):
    compile(s,"<trace sidecar>","exec")
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.splitlines(True)}
def append_after(s,anchor,lines):
    pattern=r"(?m)^([ \t]*)"+re.escape(anchor)+r"[ \t]*$"
    matches=list(re.finditer(pattern,s))
    if len(matches)!=1:raise ValueError("ambiguous instrumentation anchor")
    m=matches[0];prefix=m.group(1)
    return s[:m.end()]+"\n"+"\n".join(prefix+l for l in lines)+s[m.end():]

def build(raw, trace_source):
    if sha(raw)!=PIN:raise ValueError("unpinned v36 source")
    compile(trace_source,"<private trace module>","exec")
    nb=json.loads(raw);original=nb["cells"]
    if len(original)!=58 or sum(c.get("cell_type")=="code" for c in original)!=35:
        raise ValueError("unexpected source structure")
    selected=[i for i,c in enumerate(original) if source(c).startswith("%%writefile") and SELECTION in source(c)]
    apex=[i for i,c in enumerate(original) if "Starting Apex Grandmaster V2 Post-Processing Stage..." in source(c)]
    if len(selected)!=1 or len(apex)!=1 or selected[0]>=apex[0]:raise ValueError("unexpected stage anchors")
    si,ai=selected[0],apex[0]
    if sha(source(original[ai]))!=APEX_SHA:raise ValueError("unpinned Apex")
    setup=code(
        "# PRIVATE diagnostic only; no output-path mutation.\n"
        "import os as _nt_os, sys as _nt_sys\n"
        "from pathlib import Path as _nt_Path\n"
        "assert _nt_os.environ.get('NTL_PRIVATE_TRACE_AUTHORIZED') == '1'\n"
        "_nt_Path('/kaggle/temp').mkdir(parents=True,exist_ok=True)\n"
        "_nt_Path('/kaggle/temp/ntl_private_stage_trace.py').write_text("+repr(trace_source)+")\n"
        "_nt_sys.path.insert(0,'/kaggle/temp')\n"
        "from ntl_private_stage_trace import Trace as _NT_Trace\n")
    before=code(
        "# Capture native and V1 evidence before unmodified Apex.\n"
        "if not FAST_COMMIT:\n"
        "    import numpy as _nt_np, pandas as _nt_pd, json as _nt_json\n"
        "    _nt_base=_nt_Path('/kaggle/temp/final-logits.npz')\n"
        "    with _nt_np.load(_nt_base,allow_pickle=False) as _nt_z:\n"
        "        _nt_ids=_nt_z['ids'].astype(str); _nt_labels=_nt_z['labels'].astype(str)\n"
        "    _nt_contract=_nt_json.loads(_nt_Path('/kaggle/temp/ntl-private-trace-contract.json').read_text())\n"
        "    _nt_q=_nt_pd.read_csv(_nt_Path(DATA)/'test.csv',dtype={'id':str},usecols=['id'])\n"
        "    assert _nt_q.id.tolist()==_nt_ids.tolist()\n"
        "    _nt_trace=_NT_Trace('/kaggle/working/private-stage-trace',_nt_ids,_nt_labels,source_sha="+repr(PIN)+",cohort_sha=_nt_contract['cohort_sha'],asset_manifest_sha=_nt_contract['asset_manifest_sha'])\n"
        "    _nt_trace.capture_npz('native',_nt_base,semantics='weighted_log_score')\n"
        "    _nt_report=_nt_json.loads((V1_OUT/'report.json').read_text()) if (V1_OUT/'report.json').is_file() else {}\n"
        "    _nt_v1_status=_nt_report.get('status','missing')\n"
        "    if _nt_v1_status=='applied':\n"
        "        _nt_trace.capture_npz('post_v1','/kaggle/temp/prefix-votes.npz',semantics='logits')\n"
        "        with _nt_np.load('/kaggle/temp/ntl-trace-selected.npz',allow_pickle=False) as _nt_z:\n"
        "            _nt_selected=_nt_z['ids'].astype(str)\n"
        "        assert len(set(_nt_selected))==len(_nt_selected)==int(_nt_report['n_selected'])\n"
        "        assert set(_nt_selected)<=set(_nt_ids)\n"
        "        _nt_trace.mask('v1_selected',_nt_ids,_nt_np.isin(_nt_ids,_nt_selected))\n"
        "        _nt_trace.write('v1_selection_order',ids=_nt_selected)\n")
    changed_selection=append_after(source(original[si]),SELECTION,[
        "np.savez_compressed('/kaggle/temp/ntl-trace-selected.npz', ids=frame.iloc[sel]['id'].astype(str).to_numpy(dtype=str))"])
    compile(changed_selection.split("\n",1)[1],"<V1 trace module>","exec")
    # Original decision and parquet write are kept verbatim. Capture only after that write.
    changed_apex=append_after(source(original[ai]),FINAL,[
        "assert query_df.id.astype(str).tolist()==_nt_ids.tolist()",
        "_nt_trace.scores('apex_refined',_nt_ids,_nt_np.asarray(labels,dtype=str),_nt_np.asarray(refined_lp),semantics='logits')",
        "_nt_trace.mask('apex_protection',_nt_ids,_nt_np.asarray(_protect,dtype=bool))",
        "_nt_trace.final(current_sub.id.astype(str).to_numpy(dtype=str),current_sub.model.astype(str).to_numpy(dtype=str))",
        "# Partial trace only: runtime and independent parity receipt must be supplied by the approved runner.",
        "_nt_receipt=dict(_nt_trace.manifest,v1_status=_nt_v1_status,seal_status='UNSEALED_PENDING_RUNTIME_AND_PARITY')",
        "(_nt_trace.root/'partial_manifest.json').write_text(_nt_json.dumps(_nt_receipt,sort_keys=True))"])
    compile(changed_apex,"<Apex trace cell>","exec")
    new=[]
    for i,c in enumerate(original):
        if i==0:new.append(setup)
        if i==ai:new.append(before)
        if i in (si,ai):
            c=dict(c);c['source']=(changed_selection if i==si else changed_apex).splitlines(True)
        new.append(c)
    assert len(new)==60
    assert all(c in new for i,c in enumerate(original) if i not in (si,ai))
    nb['cells']=new
    return nb, {"source_sha":PIN,"original_cells":58,"new_cells":60,"unchanged_original_cells":56,
        "selection_cell":si,"apex_cell":ai,"selection_additions":1,"apex_additions":7,
        "source_instrumentation_only":True,"private_run_launched":False,
        "runtime_parity_verified":False,"promotion":False}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    nb,report=build(a.source.read_text(),Path("scripts/ntl_private_stage_trace.py").read_text())
    a.out.write_text(json.dumps(nb,ensure_ascii=False))
    print("V36_TRACE_BUILD="+json.dumps(report,sort_keys=True),flush=True)
