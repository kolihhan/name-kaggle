"""Build an unpushed PRIVATE frozen 0.715 scorer + 2 diagnostic-only cells.

Nothing in the scoring source is modified: one cell snapshots the S10 output
immediately before original GLM V1 stage, one appended cell summarizes the
post-stage label transitions. No new model, no Kaggle submission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

SOURCE = "kolihhan/ntl-estopadilla-glm53-0716-repro-20261122"
SOURCE_CODE_SHA = "202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8"
DEST = "kolihhan/ntl-0715-frozen-stage-trace-research-20261123"
RUNTIME = Path("scripts/ntl_frozen0715_stage_trace_20261123.py")
GLM_MARKER = "# ---- ours, V1 stage: likelihood of the test"
BASE_MARKER = "'--save_final_logits', FINAL_LOGITS"


def _source_code(nb):
    return "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source"), list) else str(cell.get("source", ""))
        for cell in nb["cells"] if cell.get("cell_type") == "code"
    )


def _cell(src):
    compile(src, "<NTL sidecar>", "exec")
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def _is_pre(c):
    return BASE_MARKER in "".join(c.get("source", [])) if isinstance(
        c.get("source"), list) else BASE_MARKER in str(c.get("source", ""))


def _is_glm(c):
    return GLM_MARKER in "".join(c.get("source", [])) if isinstance(
        c.get("source"), list) else GLM_MARKER in str(c.get("source", ""))


def build(download: Path, output: Path):
    if output.exists():
        raise FileExistsError("never overwrite a prior scientific candidate")
    md_path = download / "kernel-metadata.json"
    md = json.loads(md_path.read_text())
    if md.get("id") != SOURCE:
        raise ValueError("not the exact owned 0.715 code source")
    nb_name = Path(md["code_file"]).name
    if not nb_name.endswith(".ipynb"):
        raise ValueError("expected real pinned notebook")
    nb = json.loads((download / nb_name).read_text())
    orig_cells = nb["cells"]
    orig_code = _source_code(nb)
    sha = hashlib.sha256(orig_code.encode()).hexdigest()
    if sha != SOURCE_CODE_SHA:
        raise ValueError("original 0.715 code SHA drift: " + sha)
    infer = [i for i,c in enumerate(orig_cells) if _is_pre(c)]
    v1 = [i for i,c in enumerate(orig_cells) if _is_glm(c)]
    if len(infer)!=1 or len(v1)!=1 or infer[0] >= v1[0]:
        raise ValueError(f"could not prove unique S10->V1 boundary: {infer} {v1}")
    if any("NTL_SIDE_CAR_TRACE" in "".join(c.get("source", []))
           for c in orig_cells if isinstance(c.get("source"), list)):
        raise ValueError("not an untouched champion")
    lib = RUNTIME.read_text()
    compile(lib, str(RUNTIME), "exec")
    pre = (
        "# NTL_SIDE_CAR_TRACE_BEFORE_GLM — no edits to scoring arrays or OUT.\n"
        "try:\n"
        f"    _ntl_trace_library = {lib!r}\n"
        "    _ntl_trace_ns = {'__name__':'ntl_stage_trace_private'}\n"
        "    exec(compile(_ntl_trace_library, 'ntl_trace_library.py', 'exec'), _ntl_trace_ns)\n"
        "    _ntl_pre_meta = _ntl_trace_ns['snapshot_pre_glm']()\n"
        "    print('NTL_TRACE_PRE_GLM_METADATA=' + __import__('json').dumps(_ntl_pre_meta, sort_keys=True),flush=True)\n"
        "except Exception as _ntl_trace_exc:\n"
        "    _ntl_trace_ns = None\n"
        "    print('NTL_TRACE_PRE_GLM_NOT_VERIFIED=' + type(_ntl_trace_exc).__name__, flush=True)\n"
    )
    post = (
        "# NTL_SIDE_CAR_TRACE_AFTER_ALL_ORIGINAL_CELLS — output is untouched.\n"
        "try:\n"
        "    if not isinstance(globals().get('_ntl_trace_ns'), dict):\n"
        "        raise RuntimeError('pre_stage_snapshot_missing')\n"
        "    _ntl_trace_report = globals().get('V1_OUT', None)\n"
        "    _ntl_trace_report = (_ntl_trace_report / 'report.json') if _ntl_trace_report is not None else None\n"
        "    _ntl_result = _ntl_trace_ns['audit_post_glm'](v1_report=_ntl_trace_report)\n"
        "    print('NTL_TRACE_UNLABELLED_STAGE_CHANGES=' + __import__('json').dumps(_ntl_result, sort_keys=True), flush=True)\n"
        "except Exception as _ntl_trace_exc:\n"
        "    print('NTL_TRACE_POST_GLM_NOT_VERIFIED=' + type(_ntl_trace_exc).__name__, flush=True)\n"
    )
    revised = dict(nb)
    # The snapshot runs after the native/S10 submission writer and strictly
    # before GLM V1 can overwrite the same OUT file.
    revised["cells"] = (
        orig_cells[:v1[0]] + [_cell(pre)] + orig_cells[v1[0]:] + [_cell(post)]
    )
    preserved = [c for c in revised["cells"] if "NTL_SIDE_CAR_TRACE" not in
                 ("".join(c.get("source", [])) if isinstance(c.get("source"), list)
                  else str(c.get("source", "")))]
    if preserved != orig_cells:
        raise AssertionError("changed original scored notebook cell or ordering")
    if len(revised["cells"]) != len(orig_cells)+2:
        raise AssertionError("only two diagnostic cells may be added")
    altered_md = dict(md)
    altered_md.pop("id_no", None)
    altered_md.update({"id": DEST, "title": "NTL Frozen 0.715 Stage Trace - Private Research",
                       "is_private": True})
    untouched = ("model_sources","dataset_sources","kernel_sources",
                 "competition_sources","enable_gpu","machine_shape",
                 "enable_internet","kernel_type")
    if any(altered_md.get(k) != md.get(k) for k in untouched):
        raise AssertionError("changed inference assets or budget")
    output.mkdir(parents=True)
    (output / nb_name).write_text(json.dumps(revised,indent=1,ensure_ascii=False)+"\n")
    (output / "kernel-metadata.json").write_text(
        json.dumps(altered_md,indent=2,ensure_ascii=False)+"\n")
    reread = json.loads((output / nb_name).read_text())
    if len(reread["cells"]) != len(orig_cells)+2:
        raise AssertionError("candidate roundtrip failed")
    return {
        "status": "PRIVATE_UNPUSHED_CPU_STATIC_BUNDLE_VALID",
        "original_code_sha_verified": True,
        "original_code_sha256": sha,
        "original_cells_bytewise_unchanged": True,
        "original_cell_count": len(orig_cells),
        "sidecar_cell_count": 2,
        "infer_cell_index": infer[0],
        "pre_glm_snapshot_insert_before_original_cell": v1[0],
        "post_stage_audit_original_last_cell_index": len(orig_cells)-1,
        "private_destination_slug": DEST,
        "attached_model_counts":len(md.get("model_sources") or []),
        "attached_data_counts":len(md.get("dataset_sources") or []),
        "attached_kernel_counts":len(md.get("kernel_sources") or []),
        "unchanged_all_runtime_mounts_and_gpu":True,
        "instrumentation_never_writes_submission":True,
        "source_notebook_not_exported_or_uploaded_to_Github":True,
        "kaggle_private_kernel_pushed":False,
        "model_inference_executed":False,
        "competition_submitted":False,
    }


def smoke():
    assert _is_pre(_cell(BASE_MARKER))
    assert _is_glm(_cell(GLM_MARKER))
    assert not _is_pre(_cell(GLM_MARKER))
    assert not _is_glm(_cell(BASE_MARKER))
    print("NTL_STAGE_TRACE_BUILDER_SMOKE=PASS",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.smoke:
        smoke()
    else:
        if not a.source or not a.out:
            p.error("source and out required")
        print("NTL_0715_STAGE_TRACE_BUILDER="+json.dumps(
            build(a.source,a.out), sort_keys=True),flush=True)
