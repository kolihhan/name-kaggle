#!/usr/bin/env python3
"""Build ONE frozen 0.691 Oleg216 diagnostic with final post-V1 18-class scores.

No model training, no extra inference, no public per-row predictions. The old
216 predictions are embedded only into a NEW PRIVATE generated Kaggle notebook.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import json
import tempfile
import zlib
from pathlib import Path

import pandas as pd
from scripts import ntl_oleg18_frozen_champion_pilot_20261116 as frozen
from scripts import ntl_lightning_0694_scale300_private as helper

DEST = "kolihhan/ntl-oleg216-final18-diagnostic-nov17"
REFERENCE_PROBE_SHA = "e7ea37f4ff1f7aae0fa8dd78b29ce36e1f321b9e714f010e4caaf660333ecd51"
V1_PATH = "%%writefile /kaggle/temp/v1/v1_stage.py"
V1_ANCHOR = "    after[sel] = (zl[sel] + w * ll).argmax(1)"
V1_WRITE = "    sub.to_parquet(a.out, index=False)"
SOURCE_RUN = "    subprocess.run([sys.executable,'-u',str(RUNTIME/'ensemble.py'),'--config',str(RUNTIME/'config.json'),\n                    '--out',str(OUTPUT)],check=True)"
V1_RUN = "    subprocess.run(command,env=V1_ENV,check=True)"


def code(cell):
    return helper.source(cell)


def change_once(text: str, before: str, after: str, where: str) -> str:
    if text.count(before) != 1:
        raise ValueError("EXPECTED_EXACTLY_ONE_" + where + " anchor_count=" + str(text.count(before)))
    return text.replace(before, after, 1)


def build(source: Path, baseline: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("REFUSE_OVERWRITE")
    old = pd.read_parquet(baseline)
    if old.columns.tolist() != ["id", "model"] or len(old) != 216:
        raise ValueError("FROZEN_216_BASELINE_REQUIRED")
    if old.id.astype(str).duplicated().any() or old.model.isna().any():
        raise ValueError("INVALID_BASELINE_IDS_OR_LABELS")
    from scripts.ntl_oleg18_frozen_selector_20261116 import NTL_OLEG18_MODELS
    if not set(old.model.astype(str)).issubset(set(NTL_OLEG18_MODELS)):
        raise ValueError("OUT_OF_SCOPE_BASELINE_LABEL")
    refpairs = [[str(x.id), str(x.model)] for x in old.itertuples(index=False)]
    packed = base64.b64encode(zlib.compress(json.dumps(refpairs, separators=(",", ":")).encode())).decode()

    with tempfile.TemporaryDirectory() as t:
        orig = Path(t) / "original"
        original = frozen.build(source, orig)
        meta = json.loads((orig / "kernel-metadata.json").read_text())
        filename = meta["code_file"]
        original_bytes = (orig / filename).read_bytes()
        orig_sha = hashlib.sha256(original_bytes).hexdigest()
        if orig_sha != REFERENCE_PROBE_SHA:
            raise ValueError("FROZEN_0_691_OLEG_NOTEBOOK_SOURCE_DRIFT:" + orig_sha)
        nb = json.loads(original_bytes)

    edits = {"v1": 0, "ensemble_timeout": 0, "v1_timeout": 0}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        s = code(cell)
        if s.startswith(V1_PATH):
            # The V1 post-blend is the ACTUAL final classifier. z is 18-class
            # ensemble logits; only selected rows receive w * ll corrections.
            insert_scores = """
    _ntl_final18 = zl.copy()
    _ntl_final18[sel] += w * ll
    assert _ntl_final18.shape == (len(ids), 18)
    assert np.isfinite(_ntl_final18).all()
    assert np.array_equal(_ntl_final18.argmax(1), after), 'POST_V1_DECISION_MISMATCH'
"""
            s = change_once(s, V1_ANCHOR, V1_ANCHOR + insert_scores.rstrip(), "post_v1_final_scores")
            dump = """
    # Diagnostics only. Reference prediction pairs are private, fetched by the
    # authenticated CPU workflow and embedded into this private notebook.
    import base64 as _ntl_b64, zlib as _ntl_zlib, os as _ntl_os
    _ntl_expected = json.loads(_ntl_zlib.decompress(_ntl_b64.b64decode('PACKED_REFERENCE')).decode())
    assert len(_ntl_expected) == 216 and len(ids) == 216
    assert [x[0] for x in _ntl_expected] == [str(x) for x in ids], 'BASELINE_ID_ORDER_DRIFT'
    _ntl_float_scores = _ntl_final18.astype(np.float32)
    assert np.isfinite(_ntl_float_scores).all()
    assert np.array_equal(_ntl_float_scores.argmax(1), after), 'FLOAT32_CHANGES_ARGMAX'
    _ntl_found = np.asarray(labels)[after].astype(str).tolist()
    _ntl_match = [p == rec[1] for p, rec in zip(_ntl_found, _ntl_expected)]
    assert all(_ntl_match[:5]), 'FAIL_FAST_FIRST5_REFERENCE_MISMATCH'
    assert sum(_ntl_match) >= 214, 'LESS_THAN_99_PERCENT_BASELINE_AGREEMENT'
    _ntl_dest = Path('/kaggle/working/ntl_oleg216_final18_logits.jsonl')
    with _ntl_dest.open('w', encoding='utf-8', buffering=1) as _ntl_out:
        for _ntl_i, _ntl_id in enumerate(ids):
            _ntl_item = {
                'sample_id':str(_ntl_id),
                'labels':[str(_ntl_label) for _ntl_label in labels],
                'logits':_ntl_float_scores[_ntl_i].tolist(),
                'argmax':str(_ntl_found[_ntl_i])
            }
            _ntl_out.write(json.dumps(_ntl_item,allow_nan=False,separators=(',',':'))+'\\n')
            _ntl_out.flush()
            _ntl_os.fsync(_ntl_out.fileno())
    Path('/kaggle/working/ntl_oleg216_final18_receipt.json').write_text(
        json.dumps({'rows':len(ids),'matched':sum(_ntl_match),
                    'first5_match':True,'score_stage':'post_v1',
                    'source_unchanged_except_diagnostic':True}))
    print('NTL_OLEG216_POSTV1_18CLASS_DUMP=PASS rows=216 matched='+str(sum(_ntl_match)),flush=True)
"""
            dump = dump.replace("PACKED_REFERENCE", packed)
            s = change_once(s, V1_WRITE, V1_WRITE + dump.rstrip(), "post_v1_writer")
            edits["v1"] += 1

        # 72-minute (80% of 90 minutes) graceful watchdog for inference stages.
        # Preserve the original data path and stage configuration exactly.
        if "CONFIG['gsq_share'] = 0.12" in s and SOURCE_RUN in s:
            start = """    import time as _ntl_time
    _ntl_cutoff_monotonic = _ntl_time.monotonic() + 72*60
    _ntl_diagnostic_timeout = False
"""
            s = change_once(s, "    import subprocess\n", "    import subprocess\n" + start, "first_stage_timer")
            guarded = SOURCE_RUN.replace("check=True)", "check=True,timeout=max(1,int(_ntl_cutoff_monotonic - _ntl_time.monotonic())))")
            block = """    try:
""" + "\n".join("    " + line for line in guarded.splitlines()) + """
    except subprocess.TimeoutExpired:
        _ntl_diagnostic_timeout = True
        Path('/kaggle/working/ntl_oleg216_diag_timeout.json').write_text(
            json.dumps({'stage':'ensemble','status':'graceful_80pct_cutoff','valid_logits':False}))
        print('NTL_OLEG216_GRACEFUL_TIMEOUT_ENSEMBLE',flush=True)
"""
            s = change_once(s, SOURCE_RUN, block.rstrip(), "ensemble_run_timeout")
            # Avoid reading a missing output when the watchdog stops the ensemble.
            s = change_once(s, "    actual = pd.read_csv(DATA,dtype={'id':str})",
                            "    if not _ntl_diagnostic_timeout:\n        actual = pd.read_csv(DATA,dtype={'id':str})",
                            "after_ensemble_data_check")
            s = change_once(s, "    result = pd.read_parquet(OUTPUT)",
                            "        result = pd.read_parquet(OUTPUT)", "after_ensemble_read")
            s = change_once(s,
                "    assert result.columns.tolist() == ['id','model'] and result.id.tolist() == actual.id.tolist()",
                "        assert result.columns.tolist() == ['id','model'] and result.id.tolist() == actual.id.tolist()",
                "after_ensemble_assert1")
            s = change_once(s, "    assert result.model.notna().all()",
                            "        assert result.model.notna().all()", "after_ensemble_assert2")
            s = change_once(s, "display(pd.read_parquet(OUTPUT).head())",
                "if not globals().get('_ntl_diagnostic_timeout',False):\n    display(pd.read_parquet(OUTPUT).head())", "after_ensemble_display")
            edits["ensemble_timeout"] += 1

        if s.startswith("if not preview:\n    command = ["):
            s = change_once(s, "if not preview:",
                "if not preview and not globals().get('_ntl_diagnostic_timeout',False):", "v1_skip_timeout")
            guarded = """    try:
        subprocess.run(command,env=V1_ENV,check=True,
            timeout=max(1,int(_ntl_cutoff_monotonic - _ntl_time.monotonic())))
    except subprocess.TimeoutExpired:
        _ntl_diagnostic_timeout=True
        Path('/kaggle/working/ntl_oleg216_diag_timeout.json').write_text(
            json.dumps({'stage':'v1','status':'graceful_80pct_cutoff','valid_logits':False}))
        print('NTL_OLEG216_GRACEFUL_TIMEOUT_V1',flush=True)
"""
            s = change_once(s, V1_RUN, guarded.rstrip(), "v1_runtime")
            s = change_once(s, "    final = pd.read_parquet(OUTPUT)",
                "    if not _ntl_diagnostic_timeout:\n        final = pd.read_parquet(OUTPUT)", "v1_read")
            s = change_once(s, "    assert final.columns.tolist()==['id','model'] and final.id.tolist()==pd.read_csv(DATA,dtype={'id':str}).id.tolist()",
                "        assert final.columns.tolist()==['id','model'] and final.id.tolist()==pd.read_csv(DATA,dtype={'id':str}).id.tolist()", "v1_assert")
            s = change_once(s, "display(pd.read_parquet(OUTPUT).head())",
                "if not globals().get('_ntl_diagnostic_timeout',False):\n    display(pd.read_parquet(OUTPUT).head())",
                "v1_display")
            edits["v1_timeout"] += 1

        cell["source"] = s.splitlines(keepends=True)

    if edits != {"v1": 1, "ensemble_timeout": 1, "v1_timeout": 1}:
        raise ValueError("NOT_EXACT_THREE_INSTRUMENTATION_POINTS:" + str(edits))

    compiled = 0
    for cell in nb["cells"]:
        s = code(cell)
        if cell.get("cell_type") == "code" and s.startswith("%%writefile "):
            compile("\n".join(s.splitlines()[1:]), s.splitlines()[0], "exec")
            compiled += 1
        elif cell.get("cell_type") == "code" and "_ntl_diagnostic_timeout" in s:
            compile(s, "instrumented_notebook_cell", "exec")
    if compiled != 37:
        raise ValueError("FROZEN_MODULE_COUNT_DRIFT")

    meta["id"] = DEST
    meta["title"] = "NTL Oleg216 final 18-class diagnostic once"
    meta["is_private"] = True
    meta["enable_gpu"] = True
    meta["enable_internet"] = False
    meta["machine_shape"] = "NvidiaTeslaT4"
    meta.pop("docker_image", None)
    meta.pop("id_no", None)
    assert len(meta["dataset_sources"]) == 24 and len(meta["model_sources"]) == 6
    assert len(meta["kernel_sources"]) == 9
    output.mkdir(parents=True, exist_ok=False)
    raw = json.dumps(nb, ensure_ascii=False)
    (output / meta["code_file"]).write_text(raw, encoding="utf-8")
    (output / "kernel-metadata.json").write_text(json.dumps(meta,indent=2,sort_keys=True))
    return {
        "kernel":DEST,
        "baseline_rows":len(old),
        "source_v2_sha":helper.V2_CODE_SHA,
        "original_oleg_probe_sha":orig_sha,
        "scoring_modules_compiled":compiled,
        "only_mutated":list(edits),
        "genuine_baseline_predictions_embedded_in_private_notebook_only":True,
        "no_training":True,
        "gpu_launched":False,
        "official_submission":False,
        "notebook_sha256":hashlib.sha256(raw.encode()).hexdigest(),
    }


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print("NTL_OLEG216_FINAL18_CPU_BUILDER="+json.dumps(build(a.source,a.baseline,a.out),sort_keys=True),flush=True)
