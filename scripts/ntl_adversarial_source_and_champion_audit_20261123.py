"""NTL adversarial evidence gate: inspect public source records & 0.715 source observability.

Read-only. Always print aggregates; never emit source response texts, prompts,
request IDs, private model artifacts, competition data or API tokens.
Independent-holdout admission deliberately FAILS CLOSED.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
import pandas as pd

KNOWN = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "kimi-k3", "qwen3.8-max",
)
assert len(KNOWN) == 18 and len(set(KNOWN)) == 18

# Frozen before exploring; avoids searching for a favorable heldout score.
DATA = [
    ("ankiteshtiwari/ntl-exact-corpus-v3", "exact_corpus.csv"),
    ("ankiteshtiwari/ntl-blocks-v1", "ntl_blocks.csv"),
    ("beroli/ntl-benchmark-c2-generated-panels", None),
    ("beroli/ntl-benchmark-c1-generated-panels", None),
]
SCORER = "kolihhan/ntl-estopadilla-glm53-0716-repro-20261122"
CODE_SHA = "202bf1bf86ca6363f7f5d44786c30838904610feaf07701653d66da3495551a8"
SIZE_LIMIT_BYTES = 30_000_000
MAX_RECORDS = 75_000
FIELDS = {
    "label": ("model", "label", "target_model", "model_id", "generator",
              "model_name", "source_model"),
    "served": ("served_model", "gen_served_model", "returned_model", "actual_model",
               "response_model", "returned_model_id", "served_model_id", "gen_model"),
    "requested": ("requested_model", "request_model", "proxy_model",
                  "model_requested", "request_model_id"),
    "prompt": ("prompt_id", "prompt_group", "original_prompt_id", "prompt_hash",
               "group_id", "prompt_text", "prompt", "user_message", "instruction"),
    "text": ("text", "response", "output", "generation", "answer", "article", "content"),
    "source": ("source_dataset", "source_file", "source", "dataset", "data_source"),
    "route": ("gen_route", "route", "api_provider", "provider", "api_base"),
    "timestamp": ("timestamp", "created_at", "generation_date", "generated_at",
                  "request_date", "response_time"),
}
def emit(key, value):
    print(key+"="+json.dumps(value, sort_keys=True, ensure_ascii=False, default=str),flush=True)
def exec_cli(args,timeout=120):
    try:
        r=subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=timeout)
        return r.returncode,r.stdout
    except subprocess.TimeoutExpired:
        return -2,""
def get_files(ref):
    rc,stdout=exec_cli(["datasets","files",ref,"-v","--page-size","100"],timeout=80)
    if rc!=0:return {"status":"FILES_CLI_FAILED", "rc":rc},[]
    lines=stdout.lstrip("\ufeff").splitlines()
    ix=next((i for i,line in enumerate(lines) if line.lower().startswith("name,")),None)
    if ix is None:return {"status":"NO_FILE_CSV_HEADER"},[]
    rows=list(csv.DictReader(io.StringIO("\n".join(lines[ix:]))))
    filenames=[str(r.get("name") or r.get("Name") or "") for r in rows]
    return {"status":"OK","file_count_first_page":len(filenames),
            "potential_protocol_docs":[x for x in filenames if
                re.search(r"(^|/)(readme|audit|license|manifest|provenance|metadata)",x,re.I)][:12],
            "data_files":[x for x in filenames if x.lower().endswith((".parquet",".csv",".jsonl"))][:20]},rows
def norm(v):
    s=str(v if v is not None else "").strip().lower()
    return s.rsplit("/",1)[-1]
def read_publisher_sample(ref,filename,root):
    folder=root/("source_"+str(next(i for i,(a,_) in enumerate(DATA) if a==ref)))
    folder.mkdir(exist_ok=True)
    rc,_=exec_cli(["datasets","download","-d",ref,"-f",filename,"-p",str(folder),
                   "--unzip","-o","-q"],timeout=150)
    p=folder/filename
    if rc!=0 or not p.is_file():
        return {"status":"DOWNLOAD_FAIL","rc":rc}
    size=p.stat().st_size
    if size>SIZE_LIMIT_BYTES:
        return {"status":"TOO_LARGE_NO_READ","bytes":size}
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        df=pd.read_csv(p, nrows=MAX_RECORDS+1,low_memory=False)
    except Exception as e:
        return {"status":"CSV_READ_FAIL","exception_type":type(e).__name__}
    if len(df)>MAX_RECORDS:
        return {"status":"PARTIAL_TOO_MANY_ROWS","first_rows":MAX_RECORDS,
                "bytes":size,"sha256":sha}
    actual_cols=list(df.columns)
    selected={k:next((name for name in options if name in actual_cols),None)
              for k,options in FIELDS.items()}
    result={"status":"STRUCTURAL_ONLY_NOT_INDEPENDENT_VALIDATION",
            "bytes":size,"sha256":sha,"rows":len(df),"columns":actual_cols[:45],
            "selected_fields":selected,"source_publisher_claim_not_verified":True}
    label=selected["label"];served=selected["served"];prompt=selected["prompt"]
    if label:
        y=df[label].fillna("").astype(str).str.strip().str.lower()
        hist=y.value_counts()
        result["label_count"]=int(y.nunique())
        result["official_class_support"]=int(sum(c in hist for c in KNOWN))
        result["official_class_counts"]={c:int(hist.get(c,0)) for c in KNOWN}
        result["out_of_official_label_rows"]=int((~y.isin(KNOWN)).sum())
    else:
        result["label_count"]=None
    if label and served:
        a=df[served].fillna("").astype(str).str.strip().str.lower()
        ym=df[label].fillna("").astype(str).str.strip().str.lower()
        exact=a.eq(ym)&a.ne("")
        suffix=a.map(norm).eq(ym)&a.ne("")
        result["served_id_field_present"]=True
        result["served_nonblank"]=int(a.ne("").sum())
        result["served_exact_label_string_matches"]=int(exact.sum())
        result["served_provider_namespace_stripped_matches"]=int(suffix.sum())
        result["official_classes_with_at_least_one_suffix_match"]=[
            c for c in KNOWN if bool(((ym==c)&suffix).any())]
        # Even publisher-issued served_model metadata is not independently attested.
        result["provider_response_log_authentication"]=False
    else:
        result["served_id_field_present"]=False
        result["provider_response_log_authentication"]=False
    if prompt:
        values=df[prompt].fillna("").astype(str)
        result["prompt_field_nonempty"]=int(values.str.strip().ne("").sum())
        result["distinct_raw_prompt_or_group_keys"]=int(values.nunique())
        if label:
            blocks=df[[prompt,label]].groupby(prompt,dropna=False)[label].nunique()
            result["apparent_full18_blocks"]=int((blocks>=18).sum())
            result["prompt_blocks_not_proven_independent"]=True
    result["strong_0715_prefit_upstream_overlap_tested"]=False
    result["fresh_holdout_admitted"]=False
    return result
def notebook_observability(root):
    dest=root/"scorer"
    dest.mkdir(exist_ok=True)
    rc,_=exec_cli(["kernels","pull",SCORER,"-p",str(dest),"-m"],timeout=180)
    if rc:return {"status":"SCORER_SOURCE_DOWNLOAD_FAILED","rc":rc}
    meta=dest/"kernel-metadata.json"
    if not meta.exists():return {"status":"NO_SCORER_METADATA"}
    md=json.loads(meta.read_text())
    f=dest/md["code_file"]
    if not f.is_file():return {"status":"NO_CODE_FILE"}
    notebook=json.loads(f.read_text())
    cells=["".join(x.get("source") or []) if isinstance(x.get("source"),list)
           else str(x.get("source") or "") for x in notebook.get("cells",[])
           if x.get("cell_type")=="code"]
    code="\n".join(cells)
    sha=hashlib.sha256(code.encode()).hexdigest()
    if sha != CODE_SHA:
        return {"status":"SCORER_SOURCE_SHA_CHANGED_STOP","actual":sha,
                "expected":CODE_SHA}
    paths=sorted(set(re.findall(r"(/kaggle/(?:temp|working)/[a-zA-Z0-9_.-]+)",code)))
    signals=[
        "final-logits.npz","submission.parquet","emit_known_preview",
        "%%writefile","np.savez","fable_return","pngy_residual",
        "v1_stage","prefix_variant_stage","gauss.py",
        "preview =","if not preview",
    ]
    checkpoints={}
    for sig in signals:
        checkpoints[sig]={"occurrences":code.count(sig),
                          "code_cell_indices":[i for i,c in enumerate(cells) if sig in c][:25]}
    output={"status":"EXACT_SCORED_SOURCE_STATIC_NOT_REAL_INFERENCE",
            "source_sha_verified":True,
            "cells":len(cells),
            "code_chars":len(code),
            "notebook_mount_counts":{
                k:len(md.get(k) or [])
                for k in ("model_sources","dataset_sources","kernel_sources")},
            "intermediate_internal_paths":paths[:30],
            "signals":checkpoints,
            "exported_private_full_18way_rows":False,
            "row_matched_evaluation_by_this_audit":False,
            "no_GPU_no_model_forward_no_submission":True}
    # Use explicit file read, don't assume there is a stage unless code references it.
    return output
def main():
    with tempfile.TemporaryDirectory(prefix="ntl-adversarial-evidence-") as tmp:
        root=Path(tmp)
        source=[]
        for ref,canonical in DATA:
            inventory,files=get_files(ref)
            names=[str(x.get("name") or x.get("Name") or "") for x in files]
            selected=canonical if canonical in names else next((
                name for name in names if name.lower().endswith(".csv") and
                ("text" in name.lower() or "corpus" in name.lower() or "block" in name.lower() or "panel" in name.lower())
            ),None)
            if not selected:
                # A one-file dataset commonly uses generic "passages.csv" rather
                # than "corpus" or "panels" in its filename. Do not skip it.
                csv_files=[name for name in names if name.lower().endswith(".csv")]
                if len(csv_files)==1:
                    selected=csv_files[0]
            record={"ref":ref,"inventory":inventory,
                    "chosen_bounded_dataset_file":selected,
                    "source_metadata_not_legal_independence_proof":True}
            if selected:
                record["data_check"]=read_publisher_sample(ref,selected,root)
            source.append(record)
        emit("NTL_ADVERSARIAL_SOURCE_EVIDENCE",
             {"sources":source,"data_admitted":False,
              "selection_frozen_before_inspecting_labels":True})
        emit("NTL_ADVERSARIAL_SCORER_OBSERVABILITY",notebook_observability(root))
        emit("NTL_ADVERSARIAL_BOUNDARIES",
             {"no_new_training":True,"no_GPU":True,
              "no_model_forward":True,"no_paid_APIs":True,
              "no_competition_submission":True,
              "no_raw_text_exported":True})
def smoke():
    assert len(KNOWN)==18 and len(set(KNOWN))==18
    assert norm("openai/GPT-5.6-luna")=="gpt-5.6-luna"
    assert norm(None)==""
    print("NTL_ADVERSARIAL_EVIDENCE_SMOKE=PASS",flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    opts=ap.parse_args()
    smoke() if opts.smoke else main()
