"""Frozen 0.679 champion: one private Kaggle fresh-source diagnostic.

No model training, no competition submission, no Kaggle inputs or responses
committed into Git. Refuses to mutate any Kaggle notebook that already exists.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import zlib
import subprocess
import sys
import time
from pathlib import Path

SOURCE_REF = "kolihhan/ntl-ozertuu-0683-guarded-no-train"
DEST_REF = "kolihhan/ntl-0679-fresh-transfer-diagnostic"
SOURCE_CODE_SHA256 = "28fd128b745f0d95b70f79ea9b2c546d981edf9be08207cbf71a2d9e80ee1c86"
CONTRACT = Path("artifacts/research/ntl-0679-transfer-diagnostic/CONTRACT.md")
ADDITIONAL_SOURCES = (
    "nimithashreea/ntl-proxy-candidate-generations-20261110",
    "toanplam/ntl-fresh-generation-pilot-nov08",
)
DATA_PATH = "/kaggle/temp/ntl_external_query.csv"
TRUTH_PATH = "/kaggle/temp/ntl_external_truth.csv"
GATES = (
    "gauss.fit = _no_new_supervised_gaussian_fit",
    "NO_NEW_TRAINING: Atlas train mode forbidden",
    "NO_NEW_TRAINING: core worker train_final forbidden",
)

PREPARE_INPUTS = r'''
# Source-disjoint transfer DIAGNOSTIC only. No ground truth in model inputs.
import json as _ej
import hashlib as _eh
from pathlib import Path as _ep
import pandas as _epd

_NTL_LABELS = set([
    'claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
    'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.8-flash',
    'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra'])
def _ntl_public_attached_file(owner, slug, filename):
    # Exact public dataset bytes are in a PRIVATE Notebook source capsule.
    # Keep the champion's original 20 data sources fully unchanged.
    path = _ep('/kaggle/temp/ntl_external_capsule')/filename
    if not path.is_file():
        raise RuntimeError('NTL_EXTERNAL_PRIVATE_CAPSULE_MISSING:'+filename)
    return path

_NTL_NIM = _epd.read_csv(_ntl_public_attached_file(
    'nimithashreea', 'ntl-proxy-candidate-generations-20261110',
    'generated_proxy_candidates.csv'))
_NTL_TOAN = _epd.read_json(_ntl_public_attached_file(
    'toanplam', 'ntl-fresh-generation-pilot-nov08', 'passages.jsonl'),
    lines=True)
for _name,_frame,_rows,_prompts in (
    ('nimitha',_NTL_NIM,240,20),('toan',_NTL_TOAN,107,9)):
    if not set(['model','text','prompt_id']).issubset(_frame.columns):
        raise RuntimeError('NTL_EXTERNAL_BAD_SCHEMA:'+_name)
    if len(_frame)!=_rows or _frame['prompt_id'].nunique()!=_prompts:
        raise RuntimeError('NTL_EXTERNAL_COUNT_DRIFT:'+_name)
    if set(_frame['model'])!=_NTL_LABELS:
        raise RuntimeError('NTL_EXTERNAL_LABEL_COVERAGE_DRIFT:'+_name)
    if _frame['text'].isna().any() or _frame['model'].isna().any():
        raise RuntimeError('NTL_EXTERNAL_MISSING_CONTENT:'+_name)
    if _frame['text'].astype(str).str.strip().eq('').any():
        raise RuntimeError('NTL_EXTERNAL_EMPTY_TEXT:'+_name)
    counts=_frame.groupby('model',sort=True).size().to_dict()
    if (_name=='nimitha' and set(counts.values())!={20}) or (
        _name=='toan' and sorted(counts.values())!=[8]+[9]*11):
        raise RuntimeError('NTL_EXTERNAL_CLASS_COUNT_DRIFT:'+_name)

_NTL_PAIRS=[]
for _name,_frame in (('nimitha',_NTL_NIM),('toan',_NTL_TOAN)):
    for _i,(_index,_r) in enumerate(_frame.iterrows()):
        _NTL_PAIRS.append({'id':_name[0]+f'{_i:05d}',
                          'source':_name,'prompt_id':str(_r['prompt_id']),
                          'model':str(_r['model']),
                          'text':str(_r['text'])})
_NTL_FULL=_epd.DataFrame(_NTL_PAIRS).sample(frac=1,
    random_state=20261111).reset_index(drop=True)
if len(_NTL_FULL)!=347 or not _NTL_FULL.id.is_unique:
    raise RuntimeError('NTL_EXTERNAL_ROW_ID_DRIFT')

def _ntl_text_hash(s):
    return _eh.sha256(' '.join(str(s).split()).casefold().encode('utf-8')).hexdigest()
_NTL_EXT_HASHES=_NTL_FULL.text.map(_ntl_text_hash)
if not _NTL_EXT_HASHES.is_unique:
    raise RuntimeError('NTL_EXTERNAL_DUPLICATE_TEXT')
_NTL_OFFICIAL_TRAIN=_ep(_ntl_fix(
    '/kaggle/input/competitions/name-that-llm/train.csv'))
if not _NTL_OFFICIAL_TRAIN.is_file():
    raise RuntimeError('NTL_TRAIN_HASH_SCREEN_UNAVAILABLE')
_NTL_TRAIN_TEXT=_epd.read_csv(_NTL_OFFICIAL_TRAIN,
    usecols=['text'])['text'].dropna()
_NTL_TRAIN_HASHES=set(_NTL_TRAIN_TEXT.map(_ntl_text_hash).values)
_NTL_OVERLAP=len(set(_NTL_EXT_HASHES).intersection(_NTL_TRAIN_HASHES))
if _NTL_OVERLAP:
    raise RuntimeError('NTL_EXTERNAL_OFFICIAL_TRAIN_EXACT_OVERLAP')
_NTL_QUERY=_ep('/kaggle/temp/ntl_external_query.csv')
_NTL_TRUTH=_ep('/kaggle/temp/ntl_external_truth.csv')
_NTL_QUERY.parent.mkdir(parents=True,exist_ok=True)
_NTL_FULL[['id','text']].to_csv(_NTL_QUERY,index=False)
_NTL_FULL[['id','source','model','prompt_id']].to_csv(_NTL_TRUTH,index=False)
_NTL_WORD_COUNTS=_NTL_FULL.text.str.split().str.len()
print('NTL_EXTERNAL_PREPARED',_ej.dumps({
    'rows':len(_NTL_FULL),'source_counts':_NTL_FULL.source.value_counts().to_dict(),
    'known_target_label_count':len(_NTL_LABELS),
    'official_training_exact_text_overlap':_NTL_OVERLAP,
    'words_under_40':int((_NTL_WORD_COUNTS<40).sum()),
    'words_over_250':int((_NTL_WORD_COUNTS>250).sum()),
    'model_input_has_labels':False,
    'deterministic_shuffle_seed':20261111
},sort_keys=True),flush=True)
del _NTL_FULL,_NTL_NIM,_NTL_TOAN,_NTL_PAIRS,_NTL_TRAIN_TEXT
'''

SCORE_DIAGNOSTIC = r'''
if preview:
    raise RuntimeError('NTL_EXTERNAL_FULL_MODEL_NOT_EXECUTED')
import json as _dsj
from collections import Counter as _Counter
import pandas as _dsp
import numpy as _dsn
_truth=_dsp.read_csv('/kaggle/temp/ntl_external_truth.csv',dtype={
    'id':str,'source':str,'model':str,'prompt_id':str})
_pred=_dsp.read_parquet(OUTPUT)
if len(_truth)!=347 or len(_pred)!=347 or not _pred.id.is_unique:
    raise RuntimeError('NTL_EXTERNAL_BAD_OUTPUT_ROW_COUNT')
if list(_pred.columns)!=['id','model']:
    raise RuntimeError('NTL_EXTERNAL_OUTPUT_SCHEMA_DRIFT')
_combined=_truth.merge(_pred,on='id',how='left',
    validate='one_to_one',suffixes=('_expected','_predicted'))
if len(_combined)!=347 or _combined.model_predicted.isna().any():
    raise RuntimeError('NTL_EXTERNAL_OUTPUT_ID_MISMATCH')
_target=set(_combined.model_expected)
if len(_target)!=12:
    raise RuntimeError('NTL_EXTERNAL_EXPECTED_LABEL_SET_DRIFT')
_known_18={'claude-fable-5','claude-opus-4-6','claude-opus-4-8',
'claude-opus-5','claude-sonnet-5','gemini-3.1-pro-preview',
'gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3',
'glm-5.3-flash','gpt-5.5','gpt-5.6-luna','gpt-5.6-sol',
'gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max'}
if not set(_combined.model_predicted).issubset(_known_18):
    raise RuntimeError('NTL_EXTERNAL_PREDICTED_UNRECOGNIZED_LABEL')
def _family(label):
    return str(label).split('-')[0]
def _stats(frame):
    _n=len(frame)
    _classes=sorted(set(frame.model_expected))
    _recalls={label:round(float((frame.loc[
        frame.model_expected==label,'model_predicted']==label).mean()),6)
        for label in _classes}
    _right=frame.model_expected==frame.model_predicted
    _wrong=frame.loc[~_right]
    _pairs=_Counter(zip(_wrong.model_expected,_wrong.model_predicted))
    _families={}
    for provider in sorted({_family(x) for x in _classes}):
        sl=frame[frame.model_expected.map(_family)==provider]
        _families[provider]={'n':int(len(sl)), 'accuracy':round(
            float((sl.model_expected==sl.model_predicted).mean()),6)}
    return {
        'n':_n,'observed_true_labels':len(_classes),
        'conditional_12_target_balanced_accuracy_with_18_way_predictions':
            round(sum(_recalls.values())/len(_classes),6),
        'overall_accuracy':round(float(_right.mean()),6),
        'per_label_recall':_recalls,
        'family_slice_accuracy':_families,
        'same_family_wrong':int(sum(_family(a)==_family(b)
                                     for a,b in zip(_wrong.model_expected,
                                                    _wrong.model_predicted))),
        'cross_family_wrong':int(sum(_family(a)!=_family(b)
                                     for a,b in zip(_wrong.model_expected,
                                                    _wrong.model_predicted))),
        'predictions_outside_12_observed_targets':
            int((~frame.model_predicted.isin(_classes)).sum()),
        'top_wrong_pairs':[{'expected':a,'predicted':b,'n':n}
                           for (a,b),n in _pairs.most_common(12)]}
_result={'contract':'frozen-private-0679-external-v1',
         'score_type':'diagnostic 12 observed classes; NEVER 18-class Kaggle BA',
         'official_competition_submission':False,
         'supervised_training':False,'n_input':len(_combined),
         'sources':{k:_stats(_combined[_combined.source==k])
                    for k in ('toan','nimitha')},
         'combined_exploratory':_stats(_combined)}
_out=_ep('/kaggle/working/ntl_0679_transfer_summary.json')
_out.write_text(_dsj.dumps(_result,sort_keys=True,indent=2))
_combined[['id','source','prompt_id','model_expected',
           'model_predicted']].to_parquet(
               '/kaggle/working/ntl_0679_transfer_predictions.parquet',
               index=False)
print('NTL_TRANSFER_FINAL_AGGREGATES',
      _dsj.dumps(_result,sort_keys=True),flush=True)
'''


def _code(cell: dict) -> str:
    v=cell.get("source",[])
    return "".join(v) if isinstance(v,list) else str(v)


def _new_cell(source: str) -> dict:
    return {"cell_type":"code","metadata":{},"execution_count":None,
            "outputs":[],"source":source.strip().splitlines(keepends=True)}


def _run(args: list[str],*, timeout: int=120, check: bool=True) -> subprocess.CompletedProcess:
    p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if check and p.returncode:
        # Do not publish provider errors, which could expose auth tokens.
        raw=(p.stdout+"\n"+p.stderr)
        safe=[]
        for line in raw.splitlines():
            if re.search(r"(?i)error|failed|invalid|400|403|404|429|quota|dataset|permission|not found|limit|source",line):
                line=re.sub(r"(?i)(?:authorization|token|api[_ -]?key|secret|password)\s*[:=]\s*\S+",
                            "[redacted-credential]",line)
                line=re.sub(r"[A-Za-z0-9_=-]{46,}","[redacted-long-value]",line)
                safe.append(line.strip()[:300])
        print("NTL_TRANSFER_DEPLOY_FAILURE",json.dumps({
            "op":args[:3],"rc":p.returncode,"error_lines":safe[-8:],
            "http_400":"400" in raw,"http_403":"403" in raw,
            "http_429":"429" in raw,"quota":"quota" in raw.lower(),
            "dataset":"dataset" in raw.lower(),
        },ensure_ascii=False),flush=True)
        raise RuntimeError(f"private Kaggle operation returned code={p.returncode}; opcode={args[:3]}")
    return p


def _get_code_sha(nb: dict) -> str:
    cells=[_code(c) for c in nb.get("cells",[]) if c.get("cell_type")=="code"]
    return hashlib.sha256(json.dumps(cells,ensure_ascii=False,
           separators=(",",":")).encode("utf-8")).hexdigest()



def _make_private_capsule(destination: Path) -> str:
    """Download CC-licensed source files on ephemeral runner, embed bytes only
    in PRIVATE Kaggle Notebook; never commit or log raw public passages."""
    entries=(
        (ADDITIONAL_SOURCES[0],"generated_proxy_candidates.csv",180000,380000),
        (ADDITIONAL_SOURCES[1],"passages.jsonl",140000,360000),
    )
    source_lines=[
        "import base64 as _cb64, zlib as _cz, hashlib as _ch",
        "from pathlib import Path as _cp",
        "_CAP=_cp('/kaggle/temp/ntl_external_capsule')",
        "_CAP.mkdir(parents=True,exist_ok=False)"
    ]
    for i,(ref,filename,minimum,maximum) in enumerate(entries):
        folder=destination/("cohort_"+str(i))
        folder.mkdir(parents=True,exist_ok=False)
        _run(["kaggle","datasets","download",ref,"-f",filename,
              "-p",str(folder),"--unzip","-q"],timeout=110)
        files=list(folder.rglob(filename))
        if len(files)!=1:
            raise RuntimeError("external source file missing/ambiguous: "+filename)
        original=files[0].read_bytes()
        if not minimum<=len(original)<=maximum:
            raise RuntimeError("source corpus size outside frozen envelope")
        digest=hashlib.sha256(original).hexdigest()
        compressed=base64.b64encode(zlib.compress(original,9)).decode("ascii")
        source_lines.extend([
            "_bytes=_cz.decompress(_cb64.b64decode("+repr(compressed)+"))",
            "if _ch.sha256(_bytes).hexdigest()!="+repr(digest)+
            ":raise RuntimeError('EXTERNAL_SOURCE_CAPSULE_CORRUPTION')",
            "(_CAP/"+repr(filename)+").write_bytes(_bytes)",
            "del _bytes"
        ])
        print("NTL_TRANSFER_PUBLIC_SOURCE_CAPSULE_MANIFEST",json.dumps({
            "ref":ref,"filename":filename,"bytes":len(original),
            "sha256":digest,"compressed_base64_chars":len(compressed),
            "ephemeral_runner_only":True,"github_commit_data":False
        }),flush=True)
    return "\n".join(source_lines)+"\n"

def _clone(destination: Path)->dict:
    if not CONTRACT.is_file():
        raise RuntimeError("predeclared transfer contract missing")
    src=destination/"source"
    dst=destination/"kernel"
    src.mkdir(parents=True,exist_ok=False)
    dst.mkdir(parents=True,exist_ok=False)
    _run(["kaggle","kernels","pull",SOURCE_REF,"-p",str(src),"-m"],timeout=120)
    meta_file=src/"kernel-metadata.json"
    meta=json.loads(meta_file.read_text())
    if (meta.get("id")!=SOURCE_REF or meta.get("is_private") is not True or
        meta.get("enable_internet") is not False or
        meta.get("competition_sources")!=["name-that-llm"] or
        len(meta.get("dataset_sources") or [])!=20 or
        len(meta.get("kernel_sources") or [])!=8 or
        len(meta.get("model_sources") or [])!=6):
        raise RuntimeError("champion Kaggle metadata changed")
    for src_name in ADDITIONAL_SOURCES:
        if src_name in (meta.get("dataset_sources") or []):
            raise RuntimeError("new source was already an input to 0.679 champion")
    files=list(src.glob("*.ipynb"))
    if len(files)!=1:
        raise RuntimeError("expected one frozen private Notebook")
    nb=json.loads(files[0].read_text(encoding="utf-8"))
    digest=_get_code_sha(nb)
    if digest!=SOURCE_CODE_SHA256:
        raise RuntimeError("0.679 champion source code digest drift")
    code=[_code(c) for c in nb.get("cells",[]) if c.get("cell_type")=="code"]
    if any(sum(marker in c for c in code)!=1 for marker in GATES):
        raise RuntimeError("0.679 source not protected by three supervised-training guards")
    cells=nb["cells"]
    if (len(cells)<53 or cells[5]["cell_type"]!="code" or
        cells[7]["cell_type"]!="code" or cells[52]["cell_type"]!="code"):
        raise RuntimeError("0.679 Notebook cell lineage has drifted")
    idx5=_code(cells[5])
    old_input="DATA = Path(_ntl_fix('/kaggle/input/competitions/name-that-llm/test.csv'))"
    if idx5.count(old_input)!=1:
        raise RuntimeError("champion DATA source path changed")
    cells[5]["source"]=idx5.replace(old_input,
                   "DATA = Path('/kaggle/temp/ntl_external_query.csv')").splitlines(keepends=True)
    idx7=_code(cells[7])
    old_audit="_ntl_audit(CONFIG, EXPLICIT_PATHS)\n"
    if idx7.count(old_audit)!=1:
        raise RuntimeError("champion original input audit changed")
    idx7=idx7.replace(old_audit,old_audit+
          "CONFIG['paths']['test_csv'] = '/kaggle/temp/ntl_external_query.csv'\n",1)
    cells[7]["source"]=idx7.splitlines(keepends=True)
    idx52=_code(cells[52])
    if "if not preview:" not in idx52 or "subprocess.run(" not in idx52:
        raise RuntimeError("champion final inference entrypoint changed")
    # Appending diagnostics after the existing original full model inference.
    cells[52]["source"]=(idx52.rstrip()+"\n"+SCORE_DIAGNOSTIC.strip()+"\n").splitlines(keepends=True)
    # Insert before original DATA/preview assignment, preserving old logic.
    capsule_source=_make_private_capsule(destination)
    cells.insert(5,_new_cell(capsule_source))
    cells.insert(6,_new_cell(PREPARE_INPUTS))
    cells.insert(0,{"cell_type":"markdown","metadata":{},
       "source":["**Private source-transfer diagnostic only.** Frozen scored 0.679 "
          "champion, exact 18-class outputs, two public 12-label external cohorts; "
          "no supervised training, no Kaggle submission. Original upstream "
          "attributions preserved."]})
    for c in cells:
        if c.get("cell_type")=="code":
            c["execution_count"]=None
            c["outputs"]=[]
    meta.pop("id_no",None)
    meta["id"]=DEST_REF
    meta["title"]="NTL Frozen 0.679 Cross-Source Diagnostic (NO SUBMISSION)"
    meta["is_private"]=True
    meta["enable_internet"]=False
    # Preserve exactly the champion original 20 dataset sources.
    assert len(meta["dataset_sources"])==20
    code_file=meta.get("code_file")
    if not code_file or Path(code_file).name!=code_file or not code_file.endswith(".ipynb"):
        raise RuntimeError("invalid Notebook path")
    dest_file=dst/code_file
    dest_file.write_text(json.dumps(nb,ensure_ascii=False))
    (dst/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
    new_code=[_code(c) for c in cells if c.get("cell_type")=="code"]
    if any(sum(g in c for c in new_code)!=1 for g in GATES):
        raise RuntimeError("training guards drifted in diagnostic copy")
    if sum("NTL_TRANSFER_FINAL_AGGREGATES" in c for c in new_code)!=1:
        raise RuntimeError("missing diagnostic output")
    if sum("DATA = Path('/kaggle/temp/ntl_external_query.csv')" in c for c in new_code)!=1:
        raise RuntimeError("external query path not uniquely configured")
    if sum("CONFIG['paths']['test_csv'] = '/kaggle/temp/ntl_external_query.csv'" in c for c in new_code)!=1:
        raise RuntimeError("runtime path not uniquely configured")
    print("NTL_TRANSFER_SOURCE_CODE_SHA",digest,flush=True)
    print("NTL_TRANSFER_DIAGNOSTIC_NOTEBOOK_SHA",hashlib.sha256(dest_file.read_bytes()).hexdigest(),flush=True)
    print("NTL_TRANSFER_EXTERNAL_INPUT_REFS",json.dumps(ADDITIONAL_SOURCES),flush=True)
    print("NTL_TRANSFER_STATIC_GUARDS_PASS",flush=True)
    return {"folder":dst,"sha":hashlib.sha256(dest_file.read_bytes()).hexdigest()}


def _status()->str:
    p=_run(["kaggle","kernels","status",DEST_REF],timeout=50,check=False)
    raw=p.stdout+"\n"+p.stderr
    m=re.search(r"KernelWorkerStatus\.([A-Z_]+)",raw)
    return m.group(1) if m else ("NOT_FOUND" if p.returncode else "UNKNOWN")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true",
                        help="Build and verify private diagnostic source; no GPU")
    parser.add_argument("--run",action="store_true",
                        help="Push exactly one private Notebook; never submit competition")
    args=parser.parse_args()
    if args.preflight==args.run:
        raise RuntimeError("choose exactly one preflight or run mode")
    if args.run and _status()!="NOT_FOUND":
        raise RuntimeError("diagnostic Notebook exists or status unknown; no overwrite")
    work=Path("/tmp/ntl-0679-transfer-one-shot")
    if work.exists():
        raise RuntimeError("one-shot working directory already exists")
    built=_clone(work)
    if args.preflight:
        print("NTL_TRANSFER_PREFLIGHT_DONE_WITHOUT_GPU_OR_SUBMISSION",flush=True)
        return
    result=_run(["kaggle","kernels","push","-p",str(built["folder"]),
                 "--accelerator","NvidiaTeslaT4","--timeout","32400"],timeout=180)
    # Accept only known public-safe status lines; no token/error strings.
    lines=[x for x in (result.stdout+"\n"+result.stderr).splitlines()
           if "Kernel version" in x and "successfully pushed" in x]
    if not lines:
        raise RuntimeError("private diagnostic Kaggle push not confirmed")
    print("NTL_TRANSFER_PRIVATE_NOTEBOOK_PUSHED",json.dumps({
        "ref":DEST_REF,"messages":lines[:2],"competition_submit":False}),flush=True)
    for i in range(1,151):
        time.sleep(120)
        status=_status()
        print("NTL_TRANSFER_PRIVATE_KERNEL_STATUS",i,status,flush=True)
        if status in ("ERROR","FAILED","CANCELLED"):
            print("NTL_TRANSFER_MODEL_FAILED_CLOSED",status,flush=True)
            return
        if status=="COMPLETE":
            files=_run(["kaggle","kernels","files",DEST_REF,"-v",
                        "--page-size","200"],timeout=120)
            names=files.stdout
            ok=("ntl_0679_transfer_summary.json" in names and
                "ntl_0679_transfer_predictions.parquet" in names)
            print("NTL_TRANSFER_PRIVATE_DIAGNOSTIC_OUTPUT_READY",ok,flush=True)
            if not ok:raise RuntimeError("Notebook completed but no diagnostic outputs")
            break
    else:
        print("NTL_TRANSFER_POLL_WINDOW_EXCEEDED_RECHECK_LATER",flush=True)
        return
    # Download only our *aggregate JSON*, never raw public passages, source
    # data or per-row records to the GitHub runner.
    dest=Path("/tmp/ntl-0679-transfer-aggregate")
    dest.mkdir(exist_ok=False)
    _run(["kaggle","kernels","output",DEST_REF,
          "-p",str(dest),"--file-pattern","ntl_0679_transfer_summary.json"],
         timeout=90)
    paths=list(dest.rglob("ntl_0679_transfer_summary.json"))
    if len(paths)!=1:
        raise RuntimeError("private aggregate-only result missing")
    summary=json.loads(paths[0].read_text())
    if summary.get("n_input")!=347 or set(summary.get("sources",{}))!={"toan","nimitha"}:
        raise RuntimeError("aggregate diagnostic schema drift")
    print("NTL_TRANSFER_AGGREGATE_RESULT",json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
