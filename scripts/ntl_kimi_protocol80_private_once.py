"""One frozen private Kaggle inference of Kimi K3 reasoning high/off pairs.

Reuses EXACT 0.679 champion pipeline and no-training guards.
No competition submission, no fitted model, no public examples.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,re,subprocess,time,zlib
from pathlib import Path

from scripts import ntl_0679_transfer_private_once as base
from scripts.ntl_kimi_protocol80_source import read_pairs,SOURCE_REF,FILE

DEST_REF="kolihhan/ntl-kimi-k3-protocol80-frozen-0679"
CONTRACT=Path("artifacts/research/ntl-kimi-protocol80/CONTRACT.md")

PREPARE=r'''
# Matched high/off Kimi diagnostic. Only id,text enters frozen 18-way model.
import hashlib as _h, json as _j
from pathlib import Path as _P
import pandas as _pd
_records=[_j.loads(x) for x in _P(
    "/kaggle/temp/ntl_external_capsule/ntl_kimi_protocol80_selected.jsonl").read_text().splitlines()
    if x.strip()]
_df=_pd.DataFrame(_records)
if len(_df)!=160 or not _df["id"].is_unique:
    raise RuntimeError("NTL_KIMI_ROW_COUNT_DRIFT")
if set(_df["model"])!={"kimi-k3"} or set(_df["setting"])!={"high","off"}:
    raise RuntimeError("NTL_KIMI_TRUE_MODEL_OR_SETTING_DRIFT")
if _df["prompt_id"].nunique()!=80:
    raise RuntimeError("NTL_KIMI_DISTINCT_PROMPT_DRIFT")
for _pid,_g in _df.groupby("prompt_id"):
    if (len(_g)!=2 or set(_g["setting"])!={"high","off"} or
        _g["served_model"].nunique()!=1 or
        _g["served_provider"].nunique()!=1 or
        _g["model_id"].nunique()!=1 or _g["target_words"].nunique()!=1):
        raise RuntimeError("NTL_KIMI_PAIR_ATTRIBUTION_GUARD")
def _text_sha(s):return _h.sha256(" ".join(str(s).split()).casefold().encode("utf-8")).hexdigest()
_hh=_df["text"].map(_text_sha)
if not _hh.is_unique:raise RuntimeError("NTL_KIMI_DUPLICATE_TEXT")
_train=_P(_ntl_fix("/kaggle/input/competitions/name-that-llm/train.csv"))
if not _train.is_file():raise RuntimeError("NTL_KIMI_OFFICIAL_TRAIN_OVERLAP_SCREEN_MISSING")
_trainhash=set(_pd.read_csv(_train,usecols=["text"])["text"].dropna().map(_text_sha))
if set(_hh).intersection(_trainhash):
    raise RuntimeError("NTL_KIMI_PUBLIC_TEXT_EXACT_OFFICIAL_TRAIN_OVERLAP")
query=_P("/kaggle/temp/ntl_external_query.csv")
truth=_P("/kaggle/temp/ntl_external_truth.csv")
query.parent.mkdir(exist_ok=True,parents=True)
_df[["id","text"]].to_csv(query,index=False)
_df[["id","setting","model","prompt_id","n_words","target_words"]].to_csv(truth,index=False)
print("NTL_KIMI_PRIVATE_INPUT_READY",_j.dumps({
    "n_rows":160,"pairs":80,"true_model":"kimi-k3",
    "settings":{"high":80,"off":80},
    "exact_official_train_text_overlap":0,
    "labels_present_in_model_input":False,
    "training":False
},sort_keys=True),flush=True)
del _records,_df,_hh,_trainhash
'''

SCORE=r'''
# Frozen exact 18-way incumbent predictions, paired by source prompt ID.
if preview:raise RuntimeError("NTL_KIMI_NO_PREVIEW_SHORTCUT_ALLOWED")
import json as _sj,math as _sm
from collections import Counter as _C
import pandas as _sp
_truth=_sp.read_csv("/kaggle/temp/ntl_external_truth.csv",dtype={"id":str,"prompt_id":str})
_pred=_sp.read_parquet(OUTPUT)
if len(_pred)!=160 or len(_truth)!=160:
    raise RuntimeError("NTL_KIMI_INCOMPLETE_MODEL_INFERENCE")
if list(_pred.columns)!=["id","model"] or not _pred.id.is_unique:
    raise RuntimeError("NTL_KIMI_PREDICTION_SCHEMA_DRIFT")
_joined=_truth.merge(_pred,on="id",validate="one_to_one",
    suffixes=("_expected","_predicted"))
if len(_joined)!=160 or _joined.model_predicted.isna().any():
    raise RuntimeError("NTL_KIMI_PREDICTION_ALIGNMENT_DRIFT")
if set(_joined.model_expected)!={"kimi-k3"}:
    raise RuntimeError("NTL_KIMI_GROUND_TRUTH_IDENTITY_DRIFT")
pairs=[]
for prompt,g in _joined.groupby("prompt_id"):
    if len(g)!=2 or set(g.setting)!={"high","off"}:
        raise RuntimeError("NTL_KIMI_PREDICTION_PAIR_MISSING")
    h=g[g.setting=="high"].iloc[0]
    o=g[g.setting=="off"].iloc[0]
    pairs.append({"prompt_id":str(prompt),"high":str(h.model_predicted),
                  "off":str(o.model_predicted),"high_words":int(h.n_words),
                  "off_words":int(o.n_words)})
pairs.sort(key=lambda x:x["prompt_id"])
if len(pairs)!=80:raise RuntimeError("NTL_KIMI_PAIRED_ROWS_NOT_80")
bh=sum(x["high"]=="kimi-k3" and x["off"]!="kimi-k3" for x in pairs)
bo=sum(x["off"]=="kimi-k3" and x["high"]!="kimi-k3" for x in pairs)
both=sum(x["off"]=="kimi-k3" and x["high"]=="kimi-k3" for x in pairs)
neither=sum(x["off"]!="kimi-k3" and x["high"]!="kimi-k3" for x in pairs)
n=bh+bo
if n:
    tail=sum(_sm.comb(n,i) for i in range(min(bh,bo)+1))/2**n
    p=min(1.0,2*tail)
else:p=1.0
blocks=[]
for bi in range(4):
    seg=pairs[20*bi:20*(bi+1)]
    h=sum(x["high"]=="kimi-k3" for x in seg)
    o=sum(x["off"]=="kimi-k3" for x in seg)
    blocks.append({"block":bi+1,"n":20,"high_correct":h,
                   "off_correct":o,"delta_pp":round(100*(h-o)/20,2)})
dh=_C(x["high"] for x in pairs)
do=_C(x["off"] for x in pairs)
report={
    "scope":"Kimi K3 strict within-prompt high/off protocol sensitivity, not Kaggle 18-class BA",
    "champion_official_public_ba":0.679,
    "champion_official_ref":"58210835",
    "true_model":"kimi-k3","n_pairs":80,"n_predictions":160,
    "high_correct":bh+both,"off_correct":bo+both,
    "high_recall":round((bh+both)/80,6),
    "off_recall":round((bo+both)/80,6),
    "high_minus_off_recall_pp":round((bh-bo)/80*100,2),
    "mcnemar_high_only":bh,"mcnemar_off_only":bo,
    "both_correct":both,"both_wrong":neither,
    "mcnemar_exact_two_sided_p":round(p,9),
    "prediction_changed_within_pair":sum(x["high"]!=x["off"] for x in pairs),
    "blocks_by_sorted_prompt":blocks,
    "high_predictions":dict(dh.most_common()),
    "off_predictions":dict(do.most_common()),
    "mean_high_minus_off_words":round(
        sum(x["high_words"]-x["off_words"] for x in pairs)/80,3),
    "evidence_criterion_met":(
         abs((bh-bo)/80)>=0.15 and p<0.01 and all(
             (b["high_correct"]-b["off_correct"])*(bh-bo)>0 for b in blocks)),
    "supervised_training":False,
    "competition_submission":False,
    "causal_effect_proven":False,
}
output=_P("/kaggle/working/ntl_kimi_protocol80_summary.json")
output.write_text(_sj.dumps(report,sort_keys=True,indent=2))
_joined[["id","prompt_id","setting","model_expected",
         "model_predicted"]].to_parquet(
    "/kaggle/working/ntl_kimi_protocol80_predictions.parquet",index=False)
print("NTL_TRANSFER_FINAL_AGGREGATES",_sj.dumps(report,sort_keys=True),flush=True)
'''

def _private_capsule(folder:Path)->str:
    source=folder/"public_source"
    source.mkdir(parents=True,exist_ok=False)
    base._run(["kaggle","datasets","download",SOURCE_REF,
              "-f",FILE,"-p",str(source),"--unzip","-q"],timeout=100)
    paths=list(source.rglob(FILE))
    if len(paths)!=1:raise RuntimeError("expected one Kimi trials.jsonl file")
    records,meta=read_pairs(paths[0])
    payload="".join(json.dumps(x,ensure_ascii=False)+"\n" for x in records).encode("utf-8")
    digest=hashlib.sha256(payload).hexdigest()
    comp=base64.b64encode(zlib.compress(payload,9)).decode("ascii")
    # The CC BY-licensed 160 passages appear only in a private Kaggle Notebook,
    # not in this public Github source or log.
    lines=[
       "import base64 as _b, zlib as _z, hashlib as _s",
       "from pathlib import Path as _K",
       "_p=_K('/kaggle/temp/ntl_external_capsule')",
       "_p.mkdir(parents=True,exist_ok=False)",
       "_v=_z.decompress(_b.b64decode("+repr(comp)+"))",
       "if _s.sha256(_v).hexdigest()!="+repr(digest)+
        ":raise RuntimeError('PRIVATE_KIMI_CAPSULE_SHA_MISMATCH')",
       "(_p/'ntl_kimi_protocol80_selected.jsonl').write_bytes(_v)",
       "del _v"
    ]
    print("NTL_KIMI_FROZEN_SOURCE_AGGREGATE",json.dumps(meta,sort_keys=True),flush=True)
    print("NTL_KIMI_PRIVATE_CAPSULE_SHA",digest,flush=True)
    return "\n".join(lines)+"\n"

def status():
    proc=base._run(["kaggle","kernels","status",DEST_REF],
                   check=False,timeout=60)
    match=re.search(r"KernelWorkerStatus\.([A-Z_]+)",
                    proc.stdout+"\n"+proc.stderr)
    return match.group(1) if match else ("NOT_FOUND" if proc.returncode else "UNKNOWN")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preflight",action="store_true")
    ap.add_argument("--run",action="store_true")
    args=ap.parse_args()
    if args.preflight==args.run:raise RuntimeError("select preflight or run")
    if args.run and status()!="NOT_FOUND":
        raise RuntimeError("Kimi 80 private kernel exists or status ambiguous; refuse overwrite")
    base.DEST_REF=DEST_REF
    base.CONTRACT=CONTRACT
    base.PREPARE_INPUTS=PREPARE
    base.SCORE_DIAGNOSTIC=SCORE
    base._make_private_capsule=_private_capsule
    folder=Path("/tmp/ntl-kimi-protocol80-frozen")
    if folder.exists():raise RuntimeError("one-shot work dir already exists")
    built=base._clone(folder)
    if args.preflight:
        print("NTL_KIMI_PROTOCOL80_PREFLIGHT_DONE_NO_GPU_NO_SUBMISSION",flush=True)
        return
    result=base._run(["kaggle","kernels","push","-p",str(built["folder"]),
             "--accelerator","NvidiaTeslaT4","--timeout","32400"],timeout=180)
    lines=[line for line in (result.stdout+"\n"+result.stderr).splitlines()
           if "Kernel version" in line and "successfully pushed" in line]
    if len(lines)!=1:raise RuntimeError("Kimi private Notebook save unconfirmed")
    print("NTL_KIMI_PROTOCOL80_PRIVATE_PUSH_ACCEPTED",DEST_REF,flush=True)
    for n in range(1,181):
        time.sleep(30)
        state=status()
        print("NTL_KIMI_PROTOCOL80_PRIVATE_STATUS",n,state,flush=True)
        if state in ("ERROR","FAILED","CANCELLED"):
            raise RuntimeError("Kimi protocol GPU inference failed closed")
        if state=="COMPLETE":break
    else:
        print("NTL_KIMI_PROTOCOL80_POLL_EXHAUSTED_NO_NEW_SUBMISSION",flush=True)
        return
    files=base._run(["kaggle","kernels","files",DEST_REF,"-v",
                    "--page-size","200"],timeout=90)
    if "ntl_kimi_protocol80_summary.json" not in files.stdout:
        raise RuntimeError("Kimi kernel completed but output summary missing")
    dst=Path("/tmp/ntl-kimi-private-aggregate")
    dst.mkdir(exist_ok=False)
    base._run(["kaggle","kernels","output",DEST_REF,"-p",str(dst),
        "--file-pattern","ntl_kimi_protocol80_summary.json"],timeout=90)
    paths=list(dst.rglob("ntl_kimi_protocol80_summary.json"))
    if len(paths)!=1:raise RuntimeError("missing private Kimi aggregate")
    report=json.loads(paths[0].read_text())
    if report.get("n_pairs")!=80 or report.get("n_predictions")!=160:
        raise RuntimeError("Kimi final aggregate incomplete")
    print("NTL_KIMI_PROTOCOL80_FINAL_VERIFIED",json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
