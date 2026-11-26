"""One-shot no-new-training NTL Gemma4 31B replacement candidate.

Uses pinned scored public Lightning V52 0.724 as attributed Ensemble control,
replaces the 27B GSQ stage with 31B Q4 in-context top-two attribution, then
runs real 3-row 31B preview before precisely ONE official code submission.
No new fitting or model fine-tuning; other original lightweight readouts retained.
"""
import csv,hashlib,json,os,re,shutil,subprocess,tempfile,time
from pathlib import Path

SOURCE="lightningv08/0-724-glm-pb115-sibling-odds-qwen3-8-27b/52"
SOURCE_SHA="b4c9715e32d27463a399f8715684c8403f057c83f1f77e0039ff0da2427df343"
DEST="kolihhan/ntl-gemma4-31b-ensemble-no-train-20261126"
DESC="Gemma4 31B Q4 zero-finetune top2 rival replacing GSQ27B on Lightning V52 once 20261126"
BASELINE_SCORE=0.724
BASELINE_OWN=0.718
ROOT=Path("/tmp/ntl-gemma31b-one-shot-20261126")
ASSETS=("batuhankska/gemma-4-31b-it-q4-k-m","nixty2/llama-cpp-turing-sm75")
STAGE_PATH=Path("scripts/ntl_gemma4_31b_ensemble_stage_20261126.py")

def log(name,**data):
    print(name+"="+json.dumps(data,sort_keys=True,default=str),flush=True)

def run(*args,timeout=150):
    p=subprocess.run(["kaggle",*args],text=True,capture_output=True,timeout=timeout)
    return p

def must(p,what):
    if p.returncode:
        raise RuntimeError(what+" failed: "+(p.stderr+p.stdout)[-1000:])
    return p

def submissions():
    p=must(run("competitions","submissions","name-that-llm","-v"),"submission read")
    rows=list(csv.DictReader(p.stdout.splitlines()))
    if not rows:raise RuntimeError("no authentic submission ledger")
    return rows

def guard_account():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a=KaggleApi();a.authenticate()
    lim=a.competition_get_submission_limits("name-that-llm")
    if int(lim.num_allowed_now)<2:raise RuntimeError("reserve at least one daily submission")
    champion=a.get_submission(58755304).to_dict()
    if str(champion.get("publicScore"))!="0.718" or str(champion.get("status"))!="COMPLETE":
        raise RuntimeError("incumbent official score drift")
    prior=submissions()
    if any(DESC in str(z.get("description") or "") for z in prior):
        raise RuntimeError("this official one-shot was already submitted")
    active=[z for z in prior if str(z.get("status","")).upper() in ("PENDING","RUNNING","QUEUED","PROCESSING")]
    if active:raise RuntimeError("other active competition scorer, avoid overlap")
    log("NTL31_ACCOUNT_READY",allowed_now=int(lim.num_allowed_now),
        own_champion_score="0.718",active_scorers=0,one_submit=True)

def inject_cell(source):
    return {"cell_type":"code","execution_count":None,
            "metadata":{},"outputs":[],
            "source":source.splitlines(keepends=True)}

def build():
    dest=ROOT/"candidate"
    source_dir=ROOT/"pinned"
    dest.mkdir(parents=True);source_dir.mkdir()
    must(run("kernels","pull",SOURCE,"-p",str(source_dir),"-m"),"pinned v52 pull")
    meta=json.loads((source_dir/"kernel-metadata.json").read_text())
    raw=(source_dir/meta["code_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise RuntimeError("source SHA drift from verified scored 0.724")
    if meta.get("enable_internet") is not False or meta.get("competition_sources")!=["name-that-llm"]:
        raise RuntimeError("original notebook does not satisfy offline competition contract")
    nb=json.loads(raw)
    edits={"gsq":0,"deadline":0}
    for cell in nb["cells"]:
        if cell.get("cell_type")!="code":continue
        src="".join(cell.get("source",[]))
        if "'--gsq_share', '0.035'" in src:
            src2=src.replace("'--gsq_share', '0.035'","'--gsq_share', '0.0'")
            edits["gsq"]+=src.count("'--gsq_share', '0.035'")
            src=src2
        if "RUN_LIMIT_S = 7.0 * 3600" in src:
            edits["deadline"]+=src.count("RUN_LIMIT_S = 7.0 * 3600")
            src=src.replace("RUN_LIMIT_S = 7.0 * 3600","RUN_LIMIT_S = 6.25 * 3600")
        cell["source"]=src.splitlines(keepends=True)
    if edits!={"gsq":1,"deadline":1}:
        raise RuntimeError("pinned v52 intervention anchors wrong "+repr(edits))
    stage=STAGE_PATH.read_text(encoding="utf-8")
    compile(stage,"ntl31_stage.py","exec")
    writer=("from pathlib import Path\n"
            "Path('/kaggle/working/ntl31_stage.py').write_text("+repr(stage)+",encoding='utf-8')\n")
    apply_code=(
        "import subprocess,sys,json,time,hashlib,pathlib\n"
        "print('NTL31_TRUE_STAGE_ENTRY',bool(FAST_COMMIT),flush=True)\n"
        "subprocess.check_call([sys.executable,'/kaggle/working/ntl31_stage.py',"
        "str(pathlib.Path(DATA)/'test.csv'),str(pathlib.Path(DATA)/'train.csv'),"
        "str(NB_T0),'1' if FAST_COMMIT else '0'])\n"
        "p=pathlib.Path('/kaggle/working/ntl31_proof.json')\n"
        "z=json.loads(p.read_text())\n"
        "assert z['loaded_and_inferred'] and z['no_training'] and z['valid_calls']>0\n"
        "assert (z['preview'] is True)==bool(FAST_COMMIT)\n"
        "assert (z['changed']>0 or FAST_COMMIT) and z['n_test']>0\n"
        "print('NTL31_STAGE_VERIFIED',json.dumps(z,sort_keys=True),flush=True)\n")
    nb["cells"].append(inject_cell(writer))
    nb["cells"].append(inject_cell(apply_code))
    codefile=str(meta["code_file"])
    (dest/codefile).write_text(json.dumps(nb,ensure_ascii=False),encoding="utf-8")
    meta.pop("id_no",None)
    meta.update(id=DEST,title="NTL 31B Inference-Only Top2 Ensemble One Shot",
                is_private=True,enable_gpu=True,enable_internet=False,
                machine_shape="NvidiaTeslaT4")
    origins=list(meta.get("dataset_sources") or [])
    for a in ASSETS:
        if a in origins:raise RuntimeError("new intervention already mounted in original source")
        origins.append(a)
    meta["dataset_sources"]=origins
    (dest/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    notebook_digest=hashlib.sha256((dest/codefile).read_bytes()).hexdigest()
    log("NTL31_BUILD_OK",scored_public_source="0.724 v52",upstream_sha=SOURCE_SHA,
        candidate_code_sha=notebook_digest,original_model_sources=len(meta["model_sources"]),
        attached_new_dataset_refs=list(ASSETS),patches=edits,
        actual_gemma31b_inference_code=True,new_training=False)
    return dest,notebook_digest

def push_once(folder):
    s=run("kernels","status",DEST,timeout=40)
    if s.returncode==0:
        raise RuntimeError("candidate private notebook already exists, refuse repeat push")
    p=run("kernels","push","-p",str(folder),
          "--accelerator","NvidiaTeslaT4","--timeout","32400",timeout=190)
    blob=p.stdout+"\n"+p.stderr
    log("NTL31_KERNEL_PUSH_RECEIPT",return_code=p.returncode,excerpt=blob[-750:])
    if p.returncode or re.search(r"(?i)(quota|403|forbidden|push error)",blob):
        raise RuntimeError("Kaggle kernel push not accepted")
    match=re.search(r"Kernel version\s+(\d+)\s+successfully pushed",blob,re.I)
    if not match:
        raise RuntimeError("Kaggle push did not verify a new candidate version")
    return int(match.group(1))

def wait_preview(max_seconds=4500):
    until=time.monotonic()+max_seconds
    prev=""
    while time.monotonic()<until:
        p=run("kernels","status",DEST,timeout=55)
        s=(p.stdout+"\n"+p.stderr).strip()
        if s!=prev:
            log("NTL31_PREVIEW_STATUS",status=s[-350:])
            prev=s
        if re.search(r"KernelWorkerStatus\.COMPLETE",s):
            return True
        if re.search(r"KernelWorkerStatus\.(ERROR|FAILED|CANCELLED)",s):
            tail=run("kernels","logs",DEST)
            log("NTL31_PREVIEW_FAILURE_LOG",tail=(tail.stdout+tail.stderr)[-1200:])
            raise RuntimeError("31B real preview failed: no official submission")
        time.sleep(35)
    raise RuntimeError("31B preview not completed; no official submission")

def verify_preview():
    output=ROOT/"preview"
    output.mkdir()
    p=run("kernels","output",DEST,"-p",str(output),"-o",timeout=180)
    must(p,"private preview output fetch")
    pr=list(output.rglob("ntl31_proof.json"))
    parquet=list(output.rglob("submission.parquet"))
    if len(pr)!=1 or len(parquet)!=1:
        raise RuntimeError("missing real preview proof or submission.parquet")
    proof=json.loads(pr[0].read_text())
    import pandas as pd
    sub=pd.read_parquet(parquet[0])
    if not (proof.get("loaded_and_inferred") is True and proof.get("no_training") is True
            and proof.get("preview") is True and proof.get("valid_calls",0)>=1
            and proof.get("n_test")==len(sub) and len(sub)>0
            and sub.columns.tolist()==["id","model"] and sub.id.is_unique):
        raise RuntimeError("31B real preview proof mismatch")
    legal=set((
        "claude-fable-5","claude-opus-4-6","claude-opus-4-8","claude-opus-5",
        "claude-sonnet-5","gemini-3.1-pro-preview","gemini-3.6-flash",
        "gemini-3.7-flash","gemini-3.8-flash","glm-5.3","glm-5.3-flash",
        "gpt-5.5","gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra",
        "gpt-6-astra","kimi-k3","qwen3.8-max"))
    if not set(sub.model).issubset(legal):
        raise RuntimeError("invalid actual Kaggle preview labels")
    log("NTL31_PREVIEW_REAL_31B_VERIFIED",**proof)

def submit_once(version,sha):
    guard_account()
    from kaggle.api.kaggle_api_extended import KaggleApi
    a=KaggleApi();a.authenticate()
    p=a.competition_submit_code(file_name="submission.parquet",message=DESC,
                                  competition="name-that-llm",
                                  kernel=DEST,kernel_version=version)
    receipt=p.to_dict() if hasattr(p,"to_dict") else str(p)
    log("NTL31_ACTUAL_CODE_SUBMISSION_RESPONSE",receipt=receipt,
        candidate_sha=sha,version=version)
    rows=[r for r in submissions() if DESC in str(r.get("description") or "")]
    if len(rows)!=1:
        raise RuntimeError("official Kaggle did not register unique code submission")
    ref=rows[0].get("ref")
    log("NTL31_OFFICIAL_SUBMISSION_ID",ref=ref,status=rows[0].get("status"),
        public=rows[0].get("publicScore"),candidate_sha=sha)
    return ref

def follow(ref,seconds=900):
    until=time.monotonic()+seconds
    while time.monotonic()<until:
        match=[r for r in submissions() if str(r.get("ref"))==str(ref)]
        if len(match)!=1:raise RuntimeError("scoring receipt disappeared")
        row=match[0];score=str(row.get("publicScore") or "").strip()
        status=str(row.get("status"))
        if score:
            log("NTL31_NUMERIC_PUBLIC_RESULT",ref=ref,score=float(score),
                delta_vs_owner_0718=round(float(score)-BASELINE_OWN,6),
                passes_076=float(score)>=0.76)
            return
        if "ERROR" in status.upper():
            raise RuntimeError("official scoring ended ERROR with no Public BA")
        time.sleep(60)
    log("NTL31_SCORING_IS_PENDING",ref=ref,scoring_status="PENDING")

def main():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("missing authenticated Kaggle access")
    if ROOT.exists():raise RuntimeError("refuse to reuse local build")
    ROOT.mkdir()
    guard_account()
    bundle,code_sha=build()
    version=push_once(bundle)
    wait_preview()
    verify_preview()
    ref=submit_once(version,code_sha)
    follow(ref)

if __name__=="__main__":main()
