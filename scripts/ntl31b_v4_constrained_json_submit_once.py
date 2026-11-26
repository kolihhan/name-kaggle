"""One-shot v4: constrained 31B JSON label output, preserving same Ensemble.

v3 achieved actual 31B GPU model load + 3 real responses but all missed the
two-label exact parser. v4 explicitly disables Gemma4 thinking and uses
llama.cpp JSON-schema-constrained decoding. No new model training or weight
tuning; unchanged 31B GGUF, same top2 low-margin decisions and pinned v52.
"""
import hashlib,json,re,time
from scripts import ntl_gemma4_31b_ensemble_submit_once_20261126 as orig

BASE_BUILD=orig.build
DEST_VERSION=4
NEW_RUNTIME="artp2011/llama-cpp-bin-cuda13-sm75"

def build_v4():
    folder,_=BASE_BUILD()
    meta_path=folder/"kernel-metadata.json"
    meta=json.loads(meta_path.read_text())
    mounts=list(meta.get("dataset_sources") or [])
    old_mount="nixty2/llama-cpp-turing-sm75"
    if mounts.count(old_mount)!=1 or NEW_RUNTIME in mounts:
        raise RuntimeError("31B v4 mount count mismatch")
    mounts[mounts.index(old_mount)]=NEW_RUNTIME
    meta["dataset_sources"]=mounts
    meta_path.write_text(json.dumps(meta,indent=2))
    notebook=folder/meta["code_file"]
    data=json.loads(notebook.read_text())
    n_timer=n_writer=0
    for cell in data["cells"]:
        if cell.get("cell_type")!="code":continue
        src="".join(cell.get("source",[]))
        old="str(NB_T0),'1' if FAST_COMMIT else '0'"
        if old in src:
            if src.count(old)!=1:raise RuntimeError("timer mismatch")
            src=src.replace(old,
                  "str(globals().get('NB_T0',time.monotonic())),'1' if FAST_COMMIT else '0'")
            n_timer+=1
        if "Path('/kaggle/working/ntl31_stage.py').write_text(" in src:
            stage=orig.STAGE_PATH.read_text(encoding="utf-8")
            substitutions=[
              ('    server_bin=locate("llama-cpp-turing-sm75","llama-server")',
               '    server_bin=locate("llama-cpp-bin-cuda13-sm75","llama-server")\n'
               '    runtime_libdir=str(server_bin.parent)'),
              ('    server_bin.chmod(server_bin.stat().st_mode|0o111)',
               '    import shutil\n'
               '    writable_bin=Path("/kaggle/working/ntl31_llama_server")\n'
               '    shutil.copy2(server_bin,writable_bin)\n'
               '    writable_bin.chmod(writable_bin.stat().st_mode|0o111)\n'
               '    server_bin=writable_bin'),
              ('    server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)',
               '    env=os.environ.copy()\n'
               '    env["LD_LIBRARY_PATH"]=runtime_libdir+":"+env.get("LD_LIBRARY_PATH","")\n'
               '    server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)'),
              ('''            body={"model":"local-gemma-31b","messages":[
                 {"role":"system","content":intro},{"role":"user","content":task}],
                 "max_tokens":28,"temperature":0,"stream":False}
            result=api_request(body)
            guess=parse_label(result["choices"][0]["message"].get("content") or "")''',
               '''            schema={"type":"object","properties":{
                "label":{"type":"string","enum":[current,candidate]}},
                "required":["label"],"additionalProperties":False}
            body={"model":"local-gemma-31b","messages":[
                 {"role":"system","content":intro},{"role":"user","content":task}],
                 "max_tokens":96,"temperature":0,"stream":False,
                 "chat_template_kwargs":{"enable_thinking":False},
                 "response_format":{"type":"json_schema","schema":schema}}
            result=api_request(body)
            message=result["choices"][0]["message"]
            answer=str(message.get("content") or "")
            try:
                decoded=json.loads(answer)
                guess=decoded.get("label") if isinstance(decoded,dict) else None
            except (ValueError,TypeError):
                guess=None
            if guess not in (current,candidate):guess=None
            if preview:
                print("NTL31_V4_FORMAT_DEBUG="+json.dumps({
                    "finish_reason":result["choices"][0].get("finish_reason"),
                    "content_length":len(answer),
                    "reasoning_length":len(str(message.get("reasoning_content") or "")),
                    "valid_schema_choice":bool(guess),
                    "allowed_choices":2}),flush=True)''')
            ]
            for old,new in substitutions:
                if stage.count(old)!=1:
                    raise RuntimeError("v4 source patch anchor mismatch: "+old[:65])
                stage=stage.replace(old,new)
            compile(stage,"ntl31_stage_v4.py","exec")
            src=("from pathlib import Path\n"
                 "Path('/kaggle/working/ntl31_stage.py').write_text("+
                 repr(stage)+",encoding='utf-8')\n")
            n_writer+=1
        cell["source"]=src.splitlines(keepends=True)
    if n_timer!=1 or n_writer!=1:
        raise RuntimeError("exact v4 notebook anchors missing")
    notebook.write_text(json.dumps(data,ensure_ascii=False))
    checksum=hashlib.sha256(notebook.read_bytes()).hexdigest()
    orig.log("NTL31_V4_SCORED_SOURCE_PIN",sha=checksum,gemma31b_same_weight=True,
        runtime=NEW_RUNTIME,source="pinned Lightning 0.724 v52",
        constrained_json_enum=True,thinking_disabled=True,
        retraining=False)
    return folder,checksum

def push_v4(folder):
    p=orig.run("kernels","status",orig.DEST,timeout=50)
    if p.returncode or not re.search(r"KernelWorkerStatus\.ERROR",p.stdout+p.stderr):
        raise RuntimeError("v3 must be ERROR before exact repair; no concurrent overwrite")
    r=orig.run("kernels","push","-p",str(folder),
               "--accelerator","NvidiaTeslaT4","--timeout","32400",timeout=200)
    report=r.stdout+"\n"+r.stderr
    orig.log("NTL31_V4_KERNEL_PUSH",rc=r.returncode,reply=report[-800:])
    if r.returncode or re.search(r"quota|forbidden|push error|403",report,re.I):
        raise RuntimeError("v4 Kaggle push rejected")
    match=re.search(r"Kernel version\s+(\d+)\s+successfully pushed",report,re.I)
    if not match or int(match.group(1))!=DEST_VERSION:
        raise RuntimeError("expected precisely Kaggle v4; no official scoring")
    return DEST_VERSION

def wait_v4(max_seconds=4500):
    deadline=time.monotonic()+max_seconds
    witnessed=False
    last=None
    while time.monotonic()<deadline:
        p=orig.run("kernels","status",orig.DEST,timeout=60)
        status=p.stdout+"\n"+p.stderr
        if status!=last:
            orig.log("NTL31_V4_KERNEL_STATUS",status=status.strip()[-400:])
            last=status
        if re.search(r"KernelWorkerStatus\.(QUEUED|RUNNING)",status):
            witnessed=True
        if witnessed and re.search(r"KernelWorkerStatus\.COMPLETE",status):
            return True
        if witnessed and re.search(r"KernelWorkerStatus\.(ERROR|FAILED|CANCELLED)",status):
            lg=orig.run("kernels","logs",orig.DEST)
            orig.log("NTL31_V4_KAGGLE_PREVIEW_FAILURE",tail=(lg.stdout+lg.stderr)[-2800:])
            raise RuntimeError("v4 strict 31B real preview failed; do not submit")
        time.sleep(35)
    raise RuntimeError("v4 preview pending; no official submission")

def main():
    orig.build=build_v4
    orig.push_once=push_v4
    orig.wait_preview=wait_v4
    orig.main()

if __name__=="__main__":
    main()
