"""One last targeted 31B preview repair: genuine CUDA13 T4 runtime replacement.

v1 failed from a missing notebook timer. v2 reached real Gemma31B launch
and failed before model load: libcudart.so.12 missing. This reuses the same
weights, prompt and Ensemble decision algorithm, swapping only the offline
llama.cpp server binary and its dependent shared library directory.
"""
from __future__ import annotations
import hashlib,json,re,time
from scripts import ntl_gemma4_31b_ensemble_submit_once_20261126 as original

BASE_BUILD=original.build
OLD_RUNTIME="nixty2/llama-cpp-turing-sm75"
NEW_RUNTIME="artp2011/llama-cpp-bin-cuda13-sm75"

def build_v3():
    folder,_=BASE_BUILD()
    mdpath=folder/"kernel-metadata.json"
    md=json.loads(mdpath.read_text())
    sources=list(md.get("dataset_sources") or [])
    if sources.count(OLD_RUNTIME)!=1 or NEW_RUNTIME in sources:
        raise RuntimeError("runtime asset mount contract drift")
    sources[sources.index(OLD_RUNTIME)]=NEW_RUNTIME
    md["dataset_sources"]=sources
    mdpath.write_text(json.dumps(md,indent=2))
    notebook_path=folder/md["code_file"]
    nb=json.loads(notebook_path.read_text())
    timer=0
    writer=0
    for cell in nb["cells"]:
        if cell.get("cell_type")!="code":continue
        src="".join(cell.get("source",[]))
        needle="str(NB_T0),'1' if FAST_COMMIT else '0'"
        if needle in src:
            if src.count(needle)!=1:raise RuntimeError("multiple timer anchors")
            src=src.replace(needle,
                "str(globals().get('NB_T0',time.monotonic())),'1' if FAST_COMMIT else '0'")
            timer+=1
        if "Path('/kaggle/working/ntl31_stage.py').write_text(" in src:
            stage=original.STAGE_PATH.read_text(encoding="utf-8")
            old='    server_bin=locate("llama-cpp-turing-sm75","llama-server")'
            if stage.count(old)!=1:
                raise RuntimeError("server binary locator anchor drift")
            stage=stage.replace(old,
                '    server_bin=locate("llama-cpp-bin-cuda13-sm75","llama-server")\n'
                '    runtime_libdir=str(server_bin.parent)')
            old='    server_bin.chmod(server_bin.stat().st_mode|0o111)'
            if stage.count(old)!=1:raise RuntimeError("old chmod anchor drift")
            stage=stage.replace(old,
                '    import shutil\n'
                '    writable_bin=Path("/kaggle/working/ntl31_llama_server")\n'
                '    shutil.copy2(server_bin,writable_bin)\n'
                '    writable_bin.chmod(writable_bin.stat().st_mode|0o111)\n'
                '    server_bin=writable_bin')
            old='    server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)'
            if stage.count(old)!=1:raise RuntimeError("runtime subprocess anchor drift")
            stage=stage.replace(old,
                '    env=os.environ.copy()\n'
                '    env["LD_LIBRARY_PATH"]=runtime_libdir+":"+env.get("LD_LIBRARY_PATH","")\n'
                '    server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)')
            compile(stage,"ntl31_stage.py","exec")
            src=("from pathlib import Path\n"
                 "Path('/kaggle/working/ntl31_stage.py').write_text("+
                 repr(stage)+",encoding='utf-8')\n")
            writer+=1
        cell["source"]=src.splitlines(keepends=True)
    if timer!=1 or writer!=1:
        raise RuntimeError("v3 exact notebook anchors missing: "+str((timer,writer)))
    notebook_path.write_text(json.dumps(nb,ensure_ascii=False))
    sha=hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    original.log("NTL31_V3_CODE_PIN",sha256=sha,timer_fix=timer,server_switch=writer,
                 runtime=NEW_RUNTIME,model_weights_unchanged=True,new_training=False)
    return folder,sha

def push_v3_once(folder):
    s=original.run("kernels","status",original.DEST)
    if s.returncode or not re.search(r"KernelWorkerStatus\.ERROR",s.stdout+s.stderr):
        raise RuntimeError("v2 must have ERROR status, no concurrent/nonfailed overwrite")
    receipt=original.run("kernels","push","-p",str(folder),
         "--accelerator","NvidiaTeslaT4","--timeout","32400",timeout=190)
    message=receipt.stdout+"\n"+receipt.stderr
    original.log("NTL31_V3_KAGGLE_KERNEL_RECEIPT",returncode=receipt.returncode,text=message[-600:])
    if receipt.returncode or re.search(r"quota|push error|forbidden|403",message,re.I):
        raise RuntimeError("v3 Kaggle push refused")
    match=re.search(r"Kernel version\s+(\d+)\s+successfully pushed",message,re.I)
    if match is None or int(match.group(1))!=3:
        raise RuntimeError("expected precisely new notebook v3; no official submit")
    return 3

def await_v3(max_seconds=4500):
    until=time.monotonic()+max_seconds
    saw_current=False
    previous=None
    while time.monotonic()<until:
        p=original.run("kernels","status",original.DEST,timeout=60)
        state=p.stdout+"\n"+p.stderr
        if state!=previous:
            original.log("NTL31_V3_KAGGLE_STATUS",status=state.strip()[-350:])
            previous=state
        if re.search(r"KernelWorkerStatus\.(QUEUED|RUNNING)",state):saw_current=True
        if saw_current and re.search(r"KernelWorkerStatus\.COMPLETE",state):
            return True
        if saw_current and re.search(r"KernelWorkerStatus\.(ERROR|FAILED|CANCELLED)",state):
            p=original.run("kernels","logs",original.DEST)
            original.log("NTL31_V3_KAGGLE_FAILURE",tail=(p.stdout+p.stderr)[-1600:])
            raise RuntimeError("actual CUDA13 Gemma31B preview failed; no official submission")
        time.sleep(35)
    raise RuntimeError("v3 preview did not finish; do not submit")

def main():
    original.build=build_v3
    original.push_once=push_v3_once
    original.wait_preview=await_v3
    original.main()

if __name__=="__main__":
    main()
