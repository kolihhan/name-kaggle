"""One-time Kaggle Gemma31B private preview repair after v1 NameError.

Fixes preview-only uninitialized notebook timer and Kaggle-input executable
permission in an immutable derived candidate. Does not modify model behavior,
fusion weights, training corpus or hidden rerun decisions. Fails closed.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import time
from pathlib import Path
from scripts import ntl_gemma4_31b_ensemble_submit_once_20261126 as original

def fixed_build():
    folder,_ = original.build()
    meta=json.loads((folder/"kernel-metadata.json").read_text())
    notebook_path=folder/meta["code_file"]
    nb=json.loads(notebook_path.read_text())
    count_timer=0
    count_perm=0
    for cell in nb["cells"]:
        if cell.get("cell_type")!="code":
            continue
        src="".join(cell.get("source",[]))
        if "str(NB_T0),'1' if FAST_COMMIT else '0'" in src:
            src=src.replace("str(NB_T0),'1' if FAST_COMMIT else '0'",
                "str(globals().get('NB_T0',time.monotonic())),'1' if FAST_COMMIT else '0'")
            count_timer+=1
        # The original code writes the stage into an embedded string. Change
        # only the exact executable permission operation, not the scorer.
        needle="server_bin.chmod(server_bin.stat().st_mode|0o111)"
        if needle in src:
            replacement=(
                'import shutil\n'
                '    writable_bin=Path("/kaggle/working/ntl31_llama_server")\n'
                '    shutil.copy2(server_bin,writable_bin)\n'
                '    writable_bin.chmod(writable_bin.stat().st_mode|0o111)\n'
                '    server_bin=writable_bin'
            )
            src=src.replace(needle,replacement)
            count_perm+=1
        cell["source"]=src.splitlines(keepends=True)
    if count_timer!=1 or count_perm!=1:
        raise RuntimeError("exact preview repair anchors mismatch: "+str((count_timer,count_perm)))
    notebook_path.write_text(json.dumps(nb,ensure_ascii=False))
    sha=hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    original.log("NTL31_V2_FIXED_PREVIEW",timer=count_timer,permissions=count_perm,code_sha=sha)
    return folder,sha

def retry_failed_v1_once(folder):
    s=original.run("kernels","status",original.DEST)
    if s.returncode!=0 or not re.search(r"KernelWorkerStatus\.ERROR",s.stdout+s.stderr):
        raise RuntimeError("v1 must exist in ERROR before exactly one repair attempt")
    p=original.run("kernels","push","-p",str(folder),
        "--accelerator","NvidiaTeslaT4","--timeout","32400",timeout=190)
    output=p.stdout+"\n"+p.stderr
    original.log("NTL31_V2_REPAIR_PUSH_RESPONSE",rc=p.returncode,output=output[-700:])
    if p.returncode or re.search(r"(quota|push error|forbidden|403)",output,re.I):
        raise RuntimeError("Kaggle repair push not accepted")
    match=re.search(r"Kernel version\s+(\d+)\s+successfully pushed",output,re.I)
    if not match or int(match.group(1))!=2:
        raise RuntimeError("expected exact new notebook v2, not unverified/other version")
    return 2

def await_new_v2(max_seconds=4500):
    deadline=time.monotonic()+max_seconds
    seen_new=False
    last=None
    while time.monotonic()<deadline:
        p=original.run("kernels","status",original.DEST,timeout=55)
        state=p.stdout+"\n"+p.stderr
        if state!=last:
            original.log("NTL31_V2_REAL_KAGGLE_STATE",status=state.strip()[-300:])
            last=state
        if re.search(r"KernelWorkerStatus\.(QUEUED|RUNNING)",state):
            seen_new=True
        if seen_new and re.search(r"KernelWorkerStatus\.COMPLETE",state):
            return True
        if seen_new and re.search(r"KernelWorkerStatus\.(ERROR|FAILED|CANCELLED)",state):
            log=original.run("kernels","logs",original.DEST)
            original.log("NTL31_V2_REAL_FAILURE_LOG",tail=(log.stdout+log.stderr)[-2000:])
            raise RuntimeError("actual repaired preview failed; no official submission")
        time.sleep(35)
    raise RuntimeError("v2 preview not completed; no official submission")

def main():
    original.build=fixed_build
    original.push_once=retry_failed_v1_once
    original.wait_preview=await_new_v2
    # Call the proven preflight and one-off scorer in its original module.
    # No direct official competition submission outside original gating.
    original.main()

if __name__=="__main__":
    main()
