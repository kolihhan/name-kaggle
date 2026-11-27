"""Diagnose cache loss and retry only transient retrieval errors.
Do not accept changed hashes or missing identity metadata.
"""
import collections,hashlib,json,re,time,zipfile
from pathlib import Path
from threading import Lock
import ntl_native_nested_expert_20261127 as experiment

lock=Lock()
counts=collections.Counter()
def record(rel,status,**meta):
    group=Path(rel).name.split("__")[0]
    with lock:
        counts[(group,status)]+=1
        experiment.src.log("CACHE_FILE_STATUS",{"file":rel,"status":status,**meta})

def fetch(root,rel,sha=None):
    d=root/hashlib.sha256(rel.encode()).hexdigest()[:20]
    d.mkdir(exist_ok=True)
    for attempt in range(3):
        p=experiment.src.cmd("datasets","download",experiment.DATASET,"-f",rel,"-p",str(d),"-q")
        if p.returncode==0:break
        message=p.stdout+" "+p.stderr
        status=next((c for c in ("403","404","401","429","500","502","503","504") if re.search(r"\b"+c+r"\b",message)),None)
        transient=status in ("429","500","502","503","504") or bool(re.search(r"(?i)(timeout|connection reset|temporarily unavailable|remote disconnected|retry)",message))
        if not transient or attempt==2:
            record(rel,"HTTP_"+status if status else "CLI_FAILED",attempts=attempt+1)
            return None
        time.sleep(2**(attempt+1))
    name=Path(rel).name
    fs=list(d.rglob(name))
    if not fs:
        for f in d.rglob("*.zip"):
            with zipfile.ZipFile(f) as z:
                hits=[x for x in z.namelist() if Path(x).name==name]
                if len(hits)==1:
                    info=z.getinfo(hits[0])
                    if info.file_size>80_000_000:
                        record(rel,"SIZE_LIMIT",bytes=info.file_size)
                        return None
                    out=d/name;out.write_bytes(z.read(hits[0]));fs.append(out)
    if len(fs)!=1:
        record(rel,"MISSING_OR_AMBIGUOUS_FILE",count=len(fs));return None
    actual=hashlib.sha256(fs[0].read_bytes()).hexdigest()
    if sha and actual!=sha:
        record(rel,"SHA_MISMATCH",expected=sha,actual=actual,bytes=fs[0].stat().st_size);return None
    record(rel,"VERIFIED",bytes=fs[0].stat().st_size)
    return fs[0]

if __name__=="__main__":
    experiment.fetch=fetch
    try:
        experiment.main()
    finally:
        experiment.src.log("FAILURE_MATRIX",[{"family":g,"status":s,"count":n} for (g,s),n in sorted(counts.items())])
