"""Read-only cached OOF evidence audit. Never export row-level values or fit."""
import hashlib,json,subprocess,tempfile,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
DATASET="lightningv08/ntl-olmo2-native24-final-inference-assets"
BASE="name-that-llm-artifacts/"
FILES=[
    "ntl-oof-validation-cache/diverse_oof.npz",
    "ntl-oof-validation-cache/finetuned_oof.npz",
    "ntl-oof-validation-cache/diverse_report.json",
    "ntl-oof-validation-cache/finetuned_report.json",
    "_historical_metadata/ntl-oof-validation-cache/v1-remote-cache-manifest.json",
    "ntl-oof-q35-heads/bundle_manifest.json",

]
SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
def log(k,v):print(k+"="+json.dumps(v,sort_keys=True,default=str),flush=True)
def cmd(*args):
    return subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=100)
def get(rel,root):
    d=root/str(FILES.index(rel));d.mkdir()
    full=BASE+rel
    p=cmd("datasets","download",DATASET,"-f",full,"-p",str(d),"-q")
    if p.returncode:
        log("CACHE_DOWNLOAD",{"file":rel,"status":"failed","http":next((x for x in [403,404,401,429] if str(x) in p.stderr),None)})
        return None
    name=Path(rel).name
    candidates=[x for x in d.rglob(name) if x.is_file()]
    if not candidates:
        for f in d.rglob("*.zip"):
            with zipfile.ZipFile(f) as z:
                matches=[x for x in z.namelist() if Path(x).name==name]
                if len(matches)==1:
                    out=d/name
                    # Extract only this known member to a fixed path.
                    info=z.getinfo(matches[0])
                    if info.file_size>60_000_000:raise ValueError("cache too large for scope")
                    out.write_bytes(z.read(info));candidates.append(out)
    if len(candidates)!=1:
        log("CACHE_DOWNLOAD",{"file":rel,"status":"not_unique"});return None
    path=candidates[0]
    log("CACHE_DOWNLOAD",{"file":rel,"status":"ok","bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    return path
import re
BUNDLE="ntl-oof-q35-heads/"
EXTRA=[BUNDLE+x for x in ("oof_blend.py","oof_dense.py","oof_plan.py","fold_plan.json","nested_report.json","oof_predictions.npz")]
FILES.extend(EXTRA)
def summarize_json(obj,depth=0):
    if isinstance(obj,dict):
        return {k:summarize_json(v,depth+1) for k,v in list(obj.items())[:40]} if depth<3 else {"keys":list(obj)[:40]}
    if isinstance(obj,list):
        return {"length":len(obj),"item_type":type(obj[0]).__name__ if obj else None,"first_schema":list(obj[0])[:30] if obj and isinstance(obj[0],dict) else None}
    return type(obj).__name__
def main():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        manifest=get(BUNDLE+"bundle_manifest.json",root)
        assert manifest and hashlib.sha256(manifest.read_bytes()).hexdigest()=="f3106b4274113ee3828987051e0ec5d33746feb1b179f6fa177dc6a95b375fe9"
        m=json.loads(manifest.read_text())
        log("SCENARIOS",m.get("scenarios"))
        log("REFERENCE_SCOPE",m.get("fixed_reference_scope"))
        checks=m["checksums"]
        for rel in EXTRA[:-1]:
            p=get(rel,root)
            assert p and hashlib.sha256(p.read_bytes()).hexdigest()==checks[p.name],rel
            if p.suffix==".py":
                # Inspect upstream scoring convention and fold construction. Never execute donor code.
                import ast
                raw=p.read_text()
                tree=ast.parse(raw)
                functions=[n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                lines=[(i+1,l.strip()) for i,l in enumerate(raw.splitlines()) if len(l)<400 and re.search(r"(?i)(log_softmax|logsumexp|softmax|log.prob|predict_proba|decision_function|scenarios|outer.fold|inner.fold|groupkfold|split\(|weighted|normalize|calibrat|reference|components|nested|np.log|np.exp|scores\[|einsum|tensordot)",l)]
                log("SOURCE_CONTRACT",{"file":p.name,"functions":functions,"matches":lines[:160]})
            else:
                x=json.loads(p.read_text())
                log("JSON_SCHEMA",{"file":p.name,"structure":summarize_json(x)})
                if p.name=="nested_report.json":
                    log("REPORT_LIMITATIONS",{"method":x.get("method"),"limitations":x.get("limitations"),"reference_missing_components":x.get("reference_missing_components")})
        # Inspect one real outer and one inner feature cache. Do not unpickle or emit values.
        for name in ("qwen14__outer0.npz","qwen14__outer0_inner0.npz"):
            rel="ntl-oof-validation-cache/"+name
            FILES.append(rel)
            p=get(rel,root)
            if p:
                info={}
                with np.load(p,allow_pickle=False) as z:
                    for k in z.files:
                        try:a=z[k];info[k]={"shape":list(a.shape),"dtype":str(a.dtype)}
                        except ValueError:info[k]={"object_refused":True}
                log("FOLD_CACHE_SCHEMA",{"file":name,"arrays":info})
        log("AUDIT_DONE",{"fit":False,"gpu":False,"submission":False})
if __name__=="__main__":main()
