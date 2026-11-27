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
    "ntl-oof-q35-heads/oof_predictions.npz",
    "ntl-oof-q35-heads/nested_report.json",
    "ntl-oof-q35-heads/bundle_manifest.json",
    "ntl-oof-q35-heads/frozen_feature_manifest.json",
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
def describe_json(x,depth=0):
    # Structural metadata plus scalar experiment statistics; not arbitrary raw strings or rows.
    if depth>3:return type(x).__name__
    if isinstance(x,dict):
        out={}
        for k,v in list(x.items())[:80]:
            if isinstance(v,(dict,list)):out[k]=describe_json(v,depth+1)
            elif isinstance(v,(bool,int,float)) or v is None:out[k]=v
            elif isinstance(v,str) and ("sha" in k.lower() or "split" in k.lower() or "fold" in k.lower() or "schema" in k.lower() or "version" in k.lower()) and len(v)<100:out[k]=v
            else:out[k]=type(v).__name__
        return out
    if isinstance(x,list):return {"len":len(x),"first_structure":describe_json(x[0],depth+1) if x and isinstance(x[0],dict) else type(x[0]).__name__ if x else None}
    return type(x).__name__
def load_cache(path):
    arrays={};info={}
    with np.load(path,allow_pickle=False) as z:
        for k in z.files:
            try:
                a=z[k];info[k]={"shape":a.shape,"dtype":str(a.dtype)}
                if a.ndim==2 and np.issubdtype(a.dtype,np.number):info[k]["finite"]=bool(np.isfinite(a).all())
                arrays[k]=a
            except ValueError:
                info[k]={"status":"object array refused"}
    return arrays,info
def admit(arrays,train):
    # Explicit row IDs AND ordered class names are mandatory. No inferred row order.
    idset=set(train.id);labelset=set(train.model)
    idc=[];clc=[]
    for k,a in arrays.items():
        if a.ndim!=1 or a.dtype.kind not in "US":continue
        if len(a)==len(train) and len(set(a.astype(str)))==len(a) and set(a.astype(str))==idset:idc.append((k,a.astype(str)))
        if len(a)==18 and set(a.astype(str))==labelset:clc.append((k,a.astype(str)))
    if not idc or not clc:return {},{"status":"unadmitted_missing_explicit_ids_or_classes","ids":len(idc),"classes":len(clc)}
    ids=idc[0][1];labs=clc[0][1]
    if any(not np.array_equal(a,ids) for _,a in idc) or any(not np.array_equal(a,labs) for _,a in clc):return {},{"status":"ambiguous_axis_metadata"}
    idx=pd.Index(ids).get_indexer(train.id);perm=pd.Index(labs).get_indexer(sorted(labelset))
    scores={}
    for k,a in arrays.items():
        if a.shape==(len(ids),18) and np.issubdtype(a.dtype,np.number) and np.isfinite(a).all() and np.all(np.ptp(a,axis=1)>1e-10):
            scores[k]=a[idx][:,perm]
    return scores,{"status":"id_label_aligned_historical_not_independence_certified","matrices":list(scores)}
def main():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);cache={}
        for rel in FILES:
            path=get(rel,root)
            if not path:continue
            if path.suffix==".json":
                log("REPORT_STRUCTURE",{"file":rel,"summary":describe_json(json.loads(path.read_text()))})
            else:
                arrays,info=load_cache(path);cache[rel]=arrays
                log("CACHE_SCHEMA",{"file":rel,"arrays":info})
        train_dir=root/"train";train_dir.mkdir()
        p=cmd("competitions","download","name-that-llm","-f","train.csv","-p",str(train_dir),"-q")
        if p.returncode:log("PAIRED_METRICS_BLOCKED","authenticated train download failed");return
        paths=list(train_dir.glob("train.csv"))
        if not paths:
            for f in train_dir.glob("*.zip"):
                with zipfile.ZipFile(f) as z:
                    if "train.csv" in z.namelist():
                        f2=train_dir/"train.csv";f2.write_bytes(z.read("train.csv"));paths.append(f2)
        if len(paths)!=1 or hashlib.sha256(paths[0].read_bytes()).hexdigest()!=SHA:
            log("PAIRED_METRICS_BLOCKED","pinned train hash mismatch");return
        train=pd.read_csv(paths[0],dtype=str,usecols=["id","model","prompt_id"])
        assert len(train)==2699 and train.model.nunique()==18 and train.id.is_unique
        y=pd.Index(sorted(train.model.unique())).get_indexer(train.model)
        pgs={}
        def ba(correct):
            return float(np.mean([correct[y==k].mean() for k in range(18)]))
        for rel,arrays in cache.items():
            scores,admission=admit(arrays,train);log("CACHE_ADMISSION",{"file":rel,**admission})
            for k,a in scores.items():
                order=np.argsort(-a,axis=1,kind="stable")
                correct=order[:,0]==y
                pgs[rel+":"+k]=correct
                log("HISTORICAL_OOF_MATRIX",{"name":rel+":"+k,"n":len(y),"ba":ba(correct),"top2":ba((order[:,:2]==y[:,None]).any(1)),"top3":ba((order[:,:3]==y[:,None]).any(1)),"top5":ba((order[:,:5]==y[:,None]).any(1)),"scope":"historical publisher OOF cache, fold and training independence unverified"})
        if len(pgs)>1:
            import itertools
            for (a,p),(b,q) in itertools.combinations(pgs.items(),2):
                log("PAIRED_HISTORICAL_DIAGNOSTIC",{"a":a,"b":b,"oracle_union_ba":ba(p|q),"rescues":int((~p&q).sum()),"regressions":int((p&~q).sum()),"both_wrong":int((~p&~q).sum()),"delta_ba":ba(q)-ba(p)})
        if not pgs:log("PAIRED_METRICS_BLOCKED","no explicit full-18 ID and label axis alignment")
        log("AUDIT_DONE",{"score_matrices":len(pgs),"fit":False,"gpu":False,"submission":False})
if __name__=="__main__":main()
