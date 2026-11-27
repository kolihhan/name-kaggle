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
def analyze(arrays,train,report,scenarios):
    ids=arrays["ids"].astype(str);labs=arrays["labels"].astype(str)
    assert len(ids)==len(train)==2699 and len(set(ids))==len(ids) and set(ids)==set(train.id)
    assert len(labs)==18 and len(set(labs))==18 and set(labs)==set(train.model)
    idx=pd.Index(ids).get_indexer(train.id)
    order_labels=sorted(labs);perm=pd.Index(labs).get_indexer(order_labels)
    y=pd.Index(order_labels).get_indexer(train.model)
    assert np.array_equal(labs[arrays["y"].astype(int)][idx],train.model.to_numpy())
    names=arrays["component_names"].astype(str).tolist()
    assert len(names)==len(set(names))
    assert arrays["components"].shape==(len(names),3,2699,18)
    assert report["component_names"]==names
    def ba(ok):
        return float(np.mean([ok[y==k].mean() for k in range(18)]))
    def score(a):
        assert a.shape==(2699,18) and np.isfinite(a).all()
        a=a[idx][:,perm]
        # A constant vector provides no information. Retain and count ties rather than silently dropping them.
        rank=np.argsort(-a,axis=1,kind="stable")
        return a,rank,rank[:,0]==y
    def metrics(rank,ok):
        return dict(ba=ba(ok),top2=ba((rank[:,:2]==y[:,None]).any(1)),top3=ba((rank[:,:3]==y[:,None]).any(1)),top5=ba((rank[:,:5]==y[:,None]).any(1)),correct=int(ok.sum()))
    def backbone(name):
        # Group repeated heads from the same encoder rather than counting them as independent.
        for root in ("qwen14","qwen8","q35","deberta","roberta","style","svm"):
            if name.startswith(root):return root
        return name.split("_")[0]
    for si,scenario in enumerate(scenarios):
        base,rank,ok=score(arrays["reference"][si])
        nested,nrank,nok=score(arrays["nested"][si])
        cal,crank,cok=score(arrays["nested_calibrated"][si])
        expected=report["scenario_scores"]["fixed_reference"][str(scenario)]
        # Report may use raw accuracy instead of class balanced accuracy. Require explicit numerical consistency.
        checks={"reported":expected,"recomputed_ba":ba(ok),"recomputed_acc":float(ok.mean())}
        assert min(abs(expected-checks["recomputed_ba"]),abs(expected-checks["recomputed_acc"]))<1e-8,checks
        results=[];hits=[];ranks=[];groups={}
        for ci,name in enumerate(names):
            a,rr,c=score(arrays["components"][ci,si])
            hit=metrics(rr,c)
            m={"name":name,"backbone":backbone(name),**hit,"rescues":int((~ok&c).sum()),"regressions":int((ok&~c).sum()),"oracle_with_ref":ba(ok|c),"constant_rows":int((np.ptp(a,axis=1)<=1e-10).sum())}
            results.append(m);hits.append(c);ranks.append(rr);groups.setdefault(backbone(name),[]).append(ci)
        hits=np.asarray(hits);ranks=np.asarray(ranks)
        # Fixed, no-fit Borda control. Give each backbone equal total vote mass.
        votes=np.zeros((len(y),18),float)
        for members in groups.values():
            block=np.zeros_like(votes)
            for ci in members:
                temp=np.empty_like(votes);np.put_along_axis(temp,ranks[ci],np.broadcast_to(np.arange(18,0,-1),(len(y),18)),axis=1)
                block+=temp/len(members)
            votes+=block/len(groups)
        pred=np.argsort(-votes,axis=1,kind="stable");bok=pred[:,0]==y
        per_family=[]
        for g,members in groups.items():
            anyright=hits[members].any(0)
            others=[i for i in range(len(names)) if i not in members]
            only=anyright & (~hits[others].any(0) if others else True)
            per_family.append({"family":g,"heads":len(members),"oracle":ba(anyright),"unique_correct":int(only.sum()),"rescues_vs_ref":int((~ok&anyright).sum()),"oracle_with_ref":ba(ok|anyright)})
        top=sorted(results,key=lambda x:-x["ba"])
        log("OOF_DECOMPOSITION",{"scenario":scenario,"source_check":checks,"reference":metrics(rank,ok),"nested":metrics(nrank,nok),"nested_calibrated":metrics(crank,cok),"equal_backbone_borda":metrics(pred,bok),"borda_rescues":int((~ok&bok).sum()),"borda_regressions":int((ok&~bok).sum()),"oracle_all_heads":ba(hits.any(0)),"oracle_with_reference":ba(ok|hits.any(0)),"both_wrong":int((~ok&~hits.any(0)).sum()),"families":per_family,"heads":top,"scope":"historical publisher OOF; not exact V90 or current .718, training exposure not independently certified"})
def main():
    pins={"ntl-oof-validation-cache/diverse_oof.npz":"3db2ada2d57e074ad7507361896a2e976e8a0586f7a89f549e404b3ca4aa455f","ntl-oof-validation-cache/finetuned_oof.npz":"b414644af20295e870269a6c44c17d2209c45dd9b02dd4841894b2502a82ee06","ntl-oof-validation-cache/diverse_report.json":"153db6ef08fffe12a2a560e147efb265cc676963cd3837b1b94af3178a945e57","ntl-oof-validation-cache/finetuned_report.json":"f55b87ad5a072bbb37dfcffe639b5baaf08f2c295f3acb4d84ade00067b41637","ntl-oof-q35-heads/bundle_manifest.json":"f3106b4274113ee3828987051e0ec5d33746feb1b179f6fa177dc6a95b375fe9"}
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);files={}
        for rel,digest in pins.items():
            path=get(rel,root)
            assert path and hashlib.sha256(path.read_bytes()).hexdigest()==digest
            files[rel]=path
        manifest=json.loads(files["ntl-oof-q35-heads/bundle_manifest.json"].read_text())
        scenarios=[str(x) for x in manifest["scenarios"]]
        assert scenarios==["1","3","6"],scenarios
        log("SCENARIOS",scenarios)
        train_dir=root/"train";train_dir.mkdir()
        p=cmd("competitions","download","name-that-llm","-f","train.csv","-p",str(train_dir),"-q")
        assert p.returncode==0
        paths=list(train_dir.glob("train.csv"))
        if not paths:
            for f in train_dir.glob("*.zip"):
                with zipfile.ZipFile(f) as z:
                    if "train.csv" in z.namelist():
                        out=train_dir/"train.csv";out.write_bytes(z.read("train.csv"));paths.append(out)
        assert len(paths)==1 and hashlib.sha256(paths[0].read_bytes()).hexdigest()==SHA
        train=pd.read_csv(paths[0],dtype=str,usecols=["id","model","prompt_id"])
        assert len(train)==2699 and train.model.nunique()==18 and train.id.is_unique
        for kind in ("diverse","finetuned"):
            report=json.loads(files[f"ntl-oof-validation-cache/{kind}_report.json"].read_text())
            assert report["train_sha256"]==SHA
            arrays,_=load_cache(files[f"ntl-oof-validation-cache/{kind}_oof.npz"])
            log("ANALYZING",kind)
            analyze(arrays,train,report,scenarios)
        log("AUDIT_DONE",{"fit":False,"gpu":False,"submission":False})
if __name__=="__main__":main()
