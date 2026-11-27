"""Native nested OOF: align original cached expert scores and test a conditional residual.
No upstream model execution, no new neural training, no submission.
Only authorized public input files; row-level contents never exported.
"""
import hashlib,json,re,subprocess,tempfile,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
import ntl_decompose_historical_oof_20261127 as src

BASE="name-that-llm-artifacts/ntl-oof-validation-cache/"
PREFIX="name-that-llm-artifacts/"
DATASET=src.DATASET
ALPHA=.20

def fetch(root,rel,sha=None):
    d=root/hashlib.sha256(rel.encode()).hexdigest()[:20]
    d.mkdir(exist_ok=True)
    p=src.cmd("datasets","download",DATASET,"-f",rel,"-p",str(d),"-q")
    if p.returncode:return None
    name=Path(rel).name
    fs=list(d.rglob(name))
    if not fs:
        for f in d.rglob("*.zip"):
            with zipfile.ZipFile(f) as z:
                hits=[x for x in z.namelist() if Path(x).name==name]
                if len(hits)==1 and z.getinfo(hits[0]).file_size<80_000_000:
                    out=d/name;out.write_bytes(z.read(hits[0]));fs.append(out)
    if len(fs)!=1:return None
    if sha and hashlib.sha256(fs[0].read_bytes()).hexdigest()!=sha:return None
    return fs[0]

def npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files if k!="groups"}

def aligned(z,ids,labels,key):
    assert str(z["train_sha256"].item())==src.SHA
    zi=z["ids"].astype(str);zl=z["labels"].astype(str)
    assert len(zi)==len(ids) and len(set(zi))==len(zi) and set(zi)==set(ids)
    assert set(zl)==set(labels) and len(zl)==18
    assert z["scenarios"].astype(str).tolist()==["1","3","6"]
    a=z[key]
    if a.ndim==2:a=np.repeat(a[None],3,axis=0)
    assert a.shape==(3,len(ids),18) and np.isfinite(a).all()
    return a[:,pd.Index(zi).get_indexer(ids),:][:,:,pd.Index(zl).get_indexer(labels)]

def native_prob(a):
    if (a>=0).all() and np.allclose(a.sum(-1),1,atol=1e-5):return a
    raise ValueError("raw cache is not a probability tensor")

def log_components(q,t):
    return np.log(softmax(np.log(q.clip(1e-12))/t,axis=-1).clip(1e-9))

def match(outer,raw):
    """Identify representation by agreement with frozen scores, not truth.
    Candidate temperatures are reconstructed analytically from log odds, then
    verified against all actual outer cached scores. No selection on BA.
    """
    q=native_prob(raw)
    base=np.log(q.clip(1e-12))
    x=base-base.mean(-1,keepdims=True);y=outer-outer.mean(-1,keepdims=True)
    # Only values outside clipping regime for analytic scale.
    mask=(q>1e-7)&(outer>np.log(1e-9)+.02)
    # pairwise ratios using maximum class anchor, removes row offsets
    x=base-base.max(-1,keepdims=True);y=outer-outer.max(-1,keepdims=True)
    good=mask&(np.abs(x)>1e-3)
    if not good.any():return None
    slopes=y[good]/x[good]
    slope=float(np.median(slopes))
    if not np.isfinite(slope) or slope<=0:return None
    t=1/slope
    # Native no-temperature case may clip at 1e-9 without re-softmax.
    direct=np.log(q.clip(1e-9))
    if np.max(np.abs(outer-direct))<2e-5:return ("direct",1.)
    calc=log_components(q,t)
    if np.max(np.abs(outer-calc))<5e-5:return ("temp",t)
    return None

def transform(a,method):
    q=native_prob(a)
    return np.log(q.clip(1e-9)) if method[0]=="direct" else log_components(q,method[1])

def family(n):
    for x in ("qwen14","qwen8","q35","deberta","roberta","style","svm"):
        if n.startswith(x):return x
    return n.split("_")[0]

def ba(y,p):
    return float(np.mean([(p[y==k]==k).mean() for k in range(18)]))

def conditional(qtr,ytr,qte,context):
    # Conditional confusion likelihood from INNER predictions only.
    def bins(q):
        t=np.sort(q,axis=-1)
        return np.digitize(t[:,-1]-t[:,-2],[.05,.15,.30]) if context else np.zeros(len(q),int)
    a=qtr.argmax(1); b=qte.argmax(1); c=bins(qtr); d=bins(qte)
    k=4 if context else 1
    counts=np.zeros((18,18,k))
    np.add.at(counts,(ytr,a,c),1)
    prior=np.ones((18,18))
    np.add.at(prior,(ytr,a),1)
    prior/=prior.sum(1,keepdims=True)
    counts+=18*prior[:,:,None]/k
    counts/=counts.sum((1,2),keepdims=True)
    p=counts[:,b,d].T
    return p/p.sum(1,keepdims=True)

def main():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        pins={
          "ntl-oof-validation-cache/diverse_oof.npz":"3db2ada2d57e074ad7507361896a2e976e8a0586f7a89f549e404b3ca4aa455f",
          "ntl-oof-validation-cache/diverse_report.json":"153db6ef08fffe12a2a560e147efb265cc676963cd3837b1b94af3178a945e57",
          "ntl-oof-q35-heads/fold_plan.json":"073d5e1af3ea61d47bd55f2d91bcbad6685f693265f1777d47c583c2d3770c08",
          "_historical_metadata/ntl-oof-validation-cache/v1-remote-cache-manifest.json":"f214e33b4a6fb497a25dbd538c2b54c6d8be150914f1eea0307d4380ec4d31ab"}
        paths={k:fetch(root,PREFIX+k,v) for k,v in pins.items()}
        assert all(paths.values())
        cache=npz(paths["ntl-oof-validation-cache/diverse_oof.npz"])
        report=json.loads(paths["ntl-oof-validation-cache/diverse_report.json"].read_text())
        plan=json.loads(paths["ntl-oof-q35-heads/fold_plan.json"].read_text())
        checks=json.loads(paths["_historical_metadata/ntl-oof-validation-cache/v1-remote-cache-manifest.json"].read_text())["checksums"]
        assert plan["train_sha256"]==src.SHA
        names=cache["component_names"].astype(str).tolist()
        labels=cache["labels"].astype(str)
        ids=cache["ids"].astype(str); y=cache["y"].astype(int)
        td=root/"train";td.mkdir()
        p=src.cmd("competitions","download","name-that-llm","-f","train.csv","-p",str(td),"-q")
        assert p.returncode==0
        paths=list(td.glob("train.csv"))
        if not paths:
            for f in td.glob("*.zip"):
                with zipfile.ZipFile(f) as z:
                    if "train.csv" in z.namelist():
                        out=td/"train.csv";out.write_bytes(z.read("train.csv"));paths.append(out)
        assert len(paths)==1 and hashlib.sha256(paths[0].read_bytes()).hexdigest()==src.SHA
        train=pd.read_csv(paths[0],dtype=str,usecols=["id","model","prompt_id"])
        # Original plan indices are in official train CSV order; verify before using.
        assert np.array_equal(train.id.to_numpy(),ids)
        assert np.array_equal(train.model.to_numpy(),labels[y])
        assert list(plan["labels"])==list(labels)
        prompts=train.prompt_id.to_numpy()
        outer_names=["outer"+str(i) for i in range(5)]
        available=sorted(k for k in checks if re.fullmatch(r"[^/]+__outer[0-4](?:_inner[0-2])?\.npz",k))
        src.log("PINNED_FOLD_FILES",{"count":len(available),"roots":sorted(set(x.split("__")[0] for x in available))})
        # Read every actual authorized checksum-pinned fold cache. No checkpoint download.
        with ThreadPoolExecutor(max_workers=6) as ex:
            paths=dict(zip(available,ex.map(lambda k:fetch(root,BASE+k,checks[k]),available)))
        src.log("CACHE_ACCESS",{"available":len(available),"downloaded":sum(p is not None for p in paths.values())})
        outputs={k:np.full((3,len(ids)),-1,int) for k in ("subset_native","subset_static","subset_context","full_nested_plus_context","full_nested_plus_static")}
        folds_done=0
        for fi,fold in enumerate(outer_names):
            job=plan["jobs"][fold];va=np.array(job["valid"],int);tr=np.array(job["fit"],int)
            assert not set(prompts[va])&set(prompts[tr])
            assert set(va)|set(tr)==set(range(len(ids))) and not set(va)&set(tr)
            # Map component names to independently verified raw cache columns.
            mapping={}
            for fname,path in paths.items():
                if not path or not fname.endswith("__"+fold+".npz"):continue
                z=npz(path);rootname=fname.split("__")[0]
                for key in z:
                    if key in ("ids","labels","scenarios","train_sha256"):continue
                    if z[key].ndim not in (2,3):continue
                    try:raw=aligned(z,ids[va],labels,key)
                    except (AssertionError,KeyError):continue
                    if raw.shape!=(3,len(va),18):continue
                    for ci,name in enumerate(names):
                        if ci in mapping or family(name)!=family(rootname):continue
                        try:m=match(cache["components"][ci][:,va,:],raw)
                        except ValueError:continue
                        if m is not None:mapping[ci]=(rootname,key,m)
            src.log("MATCHED_NATIVE_COMPONENTS",{"fold":fold,"count":len(mapping),"components":[names[i] for i in mapping]})
            # Must include at least three distinct families; otherwise evidence is not adequate.
            if len({family(names[i]) for i in mapping})<3:continue
            inner_ids=[]; blocks={ci:[] for ci in mapping}
            valid_mapping=set(mapping)
            for inner in plan["outer_inner"][fold]:
                ij=plan["jobs"][inner];iv=np.array(ij["valid"],int);it=np.array(ij["fit"],int)
                assert not set(prompts[it])&set(prompts[iv])
                assert not set(prompts[it])&set(prompts[va]) and not set(prompts[iv])&set(prompts[va])
                inner_ids.extend(iv.tolist())
                for ci,(rn,key,m) in mapping.items():
                    path=paths.get(rn+"__"+inner+".npz")
                    if not path:valid_mapping.discard(ci);continue
                    try:a=transform(aligned(npz(path),ids[iv],labels,key),m)
                    except (AssertionError,ValueError,KeyError):valid_mapping.discard(ci);continue
                    blocks[ci].append(a)
            inner_ids=np.array(inner_ids,int)
            assert len(set(inner_ids))==len(inner_ids) and set(inner_ids)==set(tr)
            selected=sorted(ci for ci in valid_mapping if len(blocks[ci])==3)
            if len({family(names[i]) for i in selected})<3:continue
            weights=np.array([report["folds"][fi]["weights"].get(names[ci],0.) for ci in selected])
            if weights.sum()<=0:continue
            mass=float(weights.sum());weights/=mass
            src.log("INNER_ADMISSION",{"fold":fold,"heads":len(selected),"weight_mass":mass,"families":sorted({family(names[i]) for i in selected})})
            # Outer control uses original upstream fitted nested weights, restricted to matched heads.
            inner=np.array([np.concatenate(blocks[ci],axis=1) for ci in selected])
            outer=cache["components"][selected][:,:,va,:]
            gmap={}
            for k,ci in enumerate(selected):gmap.setdefault(family(names[ci]),[]).append(k)
            for sc in range(3):
                a=softmax(np.einsum("k,knc->nc",weights,outer[:,sc]),axis=1)
                full=softmax(cache["nested_calibrated"][sc,va],axis=1)
                ftrain=[];ftest=[]
                for members in gmap.values():
                    ftrain.append(softmax(inner[members,sc].mean(0),axis=1))
                    ftest.append(softmax(outer[members,sc].mean(0),axis=1))
                static=np.mean(ftest,axis=0)
                evidence=np.mean([conditional(q,y[inner_ids],p,True) for q,p in zip(ftrain,ftest)],axis=0)
                outputs["subset_native"][sc,va]=a.argmax(1)
                outputs["subset_static"][sc,va]=((1-ALPHA)*a+ALPHA*static).argmax(1)
                outputs["subset_context"][sc,va]=((1-ALPHA)*a+ALPHA*evidence).argmax(1)
                outputs["full_nested_plus_context"][sc,va]=((1-ALPHA)*full+ALPHA*evidence).argmax(1)
                outputs["full_nested_plus_static"][sc,va]=((1-ALPHA)*full+ALPHA*static).argmax(1)
            folds_done+=1
        if folds_done!=5 or any((x<0).any() for x in outputs.values()):
            src.log("BLOCKED",{"folds_done":folds_done,"reason":"insufficient pinned score-aligned inner cache coverage"});return
        gains=[]
        for sc in range(3):
            base=cache["nested_calibrated"][sc].argmax(1)
            ans={}
            for k,ps in outputs.items():
                p=ps[sc];ans[k]={"ba":ba(y,p),"delta_full_nested":ba(y,p)-ba(y,base),"rescues":int(((base!=y)&(p==y)).sum()),"regressions":int(((base==y)&(p!=y)).sum())}
            gains.append(ans["full_nested_plus_context"]["delta_full_nested"])
            src.log("NATIVE_NESTED_RESULT",{"scenario":[1,3,6][sc],"full_nested":ba(y,base),"methods":ans})
        src.log("GATE",{"mean_gain":float(np.mean(gains)),"min_gain":float(min(gains)),"pass":bool(np.mean(gains)>=.005 and min(gains)>=-.005),"scope":"historical nested diagnostic, no proof of contemporary leaderboard gains","gpu":False,"neural_training":False,"submission":False})
if __name__=="__main__":main()
