"""Reconstruct PB54/115 features from retained score arrays, not LLM calls.
Formula attribution: public Ozertuu v79, feature contract runs 46099762333/46100778697.
Never execute a downloaded pickle or donor source.
"""
import hashlib,io,json,pickletools,re,tempfile,time,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from numpy.lib import format as fmt
from sklearn.metrics import roc_auc_score
from scripts.ntl_pb115_paired_ablation_20261128 import HASHES,check,run,fit,log

CACHE_HASH={"flash":"05dc6e5d51d9ef0d4f9aad3ff1727560272e9e991deeb1b5efcc352b51d5d16c","glm53":"6cc84f5dbb74c7435b64d7a6dc2a14afacbcc48ec8b91ae03c30df335f4de6d6"}
PANEL_SHA="bce34670f4d1170375e7baf85eec5572d5f0ca6cf1573ce46e81c5a45c"
TOPICS={"short":(98,25,65,83,16,61,68),"median":(77,54,99,37),"long":(72,63,87)}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def id_static(blob,expected):
    b=io.BytesIO(blob);v=fmt.read_magic(b)
    if v==(1,0):shape,fort,dt=fmt.read_array_header_1_0(b)
    elif v==(2,0):shape,fort,dt=fmt.read_array_header_2_0(b)
    else:raise ValueError("unsupported npy version")
    assert shape==(75,) and dt==np.dtype("O") and not fort
    ops=list(pickletools.genops(b.read()))
    starts=[i for i,(op,arg,pos) in enumerate(ops) if op.name=="EMPTY_LIST"]
    ends=[i for i,(op,arg,pos) in enumerate(ops) if op.name=="APPENDS"]
    assert len(starts)==len(ends)==1 and starts[0]<ends[0]
    payload=ops[starts[0]+1:ends[0]]
    strings=("SHORT_BINUNICODE","BINUNICODE","BINUNICODE8")
    assert all(op.name in ("MARK","MEMOIZE","BINPUT","LONG_BINPUT",*strings) for op,arg,pos in payload)
    ids=[arg for op,arg,pos in payload if op.name in strings]
    assert sum(op.name=="MARK" for op,arg,pos in payload)==1
    assert len(ids)==len(set(ids))==75 and set(ids)<=expected
    assert all(re.fullmatch(r"fable-observation-\d{3}",i) for i in ids)
    assert [op.name for op,arg,pos in ops[ends[0]+1:] if op.name not in ("MEMOIZE","BINPUT","LONG_BINPUT")]==["TUPLE","BUILD","STOP"]
    return ids
def load_cache(path,tag,expected):
    assert digest(path)==CACHE_HASH[tag]
    with zipfile.ZipFile(path) as z:
        assert z.getinfo("ids.npy").file_size<100000
        ids=id_static(z.read("ids.npy"),expected)
    with np.load(path,allow_pickle=False) as z:
        off=z["offsets"];s=z["scores"];tokens=z["tokens"];prefix=z["n_prefix"]
    assert off.shape==(76,) and off[0]==0 and off[-1]==len(s)==len(tokens) and np.all(np.diff(off)>0)
    assert s.shape==(len(tokens),4) and np.isfinite(s).all() and np.all(s[:,3]>=0)
    assert prefix.shape==(75,) and np.all(prefix<=np.diff(off))
    return {k:(s[off[i]:off[i+1]],tokens[off[i]:off[i+1]],int(prefix[i])) for i,k in enumerate(ids)}
def split(s):
    lp,h,m2,r=(s[:,i].astype(float) for i in range(4))
    var=np.maximum(m2-h*h,1e-9)
    return lp,h,r,var,(lp+h)/np.sqrt(var)
def seq(s,n_chars):
    lp,h,r,v,z=split(s);n=len(s);half=max(n//2,1)
    d={"n_tok":n,"lp_mean":lp.mean(),"lp_std":lp.std(),"lp_min":lp.min(),
       "ent_mean":h.mean(),"ent_std":h.std(),"fdg":(lp+h).sum()/np.sqrt(v.sum()),
       "surprise_mean":(-lp-h).mean(),"z_mean":z.mean(),"logrank_mean":np.log1p(r).mean(),
       "top1":(r==0).mean(),"top5":(r<5).mean(),"top20":(r<20).mean(),"rank_ge100":(r>=100).mean(),
       "lp_mean_first":lp[:half].mean(),"lp_mean_second":lp[half:].mean() if n>half else lp.mean(),
       "bpc":-lp.sum()/max(n_chars,1)/np.log(2),"tok_per_char":n/max(n_chars,1)}
    d.update(zip(("lp_q05","lp_q10","lp_q25","lp_q50","lp_q75","lp_q90"),np.quantile(lp,[.05,.1,.25,.5,.75,.9])))
    return d
def extra(s,tag):
    lp,h,r,v,z=split(s);k=min(32,len(s));hi=h>1.5;lo=~hi
    def m(x,mask):return x[mask].mean() if mask.any() else x.mean()
    d={"lp_head32":lp[:k].mean(),"lp_tail32":lp[-k:].mean(),
       "fdg_head32":(lp[:k]+h[:k]).sum()/np.sqrt(v[:k].sum()),
       "fdg_tail32":(lp[-k:]+h[-k:]).sum()/np.sqrt(v[-k:].sum()),
       "lp_hi_ent":m(lp,hi),"z_hi_ent":m(z,hi),"top1_hi_ent":m(r==0,hi),
       "logrk_hi_ent":m(np.log1p(r),hi),"lp_lo_ent":m(lp,lo),"z_lo_ent":m(z,lo)}
    d.update(zip(("z_q10","z_q50","z_q90"),np.quantile(z,[.1,.5,.9])))
    return {tag+":"+k:v for k,v in d.items()}
def pair(f,g):
    assert len(f)==len(g)>0
    lf,hf,rf,vf,zf=split(f);lg,hg,rg,vg,zg=split(g)
    n=len(f);k=min(n,32);half=max(n//2,1);hi=(hf+hg)*.5>1.5
    a=lf-lg;b=np.log1p(rf)-np.log1p(rg);c=zf-zg;d=hf-hg
    def ts(x):return x.mean()*np.sqrt(n)/(x.std()+1e-6)
    def m(x,mask):return x[mask].mean() if mask.any() else x.mean()
    e={}
    for name,x in (("dlp",a),("dlrk",b),("dz",c),("dent",d)):
        e[name+"_mean"]=x.mean();e[name+"_std"]=x.std()
        if name!="dent":e[name+"_tstat"]=ts(x)
    e.update({"dlp_pos":(a>0).mean(),"dlp_big_pos":(a>1).mean(),"dlp_big_neg":(a<-1).mean(),
       "dlp_first":a[:half].mean(),"dlp_second":a[half:].mean() if n>half else a.mean(),
       "dlp_head32":a[:k].mean(),"dlp_tail32":a[-k:].mean(),"both_t1":((rf==0)&(rg==0)).mean(),
       "dz_pos":(c>0).mean(),"dent_pos":(d>0).mean(),"dlp_hi_ent":m(a,hi),
       "dlp_hi_ent_pos":m(a>0,hi),"dz_hi_ent":m(c,hi)})
    e.update(zip(("dlp_q10","dlp_q25","dlp_q50","dlp_q75","dlp_q90"),np.quantile(a,[.1,.25,.5,.75,.9])))
    for t in (1,5):
        x=((rf<t)&(rg>=t)).mean();y=((rg<t)&(rf>=t)).mean()
        e.update({f"excl_t{t}_f":x,f"excl_t{t}_g":y,f"excl_t{t}_diff":x-y})
    return {"p:"+k:v for k,v in e.items()}
def matrix(a,b,spec):
    f=pd.concat([a.add_prefix("flash:"),b.add_prefix("glm53:")],axis=1)
    for c in spec["diff_features"]:f["d:"+c]=f["flash:"+c]-f["glm53:"+c]
    return f[spec["features_54"]].to_numpy(float)
def main():
    start=time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);d=root/"bank";d.mkdir();o=root/"cache";o.mkdir();p=root/"panel";p.mkdir()
        run("datasets","download","-d","lumibee/ntl-prefixb-full-115","-p",str(d),"--unzip","-q")
        run("competitions","download","-c","name-that-llm","-f","train.csv","-p",str(root),"-q")
        trainfile=root/"train.csv"
        if not trainfile.exists():
            with zipfile.ZipFile(next(root.glob("*.zip"))) as z:trainfile.write_bytes(z.read("train.csv"))
        train=pd.read_csv(check(trainfile),dtype=str,usecols=["id","model"])
        assert len(train)==2699 and train.id.is_unique
        labs=sorted(train.model.unique());assert len(labs)==18
        y=pd.Index(labs).get_indexer(train.model);glm=[labs.index("glm-5.3"),labs.index("glm-5.3-flash")]
        paths={n:check(next(d.rglob(n))) for n in HASHES if n not in ("train.csv","folds.csv")}
        spec=json.loads(paths["spec.json"].read_text());assert spec["format"]=="ntl-prefixb-115-v1"
        def align(f):
            if "id" in f:f=f.set_index("id")
            f.index=f.index.astype(str)
            assert f.index.is_unique and set(f.index)==set(train.id)
            return f.loc[train.id]
        a=align(pd.read_csv(paths["flash_features.csv"],dtype={"id":str}))
        b=align(pd.read_csv(paths["glm53_features.csv"],dtype={"id":str}))
        q=align(pd.read_parquet(paths["paired_token_features.parquet"]))[spec["features_61"]]
        assert np.allclose(q["p:dlp_mean"],a.lp_mean-b.lp_mean,rtol=1e-4,atol=1e-5)
        x=matrix(a,b,spec);z=q.to_numpy(float)
        assert x.shape==(2699,54) and z.shape==(2699,61)
        run("kernels","output","kolihhan/ntl-fable-preservation-252-v2-20261125","-p",str(o),"--file-pattern",r"(^|/)(submission\.parquet|flash_scores\.npz|glm53_scores\.npz|report\.json)$","-o")
        run("datasets","download","-d","nischaydnk/ntl-topic-synthetic-18llms","-f","generations.csv","-p",str(p),"--unzip")
        source=next(p.rglob("generations.csv"));assert digest(source)==PANEL_SHA
        raw=pd.read_csv(source);blocks=[]
        for version,topics in TOPICS.items():
            for topic in topics:
                g=raw.loc[(raw.version==version)&(raw.topic_id==topic)].sort_values("model",kind="stable")
                assert len(g)==g.model.nunique()==18 and set(g.model)==set(labs)
                blocks.append(g[["text","model"]])
        panel=pd.concat(blocks,ignore_index=True)
        panel.insert(0,"id",[f"fable-observation-{i:03d}" for i in range(252)])
        panel=panel.set_index("id")
        assert panel.text.notna().all()
        pred=pd.read_parquet(next(o.rglob("submission.parquet"))).set_index("id")
        assert pred.index.is_unique and set(pred.index)==set(panel.index)
        assert int((pred.loc[panel.index,"model"]==panel.model).sum())==151
        r=json.loads(next(o.rglob("report.json")).read_text());assert r["status"]=="applied" and r["n_selected"]==75
        f=load_cache(next(o.rglob("flash_scores.npz")),"flash",set(panel.index))
        g=load_cache(next(o.rglob("glm53_scores.npz")),"glm53",set(panel.index))
        assert set(f)==set(g) and sum(len(v[0]) for v in f.values())==6987
        ids=list(f)
        aa=[];bb=[];zz=[]
        for rid in ids:
            sf,tf,pf=f[rid];sg,tg,pg=g[rid]
            assert np.array_equal(tf,tg) and pf==pg
            nc=len(panel.loc[rid,"text"])
            af=seq(sf,nc);ag=seq(sg,nc);ef=extra(sf,"flash");eg=extra(sg,"glm53");ep=pair(sf,sg)
            assert np.isclose(ep["p:dlp_mean"],af["lp_mean"]-ag["lp_mean"],atol=1e-10)
            aa.append(af);bb.append(ag);zz.append({**ef,**eg,**ep})
        xx=matrix(pd.DataFrame(aa),pd.DataFrame(bb),spec)
        zf=pd.DataFrame(zz);assert set(zf.columns)==set(spec["features_61"])
        zx=zf[spec["features_61"]].to_numpy(float)
        assert xx.shape==(75,54) and zx.shape==(75,61)
        log("RECONSTRUCTED",{"rows":75,"positions":6987,"features":115,"finite":bool(np.isfinite(xx).all() and np.isfinite(zx).all()),"ids_and_tokens_verified":True,"pickle_executed":False})
        assert np.isfinite(x).all() and np.isfinite(z).all() and np.isfinite(xx).all() and np.isfinite(zx).all()
        # Test rows have no labels during fitting; only train indices are used.
        ys=np.r_[y,np.full(75,-1)]
        tr=np.arange(len(y));va=np.arange(len(y),len(y)+75)
        p54=fit(np.vstack([x,xx]),ys,tr,va)
        p115=fit(np.vstack([np.column_stack([x,z]),np.column_stack([xx,zx])]),ys,tr,va)
        truth=pd.Index(labs).get_indexer(panel.loc[ids,"model"])
        inc=pd.Index(labs).get_indexer(pred.loc[ids,"model"])
        assert min(truth)>=0 and min(inc)>=0
        hit0=p54.argmax(1)==truth;hit1=p115.argmax(1)==truth
        support=np.unique(truth)
        def ba(h):return float(np.mean([h[truth==k].mean() for k in support]))
        def stats(p):
            h=p.argmax(1)==truth;pg=p[:,glm]
            return {"correct":int(h.sum()),"supported_class_BA":ba(h),"glm_auc":float(roc_auc_score(np.isin(truth,glm),pg.sum(1))),"glm_sibling_BA":float(np.mean([(pg[truth==k].argmax(1)==i).mean() for i,k in enumerate(glm)]))}
        log("PB115_RETAINED_TRANSFER",{"rows":75,"supported_labels":len(support),"baseline54":stats(p54),"candidate115":stats(p115),
          "rescues":int((~hit0&hit1).sum()),"regressions":int((hit0&~hit1).sum()),"v36_hard_label_correct":int((inc==truth).sum()),
          "by_label":{labs[k]:{"n":int((truth==k).sum()),"old_correct":int((hit0&(truth==k)).sum()),"new_correct":int((hit1&(truth==k)).sum())} for k in support},
          "scope":"selected spent publisher panel, standalone heads only, not v36 stage ablation","full_native_logits_available":False,
          "external_fit":False,"gpu":False,"submission":False,"seconds":round(time.monotonic()-start,2)})
if __name__=="__main__":main()
