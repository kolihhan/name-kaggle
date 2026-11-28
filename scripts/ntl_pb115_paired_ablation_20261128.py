"""PB115 fixed prompt-grouped paired ablation on spent official train.
CPU head fits only. No raw output artifacts; not current v36 evaluation.
"""
import hashlib,json,subprocess,tempfile,zipfile,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,log_loss
import lightgbm as lgb

HASHES={"train.csv":"40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707",
"folds.csv":"94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d",
"flash_features.csv":"35c2b383fb45d4797091d401f7e9d951051d68dbed8cb8eb6fd2c66ee5394f36",
"glm53_features.csv":"882b695788b06fdd5717bcab1cce663ea26628cecc6dcca237c3255f92a9817b",
"paired_token_features.parquet":"55860b37a488c5a94533524973666b25b2474df1649a86ba11f1c62f1d61210a",
"spec.json":"8b9513abdd1c07497a7a11daa3fbaa0c551bf22f120b51a974e8129124521488"}
def log(k,x):print(k+"="+json.dumps(x,sort_keys=True,default=str),flush=True)
def check(p):
    assert hashlib.sha256(p.read_bytes()).hexdigest()==HASHES[p.name],f"hash mismatch: {p.name}"
    return p
def run(*args):
    p=subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=180)
    if p.returncode:raise RuntimeError("download failed: "+next((v for v in ("403","404","429","401") if v in p.stderr),"unknown"))
def metrics(p,y,glm):
    c=p.argmax(1); hit=c==y
    m=np.isin(y,glm);g=p[:,glm].sum(1);s=p[:,glm].argmax(1)
    return {"BA":float(np.mean([hit[y==k].mean() for k in range(18)])),"correct":int(hit.sum()),
    "glm_family_auc":float(roc_auc_score(m,g)),
    "glm_sibling_BA":float(np.mean([(s[y==k]==i).mean() for i,k in enumerate(glm)])),
    "log_loss":float(log_loss(y,p,labels=np.arange(18))),
    "recall":[float(hit[y==k].mean()) for k in range(18)]}
def fit(x,y,tr,va):
    scaler=StandardScaler().fit(x[tr])
    a,b=scaler.transform(x[tr]),scaler.transform(x[va])
    lr=LogisticRegression(C=.025,max_iter=1000).fit(a,y[tr])
    assert np.array_equal(lr.classes_,np.arange(18))
    z=lr.predict_proba(b);q=[]
    for sd in (42,1042,2042):
        clf=lgb.LGBMClassifier(objective="multiclass",num_class=18,n_estimators=220,
            learning_rate=.03,num_leaves=15,min_child_samples=25,subsample=.8,
            colsample_bytree=.7,reg_alpha=1.,reg_lambda=5.,random_state=sd,verbosity=-1,n_jobs=4)
        clf.fit(a,y[tr])
        assert np.array_equal(clf.classes_,np.arange(18))
        q.append(clf.predict_proba(b))
    return .55*z+.45*np.mean(q,axis=0)
def ci_delta(y,prompt,old,new):
    groups,gi=np.unique(prompt,return_inverse=True)
    count=np.zeros((len(groups),18));gain=np.zeros_like(count)
    np.add.at(count,(gi,y),1)
    np.add.at(gain,(gi,y),new.astype(float)-old.astype(float))
    rng=np.random.default_rng(20261128)
    boot=[]
    for _ in range(2000):
        w=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))
        n=w@count
        if np.all(n>0):boot.append(float(np.mean((w@gain)/n)))
    assert len(boot)==2000
    return np.quantile(boot,[.025,.975]).tolist()
def main():
    start=time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);d=root/"features";d.mkdir()
        run("datasets","download","-d","lumibee/ntl-prefixb-full-115","-p",str(d),"--unzip","-q")
        run("competitions","download","-c","name-that-llm","-f","train.csv","-p",str(root),"-q")
        p=root/"train.csv"
        if not p.exists():
            with zipfile.ZipFile(next(root.glob("*.zip"))) as z:p.write_bytes(z.read("train.csv"))
        train=pd.read_csv(check(p),dtype=str,usecols=["id","prompt_id","model"])
        folds=pd.read_csv(check(Path("artifacts/phase1/folds.csv")),dtype=str)
        assert len(train)==len(folds)==2699 and train.id.is_unique and folds.id.is_unique
        train=train.merge(folds,on=["id","prompt_id","model"],validate="one_to_one")
        assert len(train)==2699 and train.model.nunique()==18 and train.prompt_id.nunique()==150
        assert train.groupby("prompt_id").fold_id.nunique().eq(1).all() and train.fold_id.nunique()==5
        labs=sorted(train.model.unique());y=pd.Index(labs).get_indexer(train.model)
        glm=[labs.index("glm-5.3"),labs.index("glm-5.3-flash")]
        paths={name:check(next(d.rglob(name))) for name in HASHES if name not in ("train.csv","folds.csv")}
        spec=json.loads(paths["spec.json"].read_text());assert spec["format"]=="ntl-prefixb-115-v1"
        def align(f):
            if "id" in f:f=f.set_index("id")
            f.index=f.index.astype(str)
            assert f.index.is_unique and set(f.index)==set(train.id)
            return f.loc[train.id]
        a=align(pd.read_csv(paths["flash_features.csv"],dtype={"id":str})).add_prefix("flash:")
        b=align(pd.read_csv(paths["glm53_features.csv"],dtype={"id":str})).add_prefix("glm53:")
        f=pd.concat([a,b],axis=1)
        for c in spec["diff_features"]:f["d:"+c]=f["flash:"+c]-f["glm53:"+c]
        x=f[spec["features_54"]].to_numpy(float)
        z=align(pd.read_parquet(paths["paired_token_features.parquet"]))[spec["features_61"]].to_numpy(float)
        assert x.shape==(2699,54) and z.shape==(2699,61) and np.isfinite(x).all() and np.isfinite(z).all()
        matrices=[x,np.column_stack([x,z])]
        probs=np.full((2,len(y),18),np.nan);seen=np.zeros(len(y),int)
        for fold in sorted(train.fold_id.unique()):
            va=np.flatnonzero(train.fold_id.to_numpy()==fold);tr=np.flatnonzero(train.fold_id.to_numpy()!=fold)
            assert not set(train.prompt_id.iloc[va])&set(train.prompt_id.iloc[tr])
            assert len(set(y[va]))==len(set(y[tr]))==18
            seen[va]+=1
            for j,xj in enumerate(matrices):
                t=time.monotonic();probs[j,va]=fit(xj,y,tr,va)
                log("PB115_FOLD",{"fold":fold,"features":xj.shape[1],"n":len(va),"seconds":round(time.monotonic()-t,2),**metrics(probs[j,va],y[va],glm)})
        assert np.all(seen==1) and np.isfinite(probs).all() and np.allclose(probs.sum(2),1)
        old,new=(p.argmax(1)==y for p in probs)
        res=[metrics(p,y,glm) for p in probs]
        ci=ci_delta(y,train.prompt_id.to_numpy(),old,new)
        delta=res[1]["BA"]-res[0]["BA"]
        per={lab:{"n":int((y==k).sum()),"rescues":int((~old&new&(y==k)).sum()),"regressions":int((old&~new&(y==k)).sum())} for k,lab in enumerate(labs)}
        log("PB115_PAIRED_RESULT",{"labels":labs,"rows":len(y),"prompts":150,"old54":res[0],"new115":res[1],"delta":delta,"prompt_cluster_ci":ci,"rescues":int((~old&new).sum()),"regressions":int((old&~new).sum()),"per_label":per,"diagnostic_positive":bool(delta>0 and ci[0]>0),"seconds":round(time.monotonic()-start,2),"scope":"spent 5-fold official development; standalone classifier comparison, not v36 or independent","cpu_head_fit":True,"llm_training":False,"gpu":False,"submission":False})
if __name__=="__main__":main()
