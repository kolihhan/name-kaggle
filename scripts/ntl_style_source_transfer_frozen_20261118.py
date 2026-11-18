"""NTL: one frozen style signal source-transfer test; no new feature invention.

C0: char_wb TFIDF -> class-balanced LinearSVC.
C1: exact same model/training samples + previously defined 155 style features,
    standardized on TRAIN only, clipped and per-row L2 normalized (weight=0.5).
Null: shuffle style rows separately in train/eval. NO parameter sweep.
Both external source-disjoint and historic official GroupKFold are measured.
Data source model IDs are publisher asserted, not independently confirmed.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from scripts.run_style_baseline import extract_style_features
from scripts.ntl_source_shift_linear_control_20261117 import make_weights
from scripts.ntl_prompt_residual_sparse_v1_20261118 import grouped_ci

WEIGHT=.5
SEED=20261118
STRONG_SOURCE_MIN_DELTA=.03
TRAIN_HASH="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLD_HASH="94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"

def style(train,test):
    x=extract_style_features(train)
    v=extract_style_features(test)
    scaler=StandardScaler()
    x=np.clip(scaler.fit_transform(x),-4,4)
    v=np.clip(scaler.transform(v),-4,4)
    return normalize(x),normalize(v)

def make_estimator():
    return LinearSVC(C=.5,random_state=SEED,dual="auto",max_iter=3000)

def evaluate(a,b,fold):
    txta=a.text.astype(str).tolist()
    txtb=b.text.astype(str).tolist()
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),
                        min_df=3,max_features=180000,sublinear_tf=True,
                        dtype=np.float32)
    x=vec.fit_transform(txta); z=vec.transform(txtb)
    if x.shape[1]<2000:raise ValueError("char features collapsed")
    weight=make_weights(a,False) if "source" in a else np.ones(len(a),dtype=float)
    target=a.model.to_numpy()
    base=make_estimator()
    base.fit(x,target,sample_weight=weight)
    bpred=base.predict(z)
    sx,sz=style(txta,txtb)
    c1=make_estimator()
    c1.fit(sparse.hstack([x,WEIGHT*sparse.csr_matrix(sx)],format="csr"),
           target,sample_weight=weight)
    cpred=c1.predict(sparse.hstack([z,WEIGHT*sparse.csr_matrix(sz)],format="csr"))
    # shuffled null has the same style dimension/variance but destroys relation
    # with generator identity. It is diagnostic only; never candidate selection.
    rng=np.random.default_rng(SEED+fold)
    shx=sx[rng.permutation(len(sx))]
    shz=sz[rng.permutation(len(sz))]
    nmod=make_estimator()
    nmod.fit(sparse.hstack([x,WEIGHT*sparse.csr_matrix(shx)],format="csr"),
             target,sample_weight=weight)
    npred=nmod.predict(sparse.hstack([z,WEIGHT*sparse.csr_matrix(shz)],format="csr"))
    return bpred,cpred,npred,int(x.shape[1])

def metrics(frame,base,style_pred,null_pred):
    truth=frame.model.to_numpy()
    labels=np.unique(truth)
    yy=np.searchsorted(labels,truth)
    bb=np.searchsorted(labels,base)
    cc=np.searchsorted(labels,style_pred)
    nn=np.searchsorted(labels,null_pred)
    baseline=float(balanced_accuracy_score(truth,base))
    candidate=float(balanced_accuracy_score(truth,style_pred))
    null=float(balanced_accuracy_score(truth,null_pred))
    return {
       "baseline_BA":baseline,"style_BA":candidate,
       "style_delta":candidate-baseline,
       "shuffled_style_BA":null,"shuffled_style_delta":null-baseline,
       "style_prompt_cluster_bootstrap_ci95":grouped_ci(
           yy,frame.index.to_numpy(),frame.prompt_id.to_numpy(),bb,cc,replicates=3000),
       "rescues":int(((base!=truth)&(style_pred==truth)).sum()),
       "regressions":int(((base==truth)&(style_pred!=truth)).sum()),
       "n":len(frame),"groups":frame.prompt_id.nunique(),
       "labels":len(labels)
    }

def source_transfer(root):
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["data_sha256"]!="5a3dc82288328abedf49b7dd5e866e5c69eb922f9d6008e6fd78cf5153c12332":
        raise ValueError("source manifest drift")
    for split in ("train","valid"):
        path=root/(split+".parquet")
        expected=manifest[split+"_sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(split+" data bytes drift")
    train=pd.read_parquet(root/"train.parquet")
    valid=pd.read_parquet(root/"valid.parquet").reset_index(drop=True)
    if len(train)!=5562 or len(valid)!=2700 or train.model.nunique()!=18 or valid.model.nunique()!=18:
        raise ValueError("source frozen 18-label split drift")
    if set(train.source)&set(valid.source) or set(train.prompt_id)&set(valid.prompt_id):
        raise ValueError("invalid source/prompt separation")
    bp,cp,npred,d=evaluate(train,valid,10)
    outcome=metrics(valid,bp,cp,npred)
    per={}
    for name,frame in valid.groupby("source"):
        ids=frame.index.to_numpy()
        y=valid.model.to_numpy()[ids]
        lab=np.unique(y)
        change=float(balanced_accuracy_score(y,cp[ids])-balanced_accuracy_score(y,bp[ids]))
        per[name]={"n":len(ids),"labels":len(lab),"delta":change}
    outcome["per_source"]=per
    outcome["n_char_features"]=d
    return outcome

def official_grouped():
    for path,expected in ((Path("data/train.csv"),TRAIN_HASH),
                          (Path("artifacts/phase1/folds.csv"),FOLD_HASH)):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError("official input hash drift")
    a=pd.read_csv("data/train.csv",dtype={"id":str,"prompt_id":str})
    b=pd.read_csv("artifacts/phase1/folds.csv",dtype={"id":str,"prompt_id":str})
    d=a.merge(b,on=["id","prompt_id","model"],validate="one_to_one",indicator=True)
    d=d.sort_values("id").reset_index(drop=True)
    if len(d)!=2699 or not d._merge.eq("both").all() or d.groupby("prompt_id").fold_id.nunique().ne(1).any():
        raise ValueError("invalid frozen group split")
    preds={k:np.full(len(d),"",dtype=object) for k in ("base","style","null")}
    dimensions=[]
    for fold in range(5):
        train=d[d.fold_id!=fold]
        valid=d[d.fold_id==fold]
        y0,y1,yn,num=evaluate(train,valid,fold)
        idx=valid.index.to_numpy()
        for key,values in (("base",y0),("style",y1),("null",yn)):
            preds[key][idx]=values
        dimensions.append(num)
        print("NTL_STYLE_PIVOT_FOLD",fold,"n",len(idx),
              "base",round(balanced_accuracy_score(valid.model,y0),5),
              "style",round(balanced_accuracy_score(valid.model,y1),5),flush=True)
    if any(np.any(arr=="") for arr in preds.values()):
        raise ValueError("incomplete OOF")
    out=metrics(d,preds["base"],preds["style"],preds["null"])
    out["folds"]=5
    out["char_features_per_fold"]=dimensions
    return out

def run(root):
    with threadpool_limits(limits=2):
        ext=source_transfer(root)
        off=official_grouped()
    positive_sources=sum(v["delta"]>0 for v in ext["per_source"].values())
    gate=(
        ext["style_delta"]>=STRONG_SOURCE_MIN_DELTA
        and ext["style_prompt_cluster_bootstrap_ci95"][0]>0
        and positive_sources>=2
        and ext["style_delta"]>ext["shuffled_style_delta"]
        and off["style_delta"]>=-0.01
    )
    return {
      "protocol":"ntl-style-source-transfer-frozen-v1-20261118",
      "change":"add existing frozen 155-d style features with one fixed weight .5",
      "source_transfer":ext,"official_spent_group_oof":off,
      "preregistered_gate":"external delta >=.03 and group CI lower>0 and >=2/3 source positive and beats shuffled null and official delta>=-.01",
      "gate_pass":bool(gate),
      "external_model_identity_independently_verified":False,
      "source_holdout_not_public_score":True,
      "no_new_data_selected_by_result":True,
      "no_gpu":True,"no_competition_submission":True,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("result/ntl-style-source-transfer.json"))
    a=p.parse_args()
    report=run(a.data_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True))
    # Aggregate only, never export unseen source texts or row-level prediction data.
    print("NTL_STYLE_SOURCE_TRANSFER_RESULT="+json.dumps(report,sort_keys=True),flush=True)
