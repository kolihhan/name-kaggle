"""One pre-registered source-shortcut falsifier, CPU only.

Model class fixed: TF-IDF char_wb + existing 155 style features + LinearSVC.
Treatment: remove exactly 5% of char features ranked by within-model,
between-publisher variation estimated ONLY on frozen training publishers.
Null: remove exactly 5% random char features, same frozen classifier recipe.
No inference-time source ID, new backbone, GPU, Kaggle scoring, or sweeps.

Note: the 2700-row external validation split has been used by prior studies;
this is mechanistic diagnostic, NOT a fresh independent confirmation.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from scripts.run_style_baseline import extract_style_features
from scripts.ntl_source_shift_linear_control_20261117 import make_weights
from scripts.ntl_prompt_residual_sparse_v1_20261118 import grouped_ci

SEED=20261118
REMOVE_FRACTION=.05
STYLE_SCALE=.5
EXPECTED_POOL="5a3dc82288328abedf49b7dd5e866e5c69eb922f9d6008e6fd78cf5153c12332"

def check(root):
    m=json.loads((root/"manifest.json").read_text())
    if m["data_sha256"] != EXPECTED_POOL:raise ValueError("pool changed")
    for name,n,ns in (("train",5562,8),("valid",2700,3)):
        file=root/(name+".parquet")
        if hashlib.sha256(file.read_bytes()).hexdigest()!=m[name+"_sha256"]:
            raise ValueError("frozen file checksum changed: "+name)
    train=pd.read_parquet(root/"train.parquet").reset_index(drop=True)
    valid=pd.read_parquet(root/"valid.parquet").reset_index(drop=True)
    if (len(train)!=5562 or len(valid)!=2700
            or train.source.nunique()!=8 or valid.source.nunique()!=3
            or train.model.nunique()!=18 or valid.model.nunique()!=18
            or set(train.source)&set(valid.source)
            or set(train.prompt_id)&set(valid.prompt_id)):
        raise ValueError("frozen source/label/prompt split changed")
    return m,train,valid

def nuisance_rank(x,model,source):
    # Within each model class, measure changes across known training sources.
    # Do not favor frequent source-only class priors: equal per-class averaging.
    labels=np.unique(model)
    score=np.zeros(x.shape[1],dtype=np.float64)
    support={}
    for label in labels:
        ids=np.flatnonzero(model==label)
        origin=source[ids]
        publishers=np.unique(origin)
        support[str(label)]=len(publishers)
        if len(publishers)<2:continue
        avg=np.asarray(x[ids].mean(axis=0)).reshape(-1)
        class_score=np.zeros_like(score)
        for publisher in publishers:
            part=ids[origin==publisher]
            if len(part)<3:continue
            subavg=np.asarray(x[part].mean(axis=0)).reshape(-1)
            class_score+=(len(part)/len(ids))*(subavg-avg)**2
        score+=class_score/len(labels)
    if not np.all(np.isfinite(score)) or not np.any(score>0):
        raise ValueError("Nuisance rank invalid")
    return score,support

def fit_predict(x,z,style_train,style_valid,y,weight,keep):
    a=sparse.hstack([x[:,keep],STYLE_SCALE*sparse.csr_matrix(style_train)],format="csr")
    b=sparse.hstack([z[:,keep],STYLE_SCALE*sparse.csr_matrix(style_valid)],format="csr")
    model=LinearSVC(C=.5,random_state=SEED,dual="auto",max_iter=3000)
    model.fit(a,y,sample_weight=weight)
    return model.predict(b)

def evaluate(root):
    manifest,train,valid=check(root)
    txt_train=train.text.astype(str).tolist()
    txt_valid=valid.text.astype(str).tolist()
    vectorizer=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),
               min_df=3,max_features=180000,sublinear_tf=True,dtype=np.float32)
    xx=vectorizer.fit_transform(txt_train)
    zz=vectorizer.transform(txt_valid)
    sc=StandardScaler()
    sx=normalize(np.clip(sc.fit_transform(extract_style_features(txt_train)),-4,4))
    sz=normalize(np.clip(sc.transform(extract_style_features(txt_valid)),-4,4))
    labels=train.model.to_numpy()
    source=train.source.to_numpy()
    weights=make_weights(train,False)
    with threadpool_limits(limits=2):
      score,support=nuisance_rank(xx,labels,source)
      nf=xx.shape[1]
      k=int(round(nf*REMOVE_FRACTION))
      remove=np.argsort(-score,kind="stable")[:k]
      rng=np.random.default_rng(SEED)
      null_remove=rng.choice(nf,size=k,replace=False)
      full=np.arange(nf,dtype=int)
      keep_treat=np.setdiff1d(full,remove,assume_unique=True)
      keep_null=np.setdiff1d(full,null_remove,assume_unique=True)
      pred0=fit_predict(xx,zz,sx,sz,labels,weights,full)
      pred1=fit_predict(xx,zz,sx,sz,labels,weights,keep_treat)
      prednull=fit_predict(xx,zz,sx,sz,labels,weights,keep_null)
    truth=valid.model.to_numpy()
    unique,yi=np.unique(truth,return_inverse=True)
    pred_code=lambda x:np.searchsorted(unique,x)
    b0=float(balanced_accuracy_score(truth,pred0))
    b1=float(balanced_accuracy_score(truth,pred1))
    bn=float(balanced_accuracy_score(truth,prednull))
    source_delta={}
    for item,sub in valid.groupby("source"):
        rows=sub.index.to_numpy()
        ys=truth[rows]
        source_delta[str(item)]={
            "rows":len(rows),
            "supported_labels":len(set(ys)),
            "ba_baseline":float(balanced_accuracy_score(ys,pred0[rows])),
            "ba_candidate":float(balanced_accuracy_score(ys,pred1[rows])),
            "delta":float(balanced_accuracy_score(ys,pred1[rows])-
                          balanced_accuracy_score(ys,pred0[rows]))}
    delta=b1-b0
    ci=grouped_ci(yi,valid.index.to_numpy(),valid.prompt_id.to_numpy(),
                  pred_code(pred0),pred_code(pred1),replicates=3000)
    rescues=int(((pred0!=truth)&(pred1==truth)).sum())
    regressed=int(((pred0==truth)&(pred1!=truth)).sum())
    # Gate fixed before results: material >=3pp source-BA; grouped interval
    # not crossing 0; 2/3 source-positive; >same-size random feature mask.
    positive=sum(x["delta"]>0 for x in source_delta.values())
    passes=bool(delta>=.03 and ci[0]>0 and positive>=2 and
                delta>(bn-b0) and rescues>regressed)
    return {
       "protocol":"ntl-conditional-source-nuisance-pruning-20261118-v1",
       "test_type":"retrospective_source_disjoint_reused_holdout_diagnostic",
       "pool_sha":EXPECTED_POOL,
       "train_sha":manifest["train_sha256"],"valid_sha":manifest["valid_sha256"],
       "train_rows":len(train),"train_sources":train.source.nunique(),
       "valid_rows":len(valid),"valid_sources":valid.source.nunique(),
       "valid_prompt_groups":int(valid.prompt_id.nunique()),"classes":len(unique),
       "features":nf,"removed_features":k,"fraction":REMOVE_FRACTION,
       "style_existing_features":sx.shape[1],"style_scale":STYLE_SCALE,
       "class_source_support":support,
       "baseline_char_style_ba":b0,
       "source_pruned_char_style_ba":b1,
       "delta":delta,"group_bootstrap_ci95":ci,
       "random_pruned_char_style_ba":bn,"null_delta":bn-b0,
       "rescues":rescues,"regressions":regressed,
       "source_results":source_delta,
       "gate_rule":"delta>=0.03; group-bootstrap-lower>0; >=2/3 source positive; treatment>null; rescues>regressions",
       "gate_pass":passes,
       "full_group_holdout_reused_adaptively":True,
       "original_served_model_identity_independently_proven":False,
       "new_pretrained_backbone":False,
       "base_estimator_fixed":True,
       "no_hyperparameter_sweep":True,
       "no_gpu":True,"no_kaggle_submission":True,
       "no_public_score_inference":True
    }

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,required=True)
    args=ap.parse_args()
    result=evaluate(args.data_dir)
    Path("result").mkdir(exist_ok=True)
    Path("result/ntl-source-nuisance-prune-v1.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    print("NTL_SOURCE_NUISANCE_PRUNE_RESULT="+json.dumps(result,sort_keys=True),flush=True)
