"""NTL Phase D: one fixed, prompt-centering causal test, no hidden test use.

Tests whether unlabeled prompt-group feature centering makes generator identity
more separable without enforcing one model per prompt. Evaluates oracle group,
content-inferred group, and sparse k=5 inferred group separately.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLD_SHA="94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"
WEIGHT=0.25
SEED=20261118
K_SPARSE=5

def load():
    files=[(Path("data/train.csv"),TRAIN_SHA),(Path("artifacts/phase1/folds.csv"),FOLD_SHA)]
    for path,expect in files:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expect:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    a=pd.read_csv(files[0][0],dtype={"id":str,"prompt_id":str})
    b=pd.read_csv(files[1][0],dtype={"id":str,"prompt_id":str})
    d=a.merge(b,on=["id","prompt_id","model"],how="outer",validate="one_to_one",indicator=True)
    if len(d)!=2699 or d._merge.ne("both").any():
        raise ValueError("Train/fold mismatch")
    d=d.sort_values("id").reset_index(drop=True)
    if d.prompt_id.nunique()!=150 or d.model.nunique()!=18 or d.groupby("prompt_id").fold_id.nunique().ne(1).any():
        raise ValueError("Prompt folds invalid")
    return d

def center(x, groups):
    unique,inverse=np.unique(np.asarray(groups),return_inverse=True)
    mean=np.zeros((len(unique),x.shape[1]))
    counts=np.bincount(inverse)
    np.add.at(mean,inverse,x)
    mean/=counts[:,None]
    return normalize(x-mean[inverse],norm="l2",axis=1)

def blind_clusters(text):
    if len(text)<2:return np.zeros(len(text),dtype=int)
    vectorizer=TfidfVectorizer(stop_words="english",ngram_range=(1,2),
                              min_df=1,max_features=100000,sublinear_tf=True)
    x=vectorizer.fit_transform(text)
    dist=cosine_distances(x)
    np.clip(dist,0,2,out=dist)
    return AgglomerativeClustering(n_clusters=None,distance_threshold=0.90,
            linkage="average",metric="precomputed").fit_predict(dist)

def fixed_subset(ids,prompts,k):
    selected=[]
    for prompt in np.unique(prompts):
        ind=np.flatnonzero(prompts==prompt)
        order=sorted(ind,key=lambda i:hashlib.sha256(
          ("sparse-20261118|"+str(ids[i])).encode()).digest())
        selected.extend(order[:k])
    return np.array(sorted(selected),dtype=int)

def ba(y,p):
    return float(balanced_accuracy_score(y,p))

def grouped_ci(y,ids,groups,base,alt,replicates=3000):
    labels=np.unique(y)
    gp,ix=np.unique(groups,return_inverse=True)
    n=len(gp)
    a=np.zeros((n,len(labels)),dtype=np.int16)
    b=np.zeros_like(a)
    for row,(i,label) in enumerate(zip(ix,y)):
        a[i,label]+=1
        b[i,label]+=int(alt[row]==label)-int(base[row]==label)
    rng=np.random.default_rng(SEED)
    draw=rng.multinomial(n,np.full(n,1/n),size=replicates)
    denominators=draw@a
    delta=np.mean(np.divide(draw@b,denominators,
             out=np.zeros_like(denominators,dtype=float),where=denominators>0),axis=1)
    return np.quantile(delta,[.025,.975]).tolist()

def experiment(d):
    labels,y=np.unique(d.model.to_numpy(),return_inverse=True)
    folds=d.fold_id.to_numpy()
    pred={a:np.full(len(d),-1,dtype=np.int16) for a in
         ("raw","oracle_residual","inferred_residual","inferred_blend")}
    sparse_results=[]
    ari=[]
    with threadpool_limits(limits=2):
      for fold in range(5):
        train=folds!=fold
        test=~train
        texts_train=d.text[train].tolist()
        texts_valid=d.text[test].tolist()
        tfidf=TfidfVectorizer(analyzer="char",lowercase=False,ngram_range=(2,5),
               min_df=2,max_features=30000,sublinear_tf=True)
        x=tfidf.fit_transform(texts_train)
        z=tfidf.transform(texts_valid)
        svd=TruncatedSVD(n_components=128,n_iter=5,random_state=SEED)
        x=normalize(svd.fit_transform(x),axis=1)
        z=normalize(svd.transform(z),axis=1)
        raw=LogisticRegression(C=3.0,max_iter=500,solver="lbfgs")
        residual=LogisticRegression(C=3.0,max_iter=500,solver="lbfgs")
        raw.fit(x,y[train])
        residual.fit(center(x,d.prompt_id[train].to_numpy()),y[train])
        inferred=blind_clusters(texts_valid)
        true=d.prompt_id[test].to_numpy()
        ari.append(float(adjusted_rand_score(true,inferred)))
        rprob=raw.predict_log_proba(z)
        iprob=residual.predict_log_proba(center(z,inferred))
        oprob=residual.predict_log_proba(center(z,true))
        idx=np.flatnonzero(test)
        pred["raw"][idx]=rprob.argmax(axis=1)
        pred["oracle_residual"][idx]=oprob.argmax(axis=1)
        pred["inferred_residual"][idx]=iprob.argmax(axis=1)
        # Only one preregistered mixture weight, no tuning.
        def standard(a):
            a=a-a.mean(axis=1,keepdims=True)
            return a/np.maximum(a.std(axis=1,keepdims=True),1e-6)
        pred["inferred_blend"][idx]=(
            (1-WEIGHT)*standard(rprob)+WEIGHT*standard(iprob)
        ).argmax(axis=1)
        sparse=fixed_subset(d.id[test].to_numpy(),true,K_SPARSE)
        sparse_cluster=blind_clusters([texts_valid[i] for i in sparse])
        sparse_prob=residual.predict_log_proba(center(z[sparse],sparse_cluster))
        sparse_raw=rprob[sparse]
        sparse_pred=(
            (1-WEIGHT)*standard(sparse_raw)+WEIGHT*standard(sparse_prob)
        ).argmax(axis=1)
        sparse_results.append((idx[sparse],sparse_pred,
           float(adjusted_rand_score(true[sparse],sparse_cluster))))
        print("NTL_PROMPT_RESIDUAL_FOLD",fold,"heldout",len(idx),
              "clusters",len(np.unique(inferred)),
              "ari",round(ari[-1],3),flush=True)
    if any(np.any(v<0) for v in pred.values()):
        raise ValueError("Missing predictions")
    b={a:ba(y,v) for a,v in pred.items()}
    metrics={}
    for arm in ("oracle_residual","inferred_residual","inferred_blend"):
        a=pred[arm]
        metrics[arm]={
            "ba":b[arm],
            "delta_vs_raw":b[arm]-b["raw"],
            "ci95":grouped_ci(y,d.id.to_numpy(),d.prompt_id.to_numpy(),pred["raw"],a),
            "rescued":int(((pred["raw"]!=y)&(a==y)).sum()),
            "regressed":int(((pred["raw"]==y)&(a!=y)).sum()),
        }
    ix=np.concatenate([x[0] for x in sparse_results])
    sparse_pred=np.concatenate([x[1] for x in sparse_results])
    sparse_b=pred["raw"][ix]
    sparse_y=y[ix]
    sparse={
        "k":K_SPARSE,"rows":int(len(ix)),
        "raw_ba":ba(sparse_y,sparse_b),
        "inferred_blend_ba":ba(sparse_y,sparse_pred),
        "delta_vs_raw":ba(sparse_y,sparse_pred)-ba(sparse_y,sparse_b),
        "ci95":grouped_ci(sparse_y,d.id.to_numpy()[ix],
                          d.prompt_id.to_numpy()[ix],sparse_b,sparse_pred),
        "mean_cluster_ari":float(np.mean([x[2] for x in sparse_results]))
    }
    return {
        "version":"ntl-phase-d-prompt-residual-20261118",
        "train_sha":TRAIN_SHA,"fold_sha":FOLD_SHA,
        "raw_ba":b["raw"],"full_group_ari_mean":float(np.mean(ari)),
        "metrics":metrics,"sparse":sparse,
        "grouped_folds":5,"rows":len(d),"prompts":d.prompt_id.nunique(),
        "classes":len(labels),"blend_weight":WEIGHT,"bootstrap":3000,
        "intervention":"within-prompt center SVD(128) char features; infer test prompt groups using unlabeled word TFIDF + average-linkage distance .90; no Hungarian matching",
        "pre_mortem":"hidden test group structure may be sparse, partial, or variable per model; oracle means are unattainable; evaluate inferred and sparse separately",
        "source_classification":"development proxy, not incumbent 0.691 or Public scoring",
        "scoring_run":False,"gpu":False,"kaggle_submission":False
    }

if __name__=="__main__":
    result=experiment(load())
    Path("result").mkdir(exist_ok=True)
    Path("result/ntl-prompt-residual-v1.json").write_text(json.dumps(result,indent=2))
    print("NTL_PROMPT_RESIDUAL_RESULT="+json.dumps(result,sort_keys=True),flush=True)
