"""MPCD-v1: independent matched-prompt conditional attribution, CPU diagnostic.

Scientific contract: docs/research/ntl-mpcd-v1-contract-20261123.md
No Kaggle official submissions, hidden test inputs, 0.715 artifacts or weights.
All 3 arms are single-response linear classifiers trained only on historical
official train's prompt-disjoint 5-fold training partitions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot
from scripts.run_style_baseline import extract_style_features

SEED = 20261123
CLASS_COUNT = 18
SVD_DIM = 128
CHAR_FEATURES = 45000
STYLE_WEIGHT = .30
REG_C = 1.5
COLUMN_WEIGHT = .5
MAX_ITER = 65
ARMS = ("row_ce", "true_panel_ce", "fake_panel_ce")
DRAW_COUNT = 1000


def group_index(values: np.ndarray) -> list[np.ndarray]:
    unique, inv = np.unique(np.asarray(values).astype(str), return_inverse=True)
    return [np.flatnonzero(inv == group) for group in range(len(unique))]


def fake_panels(prompts, labels, *, fold: int):
    """Destroy same-prompt response pairing while keeping group/source grid.

    Shuffle prompt assignment within *each* source label. Exact group sizes
    and per-prompt/source incidence are unchanged. Held-out labels unused.
    """
    original=np.asarray(prompts).astype(str)
    y=np.asarray(labels)
    rng=np.random.default_rng(SEED + fold * 31)
    fake=original.copy()
    for cls in np.unique(y):
        idx=np.flatnonzero(y==cls)
        fake[idx]=rng.permutation(original[idx])
    if sorted(np.unique(original,return_counts=True)[1].tolist())!=sorted(
            np.unique(fake,return_counts=True)[1].tolist()):
        raise ValueError("fake group sizes changed")
    # Each prompt contains at most one response from each source.
    if len(set(zip(fake.tolist(), y.tolist())))!=len(fake):
        raise ValueError("fake group/source grid duplicates")
    return fake


def features(texts_train:list[str], texts_val:list[str]) -> tuple[np.ndarray,np.ndarray,dict]:
    tfidf=TfidfVectorizer(analyzer="char",lowercase=False,
                         ngram_range=(2,5),min_df=2,
                         max_features=CHAR_FEATURES,sublinear_tf=True,
                         dtype=np.float32)
    x=tfidf.fit_transform(texts_train)
    v=tfidf.transform(texts_val)
    if x.shape[1]<SVD_DIM+1:
        raise ValueError("unexpected tiny vocabulary")
    svd=TruncatedSVD(n_components=SVD_DIM,n_iter=4,
                     random_state=SEED)
    a=normalize(svd.fit_transform(x),axis=1).astype(np.float64)
    b=normalize(svd.transform(v),axis=1).astype(np.float64)
    style_a=extract_style_features(texts_train)
    style_b=extract_style_features(texts_val)
    scaler=StandardScaler()
    sa=np.clip(scaler.fit_transform(style_a),-4.,4.)
    sb=np.clip(scaler.transform(style_b),-4.,4.)
    if not np.isfinite(sa).all() or not np.isfinite(sb).all():
        raise ValueError("nonfinite style features")
    sa=STYLE_WEIGHT*normalize(sa,axis=1)
    sb=STYLE_WEIGHT*normalize(sb,axis=1)
    train=np.ascontiguousarray(np.hstack([a,sa]),dtype=np.float64)
    valid=np.ascontiguousarray(np.hstack([b,sb]),dtype=np.float64)
    return train,valid,{"char_vocab":int(x.shape[1]),
                        "svd_dim":SVD_DIM,"style_dim":int(sa.shape[1]),
                        "feature_width":int(train.shape[1]),
                        "svd_explained_variance":round(float(
                            svd.explained_variance_ratio_.sum()),5)}


def ce_loss_grad(
    flat:np.ndarray, X:np.ndarray, y:np.ndarray,
    groups:list[np.ndarray]|None, *, column_weight:float=COLUMN_WEIGHT,
    C:float=REG_C
)->tuple[float,np.ndarray]:
    """Analytic full-batch gradient of raw row CE + conditional panel CE.

    True-panel CE forces matching responses to compete during FIT, but the
    resulting model can score *individual* unseen texts without test groups.
    """
    n,d=X.shape
    K=CLASS_COUNT
    p=flat.reshape(d+1,K)
    w,b=p[:d],p[d]
    logits=X@w+b
    lse=logsumexp(logits,axis=1)
    row=float((lse-logits[np.arange(n),y]).mean())
    grad=softmax(logits,axis=1)
    grad[np.arange(n),y]-=1
    grad/=n
    conditional=0.
    if groups is not None:
        if not 0<column_weight<=1:
            raise ValueError("column weight must lie in (0,1]")
        cg=np.zeros_like(logits)
        for ix in groups:
            sub=logits[ix,:]
            ld=logsumexp(sub,axis=0)
            cnt=np.bincount(y[ix],minlength=K)
            conditional+=(cnt@ld - np.sum(sub[np.arange(len(ix)),y[ix]]))
            prob=np.exp(sub-ld[None,:])
            cg[ix,:]+=prob*cnt[None,:]
            cg[ix,y[ix]]-=1.
        conditional/=n
        grad+=column_weight*cg/n
    reg=.5*np.sum(w*w)/(C*n)
    cost=row+column_weight*conditional+reg
    dW=X.T@grad+w/(C*n)
    db=grad.sum(axis=0)
    return cost,np.vstack([dW,db]).ravel()


def fit(X,y,groups):
    n,d=X.shape
    init=np.zeros((d+1)*CLASS_COUNT,dtype=np.float64)
    t0=time.perf_counter()
    result=minimize(ce_loss_grad,init,args=(X,y,groups),method="L-BFGS-B",
        jac=True,options={"maxiter":MAX_ITER,"ftol":1e-8,"gtol":1e-5,
                          "maxls":20,"maxcor":10})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("nonfinite optimization result")
    w=result.x.reshape(d+1,CLASS_COUNT)
    return w,{"iterations":int(result.nit),"optimizer_success":bool(result.success),
              "optimizer_message":str(result.message)[:130],
              "objective":round(float(result.fun),6),
              "time_seconds":round(time.perf_counter()-t0,2)}


def numerical_gradient_fixture():
    rng=np.random.default_rng(41)
    X=rng.normal(size=(8,3))*.1
    y=np.array([0,1,2,3,0,1,2,3])
    groups=[np.array([0,1,2,3]),np.array([4,5,6,7])]
    theta=rng.normal(size=(X.shape[1]+1)*CLASS_COUNT)*.03
    for group_arm in (None,groups):
        loss,analytic=ce_loss_grad(theta,X,y,group_arm)
        for j in (0,7,29,41,71):
            delta=np.zeros_like(theta)
            delta[j]=1e-5
            pp=ce_loss_grad(theta+delta,X,y,group_arm)[0]
            mm=ce_loss_grad(theta-delta,X,y,group_arm)[0]
            approx=(pp-mm)/(2e-5)
            if not np.isclose(analytic[j],approx,atol=3e-7,rtol=1e-4):
                raise AssertionError(f"bad gradient idx={j} numeric={approx} analytic={analytic[j]}")
    # fake same-prompt still exactly one source per prompt.
    fake=fake_panels(
        np.array(["p1","p1","p1","p1","p2","p2","p2","p2"]),
        y,fold=0)
    assert len(fake)==8 and sorted(np.unique(fake,return_counts=True)[1])==[4,4]
    print("NTL_MPCD_ANALYTIC_GRADIENT_AND_PANEL_NULL_SELFTEST=PASS",flush=True)


def grouped_ci(y, base, candidate, prompts, *, seed=SEED, draws=DRAW_COUNT):
    g,inv=np.unique(prompts,return_inverse=True)
    K=CLASS_COUNT
    counts=np.zeros((len(g),K),dtype=np.int32)
    diff=np.zeros_like(counts)
    for row,(grp,cls) in enumerate(zip(inv,y)):
        counts[grp,cls]+=1
        diff[grp,cls]+=int(candidate[row]==cls)-int(base[row]==cls)
    rng=np.random.default_rng(seed)
    d=rng.multinomial(len(g),np.ones(len(g))/len(g),size=draws)
    denom=d@counts
    if np.any(denom==0):raise ValueError("bootstrap lacks 18-label support")
    delta=(d@diff/denom).mean(axis=1)
    return [round(float(v),7) for v in np.quantile(delta,[.025,.975])]


def evaluate():
    numerical_gradient_fixture()
    frame=load_snapshot(Path("data/train.csv"),
                        Path("artifacts/phase1/folds.csv"))
    classes,y=np.unique(frame.model.astype(str).to_numpy(),return_inverse=True)
    if len(classes)!=CLASS_COUNT or len(frame)!=2699:
        raise ValueError("official train contract broken")
    folds=frame.fold_id.to_numpy()
    prompts=frame.prompt_id.astype(str).to_numpy()
    predictions={name:np.full(len(frame),-1,dtype=np.int16) for name in ARMS}
    fold_results=[]
    t0=time.perf_counter()
    with threadpool_limits(limits=2):
        for fold in range(5):
            train=np.flatnonzero(folds!=fold)
            held=np.flatnonzero(folds==fold)
            if set(prompts[train])&set(prompts[held]):
                raise ValueError("prompt leakage")
            train_text=frame.text.iloc[train].astype(str).tolist()
            valid_text=frame.text.iloc[held].astype(str).tolist()
            X,V,info=features(train_text,valid_text)
            true_index=group_index(prompts[train])
            null_prompt=fake_panels(prompts[train],y[train],fold=fold)
            null_index=group_index(null_prompt)
            models={}
            diagnostics={}
            for arm,panels in [
                ("row_ce",None),
                ("true_panel_ce",true_index),
                ("fake_panel_ce",null_index)
            ]:
                coef,fit_data=fit(X,y[train],panels)
                models[arm]=(V@coef[:-1]+coef[-1]).argmax(axis=1)
                diagnostics[arm]=fit_data
                predictions[arm][held]=models[arm]
            foldscores={name:round(float(balanced_accuracy_score(y[held],v)),7)
                        for name,v in models.items()}
            fold_results.append({
                "fold":fold,"train_rows":len(train),"heldout_rows":len(held),
                "train_prompt_groups":len(true_index),
                "heldout_prompt_groups":int(len(set(prompts[held]))),
                "features":info,"BA18":foldscores,"optimizer":diagnostics,
            })
            print("NTL_MPCD_FOLD_RESULT="+json.dumps({
                "fold":fold,"BA18":foldscores,
                "features":info,"fit_seconds":{k:models["time_seconds"]
                                             for k,models in diagnostics.items()}
            },sort_keys=True),flush=True)
    if any((a<0).any() for a in predictions.values()):
        raise RuntimeError("missing OOF results")
    values={k:float(balanced_accuracy_score(y,v)) for k,v in predictions.items()}
    labels=classes.tolist()
    recall={name:recall_score(y,p,labels=np.arange(CLASS_COUNT),
               average=None,zero_division=0) for name,p in predictions.items()}
    comparisons={}
    for arm in ("true_panel_ce","fake_panel_ce"):
        pred=predictions[arm]
        base=predictions["row_ce"]
        comparisons[arm]={
            "delta_vs_row_ce":round(values[arm]-values["row_ce"],7),
            "ci95":grouped_ci(y,base,pred,prompts),
            "rescues":int(((base!=y)&(pred==y)).sum()),
            "regressions":int(((base==y)&(pred!=y)).sum()),
            "changed_predictions":int((base!=pred).sum()),
            "per_model_recall_delta":{
                label:round(float(recall[arm][i]-recall["row_ce"][i]),5)
                for i,label in enumerate(labels)},
        }
    true=predictions["true_panel_ce"]
    fake=predictions["fake_panel_ce"]
    comparisons["true_vs_fake"]={
        "delta_BA18":round(values["true_panel_ce"]-values["fake_panel_ce"],7),
        "ci95":grouped_ci(y,fake,true,prompts),
        "rescues":int(((fake!=y)&(true==y)).sum()),
        "regressions":int(((fake==y)&(true!=y)).sum()),
    }
    class_recall_fall=(
        recall["row_ce"]-recall["true_panel_ce"]
    )
    wins=sum(row["BA18"]["true_panel_ce"]>row["BA18"]["row_ce"]
             for row in fold_results)
    passed=bool(
        values["true_panel_ce"]-values["row_ce"]>=.02 and
        values["true_panel_ce"]-values["fake_panel_ce"]>=.01 and
        comparisons["true_panel_ce"]["ci95"][0]>0 and
        comparisons["true_vs_fake"]["ci95"][0]>0 and
        wins>=4 and
        np.max(class_recall_fall)<=.15
    )
    return {
        "experiment":"MPCD-v1 independent single-response matched-prompt conditional loss",
        "scope":"2699 official train, historical 150 prompts, frozen five-fold GroupKFold DEV",
        "same_rows_all_arms":True,"no_test_prompt_id_used":True,
        "no_0715_champion_artifacts_used":True,
        "not_fresh_external_transfer":True,"not_official_kaggle_submission":True,
        "arms":{
            k:{"OOF_BA18":round(v,7),
               "per_model_recall":{label:round(float(recall[k][i]),5)
                                   for i,label in enumerate(labels)}}
            for k,v in values.items()
        },
        "comparisons":comparisons,
        "folds":fold_results,
        "fold_wins_c1_over_c0":wins,
        "max_class_recall_drop_vs_c0":round(float(class_recall_fall.max()),6),
        "historical_dev_mechanism_gate_pass":passed,
        "official_best_BA":0.715,
        "not_comparable_to_0715_on_same_rows":True,
        "training_seconds":round(time.perf_counter()-t0,1),
        "CPU_only":True,
        "No_GPU":True,
        "Official_submission":False,
    }


if __name__=="__main__":
    result=evaluate()
    print("NTL_MPCD_V1_RESULT="+json.dumps(result,sort_keys=True),flush=True)
