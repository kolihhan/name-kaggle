"""NTL F1 fixed-prompt grouped decoder-layer probe, CPU scorer.

C0 fold-local char_wb(3,5)+balanced LR.
C1 four pooled layers of frozen Qwen3-1.7B-Base + balanced LR.
C2 C1 with training-fold prompt-centroid PCA nuisance projection k=8.
Reused official folds are DEV-only; never compare their BA numerically to
the 0.702 attributed Public champion or claim independent generalization.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLD_SHA="94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"
MODEL_ID="Qwen/Qwen3-1.7B-Base"
MODEL_REV="658b6ee46470b73f3e0e04e57fc9fb7031fd7daa"
VERSION="ntl-decoder-fourlayer-prompt-pca8-v1-20261119"
C=1.0
SEED=20261119
N_BOOT=5000
PCA_K=8
ARMS=("C0_char_lr","C1_decoder_lr","C2_decoder_prompt_pca8")

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_official(train,folds):
    if sha(train)!=TRAIN_SHA or sha(folds)!=FOLD_SHA:
        raise ValueError("PINNED_TRAIN_OR_FOLD_SHA_MISMATCH")
    a=pd.read_csv(train,dtype={"id":str,"prompt_id":str})
    b=pd.read_csv(folds,dtype={"id":str,"prompt_id":str})
    if a.id.duplicated().any() or b.id.duplicated().any():
        raise ValueError("DUPLICATE_IDS")
    f=a.merge(b,on=["id","prompt_id","model"],validate="one_to_one",
              how="outer",indicator=True)
    if (len(f)!=2699 or (f._merge!="both").any() or f.isna().any().any()
        or f.model.nunique()!=18 or f.prompt_id.nunique()!=150
        or sorted(f.fold_id.unique().tolist())!=list(range(5))
        or f.groupby("prompt_id").fold_id.nunique().ne(1).any()
        or f.groupby("fold_id").prompt_id.nunique().to_dict()!=dict.fromkeys(range(5),30)
        or f.duplicated(["prompt_id","model"]).any()):
        raise ValueError("OFFICIAL_GROUPED_FOLDS_INVALID")
    return f.sort_values("id").reset_index(drop=True)

def read_features(path,frame):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={"ids","features","metadata"}:
            raise ValueError("PRIVATE_FEATURE_SCHEMA_BAD")
        meta=json.loads(str(z["metadata"].item()))
        expected={"schema":VERSION,"model":MODEL_ID,"revision":MODEL_REV,
            "train_sha":TRAIN_SHA,"fold_sha":FOLD_SHA,"max_length":512,
            "pooling":"attention_mask_mean","row_count":len(frame)}
        for k,v in expected.items():
            if meta.get(k)!=v: raise ValueError("FEATURE_PROVENANCE_DRIFT_"+k)
        if len(meta.get("layer_indices",[]))!=4 or len(set(meta["layer_indices"]))!=4:
            raise ValueError("LAYER_INDICES_BAD")
        if z["ids"].astype(str).tolist()!=frame.id.tolist():
            raise ValueError("PRIVATE_EMBEDDING_ID_ALIGNMENT_FAILED")
        x=np.asarray(z["features"],dtype=np.float32)
    if x.ndim!=2 or x.shape[0]!=2699 or x.shape[1]<2048 or not np.isfinite(x).all():
        raise ValueError("PRIVATE_EMBEDDING_SHAPE_OR_NAN")
    return x

def classifier():
    return LogisticRegression(C=C,class_weight="balanced",solver="lbfgs",
                              max_iter=1200,random_state=SEED)

def training_only_project(train,valid,prompts,k):
    """Fit all scaling/PCA exclusively on training rows. Never use heldout prompt IDs."""
    scaler=StandardScaler()
    a=np.asarray(scaler.fit_transform(train),dtype=np.float32)
    b=np.asarray(scaler.transform(valid),dtype=np.float32)
    if k:
        unique=np.unique(prompts)
        if k>=len(unique):raise ValueError("NOT_ENOUGH_TRAIN_PROMPTS")
        means=np.stack([a[prompts==p].mean(axis=0) for p in unique])
        u=PCA(n_components=k,svd_solver="randomized",random_state=SEED).fit(means).components_
        u=np.asarray(u,dtype=np.float32)
        a=a-(a@u.T)@u
        b=b-(b@u.T)@u
    return a,b

def paired_ci(frame,y,old,new,labels,reps=N_BOOT):
    groups,inv=np.unique(frame.prompt_id.astype(str),return_inverse=True)
    idx={label:i for i,label in enumerate(labels)}
    n=len(groups);m=len(labels)
    den=np.zeros((n,m),np.int32)
    delta=np.zeros((n,m),np.int32)
    for j in range(len(y)):
        c=idx[y[j]]
        den[inv[j],c]+=1
        delta[inv[j],c]+=int(new[j]==y[j])-int(old[j]==y[j])
    rng=np.random.default_rng(SEED)
    values=[]
    for lo in range(0,reps,250):
        draw=rng.integers(0,n,size=(min(250,reps-lo),n))
        counts=den[draw].sum(axis=1)
        diffs=delta[draw].sum(axis=1)
        good=np.all(counts>0,axis=1)
        values.extend(np.mean(diffs[good]/counts[good],axis=1).tolist())
    if len(values)<.95*reps:raise RuntimeError("INSUFFICIENT_BOOTSTRAP_VALID_REPLICATES")
    return [float(x) for x in np.quantile(values,[.025,.975])]

def evaluate(frame,features,control_only=False):
    y=frame.model.astype(str).to_numpy()
    labels=sorted(set(y))
    arms=ARMS[:1] if control_only else ARMS
    preds={arm:np.full(len(frame),"",dtype=object) for arm in arms}
    by_fold=[]
    for fold in range(5):
        tr=np.flatnonzero(frame.fold_id.to_numpy()!=fold)
        va=np.flatnonzero(frame.fold_id.to_numpy()==fold)
        texts=frame.text.astype(str)
        tf=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,
            max_features=100000,sublinear_tf=True,dtype=np.float32)
        x=tf.fit_transform(texts.iloc[tr].tolist())
        z=tf.transform(texts.iloc[va].tolist())
        preds[ARMS[0]][va]=classifier().fit(x,y[tr]).predict(z)
        if not control_only:
            if features is None:raise ValueError("NEED_PRIVATE_FEATURES")
            for arm,k in ((ARMS[1],0),(ARMS[2],PCA_K)):
                a,b=training_only_project(features[tr],features[va],
                                          frame.prompt_id.to_numpy()[tr],k)
                preds[arm][va]=classifier().fit(a,y[tr]).predict(b)
        row={"fold":fold,"rows":int(len(va)),
             "prompt_groups":int(frame.prompt_id.iloc[va].nunique())}
        by_fold.append(row)
        print("NTL_F1_ACTUAL_FOLD_DONE="+json.dumps(row,sort_keys=True),flush=True)
    if any(np.any(p=="") for p in preds.values()):raise RuntimeError("INCOMPLETE_OOF")
    stats={}
    yfam=np.array([v.split("-")[0] for v in y])
    for arm,p in preds.items():
        pfam=np.array([str(v).split("-")[0] for v in p])
        stats[arm]={"ba":float(balanced_accuracy_score(y,p)),
            "correct":int((p==y).sum()),
            "recall":{c:float((p[y==c]==c).mean()) for c in labels},
            "wrong_cross_family":int(((p!=y)&(pfam!=yfam)).sum()),
            "wrong_within_family":int(((p!=y)&(pfam==yfam)).sum())}
    pairs={}
    if not control_only:
        for old,new in ((ARMS[0],ARMS[1]),(ARMS[1],ARMS[2]),(ARMS[0],ARMS[2])):
            a=preds[old]==y
            b=preds[new]==y
            pairs[new+"_minus_"+old]={
                "delta_ba":stats[new]["ba"]-stats[old]["ba"],
                "rescues":int((~a&b).sum()),
                "regressions":int((a&~b).sum()),
                "prompt_cluster_ci95":paired_ci(frame,y,preds[old],preds[new],labels)}
    if control_only:
        decision="ACTUAL_LEXICAL_CONTROL_ONLY_NO_DECODER_RESULT"
    else:
        d=pairs[ARMS[1]+"_minus_"+ARMS[0]]
        gain=d["delta_ba"]
        good_classes=sum(stats[ARMS[1]]["recall"][c]>=stats[ARMS[0]]["recall"][c]-.02 for c in labels)
        decision=("ADVANCE_ONLY_TO_INDEPENDENT_SOURCE_TRANSFER" if
                  gain>=.08 and d["prompt_cluster_ci95"][0]>=.04 and good_classes>=12
                  else "NO_COMPETITION_PROMOTION_FROM_REUSED_DEV")
        if gain<=.03 or d["prompt_cluster_ci95"][1]<=0:
            decision="STOP_FIXED_DECODER_LAYER_PROBE"
    return {"experiment":VERSION,"evidence":"SPENT_OFFICIAL_TRAIN_GROUPED_OOF_DEV_ONLY",
        "official_best_public_ba":0.702,"official_best_oof":"NOT_AVAILABLE",
        "rows":len(frame),"prompts":int(frame.prompt_id.nunique()),"classes":len(labels),
        "folds":by_fold,"fixed":{"LR_C":C,"PCA_k":PCA_K,"bootstrap":N_BOOT},
        "results":stats,"comparisons":pairs,"decision":decision,
        "new_independent_holdout":False,"no_public_submission":True,
        "no_champion_paired_score":True,"no_GPU_during_evaluation":True}

def self_test():
    rng=np.random.default_rng(SEED)
    p=np.repeat([f"p{i:02}" for i in range(24)],3)
    x=rng.normal(size=(72,15)).astype(np.float32)
    a,b=training_only_project(x[:60],x[60:],np.asarray(p[:60]),8)
    a2,b2=training_only_project(x[:60],x[60:]*3,np.asarray(p[:60]),8)
    assert np.array_equal(a,a2) and a.shape==(60,15) and b.shape==(12,15)
    y=np.tile(np.asarray(["a","b","c"]),24)
    frame=pd.DataFrame({"prompt_id":p})
    assert paired_ci(frame,y,y,y,["a","b","c"],100)==[0.,0.]
    print("NTL_F1_CPU_FOLD_LOCAL_PCA_AND_BOOTSTRAP_SELFTEST=PASS",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,default=Path("data/train.csv"))
    p.add_argument("--folds",type=Path,default=Path("artifacts/phase1/folds.csv"))
    p.add_argument("--features",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--real-control-only",action="store_true")
    a=p.parse_args()
    if a.self_test:self_test();return
    if not a.out:p.error("--out required")
    frame=load_official(a.train,a.folds)
    features=read_features(a.features,frame) if a.features else None
    if not a.real_control_only and features is None:
        p.error("--features required to score decoder C1/C2")
    r=evaluate(frame,features,a.real_control_only)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("NTL_F1_AGGREGATE="+json.dumps({
        "scores":{a:round(b["ba"],6) for a,b in r["results"].items()},
        "comparisons":r["comparisons"],"decision":r["decision"]},sort_keys=True),flush=True)

if __name__=="__main__":main()
