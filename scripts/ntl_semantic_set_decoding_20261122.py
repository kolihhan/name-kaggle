"""NTL semantic-set decoding (SSD): pretrained semantic representation -> learned
same-prompt graph -> optional injective model assignment over confirmed groups.

A different inference mechanism than prompt residuals: it changes 18-class
decisions jointly across multiple texts, not a per-row representation.

TRAIN labels: fit only same-prompt pair scorer and 18-class local surrogate.
TEST labels: used only for retrospective measurement after all decisions.
No oracle prompt_id in production clustering/decoding. "oracle" arm is a
non-deployable ceiling, not a proposed competition submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot

SEED=20261122
MODEL="BAAI/bge-small-en-v1.5"  # MIT, 33.4M, pretrained ONNX CPU
MAX_GROUP=18
NEG_NEIGHBORS=6
NEG_QUANTILE=0.995
MIN_TAU=0.80
BOOT=1000
K_SPARSE=5


def embed(texts):
    from fastembed import TextEmbedding
    enc=TextEmbedding(model_name=MODEL, max_length=512, threads=2)
    z=np.stack(list(enc.embed(texts, batch_size=32))).astype(np.float32)
    z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
    if not np.isfinite(z).all() or z.shape[0]!=len(texts):
        raise ValueError("bad pretrained embedding extraction")
    return z


def features(semantic, lexical, lengths, i,j):
    sem=np.sum(semantic[i]*semantic[j],axis=1)
    lex=np.asarray(lexical[i].multiply(lexical[j]).sum(axis=1)).reshape(-1)
    ln=np.minimum(lengths[i],lengths[j])/np.maximum(np.maximum(lengths[i],lengths[j]),1)
    return np.column_stack([sem,lex,ln]).astype(np.float32)


def fit_linker(x, lexical, lengths, groups,seed):
    """Supervised pair relevance fit uses only TRAIN prompt IDs."""
    rng=np.random.default_rng(seed)
    unique,inv=np.unique(groups,return_inverse=True)
    if len(unique)<25: raise ValueError("insufficient source prompts")
    pos_a=[];pos_b=[]
    for k in range(len(unique)):
        ids=np.flatnonzero(inv==k)
        a,b=np.triu_indices(len(ids),1)
        pos_a.extend(ids[a].tolist())
        pos_b.extend(ids[b].tolist())
    pos_a=np.asarray(pos_a,dtype=int)
    pos_b=np.asarray(pos_b,dtype=int)

    sims=x@x.T
    np.fill_diagonal(sims,-2)
    neg_pairs=set()
    for i in range(len(groups)):
        candidates=np.flatnonzero(inv!=inv[i])
        if len(candidates)<NEG_NEIGHBORS:raise ValueError("too few cross-prompt negatives")
        scores=sims[i,candidates]
        local=np.argpartition(scores,-NEG_NEIGHBORS)[-NEG_NEIGHBORS:]
        for j in candidates[local]:
            neg_pairs.add((min(i,int(j)),max(i,int(j))))
    while len(neg_pairs)<len(pos_a):
        i,j=rng.integers(0,len(groups),size=2)
        if inv[i]!=inv[j]:neg_pairs.add((min(int(i),int(j)),max(int(i),int(j))))
    neg=np.asarray(sorted(neg_pairs),dtype=int)
    # Pair classes balanced for training; threshold from hard-negative tail,
    # not validation labels, so all subsequent fold scores are untouched.
    p=features(x,lexical,lengths,pos_a,pos_b)
    n=features(x,lexical,lengths,neg[:,0],neg[:,1])
    xx=np.vstack([p,n])
    yy=np.r_[np.ones(len(p),dtype=int),np.zeros(len(n),dtype=int)]
    clf=LogisticRegression(C=1.,max_iter=350,random_state=seed).fit(xx,yy)
    neg_prob=clf.predict_proba(n)[:,1]
    tau=max(MIN_TAU,float(np.quantile(neg_prob,NEG_QUANTILE)))
    tau=min(tau,0.999999)
    return clf,tau,{
        "train_positive_pairs":len(p),
        "train_hard_and_random_negative_pairs":len(n),
        "train_only_threshold":round(tau,6),
    }


def probability_matrix(clf,z,lexical,lengths):
    n=len(z)
    i,j=np.triu_indices(n,1)
    p=clf.predict_proba(features(z,lexical,lengths,i,j))[:,1]
    mat=np.eye(n,dtype=np.float32)
    mat[i,j]=p.astype(np.float32)
    mat[j,i]=p.astype(np.float32)
    return mat


def cluster_prob(mat,tau):
    if len(mat)==1: return np.zeros(1,dtype=int)
    d=np.clip(1.-mat,0.,1.).astype(np.float64)
    np.fill_diagonal(d,0)
    return AgglomerativeClustering(
        n_clusters=None,distance_threshold=1.-tau,
        linkage="complete",metric="precomputed"
    ).fit_predict(d)


def lexical_cluster(x):
    if x.shape[0]==1:return np.zeros(1,dtype=int)
    similarity=(x@x.T).toarray()
    d=np.clip(1-similarity,0,2)
    np.fill_diagonal(d,0)
    return AgglomerativeClustering(
        n_clusters=None,distance_threshold=.90,
        linkage="average",metric="precomputed"
    ).fit_predict(d)


def graph_stats(truth,assigned):
    a,b=np.triu_indices(len(truth),1)
    match=(truth[a]==truth[b])
    same=(assigned[a]==assigned[b])
    tp=np.count_nonzero(match&same)
    pred=np.count_nonzero(same)
    relevant=np.count_nonzero(match)
    precision=tp/pred if pred else 1.
    recall=tp/relevant if relevant else 0.
    f1=2*tp/(pred+relevant) if pred+relevant else 0.
    return {
        "ari":float(adjusted_rand_score(truth,assigned)),
        "precision":float(precision),
        "recall":float(recall),
        "pair_f1":float(f1),
        "predicted_clusters":int(len(set(assigned))),
        "predicted_multi_groups":int(sum(np.sum(assigned==k)>=2 for k in set(assigned)))
    }


def assign(scores,clusters,*, max_group=MAX_GROUP):
    """Each *predicted* group uses each source label at most once.

    No label counts, true prompt IDs, or actual test labels enter this method.
    Score comparisons are within a group; no model weights are refitted.
    """
    result=np.argmax(scores,axis=1).copy()
    for c in np.unique(clusters):
        idx=np.flatnonzero(clusters==c)
        if not (2<=len(idx)<=max_group):continue
        row,col=linear_sum_assignment(-scores[idx])
        result[idx[row]]=col
    return result


def subset_k(frame,positions,k=K_SPARSE):
    ids=frame.id.to_numpy().astype(str)
    g=frame.prompt_id.to_numpy().astype(str)
    selected=[]
    for name in np.unique(g[positions]):
        members=positions[g[positions]==name]
        rank=sorted(members,key=lambda i:hashlib.sha256(
            (str(SEED)+"|"+ids[i]).encode()).digest())
        selected.extend(rank[:k])
    return np.asarray(sorted(selected),dtype=int)


def bootstrap(y,base,new,prompts):
    keys,ix=np.unique(prompts,return_inverse=True)
    c=int(np.max(y))+1
    counts=np.zeros((len(keys),c),np.int16)
    delta=np.zeros_like(counts)
    for i,k in enumerate(y):
        counts[ix[i],k]+=1
        delta[ix[i],k]+=int(new[i]==k)-int(base[i]==k)
    rng=np.random.default_rng(SEED)
    out=[]
    for _ in range(BOOT):
        draw=rng.integers(0,len(keys),len(keys))
        denom=counts[draw].sum(0)
        if np.any(denom==0):continue
        out.append((delta[draw].sum(0)/denom).mean())
    if len(out)<.95*BOOT:raise ValueError("bootstrap support too weak")
    return [float(z) for z in np.quantile(out,[.025,.975])]


def evaluate(frame,z):
    labels,y=np.unique(frame.model.to_numpy(),return_inverse=True)
    prompts=frame.prompt_id.astype(str).to_numpy()
    folds=frame.fold_id.to_numpy()
    if len(frame)!=2699 or len(labels)!=18 or len(np.unique(prompts))!=150:
        raise ValueError("official training dimensions drift")
    ans={k:np.full(len(y),-1,dtype=np.int16)
         for k in ("raw","joint_semantic","joint_lexical","joint_random","joint_oracle")}
    outputs={"full":[],"k5":[]}
    rng=np.random.default_rng(SEED)
    for fold in range(5):
        tr,va=folds!=fold,folds==fold
        if set(prompts[tr]) & set(prompts[va]):
            raise ValueError("training and validation prompt overlap")
        texts_tr=frame.text[tr].astype(str).tolist()
        texts_va=frame.text[va].astype(str).tolist()
        tf=TfidfVectorizer(analyzer="word",ngram_range=(1,2),
          stop_words="english",min_df=2,max_features=50000,sublinear_tf=True)
        a=tf.fit_transform(texts_tr)
        b=tf.transform(texts_va)
        lens=np.asarray(frame.text.astype(str).str.len(),dtype=int)
        group=prompts[tr]
        clf,tau,fit=fit_linker(z[tr],a,lens[tr],group,SEED+fold)
        valp=probability_matrix(clf,z[va],b,lens[va])
        semantic=cluster_prob(valp,tau)
        lexical=lexical_cluster(b)
        char=TfidfVectorizer(analyzer="char",lowercase=False,
           ngram_range=(2,5),min_df=2,max_features=45000,
           sublinear_tf=True)
        ca=char.fit_transform(texts_tr)
        cb=char.transform(texts_va)
        svc=LinearSVC(C=.5,class_weight="balanced",random_state=SEED,
          max_iter=3000,dual="auto").fit(ca,y[tr])
        scores=np.asarray(svc.decision_function(cb))
        original=scores.argmax(1)
        truegroup=prompts[va]
        # Oracle is a diagnostic upper bound, never a deployable arm.
        randomperm=rng.permutation(len(original))
        randomgroups=semantic[randomperm]
        trial={
            "raw":original,
            "joint_semantic":assign(scores,semantic),
            "joint_lexical":assign(scores,lexical),
            "joint_random":assign(scores,randomgroups),
            "joint_oracle":assign(scores,truegroup),
        }
        indices=np.flatnonzero(va)
        for key,prd in trial.items():
            ans[key][indices]=prd
        outputs["full"].append({
            "fold":fold,"pairwise_fit":fit,
            "semantic":graph_stats(truegroup,semantic),
            "lexical":graph_stats(truegroup,lexical),
            "ba":{k:float(balanced_accuracy_score(y[va],v)) for k,v in trial.items()}
        })
        sparse_abs=subset_k(frame,indices)
        local=np.searchsorted(indices,sparse_abs)
        sparse_g=prompts[sparse_abs]
        ps=valp[np.ix_(local,local)]
        sparse_sem=cluster_prob(ps,tau)
        sparse_lex=lexical_cluster(b[local])
        sr=scores[local]
        rr=sparse_sem[rng.permutation(len(local))]
        sparse_trial={
            "raw":sr.argmax(1),
            "joint_semantic":assign(sr,sparse_sem),
            "joint_lexical":assign(sr,sparse_lex),
            "joint_random":assign(sr,rr),
            "joint_oracle":assign(sr,sparse_g),
        }
        outputs["k5"].append({
            "fold":fold,"rows":len(local),
            "semantic":graph_stats(sparse_g,sparse_sem),
            "lexical":graph_stats(sparse_g,sparse_lex),
            "ba":{k:float(balanced_accuracy_score(y[sparse_abs],v))
                  for k,v in sparse_trial.items()},
        })
        print("NTL_SEMANTIC_SET_REAL_FOLD_DONE="+str(fold),flush=True)
    if any(np.any(v<0) for v in ans.values()):
        raise ValueError("OOF model predictions not complete")
    ba={k:float(balanced_accuracy_score(y,v)) for k,v in ans.items()}
    cmp={}
    for key in ("joint_semantic","joint_lexical","joint_random","joint_oracle"):
        before=ans["raw"]==y
        after=ans[key]==y
        cmp[key]={
            "delta_ba":ba[key]-ba["raw"],
            "prompt_ci95":bootstrap(y,ans["raw"],ans[key],prompts),
            "rescues":int((~before&after).sum()),
            "regressions":int((before&~after).sum())
        }
    full=np.mean([r["semantic"]["pair_f1"] for r in outputs["full"]])
    sparse=np.mean([r["semantic"]["pair_f1"] for r in outputs["k5"]])
    k5p=np.mean([r["semantic"]["precision"] for r in outputs["k5"]])
    k5r=np.mean([r["semantic"]["recall"] for r in outputs["k5"]])
    # Predeclared high specificity: avoid invalid hidden-test assumptions.
    gate=(k5p>=.95 and k5r>=.60 and sparse>=.70 and full>=.70
      and cmp["joint_semantic"]["delta_ba"]>=.02
      and cmp["joint_semantic"]["prompt_ci95"][0]>0
      and cmp["joint_semantic"]["delta_ba"]>cmp["joint_random"]["delta_ba"])
    return {
      "method":"pretrained BGE prompt-linker plus group-injective 18-way decoding",
      "representation":MODEL,
      "target":"single text->model output becomes jointly decoded set assignment",
      "scope":"reused 2699 official rows and 150 prompt-disjoint groups: DEVELOPMENT",
      "test_group_requirement":"none: clusters discovered unlabeled",
      "oracle_only_upper_bound_not_deployable":True,
      "rows":len(y),"prompts":len(np.unique(prompts)),"classes":len(labels),
      "full_ba":ba,"comparisons":cmp,
      "mean_full_pair_f1":float(full),"mean_k5_pair_f1":float(sparse),
      "mean_k5_pair_precision":float(k5p),"mean_k5_pair_recall":float(k5r),
      "details":outputs,"preregistered_gate_pass":bool(gate),
      "champion_0p715_same_row":False,"official_score":None,
      "gpu":False,"new_submission":False
    }


def self_test():
    score=np.array([[10,9,0],[11,8,0]],float)
    got=assign(score,np.array([0,0]))
    assert len(set(got))==2 and got[0]==1 and got[1]==0
    assert np.array_equal(assign(score,np.array([0,1])),[0,0])
    g=np.array(["a","a","b","b"])
    mat=np.array([[1,.99,.001,.001],[.99,1,.001,.001],
                  [.001,.001,1,.99],[.001,.001,.99,1]])
    p=cluster_prob(mat,.90)
    assert graph_stats(g,p)["pair_f1"]==1.
    print("NTL_SEMANTIC_SET_SYNTHETIC=PASS")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--synthetic-only",action="store_true")
    parser.add_argument("--summary",type=Path,required=True)
    args=parser.parse_args()
    self_test()
    if args.synthetic_only:return
    d=load_snapshot(Path("data/train.csv"),Path("artifacts/phase1/folds.csv"))
    print("NTL_SEMANTIC_SET_PRETRAINED_CPU_EMBEDDING_START=TRUE",flush=True)
    with threadpool_limits(limits=2):
        z=embed(d.text.astype(str).tolist())
        print("NTL_SEMANTIC_SET_PRETRAINED_CPU_EMBEDDING_DONE="+str(z.shape),flush=True)
        result=evaluate(d,z)
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NTL_SEMANTIC_SET_AGGREGATE="+json.dumps({
      "full_ba":result["full_ba"],"comparisons":result["comparisons"],
      "mean_full_pair_f1":result["mean_full_pair_f1"],
      "mean_k5_pair_f1":result["mean_k5_pair_f1"],
      "mean_k5_pair_precision":result["mean_k5_pair_precision"],
      "mean_k5_pair_recall":result["mean_k5_pair_recall"],
      "gate_pass":result["preregistered_gate_pass"]
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
