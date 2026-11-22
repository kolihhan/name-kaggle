"""NTL label-only strong-model reconciliation (LOR) on real grouped OOF.

Can a weaker likelihood surface and blind same-prompt structure correct a
STRONGER model when that model exposes only final class labels, NOT logits?
This is the exact integration problem with public 0.715 attributed champion.

C0 strong char_wb+word LinearSVC alone
C1 C0 own-score Hungarian (not feasible with label-only champion)
C2 weaker char-only 18-way SVC alone
C3 weaker-score Hungarian
C4 weak scores + strong hard-label bonus (1.5) Hungarian
C5 same as C4 but randomized group IDs (negative control)

Test-time grouping uses only word TFIDF of unlabeled texts. No prompt ID,
truth label, or validation class prior in inference. Fixed 5 group folds,
fixed lambda, no other hyperparameters or Kaggle submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot
from scripts.ntl_semantic_set_decoding_20261122 import (
    lexical_cluster,assign,subset_k,bootstrap,
)

SEED=20261122
BONUS=1.5
ARMS=("strong_raw","strong_joint","weak_raw","weak_joint",
      "strong_label_weak_scores_joint","randomized_group_joint","oracle_joint")


def linear_svc():
    return LinearSVC(C=.5,class_weight="balanced",
                     random_state=SEED,max_iter=3000,dual="auto")


def anchored_scores(weak_scores, strong_labels):
    if weak_scores.ndim!=2 or weak_scores.shape[0]!=len(strong_labels):
        raise ValueError("weak score matrix / anchor mismatch")
    if weak_scores.shape[1]!=18:
        raise ValueError("expected 18 outputs")
    if np.any(strong_labels<0) or np.any(strong_labels>=18):
        raise ValueError("out-of-range anchor labels")
    center=weak_scores-weak_scores.mean(axis=1,keepdims=True)
    scaled=center/np.maximum(weak_scores.std(axis=1,keepdims=True),1e-6)
    scaled[np.arange(len(strong_labels)),strong_labels]+=BONUS
    return scaled


def self_test():
    z=np.zeros((2,18),dtype=float)
    a=np.array([4,7])
    q=anchored_scores(z,a)
    assert list(q.argmax(1))==[4,7]
    assert q[0,4]==BONUS and q[1,7]==BONUS
    print("NTL_LABEL_ONLY_RECONCILIATION_SYNTHETIC=PASS",flush=True)


def experiment():
    self_test()
    d=load_snapshot(Path("data/train.csv"),Path("artifacts/phase1/folds.csv"))
    classes,y=np.unique(d.model.to_numpy(),return_inverse=True)
    groups=d.prompt_id.astype(str).to_numpy()
    folds=d.fold_id.to_numpy()
    if len(d)!=2699 or len(classes)!=18 or len(set(groups))!=150:
        raise ValueError("official data contract drift")
    pred={a:np.full(len(y),-1,dtype=int) for a in ARMS}
    sparse_results=[]
    rng=np.random.default_rng(SEED)
    with threadpool_limits(limits=2):
        for fold in range(5):
            tr,va=folds!=fold,folds==fold
            if set(groups[tr])&set(groups[va]):
                raise ValueError("prompt overlap")
            train_text=d.text[tr].astype(str).tolist()
            valid_text=d.text[va].astype(str).tolist()

            char=TfidfVectorizer(analyzer="char",lowercase=False,
                ngram_range=(2,5),max_features=45000,min_df=2,
                sublinear_tf=True)
            ca=char.fit_transform(train_text)
            cb=char.transform(valid_text)
            weak=linear_svc().fit(ca,y[tr])
            ws=np.asarray(weak.decision_function(cb))

            cwb=TfidfVectorizer(analyzer="char_wb",
                ngram_range=(3,5),max_features=65000,min_df=2,
                sublinear_tf=True)
            wa=cwb.fit_transform(train_text)
            wb=cwb.transform(valid_text)
            word=TfidfVectorizer(analyzer="word",ngram_range=(1,2),
                stop_words="english",max_features=50000,min_df=2,
                sublinear_tf=True)
            xa=word.fit_transform(train_text)
            xb=word.transform(valid_text)
            strong=linear_svc().fit(sparse.hstack([wa,xa],format="csr"),
                                    y[tr])
            ss=np.asarray(strong.decision_function(
                sparse.hstack([wb,xb],format="csr")))

            anchor=ss.argmax(axis=1)
            wraw=ws.argmax(axis=1)
            blind=lexical_cluster(xb)
            perm=blind[rng.permutation(len(blind))]
            joined=anchored_scores(ws,anchor)
            oracle=groups[va]
            trial={
                "strong_raw":anchor,
                "strong_joint":assign(ss,blind),
                "weak_raw":wraw,
                "weak_joint":assign(ws,blind),
                "strong_label_weak_scores_joint":assign(joined,blind),
                "randomized_group_joint":assign(joined,perm),
                "oracle_joint":assign(joined,oracle),
            }
            idx=np.flatnonzero(va)
            for k,v in trial.items():
                pred[k][idx]=v
            sample=subset_k(d,idx,5)
            rel=np.searchsorted(idx,sample)
            small_clusters=lexical_cluster(xb[rel])
            small_scores=ss[rel]
            small_anchor=small_scores.argmax(axis=1)
            small_weak=ws[rel]
            small_anchor_scores=anchored_scores(small_weak,small_anchor)
            small_random=small_clusters[rng.permutation(len(rel))]
            small_trial={
                "strong_raw":small_anchor,
                "strong_joint":assign(small_scores,small_clusters),
                "weak_raw":small_weak.argmax(1),
                "weak_joint":assign(small_weak,small_clusters),
                "strong_label_weak_scores_joint":assign(small_anchor_scores,small_clusters),
                "randomized_group_joint":assign(small_anchor_scores,small_random),
                "oracle_joint":assign(small_anchor_scores,groups[sample]),
            }
            sparse_results.append({
                "fold":fold,"rows":len(sample),
                "ba":{k:float(balanced_accuracy_score(y[sample],v))
                      for k,v in small_trial.items()}
            })
            print("NTL_LABEL_ONLY_REAL_FOLD="+str(fold),flush=True)
    if any((v<0).any() for v in pred.values()):
        raise ValueError("OOF rows missing")
    ba={k:float(balanced_accuracy_score(y,v)) for k,v in pred.items()}
    diffs={}
    for arm in ARMS:
        if arm=="strong_raw":continue
        new=pred[arm]
        baseline=pred["strong_raw"]
        diffs[arm]={
            "delta_BA18":ba[arm]-ba["strong_raw"],
            "ci95":bootstrap(y,baseline,new,groups),
            "rescues":int(((baseline!=y)&(new==y)).sum()),
            "regressions":int(((baseline==y)&(new!=y)).sum())
        }
    sparse_scores={k:float(np.mean([fold["ba"][k] for fold in sparse_results]))
                   for k in ARMS}
    # The scientific question only passes if the weaker label-only rescue
    # improves the stronger model across complete and sparse sets, and
    # beats shuffled grouping. Still cannot claim 0.715 transfer.
    viable=(ba["strong_raw"]>ba["weak_raw"]
       and diffs["strong_label_weak_scores_joint"]["delta_BA18"]>=.01
       and diffs["strong_label_weak_scores_joint"]["ci95"][0]>0
       and sparse_scores["strong_label_weak_scores_joint"]
                >sparse_scores["strong_raw"]
       and ba["strong_label_weak_scores_joint"]
                >ba["randomized_group_joint"])
    return {
      "scope":"2699 official train GROUPED 5-fold historical DEV only",
      "question":"can a weaker score-vector model improve a STRONGER hard-label-only model?",
      "strong_model":"char_wb+word TFIDF LinearSVC",
      "weak_model":"char TFIDF LinearSVC",
      "hidden_prompt_unique_source_unverified":True,
      "fully_frozen_label_bonus":BONUS,
      "rows":len(y),"prompts":len(set(groups)),"classes":len(classes),
      "full_OOF_BA18":ba,
      "paired_vs_strong":diffs,
      "sparse_k5_fold_mean_BA18":sparse_scores,
      "sparse_k5_each_fold":sparse_results,
      "gate_pass":bool(viable),
      "champion_0p715_inference_not_run":True,
      "no_GPU":True,"no_competition_submission":True
    }


if __name__=="__main__":
    r=experiment()
    print("NTL_LABEL_ONLY_REAL_RESULT="+json.dumps(r,sort_keys=True),flush=True)
