"""Inference-only label+weaker-score joint decoding for a scored 0.715 notebook.

Executed as FINAL cell after the untouched scored notebook has already written
/kaggle/working/submission.parquet. This never requests champion logits,
never refits/changes any champion component, and never reads hidden labels.

Scientific gate: historical Phase5b strong0.367 source/weak0.316 label-only
five-fold group OOF gain +0.0556; UNKNOWN transfer to 0.715 hidden Kaggle.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ALLOWED_INPUTS=(
    Path("/kaggle/input/competitions/name-that-llm"),
    Path("/kaggle/input/name-that-llm"),
)
SUBMISSION=Path("/kaggle/working/submission.parquet")
SEED=20261122
BONUS=1.5
MAX_GROUP=18
EXPECTED_CLASSES=18
MAX_TEST_ROWS=25000


def input_root():
    options=[
        root for root in ALLOWED_INPUTS
        if (root/"train.csv").is_file() and (root/"test.csv").is_file()
    ]
    if len(options)!=1:
        raise RuntimeError("OFFICIAL_COMPETITION_DATASET_PATH_NOT_UNIQUE")
    return options[0]


def infer_group(text_train,text_test):
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer=TfidfVectorizer(analyzer="word",ngram_range=(1,2),
        stop_words="english",max_features=50000,min_df=2,sublinear_tf=True)
    vectorizer.fit(text_train)
    x=vectorizer.transform(text_test)
    n=len(text_test)
    if n<2:
        return np.zeros(n,dtype=int)
    # Original frozen test method: same average-linkage cosine cut 0.90.
    similarities=(x@x.T).toarray()
    dist=np.clip(1.0-similarities,0.0,2.0)
    np.fill_diagonal(dist,0.0)
    return AgglomerativeClustering(n_clusters=None,
        distance_threshold=.90,linkage="average",
        metric="precomputed").fit_predict(dist)


def class_scores(train_text,train_labels,test_text):
    vec=TfidfVectorizer(analyzer="char",lowercase=False,
        ngram_range=(2,5),max_features=45000,min_df=2,
        sublinear_tf=True)
    x=vec.fit_transform(train_text)
    v=vec.transform(test_text)
    clf=LinearSVC(C=.5,class_weight="balanced",max_iter=3000,
        random_state=SEED,dual=True)
    clf.fit(x,train_labels)
    scores=np.asarray(clf.decision_function(v),dtype=float)
    return scores,clf.classes_


def reconcile(model_labels,weak,weak_classes,groups):
    model_labels=np.asarray(model_labels,dtype=str)
    classes=np.asarray(weak_classes,dtype=str)
    if len(classes)!=EXPECTED_CLASSES or len(set(classes))!=len(classes):
        raise RuntimeError("WEAK_MODEL_CLASS_COVERAGE_INVALID")
    if not set(model_labels).issubset(set(classes)):
        raise RuntimeError("CHAMPION_CLASS_LABEL_OUT_OF_RANGE")
    if weak.shape!=(len(model_labels),len(classes)):
        raise RuntimeError("WEAK_SCORE_MATRIX_SHAPE_INVALID")
    label_index={name:i for i,name in enumerate(classes)}
    original=np.asarray([label_index[z] for z in model_labels],dtype=int)

    centered=weak-weak.mean(axis=1,keepdims=True)
    scores=centered/np.maximum(weak.std(axis=1,keepdims=True),1e-6)
    scores[np.arange(len(scores)),original]+=BONUS
    assigned=original.copy()
    updated_group_count=0
    for cluster in np.unique(groups):
        idx=np.flatnonzero(groups==cluster)
        if not (2<=len(idx)<=MAX_GROUP):
            continue
        r,c=linear_sum_assignment(-scores[idx])
        proposed=c
        if np.any(proposed!=original[idx[r]]):
            updated_group_count+=1
        assigned[idx[r]]=proposed
    pred=classes[assigned]
    if len(pred)!=len(model_labels):
        raise RuntimeError("WRONG_OUTPUT_ROW_COUNT")
    return pred,{"changed_rows":int(np.sum(pred!=model_labels)),
                 "modified_groups":int(updated_group_count),
                 "total_inferred_groups":int(len(np.unique(groups)))}


def smoke():
    labs=np.asarray([f"M{i:02d}" for i in range(18)])
    # Fake source labels and scores only, no training/test content.
    scores=np.zeros((2,18),dtype=float)
    scores[0,0]=2.;scores[0,1]=1.
    scores[1,0]=2.;scores[1,1]=1.
    pred,_=reconcile(["M00","M00"],scores,labs,np.array([0,0]))
    assert len(pred)==2 and len(set(pred))==2
    print("NTL_0715_LABEL_ONLY_JOINT_SYNTHETIC=PASS",flush=True)


def main():
    smoke()
    t0=time.monotonic()
    if not SUBMISSION.is_file():
        raise FileNotFoundError("scored notebook output submission.parquet missing")
    root=input_root()
    df_train=pd.read_csv(root/"train.csv",dtype={"id":str,"prompt_id":str})
    df_test=pd.read_csv(root/"test.csv",dtype={"id":str})
    previous=pd.read_parquet(SUBMISSION)
    if (set(previous.columns)!={"id","model"} or
        len(df_train)!=2699 or df_train.model.nunique()!=EXPECTED_CLASSES or
        len(df_test)<1 or len(df_test)>MAX_TEST_ROWS or
        len(previous)!=len(df_test) or
        df_test.id.duplicated().any() or previous.id.duplicated().any() or
        set(df_test.id)!=set(previous.id)):
        raise RuntimeError("FROZEN_BASE_NOTEBOOK_OUTPUT_CONTRACT_INVALID")
    # Preserve exact scored champion test output IDs and ordering.
    old=previous.copy()
    aligned=old.set_index("id").loc[df_test.id.to_numpy()]["model"].astype(str).to_numpy()
    texts_train=df_train.text.astype(str).tolist()
    texts_test=df_test.text.astype(str).tolist()

    print("NTL_0715_JOINT_INPUT_COUNT="+str(len(df_test)),flush=True)
    weak,classes=class_scores(texts_train,
        df_train.model.astype(str).to_numpy(),texts_test)
    groups=infer_group(texts_train,texts_test)
    result,stats=reconcile(aligned,weak,classes,groups)
    lookup=dict(zip(df_test.id.astype(str),result))
    new=old.copy()
    new["model"]=old.id.astype(str).map(lookup)
    if new.model.isna().any() or not set(new.model).issubset(set(classes)):
        raise RuntimeError("OUTPUT_LABEL_FORMAT_INVALID")
    if not np.array_equal(new.id.to_numpy(),old.id.to_numpy()):
        raise RuntimeError("CHAMPION_OUTPUT_ORDER_CHANGED")

    # Output changed final labels only, preserving public scored provenance.
    new.to_parquet(SUBMISSION,index=False)
    print("NTL_0715_LABEL_ONLY_JOINT_INFERENCE="+json.dumps({
        "test_rows":len(new),**stats,
        "model":"weak-charSVC+topic-group+Hungarian anchored to champion hard labels",
        "model_0p715_base_changed":False,
        "true_hidden_score":None,
        "no_reference_labels":True,
        "seconds":round(time.monotonic()-t0,1),
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
