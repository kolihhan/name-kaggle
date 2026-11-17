"""Pre-registered GPT three-sibling rescue experiment, without test labels in fit.

CPU only; actual current 0.691 Lightning Oleg216 frozen predictions.
No new Lightning GPU inference, no competition submission.
Do not promote the spent external Oleg panel into independent evidence.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scripts.ntl_oleg18_frozen_selector_20261116 import (
    ntl_oleg18_select, NTL_OLEG18_SHA, NTL_OLEG18_MODELS
)

SEED=20261118
TRIAD=("gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra")
GAP=.20
MIN_BA_DELTA=.03
BOOTSTRAPS=5000
POOL="olegzholobov/name-that-llm-response-corpus"


def normalize(s):
    return " ".join(str(s).casefold().split())


def calc_ba(y,p,labels):
    return float(np.mean([np.mean(p[y==m]==m) for m in labels]))


def audited_frame(train_path,oleg_dir,pred_path):
    t=pd.read_csv(train_path,dtype={"id":str,"prompt_id":str})
    assert {"id","model","text","prompt_id"}.issubset(t.columns)
    if t.id.duplicated().any() or t.model.isna().any() or t.text.isna().any():
        raise RuntimeError("OFFICIAL_TRAIN_BAD_VALUES")
    if not set(t.model.astype(str))==set(NTL_OLEG18_MODELS):
        raise RuntimeError("OFFICIAL_TRAIN_NOT_18_CLASSES")
    if len(t)<2600 or t.prompt_id.nunique()<100:
        raise RuntimeError("OFFICIAL_TRAIN_TOO_SMALL_OR_PROMPT_METADATA_MISSING")
    for filename,digest in NTL_OLEG18_SHA.items():
        if hashlib.sha256((oleg_dir/filename).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("OLEG_INPUT_SHA_DRIFT")
    truth=ntl_oleg18_select(pd.read_parquet(oleg_dir/"external_short_wide.parquet"),
                            pd.read_parquet(oleg_dir/"external_cell_metadata.parquet"))
    pred=pd.read_parquet(pred_path)
    if pred.columns.tolist()!=["id","model"] or len(pred)!=216 or pred.id.duplicated().any():
        raise RuntimeError("WRONG_FROZEN_CHAMPION_SHAPE")
    a=truth[["id","text","model","prompt_group_id"]].merge(pred,on="id",suffixes=("_true","_pred"),validate="one_to_one")
    if len(a)!=216 or a.id.duplicated().any() or a.prompt_group_id.duplicated().any():
        raise RuntimeError("NOT_EXACT_OLEG216_UNIQUE_GROUPS")
    actual=a.model_true.to_numpy(dtype=str)
    incumbent=a.model_pred.to_numpy(dtype=str)
    if int(np.sum(actual==incumbent))!=89:
        raise RuntimeError("EXACT_0691_CHAMPION_FINGERPRINT_DRIFT")
    train_tri=t.loc[t.model.astype(str).isin(TRIAD),["text","model","prompt_id"]]
    if train_tri.model.nunique()!=3 or train_tri.groupby("model").size().min()<50:
        raise RuntimeError("NO_3CLASS_TRAIN_SUPPORT")
    # All raw Oleg text is held out from the specialist fit. A matching response
    # from the official train, even outside the GPT trio, invalidates external independence.
    orig_hash={hashlib.sha256(normalize(x).encode()).digest() for x in t.text.astype(str)}
    overlaps=sum(hashlib.sha256(normalize(x).encode()).digest() in orig_hash
                 for x in a.text.astype(str))
    if overlaps:
        raise RuntimeError("EXACT_RESPONSE_OVERLAP_WITH_OFFICIAL_TRAIN="+str(overlaps))
    return a,train_tri,t


def evaluate(a, train_tri, train_all):
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,
        max_features=100000,sublinear_tf=True,dtype=np.float32)
    x=vec.fit_transform(train_tri.text.astype(str).tolist())
    clf=LinearSVC(C=.5,class_weight="balanced",random_state=SEED,max_iter=3000,dual="auto")
    clf.fit(x,train_tri.model.astype(str).to_numpy())
    if not set(clf.classes_)==set(TRIAD):
        raise RuntimeError("TRIAD_CLASS_ORDER_DRIFT")
    current=a.model_pred.astype(str).to_numpy()
    y=a.model_true.astype(str).to_numpy()
    proposed=current.copy()
    original_gate=np.flatnonzero(np.isin(current,TRIAD))
    if not len(original_gate):
        raise RuntimeError("CHAMPION_HAS_NO_TRIAD_PREDICTIONS")
    z=vec.transform(a.iloc[original_gate].text.astype(str).tolist())
    decision=clf.decision_function(z)
    assert decision.shape==(len(original_gate),3)
    sorted_score=np.sort(decision,axis=1)
    margins=sorted_score[:,-1]-sorted_score[:,-2]
    chosen=clf.classes_[decision.argmax(axis=1)]
    changed_indices=original_gate[(margins>=GAP)&(chosen!=current[original_gate])]
    proposed[changed_indices]=chosen[(margins>=GAP)&(chosen!=current[original_gate])]
    assert set(proposed).issubset(set(NTL_OLEG18_MODELS))
    assert np.array_equal(current[~np.isin(current,TRIAD)],
                          proposed[~np.isin(current,TRIAD)])
    assert all(proposed[i] in TRIAD for i in changed_indices)
    labels=list(NTL_OLEG18_MODELS)
    ba0=calc_ba(y,current,labels)
    ba1=calc_ba(y,proposed,labels)
    delta=ba1-ba0
    rescues=(current!=y)&(proposed==y)
    regressions=(current==y)&(proposed!=y)
    # Stratified bootstrap by true model label (each selected row is a distinct
    # prompt_group_id). Fixed 12 true rows/class per replicate.
    rng=np.random.default_rng(SEED)
    label_indices=[np.flatnonzero(y==lab) for lab in labels]
    diffs=np.empty(BOOTSTRAPS)
    for k in range(BOOTSTRAPS):
        picks=np.concatenate([rng.choice(idx,size=len(idx),replace=True) for idx in label_indices])
        diffs[k]=calc_ba(y[picks],proposed[picks],labels)-calc_ba(y[picks],current[picks],labels)
    lo,hi=np.quantile(diffs,[.025,.975])
    count_by_label={l:{"n":int((y==l).sum()),
                       "C0_recall":round(float(np.mean(current[y==l]==l)),6),
                       "C1_recall":round(float(np.mean(proposed[y==l]==l)),6),
                       "rescues":int((rescues&(y==l)).sum()),
                       "regressions":int((regressions&(y==l)).sum())}
                    for l in labels}
    eligible=(delta>=MIN_BA_DELTA and lo>0 and rescues.sum()>regressions.sum()
              and not np.any(proposed[~np.isin(current,TRIAD)]!=current[~np.isin(current,TRIAD)]))
    outcome={
        "contract":"ntl-gpt-triad-probe-pre-registration-20261118",
        "official_score_incumbent":0.691,
        "frozen_incumbent_submission_ref":58375806,
        "external_source":POOL,
        "training_rows_all_official":int(len(train_all)),
        "training_rows_three_model":int(len(train_tri)),
        "training_prompts_three_model":int(train_tri.prompt_id.nunique()),
        "frozen_holdout_rows":int(len(a)),
        "frozen_holdout_prompt_groups":int(a.prompt_group_id.nunique()),
        "balanced_classes":18,
        "classes":labels,
        "C0_correct":int(np.sum(current==y)),
        "C1_correct":int(np.sum(proposed==y)),
        "C0_BA":round(ba0,6),"C1_BA":round(ba1,6),"delta_BA":round(delta,6),
        "rescues":int(rescues.sum()),"regressions":int(regressions.sum()),
        "original_gpt_triad_prediction_count":int(len(original_gate)),
        "changed_label_count":int(len(changed_indices)),
        "non_target_label_changes":0,
        "confidence_gap":GAP,
        "bootstrap_95_CI":[round(float(lo),6),round(float(hi),6)],
        "bootstrap_repeats":BOOTSTRAPS,
        "model_train_only_official_train":True,
        "exact_text_overlap_with_holdout":0,
        "target_only_previous_gpt_prediction":True,
        "per_true_label":count_by_label,
        "candidate_score_not_official":True,
        "spent_holdout":True,
        "status":"ELIGIBLE_TO_BUILD_EXPLICIT_PROBE_ONLY" if eligible else "NO_CANDIDATE_GATE_FAILED",
        "new_competition_submission":False,
    }
    return outcome


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--official-train",type=Path,required=True)
    p.add_argument("--oleg-source",type=Path,required=True)
    p.add_argument("--champion",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    frame,subtrain,train_all=audited_frame(args.official_train,args.oleg_source,args.champion)
    report=evaluate(frame,subtrain,train_all)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("NTL_GPT_TRIAD_FROZEN_PAIRED_CPU_RESULT="+json.dumps({
        k:report[k] for k in ("C0_BA","C1_BA","delta_BA","rescues","regressions",
              "changed_label_count","original_gpt_triad_prediction_count",
              "bootstrap_95_CI","status")
    },sort_keys=True),flush=True)
if __name__=="__main__":main()
