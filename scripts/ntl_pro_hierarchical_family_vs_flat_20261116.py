"""One fixed CPU supervised model-class pivot: hierarchical vs flat attribution.

Use only historic official train for fitting; test on the pre-existing, spent,
prompt-disjoint Oleg18 216 cohort, pairing against real frozen winner labels.
This is an exploratory transfer/mechanism test, NOT independent validation,
not permission for Kaggle submission. No high-powered teacher / GPU.

The competing hypothesis is whether family-conditioned 18-label classification
better separates source-dependent model siblings than flat classification.
Same char_wb TF-IDF representation, C=1.0, data, label distribution, and
estimator class in both arms. The hierarchy is the sole treatment variable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from scripts.ntl_oleg18_frozen_selector_20261116 import (
    NTL_OLEG18_MODELS, NTL_OLEG18_SHA, ntl_oleg18_select
)

TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
SEED=20261116
C=1.0
NGRAM=(3,5)
MAX_FEATURES=150000
MIN_DF=2
REQUIRED_EXTERNAL=216
BASELINE_CORRECT=89


def family(label:str)->str:
    for prefix,fam in (
        ("claude-","claude"),("gemini-","gemini"),("glm-","glm"),
        ("gpt-","gpt"),("kimi-","kimi"),("qwen","qwen")
    ):
        if label.startswith(prefix):return fam
    raise ValueError("unknown model family")


def svc()->LinearSVC:
    return LinearSVC(C=C, class_weight="balanced",random_state=SEED,
                     dual="auto",max_iter=3000)


def score(pred:np.ndarray, truth:np.ndarray, base:np.ndarray)->dict:
    hits=(pred==truth)
    reference=(base==truth)
    labels=list(NTL_OLEG18_MODELS)
    per={l:int(np.sum(hits[truth==l])) for l in labels}
    return {
        "correct":int(hits.sum()),"n":len(truth),
        "balanced_accuracy":round(float(np.mean([np.mean(hits[truth==l]) for l in labels])),6),
        "reference_correct":int(reference.sum()),
        "rescues_vs_0691_frozen":int(np.sum(hits & ~reference)),
        "regressions_vs_0691_frozen":int(np.sum(~hits & reference)),
        "net_vs_0691_frozen":int(hits.sum()-reference.sum()),
        "cross_family_errors":int(np.sum([family(a)!=family(b) for a,b in zip(truth,pred)])),
        "per_18_label_correct_out_of_12":per,
    }


def run(train:Path,external_root:Path,frozen:Path)->dict:
    if hashlib.sha256(train.read_bytes()).hexdigest()!=TRAIN_SHA:
        raise ValueError("official training SHA changed")
    tr=pd.read_csv(train,usecols=["id","prompt_id","model","text"],dtype=str)
    if len(tr)!=2699 or tr.model.nunique()!=18 or tr.prompt_id.nunique()!=150:
        raise ValueError("training population drift")
    for name,digest in NTL_OLEG18_SHA.items():
        if hashlib.sha256((external_root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("external source SHA changed")
    cohort=ntl_oleg18_select(
        pd.read_parquet(external_root/"external_short_wide.parquet"),
        pd.read_parquet(external_root/"external_cell_metadata.parquet"))
    if (len(cohort)!=REQUIRED_EXTERNAL or cohort.model.nunique()!=18
        or cohort.model.value_counts().ne(12).any()
        or cohort.prompt_group_id.nunique()!=216):
        raise ValueError("external test population changed")
    old=pd.read_parquet(frozen)
    if (len(old)!=216 or list(old.columns)!=["id","model"]
        or old.id.duplicated().any() or set(old.id)!=set(cohort.id)):
        raise ValueError("frozen champion prediction mismatch")
    old=cohort[["id"]].merge(old,on="id",validate="one_to_one")
    truth=cohort.model.to_numpy(dtype=str)
    base=old.model.to_numpy(dtype=str)
    if int((base==truth).sum()) != BASELINE_CORRECT:
        raise ValueError("frozen champion historical 89/216 check failed")

    # Make a SINGLE, source-blind representation, fit on official train only.
    vectorizer=TfidfVectorizer(analyzer="char_wb",ngram_range=NGRAM,
        min_df=MIN_DF,max_features=MAX_FEATURES,sublinear_tf=True,
        dtype=np.float32,lowercase=False)
    x=vectorizer.fit_transform(tr.text.astype(str))
    xh=vectorizer.transform(cohort.text.astype(str))
    y=tr.model.to_numpy(dtype=str)
    unique=np.asarray(sorted(set(y)))
    if set(unique)!=set(NTL_OLEG18_MODELS):
        raise ValueError("18-class train/test label contract drift")

    direct=svc().fit(x,y).predict(xh)
    yf=np.asarray([family(z) for z in y],dtype=str)
    stage1=svc().fit(x,yf).predict(xh)
    stage2=np.empty(len(cohort),dtype=object)
    for fam in sorted(set(yf)):
        idx=np.flatnonzero(stage1==fam)
        if not len(idx):continue
        trainmask=yf==fam
        classes=np.unique(y[trainmask])
        if len(classes)==1:
            stage2[idx]=classes[0]
        else:
            stage2[idx]=svc().fit(x[trainmask],y[trainmask]).predict(xh[idx])
    if any(s is None for s in stage2):raise RuntimeError("missing predictions")

    flat=score(direct,truth,base)
    hierarchical=score(np.asarray(stage2,dtype=str),truth,base)
    f=(direct==truth);h=(np.asarray(stage2,dtype=str)==truth)
    rescued=int((h&~f).sum());regressed=int((~h&f).sum())
    discord=rescued+regressed
    return {
        "protocol":"ntl-18-class-flat-vs-family-hierarchy-charwb-v0-20261116",
        "hypotheses":{
            "H_hierarchy":"explicit family decomposition increases Oleg source-transfer BA by >=0.05 vs same-feature flat classifier",
            "H_flat":"hierarchical routing compounds family mistakes and fails to improve real source-transfer correctness"
        },
        "model":"CPU-only LinearSVC char_wb(3,5), C=1, 150k terms",
        "same_data_and_features":True,
        "training":"historical official train 2699 rows; none from Oleg",
        "testing":"previously inspected Oleg18 216, not a clean untouched source",
        "frozen_champion":{"correct":BASELINE_CORRECT,"n":216,"BA":round(BASELINE_CORRECT/216,6)},
        "flat":flat,
        "hierarchical":hierarchical,
        "hierarchy_rescues_vs_flat":rescued,
        "hierarchy_regressions_vs_flat":regressed,
        "hierarchy_delta_BA":round((hierarchical["correct"]-flat["correct"])/216,6),
        "sign_test_two_sided_exploratory":float(binomtest(rescued,discord).pvalue) if discord else 1.0,
        "predeclared_pilot_gate":"hierarchy minus flat >=0.05 BA AND >=frozen champion 89/216 correct AND at least 10 net rescues vs champion",
        "gate_pass":bool(hierarchical["correct"]-flat["correct"]>=11 and hierarchical["correct"]>=99),
        "source_overlap_verified":False,
        "new_model_training":True,"GPU":False,"official_kaggle_submission":False,
        "candidate_for_automatic_official_submit":False,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--external",type=Path,required=True)
    p.add_argument("--champion",type=Path,required=True)
    a=p.parse_args()
    print("NTL_PRO_HIERARCHICAL_REAL_TRANSFER="+json.dumps(run(a.train,a.external,a.champion),sort_keys=True),flush=True)
