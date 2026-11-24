"""NTL 18-way cross-prompt/source *fingerprint evidence* diagnostic, no model training.

Compare per-prompt model-style rank vectors on official train versus the
previously used 252-row synthetic 18-model source. All 18 truth labels used
only to interpret matched prompt response sets in this descriptive study.
These rankings CANNOT be applied to the unlabeled hidden test, whose
prompt grouping/model assignments are not provided.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

OFFICIAL_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXTERNAL_SHA="bce34670f4d1170375e7baf85eec5577af4282d5f0ca6cf1573ce46e81c5a45c"
SEED=20261124
PROMPTS={"short":(98,25,65,83,16,61,68),"median":(77,54,99,37),"long":(72,63,87)}
FEATURES=["log_words","mean_word_length","sentence_density","comma_per100",
          "colon_per100","semicolon_per100","emdash_per100","curly_per100",
          "hyphen_per100","digits_per100","first_person_per100","contraction_per100",
          "newline_per100","type_token_ratio"]
F=len(FEATURES)
LABELS=18

def sha(path):
    with Path(path).open("rb") as h:return hashlib.file_digest(h,"sha256").hexdigest()

def featurize(text):
    s=str(text)
    words=re.findall(r"\b[\w]+(?:['’][\w]+)?\b",s,flags=re.UNICODE)
    n=max(len(words),1)
    lens=np.array([len(w) for w in words],dtype=np.float64)
    wlower=[w.casefold() for w in words]
    return np.array([
        np.log1p(len(words)),float(lens.mean()) if len(lens) else 0.0,
        100.0*(s.count(".")+s.count("?")+s.count("!"))/n,
        100.0*s.count(",")/n,100.0*s.count(":")/n,100.0*s.count(";")/n,
        100.0*s.count("—")/n,100.0*(s.count("’")+s.count("“")+s.count("”"))/n,
        100.0*s.count("-")/n,100.0*sum(ch.isdigit() for ch in s)/n,
        100.0*sum(w in {"i","me","my","mine","we","us","our","ours"} for w in wlower)/n,
        100.0*sum("'" in w or "’" in w for w in words)/n,
        100.0*s.count("\n")/n,
        len(set(wlower))/n,
    ],dtype=np.float64)

def make_cubes(official,external):
    train=pd.read_csv(official,dtype={"id":str,"prompt_id":str,"model":str})
    if len(train)!=2699 or train.model.nunique()!=18 or train.prompt_id.nunique()!=150 or not train.id.is_unique:
        raise ValueError("official cohort schema drift")
    labels=sorted(train.model.unique())
    official_cubes=[]
    for pid,g in train.groupby("prompt_id"):
        if g.model.nunique()!=18:continue
        if len(g)!=18:raise ValueError("duplicate official model within prompt")
        official_cubes.append(np.stack([featurize(g.set_index("model").loc[x,"text"]) for x in labels]))
    ext=pd.read_csv(external)
    ext_cubes=[]
    ext_subsets=[]
    for version,topics in PROMPTS.items():
        for topic in topics:
            q=ext.loc[(ext.version==version)&(ext.topic_id==topic)]
            if len(q)!=18 or set(q.model)!=set(labels) or q.model.duplicated().any():
                raise ValueError("invalid external matched all18 source block")
            ext_cubes.append(np.stack([featurize(q.set_index("model").loc[x,"text"]) for x in labels]))
            ext_subsets.append(version)
    if len(official_cubes)!=149 or len(ext_cubes)!=14:
        raise ValueError("fixed prompt block number drift")
    if len(train)==0:raise ValueError("no official rows")
    return np.stack(official_cubes),np.stack(ext_cubes),labels,ext_subsets,train,ext

def prompt_ranks(cube):
    # [prompt, model, stylefeature] rank 1..18, scaled mean-zero
    ranks=np.stack([rankdata(cube[:,:,k],axis=1,method="average") for k in range(cube.shape[2])],axis=2)
    return (ranks-9.5)/8.5

def correlations(A,B):
    a=A.mean(axis=0);b=B.mean(axis=0)
    v=[]
    for k in range(F):
        x=spearmanr(a[:,k],b[:,k]).statistic
        v.append(float(x) if np.isfinite(x) else 0.0)
    return np.asarray(v),a,b

def pairwise_match(a,b):
    # Prompt-conditional descriptive model-centroid resemblance.
    # NOT a usable single-row Kaggle test-time classifier.
    a=a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-9)
    b=b/(np.linalg.norm(b,axis=1,keepdims=True)+1e-9)
    sim=b@a.T
    return sim

def main(path1,path2):
    start=time.monotonic()
    if sha(path1)!=OFFICIAL_SHA or sha(path2)!=EXTERNAL_SHA:
        raise ValueError("unrecognized official/external bytes")
    official,external,labels,versions,train,ext=make_cubes(path1,path2)
    a=prompt_ranks(official);b=prompt_ranks(external)
    corr,ta,tb=correlations(a,b)
    similarity=pairwise_match(ta,tb)
    neighbors=np.argsort(-similarity,axis=1)
    exact=int(sum(i in [neighbors[i,0]] for i in range(18)))
    top3=int(sum(i in neighbors[i,:3] for i in range(18)))
    margins=np.diag(similarity)-np.array([np.max(np.delete(similarity[i],i)) for i in range(18)])
    rng=np.random.default_rng(SEED)
    reps=2000
    null_top=np.zeros(reps,dtype=np.int32)
    null_maxcorr=np.zeros(reps)
    for j in range(reps):
        # Shuffle each external prompt's label mapping while preserving 18-way
        # prompt composition; this destroys stable model identity only.
        shuffled=np.stack([row[rng.permutation(18)] for row in b])
        v,_,hb=correlations(a,shuffled)
        null_maxcorr[j]=np.max(np.abs(v))
        m=pairwise_match(ta,hb)
        null_top[j]=int((m.argmax(1)==np.arange(18)).sum())
    # Paired prompt-block bootstrap (source imbalance 149 v 14 captured)
    boots=np.zeros((1000,F))
    for j in range(1000):
        ia=rng.integers(0,len(a),len(a))
        ib=rng.integers(0,len(b),len(b))
        boots[j]=correlations(a[ia],b[ib])[0]
    rankres=[]
    for k,name in enumerate(FEATURES):
        adjusted=(1+np.sum(null_maxcorr>=abs(corr[k])))/(1+reps)
        rankres.append({
          "feature":name,"cross_source_spearman18":round(float(corr[k]),4),
          "bootstrap_95":[round(float(z),4) for z in np.quantile(boots[:,k],[.025,.975])],
          "permutation_max_abs_corr_FWER_p":round(float(adjusted),4),
          "mean_abs_label_rank_drift":round(float(np.mean(np.abs(ta[:,k]-tb[:,k]))),4),
        })
    total_emds={}
    for label in labels:
        x=train.loc[train.model==label].text.astype(str)
        y=ext.loc[ext.model==label].text.astype(str)
        total_emds[label]={"official_emdash_pct":round(100*float(x.str.contains("—",regex=False).mean()),2),
                           "external_all_rows_emdash_pct":round(100*float(y.str.contains("—",regex=False).mean()),2)}
    result={
      "experiment":"NTL_all18_source_invariant_prompt_rank_fingerprint_nontrained",
      "official_complete_prompts":len(official),
      "external_complete_prompt_blocks":len(external),
      "official_complete_rows":int(len(official)*18),
      "external_selected_rows":int(len(external)*18),
      "external_selected_versions":dict(pd.Series(versions).value_counts()),
      "truth_classes":18,
      "source_publisher_reported_not_authenticated":True,
      "official_historical_prompt_data_spent":True,
      "source_transfer_not_independent_hidden_test":True,
      "uses_privileged_prompt_model_grouping_not_usable_hidden_inference":True,
      "source_correlation_features":rankres,
      "descriptive_template_self_top1_of18":exact,
      "descriptive_template_self_top3_of18":top3,
      "permutation_top1_p":round(float((1+np.sum(null_top>=exact))/(reps+1)),4),
      "template_margin_positive_classes":int(sum(margins>0)),
      "template_positive_margin_labels":[labels[i] for i in range(18) if margins[i]>0],
      "external_vs_official_emdash_rates_by_true_model":total_emds,
      "no_fit_no_neural_train_no_gpu_no_official_submission":True,
      "runtime_s":round(time.monotonic()-start,1)
    }
    print("NTL_18SOURCE_REAL_FINGERPRINT="+json.dumps(result,sort_keys=True),flush=True)

def selftest():
    x=np.array([[[1.,5.],[2.,4.],[3.,3.]]])
    r=prompt_ranks(np.tile(x,(1,6,1)))
    assert r.shape==(1,18,2)
    assert np.allclose(r.mean(1),0)
    t=featurize("I'm here — hello, world.\nHow are you?")
    assert len(t)==F and t[6]>0 and t[7]==0
    assert np.all(np.isfinite(t))
    print("NTL_18SOURCE_FINGERPRINT_SELFTEST=PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--official",type=Path)
    p.add_argument("--external",type=Path)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    selftest()
    if not args.smoke:
        if args.official is None or args.external is None:p.error("both pinned inputs required")
        main(args.official,args.external)
