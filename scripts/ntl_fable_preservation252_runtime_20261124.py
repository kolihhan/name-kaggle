"""Observational REAL 18-label Fable preservation, on pre-existing 252 panel.

NO new model, GPU work beyond incumbent private Kaggle scorer, or official
competition submission. Labels are sequestered from inference until the
original Apex stage already produced its final output.

Original pre-Apex and original protected Apex are untouched. The counterfactual
Apex only replays the final decision logic on existing in-memory refined_lp.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

WORK=Path("/kaggle/temp/ntl-fable-preservation252")
OFFICIAL_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
PANEL_SHA="bce34670f4d1170375e7baf85eec5577af4282d5f0ca6cf1573ce46e81c5a45c"
TOPICS={'short':(98,25,65,83,16,61,68),'median':(77,54,99,37),'long':(72,63,87)}
GLM=("glm-5.3","glm-5.3-flash")
FABLE="claude-fable-5"
OUTPUT=Path("/kaggle/working/ntl_fable_preservation_252_aggregate.json")
PRE=WORK/"pre_apex_submission.parquet"
RESULT_MODE="OBSERVATIONAL_SPENT_PUBLISHER_LABELED_18CLASS_NOT_GOLD"

def sha(path):
    with Path(path).open("rb") as fd:return hashlib.file_digest(fd,"sha256").hexdigest()

def norm(text):
    return hashlib.sha256(" ".join(str(text).split()).casefold().encode()).hexdigest()

def prepare(original_data):
    """Only swaps inference test.csv while preserving official TRAIN and weights."""
    official=Path(original_data)/"train.csv"
    if sha(official)!=OFFICIAL_SHA:raise ValueError("pinned official TRAIN bytes changed")
    df=pd.read_csv(official,usecols=["model","text"],dtype={"model":str})
    labels=set(df.model)
    if len(df)!=2699 or len(labels)!=18:raise ValueError("official identity schema drift")
    found=[p for p in Path("/kaggle/input").rglob("generations.csv")
           if "ntl-topic-synthetic-18llms" in p.parts and sha(p)==PANEL_SHA]
    if len(found)!=1:raise ValueError("pinned external 18-class panel not mounted")
    data=pd.read_csv(found[0])
    blocks=[]
    for version,topics in TOPICS.items():
        for topic in topics:
            b=data.loc[(data.version==version)&(data.topic_id==topic)].copy()
            if len(b)!=18 or set(b.model)!=labels:raise ValueError("expected complete 18-way block")
            blocks.append(b.sort_values("model",kind="stable"))
    panel=pd.concat(blocks,ignore_index=True)
    if len(panel)!=252 or panel.text.isna().any() or panel.model.value_counts().ne(14).any():
        raise ValueError("panel size/model balance drift")
    if not panel.text.map(norm).is_unique or set(panel.text.map(norm))&set(df.text.dropna().map(norm)):
        raise ValueError("contamination or duplicate external response")
    panel.insert(0,"id",[f"fable-observation-{i:03d}" for i in range(len(panel))])
    WORK.mkdir(parents=True,exist_ok=False)
    d=WORK/"data"
    d.mkdir()
    (d/"train.csv").symlink_to(official.resolve())
    panel[["id","text"]].to_csv(d/"test.csv",index=False)
    # Sequester labels in diagnostic file. This path is never passed to the
    # original scoring notebook or queried by its inference stages.
    panel[["id","model","topic_id","version"]].to_csv(WORK/"truth.csv",index=False)
    print("NTL_FABLE_252_PREPARED="+json.dumps({
       "rows":252,"labels":18,"per_label":14,"prompt_topics":14,
       "original_train_pinned":True,"test_labels_inference_hidden":True,
       "publisher_identity_independent_verified":False,
       "panel_spent_prior_experiment":True},sort_keys=True),flush=True)
    return str(d)

def snapshot_pre_apex(data,out):
    """Only executed AFTER native model + GLM and before original Apex."""
    q=pd.read_csv(Path(data)/"test.csv",dtype={"id":str},usecols=["id"])
    p=pd.read_parquet(out)
    if len(q)!=252 or len(p)!=252 or p.columns.tolist()!=["id","model"]:
        raise ValueError("wrong pre-Apex prediction shape")
    if p.id.astype(str).tolist()!=q.id.tolist():raise ValueError("pre-Apex ID mismatch")
    base=Path("/kaggle/temp/final-logits.npz")
    if not base.exists():raise FileNotFoundError("no genuine full native logits")
    with np.load(base,allow_pickle=False) as z:
        ids=z["ids"].astype(str)
        labels=z["labels"].astype(str)
        scores=z["logits"]
        if ids.tolist()!=q.id.tolist() or scores.shape!=(252,18) or len(set(labels))!=18:
            raise ValueError("pre-V1 native logits not aligned")
    PRE.parent.mkdir(parents=True,exist_ok=True)
    p.to_parquet(PRE,index=False)
    print("NTL_FABLE_252_PRE_APEX_SNAPSHOT="+json.dumps({
       "rows":252,"labels":18,"pre_glm_native_logits_aligned":True,
       "post_glm_labels_saved_ephemerally":True,
       "no_model_scores_or_labels_leaked":True},sort_keys=True),flush=True)

def replay_typography_without_protect(pre,refined_preds,refined_lp,labels,texts):
    """Frozen original Apex deterministic logic, omitting NEW protection only.

    This function never fits an estimator or calls a language model.
    """
    p=np.asarray(pre,dtype=str).copy()
    r=np.asarray(refined_preds,dtype=str)
    z=np.asarray(refined_lp,dtype=np.float64)
    labels=np.asarray(labels,dtype=str)
    if z.shape!=(len(p),len(labels)) or len(texts)!=len(p):raise ValueError("Apex logits input drift")
    if not np.isfinite(z).all() or not set(p).issubset(set(labels)) or not set(r).issubset(set(labels)):
        raise ValueError("invalid Apex source scores")
    # Identical to original 0.715 Apex: only original GLM-labelled rows are preserved.
    glm=np.isin(p,GLM)
    p[~glm]=r[~glm]
    P=np.exp(z-z.max(axis=1,keepdims=True))
    P/=P.sum(1,keepdims=True)
    spaced_allowed={"claude-opus-4-6","claude-opus-5",*GLM}
    spaced_forbidden=np.array([x not in spaced_allowed for x in labels])
    no_curly=np.array([x.startswith(("claude-","glm-")) for x in labels])
    no_gemini=np.array([x.startswith("gemini-") for x in labels])
    for i,t in enumerate(texts):
        if " — " in t and p[i] not in spaced_allowed:
            P[i,spaced_forbidden]=0.0
            if P[i].sum()>0:
                P[i]/=P[i].sum()
                p[i]=labels[P[i].argmax()]
        if "’" in t and p[i].startswith(("claude-","glm-")):
            P[i,no_curly]=0.0
            if P[i].sum()>0:
                P[i]/=P[i].sum()
                p[i]=labels[P[i].argmax()]
        if "—" in t and p[i].startswith("gemini-"):
            P[i,no_gemini]=0.0
            if P[i].sum()>0:
                P[i]/=P[i].sum()
                p[i]=labels[P[i].argmax()]
    if not set(p).issubset(set(labels)):raise ValueError("counterfactual output label drift")
    return p

def counts(y,old,new):
    a=(np.asarray(old)==np.asarray(y))
    b=(np.asarray(new)==np.asarray(y))
    return {"correct_A":int(a.sum()),"correct_B":int(b.sum()),
      "rescues_B_over_A":int((~a&b).sum()),"regressions_B_over_A":int((a&~b).sum()),
      "changed":int((np.asarray(old)!=np.asarray(new)).sum())}

def evaluate(data,out,refined_lp,refined_preds,labels):
    if not PRE.exists():raise FileNotFoundError("snapshot was skipped; do not infer score")
    q=pd.read_csv(Path(data)/"test.csv",dtype={"id":str})
    truth=pd.read_csv(WORK/"truth.csv",dtype={"id":str,"model":str}).set_index("id")
    pre=pd.read_parquet(PRE)
    real=pd.read_parquet(out)
    if not (len(q)==len(pre)==len(real)==252):raise ValueError("not 252 real inference outputs")
    if not (pre.id.astype(str).tolist()==real.id.astype(str).tolist()==q.id.tolist()):
        raise ValueError("pre/post-Apex ID mismatch")
    labs=np.asarray(labels,dtype=str)
    if len(labs)!=18 or set(labs)!=set(truth.model):raise ValueError("class schema mismatch")
    y=truth.loc[q.id,"model"].to_numpy(str)
    if set(y)!=set(labs) or not np.all(np.unique(y,return_counts=True)[1]==14):
        raise ValueError("balanced 18-class real truth mismatch")
    cf=replay_typography_without_protect(
        pre.model.to_numpy(str),refined_preds,refined_lp,labs,q.text.astype(str).tolist())
    actual=real.model.to_numpy(str)
    if not set(actual).issubset(set(labs)):raise ValueError("new final output has invalid model")

    status="MISSING"
    report=Path("/kaggle/working/v1/report.json")
    if report.exists():
        obj=json.loads(report.read_text())
        status=str(obj.get("status","MISSING"))
    protect=np.zeros(len(q),dtype=bool)
    actual_v1_mask=np.isin(pre.model.to_numpy(str),GLM)
    if status=="applied":
        votes=Path("/kaggle/temp/prefix-votes.npz")
        if not votes.exists():raise FileNotFoundError("V1 applied but its post likelihood votes missing")
        with np.load(votes,allow_pickle=False) as z:
            if z["ids"].astype(str).tolist()!=q.id.tolist():raise ValueError("post V1 id drift")
            names=z["labels"].astype(str).tolist()
            if set(names)!=set(labs):raise ValueError("post V1 label drift")
            post=z["logits"][:,[names.index(k) for k in labs]]
        with np.load("/kaggle/temp/final-logits.npz",allow_pickle=False) as z:
            names=z["labels"].astype(str).tolist()
            base=z["logits"][:,[names.index(k) for k in labs]]
        # same method as exact 0.718 source, without training.
        f=labs.tolist().index(FABLE)
        glm=[labs.tolist().index(k) for k in GLM]
        before=base[:,f]-base[:,glm].max(1)
        after=post[:,f]-post[:,glm].max(1)
        protect=(np.isin(base.argmax(1),glm)&(post.argmax(1)==f)
                 &(after-before>=.30)&(after>=.25))
    # Real final score must differ from unprotected counterfactual only
    # on the exact same protected mask.
    altered=actual!=cf
    if np.any(altered &~protect):
        raise AssertionError("new Apex differs outside the fable protection mask: unreconciled decision logic")
    if status!="applied" and np.any(protect):raise AssertionError("unsupported V1 protect")
    ca=counts(y,cf,actual)
    preq=counts(y,pre.model.to_numpy(str),actual)
    all_glm_fable=[FABLE,*GLM]
    subset={}
    for model in all_glm_fable:
        ix=(y==model)
        subset[model]={"truth_rows":int(ix.sum()),
           "baseline_unprotected_correct":int(np.sum(cf[ix]==y[ix])),
           "actual_protected_correct":int(np.sum(actual[ix]==y[ix])),
           "real_changed":int(altered[ix].sum())}
    dataout={
      "status":RESULT_MODE,"no_new_training_or_model_heads":True,
      "source_champion_scored_public_ba":0.718,"source_exact_public_reproduction":True,
      "source_identity_protocol_unverified":True,"historically_spent_panel":True,
      "source_18_label_support":True,"source_rows":len(y),"truth_rows_per_label":14,
      "v1_status":status,"v1_glm_mask_count":int(actual_v1_mask.sum()),
      "fable_protection_eligible_rows":int(protect.sum()),
      "protected_final_predictions_changed":int(altered.sum()),
      "old_unprotected_balanced_accuracy":float((cf==y).mean()),
      "new_protected_balanced_accuracy":float((actual==y).mean()),
      "pre_apex_balanced_accuracy":float((pre.model.to_numpy(str)==y).mean()),
      "paired_old_unprotected_to_real_protected":ca,
      "paired_pre_apex_to_real_protected":preq,
      "specific_truth_classes":subset,
      "real_glm_stage_run":status=="applied",
      "candidate_promote":False,"official_submission":False,
      "no_hidden_test_labels_or_predictions_exported":True,
    }
    OUTPUT.write_text(json.dumps(dataout,indent=2,sort_keys=True)+"\n")
    print("NTL_FABLE_PRESERVATION_REAL252="+json.dumps(dataout,sort_keys=True),flush=True)
    if not protect.any():
        print("NTL_FABLE_PRESERVATION_NO_ELIGIBLE_ROWS_NON_DISCRIMINATING=TRUE",flush=True)
    else:
        print("NTL_FABLE_PRESERVATION_OBSERVED_ONLY_NO_COMPETITION_SUBMIT=TRUE",flush=True)

def selftest():
    labels=["claude-fable-5","claude-opus-5","gemini-3.6-flash","glm-5.3","glm-5.3-flash"]
    pre=["glm-5.3","claude-opus-5","gemini-3.6-flash"]
    refined=["claude-fable-5","claude-fable-5","gemini-3.6-flash"]
    logits=np.zeros((3,len(labels)),np.float64)
    out=replay_typography_without_protect(pre,refined,logits,labels,["simple","simple","simple"])
    assert out.tolist()==["glm-5.3","claude-fable-5","gemini-3.6-flash"],out
    out2=replay_typography_without_protect(pre,refined,logits,labels,["simple","text — here","simple"])
    assert out2[0]=="glm-5.3" and out2[2]=="gemini-3.6-flash"
    assert counts(np.asarray(pre),np.asarray(pre),out)["changed"]==1
    assert OUTPUT.name=="ntl_fable_preservation_252_aggregate.json"
    print("NTL_FABLE_PRESERVATION_CPU_SELFTEST=PASS",flush=True)

if __name__=="__main__":
    if os.getenv("NTL_FABLE_SELFTEST")=="1":selftest()
    else:raise RuntimeError("this module is for the private notebook, not direct command execution")
