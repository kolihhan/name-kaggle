"""One-shot syntax-order *mechanism falsifier*, NOT competitive NTL model.

Compare lexical-free sentence POS bags (C0) vs real grammatical sequence (C1)
vs deterministic within-sentence order-shuffled POS control (C2); same 18 labels,
same prompt-heldout folds, same model family and vectorization hyperparameters.
Also evaluate frozen train-fit on external true 18-class Oleg216 and compare
row-matched with the old 0.691 complete strong-model predictions.

No paid API, no QLoRA, no foundation-model inference, no hidden test, no
competition submission. Oleg216 has already been studied, and all its topics
are narrow; results can falsify but not validate prospective transfer.
"""
from __future__ import annotations
import hashlib
import json
import re
import time
import argparse
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC
from scripts.ntl_oleg18_frozen_selector_20261116 import (
    NTL_OLEG18_SHA, NTL_OLEG18_MODELS, ntl_oleg18_select,
)

TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
LABELS=tuple(sorted(NTL_OLEG18_MODELS))
SEED=20261123
FOLDS=5
# Universal Penn tagset mapped to coarse grammatical classes; not lexical.
TAG_CLASSES={
    "CC":"CONJ","CD":"NUM","DT":"DET","EX":"OTHER","FW":"OTHER",
    "IN":"PREP","JJ":"ADJ","JJR":"ADJ","JJS":"ADJ","LS":"OTHER",
    "MD":"AUX","NN":"NOUN","NNS":"NOUN","NNP":"PROPN","NNPS":"PROPN",
    "PDT":"DET","POS":"PRON","PRP":"PRON","PRP$":"PRON","RB":"ADV",
    "RBR":"ADV","RBS":"ADV","RP":"PART","SYM":"OTHER","TO":"PREP",
    "UH":"INTJ","VB":"VERB","VBD":"VERB","VBG":"VERB","VBN":"VERB",
    "VBP":"VERB","VBZ":"VERB","WDT":"DET","WP":"PRON","WP$":"PRON",
    "WRB":"ADV"
}
ARM=["C0_POS_bag","C1_POS_order","C2_POS_permuted_order"]
WORD_RE=re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
SENT_RE=re.compile(r"(?<=[.!?])\s+|\n+|(?<=;)\s+")

def emit(name,obj):
    print("NTL_SYNTAX_"+name+"="+json.dumps(obj,sort_keys=True,ensure_ascii=False),flush=True)

def freeze(path,sha):
    with path.open("rb") as f:
        actual=hashlib.file_digest(f,"sha256").hexdigest()
    if actual!=sha:
        raise ValueError("FROZEN_SOURCE_DRIFT:"+path.name)

def load_data(train_path,fold_path,external_dir,oldpred_path):
    freeze(train_path,TRAIN_SHA)
    tr=pd.read_csv(train_path,dtype={"id":str,"prompt_id":str})
    fs=pd.read_csv(fold_path,dtype={"id":str,"prompt_id":str})
    assert (len(tr)==2699 and tr.prompt_id.nunique()==150 and
            set(tr.model)==set(LABELS) and len(fs)==2699 and fs.id.is_unique)
    need={"id","prompt_id","model","fold_id"}
    assert need.issubset(fs.columns)
    fs=fs[list(need)].copy()
    m=tr.merge(fs,on="id",suffixes=("","_fold"),validate="one_to_one")
    assert len(m)==2699
    assert (m.model==m.model_fold).all() and (m.prompt_id==m.prompt_id_fold).all()
    assert set(m.fold_id)==set(range(FOLDS))
    assert m.groupby("prompt_id").fold_id.nunique().eq(1).all()
    for file,sha in NTL_OLEG18_SHA.items():freeze(external_dir/file,sha)
    wide=pd.read_parquet(external_dir/"external_short_wide.parquet")
    meta=pd.read_parquet(external_dir/"external_cell_metadata.parquet")
    ext=ntl_oleg18_select(wide,meta)
    assert (len(ext)==216 and ext.model.value_counts().eq(12).all() and
            ext.prompt_group_id.nunique()==216 and set(ext.model)==set(LABELS))
    old=pd.read_parquet(oldpred_path)
    assert (set(old.columns)=={"id","model"} and len(old)==216
            and old.id.is_unique and set(old.id)==set(ext.id))
    ext=ext[["id","model","text","prompt_group_id"]].merge(
        old.rename(columns={"model":"old_strong_pred"}),on="id",validate="one_to_one")
    assert int((ext.model==ext.old_strong_pred).sum())==89
    return m,ext

def suffix_tag(sentences):
    # One NLTK pretrained, frozen POS tagger provides only 12-class coarse tags.
    # The original actual model labels are NOT used during feature extraction.
    result=[]
    for item in sentences:
        segs=[]
        for sentence in SENT_RE.split(str(item)):
            words=WORD_RE.findall(sentence)
            if not words:continue
            tags=nltk.pos_tag(words)
            segs.append([TAG_CLASSES.get(tag,"OTHER") for _,tag in tags])
        result.append(segs)
    return result

def permute_tags(parts,index):
    rows=[]
    for i,part in enumerate(parts):
        row=[]
        for j,sent in enumerate(part):
            ordered=list(sent)
            if len(ordered)>1:
                h=hashlib.sha256((str(SEED)+"|"+str(index[i])+"|"+str(j)).encode()).digest()
                rng=np.random.default_rng(int.from_bytes(h[:8],"big"))
                ordered=list(rng.permutation(ordered))
            row.extend(ordered+["SENT"])
        rows.append(row)
    return rows

def real_tags(parts):
    return [sum((list(seg)+["SENT"] for seg in text),[]) for text in parts]

def make_arm_docs(tags,control,index):
    if control=="C2_POS_permuted_order":
        return [" ".join(t) for t in permute_tags(tags,index)]
    return [" ".join(t) for t in real_tags(tags)]

def fit_predict(train_docs,train_y,val_docs,arm):
    # All three arms: same C, class balance, IDF, vocab cap and no leakage.
    ngrams=(1,1) if arm=="C0_POS_bag" else (1,3)
    vec=TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",
        ngram_range=ngrams,sublinear_tf=True,min_df=2,max_features=4096,
        lowercase=False,norm="l2")
    Xt=vec.fit_transform(train_docs)
    Xv=vec.transform(val_docs)
    estimator=LinearSVC(C=1.0,class_weight="balanced",dual="auto",
                        max_iter=3000,random_state=SEED)
    estimator.fit(Xt,train_y)
    return estimator.predict(Xv),len(vec.vocabulary_)

def ba(y,p):
    return round(float(balanced_accuracy_score(y,p)),8)

def scorer(y,p):
    return {"accuracy":round(float(np.mean(y==p)),8),
            "BA18":ba(y,p),
            "correct":int(np.sum(y==p))}

def paired_stats(y,base,new,groups,seed=SEED):
    a=base==y;b=new==y
    keys=list(sorted(set(groups)))
    inds={g:np.flatnonzero(np.asarray(groups)==g) for g in keys}
    rng=np.random.default_rng(seed)
    boots=[]
    for _ in range(600):
        select=rng.integers(0,len(keys),size=len(keys))
        ix=np.concatenate([inds[keys[k]] for k in select])
        if len(set(y[ix]))<len(LABELS):continue
        boots.append(balanced_accuracy_score(y[ix],new[ix])
                     -balanced_accuracy_score(y[ix],base[ix]))
    return {"rescues":int((b&(~a)).sum()),
            "regressions":int(((~b)&a).sum()),
            "net_correct":int(b.sum()-a.sum()),
            "BA_delta":round(ba(y,new)-ba(y,base),8),
            "resampling_units":len(keys),
            "bootstrap_ci95_conditional_diagnostic":[round(float(z),7)
                for z in np.quantile(boots,[.025,.975])] if boots else None}

def run(train_path,folds_path,external_dir,oldpred_path):
    start=time.monotonic()
    data,ext=load_data(train_path,folds_path,external_dir,oldpred_path)
    y=data.model.to_numpy(dtype=str)
    y_external=ext.model.to_numpy(dtype=str)
    old=ext.old_strong_pred.to_numpy(dtype=str)
    # One source-neutral extraction pass before labels are used for training.
    parts=suffix_tag(data.text.fillna("").astype(str).tolist()+
                     ext.text.fillna("").astype(str).tolist())
    ids=list(data.id.astype(str))+list(ext.id.astype(str))
    docs={arm:make_arm_docs(parts,arm,ids) for arm in ARM}
    n=len(data)
    oof={arm:np.empty(n,dtype=object) for arm in ARM}
    vocabs={arm:[] for arm in ARM}
    for fold in range(FOLDS):
        tr=np.where(data.fold_id.to_numpy()!=fold)[0]
        va=np.where(data.fold_id.to_numpy()==fold)[0]
        for arm in ARM:
            pred,k=fit_predict([docs[arm][i] for i in tr],y[tr],
                               [docs[arm][i] for i in va],arm)
            oof[arm][va]=pred
            vocabs[arm].append(k)
    exts={}
    for arm in ARM:
        exts[arm],_=fit_predict(docs[arm][:n],y,docs[arm][n:],arm)
    assert all(set(oof[a])<=set(LABELS) and
               set(exts[a])<=set(LABELS) for a in ARM)
    # Both rows and labels are truly aligned; do not score the old model on TRAIN.
    oscores={arm:scorer(y,oof[arm]) for arm in ARM}
    escores={arm:scorer(y_external,exts[arm]) for arm in ARM}
    oof_decision=paired_stats(y,oof[ARM[0]],oof[ARM[1]],
                              data.prompt_id.to_numpy(dtype=str))
    oof_shuffle=paired_stats(y,oof[ARM[2]],oof[ARM[1]],
                             data.prompt_id.to_numpy(dtype=str))
    external_decision=paired_stats(y_external,exts[ARM[0]],exts[ARM[1]],
                        ext.prompt_group_id.to_numpy(dtype=str))
    external_shuffle=paired_stats(y_external,exts[ARM[2]],exts[ARM[1]],
                        ext.prompt_group_id.to_numpy(dtype=str))
    strong_vs_new=paired_stats(y_external,old,exts[ARM[1]],
                        ext.prompt_group_id.to_numpy(dtype=str))
    # OOD labels/source already spent; no prospective transfer claim.
    external_family_correct={}
    family_map={"claude-":"anthropic","gemini-":"google","glm-":"zai",
                "gpt-":"openai","kimi-":"moonshot","qwen":"alibaba"}
    for prefix,fn in family_map.items():
        ix=np.array([str(v).startswith(prefix) for v in y_external])
        if ix.sum()==0:continue
        external_family_correct[fn]={
            "n":int(ix.sum()),
            "historical_strong_correct":int((old[ix]==y_external[ix]).sum()),
            "new_syntax_correct":int((exts[ARM[1]][ix]==y_external[ix]).sum())}
    emit("RESULT",{
        "study":"predeclared coarse-POS syntax temporal-order probe",
        "view":"lexical_content_removed_universal_penn_tag_classes",
        "official_rows":n,"official_prompts":int(data.prompt_id.nunique()),
        "external_rows":len(ext),"external_class_count":18,
        "external_all_sensitive_topics":True,
        "external_scope_is_spent_historical":True,
        "strong_reference":"frozen historical 0.691 Lightning V2, NOT current 0.715",
        "strong_external":scorer(y_external,old),
        "official_prompt_grouped_OOF":oscores,
        "external_source_transfer":escores,
        "official_C1_vs_C0":oof_decision,
        "official_C1_vs_permuted":oof_shuffle,
        "external_C1_vs_C0":external_decision,
        "external_C1_vs_permuted":external_shuffle,
        "external_syntax_C1_vs_historic_strong_replacement":strong_vs_new,
        "external_family_correct":external_family_correct,
        "feature_vocab_sizes_per_fold":vocabs,
        "elapsed_seconds":round(time.monotonic()-start,2),
        "no_Qlora":True,"no_paid_API":True,
        "no_frozen_LLM_inference":True,
        "no_champion_modification":True,
        "no_Kaggle_submission":True,
        "no_comparison_to_current_0715":True,
        "negative_result_veto":"if syntax C1 <= POS bag C0 or fails external transfer, RETIRE"
        })
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--train",type=Path,required=True)
    a.add_argument("--folds",type=Path,required=True)
    a.add_argument("--external",type=Path,required=True)
    a.add_argument("--old-pred",type=Path,required=True)
    p=a.parse_args()
    run(p.train,p.folds,p.external,p.old_pred)
