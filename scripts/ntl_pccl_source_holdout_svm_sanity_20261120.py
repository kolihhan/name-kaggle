"""One post-hoc CPU sanity check: is the 18-label source-holdout learnable?

Fully supervised sparse SVC trained on either the EXACT 640 presentations seen
by the real QLoRA pilot (C0/C1 shared), or the whole 8347-row frozen training
cohort. Same 1800-row source-disjoint validation. NO hyperparameter selection
or new candidate promotion from repeatedly inspected evidence.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def score(train,valid,labels):
    x1=TfidfVectorizer(analyzer="char",ngram_range=(3,5),max_features=85000,
                      min_df=2,sublinear_tf=True,dtype=np.float32)
    x2=TfidfVectorizer(analyzer="word",ngram_range=(1,2),max_features=35000,
                      min_df=2,sublinear_tf=True,dtype=np.float32,
                      token_pattern=r"(?u)\b\w+\b")
    a=x1.fit_transform(train.text.astype(str))
    b=x2.fit_transform(train.text.astype(str))
    v1=x1.transform(valid.text.astype(str))
    v2=x2.transform(valid.text.astype(str))
    X=sparse.hstack((a,b),format="csr",dtype=np.float32)
    Y=sparse.hstack((v1,v2),format="csr",dtype=np.float32)
    model=LinearSVC(C=1.,class_weight="balanced",dual="auto",
                    random_state=20261120,max_iter=1500)
    model.fit(X,train.model.astype(str))
    pred=np.asarray(model.predict(Y),dtype=str)
    truth=valid.model.to_numpy(dtype=str)
    by={label:float((pred[truth==label]==label).mean()) for label in labels}
    source={}
    for src in sorted(valid.source.unique()):
        rows=valid.source.astype(str).eq(src).to_numpy()
        cats=sorted(set(truth[rows]))
        source[src]={"rows":int(rows.sum()),"n_classes":len(cats),
                     "macro_recall_for_present_classes":round(float(np.mean([
                         (pred[rows][truth[rows]==c]==c).mean() for c in cats])),6)}
    return {
        "BA18":round(float(np.mean(list(by.values()))),6),
        "correct":int((pred==truth).sum()),
        "train_rows":len(train),
        "train_unique_texts":int(train.text.nunique()),
        "train_supported_classes":int(train.model.nunique()),
        "feature_dim":X.shape[1],
        "per_class_recall":{k:round(v,6) for k,v in by.items()},
        "source_breakdown":source,
        "train_fitted_accuracy":round(float(np.mean(model.predict(X)==train.model.astype(str))),6),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    root=args.input
    m=json.loads((root/'manifest.json').read_text())
    for name in ('train','valid'):
        p=root/f'{name}.parquet'
        if sha(p)!=m[f'{name}_sha256']:raise RuntimeError('DATA_HASH_MISMATCH:'+name)
    if sha(root/'batches.json')!=m['batches_sha256']:
        raise RuntimeError('SCHEDULE_SHA_MISMATCH')
    train=pd.read_parquet(root/'train.parquet')
    valid=pd.read_parquet(root/'valid.parquet')
    if len(train)!=8347 or len(valid)!=1800 or len(m['labels'])!=18:
        raise RuntimeError('PCCL_COHORT_CHANGED')
    batches=json.loads((root/'batches.json').read_text())
    rows=[int(idx) for batch in batches for idx in batch['indices']]
    if len(rows)!=640:raise RuntimeError('PCCL_640_PRESENTATION_MISMATCH')
    small=train.iloc[rows].reset_index(drop=True)
    print('NTL_PCCL_CPU_640_INPUT='+json.dumps({
        'exposures':len(small),'unique':int(small.text.nunique()),
        'covered_labels':int(small.model.nunique())
    },sort_keys=True),flush=True)
    a=score(small,valid,m['labels'])
    print('NTL_PCCL_CPU_SAME_EXPOSURE_SVM='+json.dumps({
        'BA18':a['BA18'],'correct':a['correct'],'source':a['source_breakdown']
    },sort_keys=True),flush=True)
    b=score(train,valid,m['labels'])
    print('NTL_PCCL_CPU_ALL_DATA_SVM='+json.dumps({
        'BA18':b['BA18'],'correct':b['correct'],'source':b['source_breakdown']
    },sort_keys=True),flush=True)
    r={
      'study':'NTL_PCCL_POSTHOC_SAME_COHORT_SVM_CAN_WE_LEARN',
      'source_sha':m['source_pool_sha256'],
      'training_rows':len(train),'valid_rows':len(valid),
      'validation_sources':len(m['valid_sources']),
      'all18':True,
      'SVM_same_640_presentations':a,
      'SVM_all_8347_rows':b,
      'published_Qwen17B_CE_BA':0.078889,
      'published_Qwen17B_CEplusCL_BA':0.071667,
      'posthoc_not_independent_selection':True,
      'not_a_test_of_contrastive_objective':True,
      'not_paired_equal_model_or_backbone':True,
      'no_gpu':True,'no_competition_submission':True,
      'no_model_weights_or_raw_text_exported':True
    }
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print('NTL_PCCL_CPU_SOURCE_SHIFT_FALSIFIER_RESULT='+json.dumps({
       'SVM640':a['BA18'],'SVM8347':b['BA18'],
       'QwenCE':0.078889,'QwenPCCL':0.071667,
       'decision_scope':'Posthoc source-learnability check only'
    },sort_keys=True),flush=True)
if __name__=='__main__':main()
