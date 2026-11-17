#!/usr/bin/env python3
"""Zero-GPU, descriptive Oleg216 post-V1 logit audit. No training, no submission.

Uses one spent sensitive-topics source, not an independent Kaggle test.
No raw per-row model output or source text is printed.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

N=216
K=18
SEED=20261117
TAUS=(0.0,0.02,0.05,0.1)
MIN_AUC=0.65
BOOT=1500

def table(path):
    x=(pd.read_parquet(path) if path.suffix=='.parquet' else
       pd.read_csv(path,dtype={'id':str}))
    if not {'id','model'}<=set(x.columns) or x.id.isna().any() or x.model.isna().any():
        raise ValueError('bad table schema')
    x=x[['id','model']].astype(str)
    if x.id.duplicated().any():raise ValueError('duplicate row')
    return x.set_index('id')

def ba(y,p,labels):
    return float(np.mean([np.mean(p[y==c]==c) for c in labels]))

def choose(scores,labels,tau):
    order=np.argsort(scores,axis=1,kind='stable')
    top=order[:,-1];second=order[:,-2]
    prob=softmax(scores.astype(np.float64),axis=1)
    margin=prob[np.arange(len(top)),top]-prob[np.arange(len(top)),second]
    return labels[np.where(margin<tau,second,top)],margin,prob

def bootstrap_delta(y,base,alt,labels,reps=BOOT):
    rng=np.random.default_rng(SEED)
    locations=[np.flatnonzero(y==c) for c in labels]
    values=[]
    for _ in range(reps):
        ix=np.concatenate([rng.choice(i,size=len(i),replace=True) for i in locations])
        values.append(ba(y[ix],alt[ix],labels)-ba(y[ix],base[ix],labels))
    return [round(float(v),6) for v in np.quantile(values,[.025,.5,.975])]

def eval_oof(y,base,scores,labels):
    # ONE parameter selected inside folds, then evaluated on held-out rows.
    cv=RepeatedStratifiedKFold(n_splits=3,n_repeats=5,random_state=SEED)
    candidates={tau:choose(scores,labels,tau)[0] for tau in TAUS}
    gains=np.zeros(N);counts=np.zeros(N,dtype=int)
    choices=[]
    for train,test in cv.split(np.zeros(N),y):
        val={tau:ba(y[train],pred[train],labels)-ba(y[train],base[train],labels)
             for tau,pred in candidates.items()}
        tau=max(TAUS,key=lambda t:(val[t],-t))
        choices.append(tau)
        gains[test]+=(candidates[tau][test]==y[test])-(base[test]==y[test])
        counts[test]+=1
    if not (counts==5).all():raise ValueError('bad split')
    gains/=counts
    estimate=float(np.mean([gains[y==c].mean() for c in labels]))
    rng=np.random.default_rng(SEED+1)
    per_class=[gains[y==c] for c in labels]
    bs=[]
    for _ in range(BOOT):
        bs.append(np.mean([rng.choice(z,size=len(z),replace=True).mean() for z in per_class]))
    interval=[round(float(v),6) for v in np.quantile(bs,[.025,.5,.975])]
    return {'paired_macro_oof_delta':round(estimate,6),'bootstrap95_descriptive':interval,
            'selected_threshold_counts':{str(t):choices.count(t) for t in TAUS},
            'kill_global_rank_switch':bool(interval[0]<=0),'not_independent':True}

def analyze(dump,baseline,truth):
    rows=[json.loads(line) for line in Path(dump).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows)!=N:raise ValueError('INCOMPLETE_216_LOGITS')
    labels=np.asarray(rows[0]['labels'],dtype=str)
    if len(labels)!=K or len(set(labels))!=K:raise ValueError('BAD_18_LABELS')
    if any(x['labels']!=rows[0]['labels'] for x in rows):raise ValueError('LABEL_ORDER_DRIFT')
    ids=[str(x['sample_id']) for x in rows]
    if len(set(ids))!=N:raise ValueError('DUPLICATE_ID')
    z=np.asarray([x['logits'] for x in rows],dtype=np.float32)
    if z.shape!=(N,K) or not np.isfinite(z).all():raise ValueError('BAD_FLOAT32_LOGITS')
    pred=np.asarray(labels)[z.argmax(1)]
    if not np.array_equal(pred,np.asarray([x['argmax'] for x in rows])):
        raise ValueError('FLOAT32_ARGMAX_DRIFT')
    oldtable=table(Path(baseline));truthtable=table(Path(truth))
    if len(oldtable)!=N or set(ids)!=set(oldtable.index) or not set(ids)<=set(truthtable.index):
        raise ValueError('BASELINE_OR_TRUTH_ID_DRIFT')
    old=oldtable.loc[ids].model.to_numpy();y=truthtable.loc[ids].model.to_numpy()
    if set(y)!=set(labels) or any((y==c).sum()!=12 for c in labels):
        raise ValueError('NOT_FROZEN_12_PER_CLASS_OLEG')
    agreement=float((pred==old).mean())
    out={'n':N,'n_classes':K,'baseline_agreement':round(agreement,6),
         'first5_match':bool(np.array_equal(pred[:5],old[:5])),
         'official_0_691_reference_is_not_this_dataset':True,
         'new_gpu':False,'training':False,'new_submission':False}
    if agreement<.99 or not out['first5_match']:
        return dict(out,decision='CHOOSE_B',reason='original_argmax_consistency_failed')
    correct=(pred==y).astype(int)
    _,margin,probs=choose(z,labels,0)
    auc=(float(roc_auc_score(correct,margin)) if len(set(correct))==2 else None)
    conf=np.max(probs,axis=1)
    ece=0.
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        sel=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if sel.any():ece+=sel.mean()*abs(float(correct[sel].mean())-float(conf[sel].mean()))
    q=lambda flag:([round(float(v),6) for v in np.quantile(margin[flag],[.1,.25,.5,.75,.9])] if flag.any() else [])
    out.update({'old_Oleg_BA':round(ba(y,old,labels),6),
        'new_Oleg_BA':round(ba(y,pred,labels),6),
        'correct':int(correct.sum()),'wrong':int(N-correct.sum()),
        'margin_correctness_AUROC':(round(auc,6) if auc is not None else None),
        'ECE15':round(ece,6),'margin_correct_q10_25_50_75_90':q(correct==1),
        'margin_wrong_q10_25_50_75_90':q(correct==0),
        'threshold_predeclared':MIN_AUC})
    if auc is None or auc<MIN_AUC:
        return dict(out,decision='CHOOSE_B',reason='margin_AUROC_below_0.65')
    variants={}
    for tau in TAUS:
        alt,_,_=choose(z,labels,tau)
        variants[str(tau)]={'delta_BA_same_Oleg':round(ba(y,alt,labels)-ba(y,pred,labels),6),
            'rescues':int(((pred!=y)&(alt==y)).sum()),
            'regressions':int(((pred==y)&(alt!=y)).sum()),
            'changes':int((pred!=alt).sum()),
            'paired_bootstrap95':bootstrap_delta(y,pred,alt,labels)}
    oof=eval_oof(y,pred,z,labels)
    out.update({'global_one_parameter_rank_switch':variants,'repeated_CV':oof})
    out['decision']='KILL_RANK_SWITCH' if oof['kill_global_rank_switch'] else 'EXPLORATORY_GLOBAL_CANDIDATE'
    out['note']='Oleg is spent; even positive OOF does not establish Public/Private uplift'
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dump',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--truth',type=Path,required=True)
    args=p.parse_args()
    print('NTL_OLEG216_GLOBAL_CPU_AUDIT='+json.dumps(
        analyze(args.dump,args.baseline,args.truth),sort_keys=True,allow_nan=False),flush=True)

if __name__=='__main__':main()
