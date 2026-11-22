"""Original NTL GLM sibling study: distributed token-evidence vs matched controls.

Hypothesis: independent, spatially separated bursts of aligned likelihood
disagreement carry GLM-5.3-vs-Flash information beyond global likelihood stats.
Not a scorer candidate. No champion (0.715) logits are available here; results
are historical official-train prompt-held-out GLM-only *mechanism diagnostics*.

C0: published fixed full+flash feature tables + ordinary paired statistics.
C1: exactly C0 + fixed windowed-disagreement features (real token alignment).
C2: exactly C0 + identical features after shuffling Flash token scores within
     each text half. This preserves all per-text marginal Flash statistics.

No model fitting outside the historical training folds, no Kaggle submission.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED=20261122
FULL="glm-5.3"
FLASH="glm-5.3-flash"
WINDOW=8
PERCENTILE=80
STATS=("n_tok","lp_mean","lp_std","lp_q05","lp_q10","lp_q25",
       "lp_q50","lp_q75","lp_q90","lp_min","ent_mean","ent_std",
       "fdg","surprise_mean","z_mean","logrank_mean","top1","top5",
       "top20","rank_ge100","lp_mean_first","lp_mean_second",
       "bpc","tok_per_char")

def safe_pickle_object_string_ids(path:Path, expected_ids:set[str], expected_n:int):
    """Read string payloads from numpy object-array pickle *without executing it*.

    Numpy's object 'ids' uses pickle. Never call np.load(...allow_pickle=True)
    on a third-party artifact. pickletools.genops parses opcodes without running
    REDUCE / GLOBAL / BUILD. Match only known official IDs, preserve their order.
    """
    import io
    import pickletools
    import zipfile
    from numpy.lib import format as npyformat
    with zipfile.ZipFile(path) as archive:
        raw=archive.read("ids.npy")
    stream=io.BytesIO(raw)
    version=npyformat.read_magic(stream)
    if version==(1,0): shape,fortran,dtype=npyformat.read_array_header_1_0(stream)
    elif version==(2,0):shape,fortran,dtype=npyformat.read_array_header_2_0(stream)
    else:raise ValueError("unknown_numpy_id_header_version")
    if tuple(shape)!=(expected_n,) or not dtype.hasobject or fortran:
        raise ValueError("unexpected_object_ID_array_contract")
    picked=stream.read()
    ids=[]
    for op,arg,_ in pickletools.genops(picked):
        if op.name in ("SHORT_BINUNICODE","BINUNICODE","BINUNICODE8","UNICODE",
                       "SHORT_BINSTRING","BINSTRING","BINBYTES","SHORT_BINBYTES"):
            if isinstance(arg,bytes):
                try:arg=arg.decode("utf-8")
                except UnicodeDecodeError:continue
            if isinstance(arg,str) and arg in expected_ids:
                ids.append(arg)
    if len(ids)!=expected_n or len(set(ids))!=expected_n:
        raise ValueError("pickle_static_ID_extraction_not_unique_or_incomplete")
    return np.asarray(ids,dtype=str)

def load_cache(path:Path, feature_table:pd.DataFrame):
    expected=set(feature_table.id.astype(str))
    expected_n=len(expected)
    ids=safe_pickle_object_string_ids(path,expected,expected_n)
    with np.load(path,allow_pickle=False) as f:
        matrix=np.asarray(f["scores"],dtype=np.float64)
        offsets=np.asarray(f["offsets"],dtype=np.int64)
        prefix=np.asarray(f["n_prefix"],dtype=np.int64)
        tokens=np.asarray(f["tokens"])
    n=len(ids)
    if n!=expected_n or n<100 or not expected.issuperset(ids):
        raise ValueError("cache and published feature-table IDs differ")
    if matrix.ndim!=2 or matrix.shape[1]!=4 or offsets.ndim!=1 or len(offsets)!=n+1 or offsets[0]!=0 or offsets[-1]!=len(matrix):
        raise ValueError("unsupported flattened 4-column score/offset contract")
    if tokens.ndim!=1 or len(tokens)!=len(matrix) or np.any(np.diff(offsets)<0):
        raise ValueError("aligned tokens missing or offsets malformed")
    if prefix.shape!=(n,):
        raise ValueError("unexpected per-passage prefix count")
    if (prefix<0).any() or np.any(prefix>np.diff(offsets)):
        raise ValueError("invalid prefix token counts")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("nonfinite score values")
    # Determine which numeric column is next-token log probability by comparing
    # against the publisher's existing per-row lp_mean feature, without labels.
    published=feature_table.set_index("id").loc[ids,"lp_mean"].to_numpy(dtype=float)
    differences=[]
    for col in range(matrix.shape[1]):
        means=np.asarray([matrix[offsets[i]+prefix[i]:offsets[i+1],col].mean()
            if offsets[i]+prefix[i]<offsets[i+1] else np.nan for i in range(n)])
        if not np.isfinite(means).all():
            raise ValueError("empty_text_suffix_or_nonfinite")
        differences.append(float(np.median(np.abs(means-published))))
    best=int(np.argmin(differences))
    others=sorted(differences)
    if others[0]>.50 or others[1]-others[0]<1.0:
        raise ValueError("cannot_unequivocally_identify_lp_column:"+str([round(x,3) for x in differences]))
    return dict(ids=ids,scores=matrix[:,best],offsets=offsets,
                prefix=prefix,tokens=tokens,lp_column=best,
                lp_median_abs_error=round(others[0],6))

def row_values(c,i):
    a,b=c["offsets"][i:i+2]
    k=int(c["prefix"][i])
    return c["scores"][a+k:b],c["tokens"][a+k:b]

def paired_rows(full,flash):
    if not np.array_equal(full["ids"],flash["ids"]):
        raise ValueError("full and flash row IDs must be identical and ordered")
    if not np.array_equal(full["offsets"],flash["offsets"]):
        raise ValueError("token offsets must be identical")
    if not np.array_equal(full["prefix"],flash["prefix"]):
        raise ValueError("prefix boundaries must be identical")
    if not np.array_equal(full["tokens"],flash["tokens"]):
        raise ValueError("token identities must be exactly aligned")
    for i,key in enumerate(full["ids"]):
        a,t=row_values(full,i)
        b,u=row_values(flash,i)
        if not np.array_equal(t,u):
            raise ValueError("non-identical suffix tokens")
        yield str(key),a,b

def fixed_summary(a,b):
    d=a-b
    return [float(d.mean()),
            float((d>0).mean()),
            float(np.mean(np.sign(d))),
            float(np.percentile(d,90)-np.percentile(d,10)),
            float(np.corrcoef(a,b)[0,1]) if a.std()>1e-7 and b.std()>1e-7 else 0.0]

def shuffled_flash(b,key):
    # Shuffle scores only, not tokens, inside each half; all global statistics
    # including mean log-likelihood remain unchanged.
    seed=int.from_bytes(hashlib.sha256((str(SEED)+"|"+key).encode()).digest()[:8],"big")
    rng=np.random.default_rng(seed)
    c=b.copy()
    middle=len(c)//2
    c[:middle]=rng.permutation(c[:middle])
    c[middle:]=rng.permutation(c[middle:])
    return c

def distributed_features(a,b):
    d=a-b
    if len(d)<16:raise ValueError("too short")
    cutoff=float(np.percentile(np.abs(d),PERCENTILE))
    wins=[]
    strength=[]
    for i in range(0,len(d),WINDOW):
        z=d[i:i+WINDOW]
        strong=np.abs(z)>=cutoff
        if not strong.any():
            wins.append(0.0); strength.append(0.0);continue
        j=int(np.argmax(np.abs(z)))
        wins.append(float(np.sign(z[j])))
        strength.append(float(abs(z[j])))
    w=np.asarray(wins)
    pos=(w>0).astype(float);neg=(w<0).astype(float)
    abs_v=np.asarray(strength)
    def spread(v):
        s=float(v.sum())
        return float((s*s)/(len(v)*np.dot(v,v))) if s>0 else 0.0
    def longest(v):
        cur=best=0
        for item in v:
            cur=cur+1 if item else 0
            best=max(best,cur)
        return float(best)/len(v)
    halves=np.array_split(w,2)
    return [float(w.mean()),float(pos.mean()),float(neg.mean()),
            spread(pos),spread(neg),
            longest(pos),longest(neg),
            float(np.mean(halves[0])-np.mean(halves[1])),
            float(np.mean(np.abs(w))),
            float(np.mean(abs_v>np.median(abs_v)))]

def evaluate(train,folds,fullfeat,flashfeat,full,flash):
    if len(train)!=2699 or train.prompt_id.nunique()!=150:
        raise ValueError("official train size or prompt count differs")
    if len(folds)!=2699:raise ValueError("frozen fold size differs")
    if set(folds.id)!=set(train.id):
        raise ValueError("frozen fold and official train ID mismatch")
    if set(fullfeat.id)!=set(flashfeat.id) or set(fullfeat.id)!=set(full["ids"]) or len(fullfeat)<100:
        raise ValueError("cache and both published feature-table IDs must exactly match")
    train=train.merge(folds[['id','prompt_id','model','fold_id']],on='id',suffixes=('','_fold'),
                      validate='one_to_one')
    if (train.prompt_id!=train.prompt_id_fold).any() or (train.model!=train.model_fold).any():
        raise ValueError("frozen fold prompt/model mismatch")
    ids = list(full["ids"].astype(str))
    f=fullfeat.set_index('id').loc[ids]
    l=flashfeat.set_index('id').loc[ids]
    xbase=np.array([np.r_[f.loc[k,list(STATS)].to_numpy(dtype=float),
                            l.loc[k,list(STATS)].to_numpy(dtype=float)] for k in ids],dtype=float)
    byid=train.set_index('id')
    label=np.array([byid.loc[k,'model'] for k in ids])
    fold=np.array([int(byid.loc[k,'fold_id']) for k in ids])
    prompt=np.array([str(byid.loc[k,'prompt_id']) for k in ids])
    for p in set(prompt):
        if len(set(fold[prompt==p]))!=1:
            raise ValueError("prompt crosses frozen folds")
    allowed=np.isin(label,[FULL,FLASH])
    if allowed.sum()<80 or set(np.unique(fold[allowed]))!=set(range(5)):
        raise ValueError("need enough GLM sibling rows from published cache across 5 frozen folds")
    diff=np.zeros((len(ids),5),dtype=float)
    local=np.zeros((len(ids),10),dtype=float)
    shuffled=np.zeros((len(ids),10),dtype=float)
    for i,(_id,a,b) in enumerate(paired_rows(full,flash)):
        if not allowed[i]:continue
        if len(a)<16:
            raise ValueError("target_GLM_passage_too_short_for_two_windows")
        diff[i]=fixed_summary(a,b)
        local[i]=distributed_features(a,b)
        shuffled[i]=distributed_features(a,shuffled_flash(b,_id))
    base=np.concatenate([xbase,diff],axis=1)[allowed]
    inputs={"C0_global":base,"C1_distributed":np.c_[base,local[allowed]],
            "C2_misaligned":np.c_[base,shuffled[allowed]]}
    ys=(label[allowed]==FULL).astype(int)
    ff=fold[allowed]
    pr=prompt[allowed]
    if ys.sum()<35 or (len(ys)-ys.sum())<35:raise ValueError("not enough GLM examples in cache")
    preds={}
    for name,x in inputs.items():
        if not np.isfinite(x).all():raise ValueError("nonfinite features")
        oof=np.full(len(ys),-1,dtype=int)
        for k in range(5):
            tr,va=ff!=k,ff==k
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,random_state=SEED,
                class_weight="balanced",max_iter=2000))
            clf.fit(x[tr],ys[tr])
            oof[va]=clf.predict(x[va])
        if -1 in oof:raise ValueError("missing heldout predictions")
        preds[name]=oof
    def ba(pred):
        return .5*(np.mean(pred[ys==1]==1)+np.mean(pred[ys==0]==0))
    scores={k:round(float(ba(v)),7) for k,v in preds.items()}
    basepred=preds["C0_global"]; newpred=preds["C1_distributed"]
    rescues=int(np.sum((newpred==ys)&(basepred!=ys)))
    regressions=int(np.sum((newpred!=ys)&(basepred==ys)))
    fold_gain=[round(float(np.mean(newpred[ff==k]==ys[ff==k])
                           -np.mean(basepred[ff==k]==ys[ff==k])),6) for k in range(5)]
    # Prompt-level paired bootstrap is diagnostic on spent official prompts, not
    # evidence of independent competition transfer.
    clusters={p:np.flatnonzero(pr==p) for p in np.unique(pr)}
    keys=list(clusters)
    rng=np.random.default_rng(SEED)
    deltas=[]
    for _ in range(1000):
        sample=np.concatenate([clusters[keys[j]] for j in rng.integers(0,len(keys),size=len(keys))])
        if set(ys[sample])!={0,1}:continue
        deltas.append(ba(newpred[sample]) - ba(basepred[sample]) if False else
            .5*sum(np.mean(newpred[sample][ys[sample]==y]==y)-
                   np.mean(basepred[sample][ys[sample]==y]==y) for y in (0,1)))
    ci=np.percentile(deltas,[2.5,97.5]) if deltas else [float('nan')]*2
    gain=scores["C1_distributed"]-max(scores["C0_global"],scores["C2_misaligned"])
    gate=(gain>=.02 and sum(v>0 for v in fold_gain)>=4 and ci[0]>0)
    return dict(scope="spent_official_train_glm_only",
                cases=int(len(ys)),prompts=int(len(keys)),folds=5,
                base_0p715_full18_comparison="NOT_AVAILABLE",
                binary_BA=scores,
                rescues=rescues,regressions=regressions,net=rescues-regressions,
                fold_accuracy_delta=fold_gain,paired_bootstrap_ci=[round(float(x),5) for x in ci],
                c1_vs_best_control=round(float(gain),7),
                mechanism_gate_pass=bool(gate),
                full18_if_only_two_classes_change_upper_bound_note="binary BA gain times 2/18; cannot transfer directly to champion")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("train","folds","glm53_features","flash_features","glm53_scores","flash_scores"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    train=pd.read_csv(a.train,dtype={"id":str,"prompt_id":str,"model":str})
    folds=pd.read_csv(a.folds,dtype={"id":str,"prompt_id":str,"model":str})
    fullfeat=pd.read_csv(a.glm53_features,dtype={"id":str})
    flashfeat=pd.read_csv(a.flash_features,dtype={"id":str})
    f=load_cache(a.glm53_scores,fullfeat);l=load_cache(a.flash_scores,flashfeat)
    print("NTL_GLM_SAFE_CACHE_CONTRACT="+json.dumps({"rows":len(f["ids"]),
        "full_lp_column":f["lp_column"],"flash_lp_column":l["lp_column"],
        "full_mean_abs_err":f["lp_median_abs_error"],
        "flash_mean_abs_err":l["lp_median_abs_error"]}))
    result=evaluate(train,folds,fullfeat,flashfeat,f,l)
    print("NTL_DISTRIBUTED_GLM_REAL_OOF="+json.dumps(result,sort_keys=True))
