"""Frozen pretrained causal LM token-surprisal fingerprint v1.

No proprietary target generator weights, Champion predictions, hidden test
texts, Kaggle submission, fine tuning, or public-score calibration.
The only fitted components are train-fold-local 18-class linear readouts.
See docs/research/ntl-fcs-v1-contract-20261123.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot
from scripts.ntl_mpcd_v1_20261123 import grouped_ci
from scripts.run_strong_sparse_anchor import build_fold_features, make_svc

SEED=20261123
MODEL="distilbert/distilgpt2"
REV="f241065e938b44ac52db2c5de82c8bd2fafc76d0"
K=18
LIMIT=127
BATCH=6
ARMS=("bos_unigram","real_context","shuffled_context","strong_sparse")
Q=(.10,.25,.50,.75,.90)


def deterministic_shuffle(tokens: list[int], rowid: str):
    rng=np.random.default_rng(int.from_bytes(
        hashlib.sha256(f"ntl-fcs-v1|{SEED}|{rowid}".encode()).digest()[:8],
        "little"))
    return [tokens[int(i)] for i in rng.permutation(len(tokens))]


def stat(a: np.ndarray):
    if not len(a):return np.zeros(7,dtype=np.float64).tolist()
    a=np.asarray(a,dtype=np.float64)
    return [float(np.mean(a)),float(np.std(a)),
            *(float(x) for x in np.quantile(a,Q))]


def id_category(token: str):
    """Five disjoint broad categories, fixed without outcome labels."""
    stem=token.replace("Ġ","").replace("▁","")
    if not any(ch.isalnum() for ch in stem):return 2 # punct
    if any(ch.isdigit() for ch in stem):return 1 # numeric
    if token.startswith("Ġ") or token.startswith("▁"):return 0 # word-start
    if any(ch.isalpha() for ch in stem):return 3 # alphabetic inner
    return 4 # other


def make_feature(tokens: list[int], base: np.ndarray,
                 conditioned: np.ndarray, categories: np.ndarray):
    """Same fixed feature slots for all three causal/no-context arms."""
    n=len(tokens)
    if n<1 or len(base)!=n or len(conditioned)!=n:
        raise AssertionError("feature input misalignment")
    if not np.isfinite(base).all() or not np.isfinite(conditioned).all():
        raise ValueError("nonfinite log-likelihood input")
    uplift=conditioned-base
    components=(conditioned,base,uplift)
    f=[]
    for v in components:
        f+=stat(v)
    for v in components:
        for idx in np.array_split(np.arange(n),4):
            f.append(float(np.mean(v[idx])) if len(idx) else 0.)
    shape=categories[np.asarray(tokens,dtype=np.int64)]
    for v in components:
        for typ in range(5):
            found=shape==typ
            f.append(float(np.mean(v[found])) if found.any() else 0.)
    f += [float(np.mean(shape==typ)) for typ in range(5)]
    f += [float(np.log1p(n)),float(len(set(tokens))/n)]
    if len(f)!=55:raise AssertionError("feature dimension changed")
    a=np.asarray(f,dtype=np.float64)
    if not np.isfinite(a).all():raise AssertionError("bad feature")
    return a


def selftest():
    arr=[9,3,6,3,5,9,4,2]
    sh=deterministic_shuffle(arr,"synthetic-id")
    assert sorted(sh)==sorted(arr)
    cat=np.asarray([i%5 for i in range(12)],dtype=np.int8)
    p=np.array([-4.,-2.,-1.,-4.,-3.,-5.,-2.,-3.])
    f0=make_feature(arr,p,p,cat)
    f1=make_feature(arr,p,p+.2,cat)
    assert len(f0)==55 and len(f1)==55
    assert np.allclose(f0[14:21],0)
    assert np.isclose(f1[14],.2)
    print("NTL_FCS_V1_SYNTHETIC_ALIGNMENT_AND_DIMENSION_TEST=PASS",flush=True)


def pretrained_features(frame):
    """Read texts without fitting foundation model or saving raw passages."""
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer

    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(
        MODEL,revision=REV,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REV,trust_remote_code=False,
        use_safetensors=True,torch_dtype=torch.float32)
    model.eval()
    if not hasattr(tokenizer,"eos_token_id") or tokenizer.eos_token_id is None:
        raise RuntimeError("causal GPT BOS token missing")
    bos=int(tokenizer.eos_token_id)
    raw=frame.text.astype(str).tolist()
    ids=frame.id.astype(str).tolist()
    enc=[tokenizer.encode(v,add_special_tokens=False,
               truncation=True,max_length=LIMIT) for v in raw]
    if not all(len(x)>=1 for x in enc):
        raise RuntimeError("empty tokenized passage")
    shuffled=[deterministic_shuffle(row,id_) for row,id_ in zip(enc,ids)]
    if not all(sorted(x)==sorted(y) for x,y in zip(enc,shuffled)):
        raise RuntimeError("token multiset changed in shuffled control")
    cats=np.asarray([id_category(t) for t in tokenizer.convert_ids_to_tokens(
        list(range(tokenizer.vocab_size)))],dtype=np.int8)
    truncated=sum(len(tokenizer.encode(t,add_special_tokens=False))>LIMIT
                  for t in raw)
    # A single BOS forward gives an ACTUAL, model-native token-marginal
    # distribution over its entire vocabulary (not corpus-empirical counts).
    with torch.inference_mode():
        logits=model(input_ids=torch.tensor([[bos]])).logits[0,-1]
        bos_logp=(logits-torch.logsumexp(logits,dim=-1)).cpu().numpy()
    results={}
    runtime={}
    for variant,orders in (("real",enc),("shuffle",shuffled)):
        begin=time.perf_counter()
        f=np.full((len(enc),55),np.nan,dtype=np.float64)
        bos_f=np.full_like(f,np.nan) if variant=="real" else None
        with torch.inference_mode():
            for start in range(0,len(enc),BATCH):
                sub=orders[start:start+BATCH]
                width=max(len(r) for r in sub)+1
                data=np.full((len(sub),width),bos,dtype=np.int64)
                mask=np.zeros((len(sub),width),dtype=np.int64)
                for k,row in enumerate(sub):
                    data[k,:len(row)+1]=[bos]+row
                    mask[k,:len(row)+1]=1
                input_ids=torch.from_numpy(data)
                attention_mask=torch.from_numpy(mask)
                output=model(input_ids=input_ids,attention_mask=attention_mask)
                pred=output.logits[:,:-1,:]
                target=input_ids[:,1:]
                logp=(torch.gather(pred,-1,target.unsqueeze(-1)).squeeze(-1)
                     -torch.logsumexp(pred,dim=-1)).cpu().numpy()
                for k,row in enumerate(sub):
                    ix=start+k
                    actual=logp[k,:len(row)].astype(np.float64)
                    base=bos_logp[np.asarray(row,dtype=np.int64)].astype(np.float64)
                    f[ix]=make_feature(row,base,actual,cats)
                    if bos_f is not None:
                        bos_f[ix]=make_feature(row,base,base,cats)
                del output,pred,target,logp
                if (start//BATCH+1)%75==0:
                    print("NTL_FCS_V1_REAL_PRETRAINED_FORWARD_PROGRESS="+json.dumps({
                      "arm":variant,"processed":min(len(enc),start+BATCH),
                      "total":len(enc)},sort_keys=True),flush=True)
        if not np.isfinite(f).all():raise RuntimeError("feature extraction incomplete")
        results[variant]=f
        if bos_f is not None:results["bos"]=bos_f
        runtime[variant]=round(time.perf_counter()-begin,1)
        print("NTL_FCS_V1_PRETRAINED_FEATURE_STAGE="+json.dumps({
            "variant":variant,"rows":len(enc),"seconds":runtime[variant],
            "shape":[len(f),55],"num_truncated":truncated,
            "source":MODEL,"model_revision":REV,
            "torch_version":torch.__version__},sort_keys=True),flush=True)
    assert results["real"].shape==results["shuffle"].shape==results["bos"].shape
    return results,runtime,truncated


def paired(y,ref,pred,prompt):
    return {"delta_BA18":round(float(balanced_accuracy_score(y,pred)-
                      balanced_accuracy_score(y,ref)),7),
      "CI95_prompt_bootstrap":grouped_ci(
          y,ref,pred,prompt,seed=SEED,draws=1000),
      "rescues":int(((ref!=y)&(pred==y)).sum()),
      "regressions":int(((ref==y)&(pred!=y)).sum()),
      "prediction_changes":int(np.count_nonzero(ref!=pred))}


def main():
    selftest()
    d=load_snapshot(Path("data/train.csv"),Path("artifacts/phase1/folds.csv"))
    labels,y=np.unique(d.model.astype(str),return_inverse=True)
    if len(d)!=2699 or len(labels)!=K or d.prompt_id.nunique()!=150:
        raise RuntimeError("historic exact official DEV alignment failed")
    folds=d.fold_id.to_numpy()
    prompt=d.prompt_id.astype(str).to_numpy()
    t0=time.perf_counter()
    with threadpool_limits(limits=2):
        features,runtime,truncated=pretrained_features(d)
        predictions={arm:np.full(len(d),-1,dtype=np.int16) for arm in ARMS}
        per_fold=[]
        for fold in range(5):
            tr=np.flatnonzero(folds!=fold)
            va=np.flatnonzero(folds==fold)
            if set(prompt[tr])&set(prompt[va]):
                raise RuntimeError("prompt leakage")
            for arm,key in (("bos_unigram","bos"),
                            ("real_context","real"),
                            ("shuffled_context","shuffle")):
                clf=make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=1.,class_weight="balanced",
                        max_iter=400,solver="lbfgs",random_state=SEED))
                clf.fit(features[key][tr],y[tr])
                predictions[arm][va]=clf.predict(features[key][va]).astype(np.int16)
            a,z,details=build_fold_features(
                d.text.iloc[tr].astype(str).tolist(),
                d.text.iloc[va].astype(str).tolist())
            anchor=make_svc(SEED+fold)
            anchor.fit(a,y[tr])
            predictions["strong_sparse"][va]=anchor.predict(z).astype(np.int16)
            b={arm:round(float(balanced_accuracy_score(
                            y[va],predictions[arm][va])),7) for arm in ARMS}
            per_fold.append({"fold":fold,"BA18":b,"train_rows":len(tr),
                "heldout_rows":len(va),"features":55,
                "sparse_width":details["total_features"]})
            print("NTL_FCS_V1_FOLD_RESULT="+json.dumps(per_fold[-1],sort_keys=True),
                  flush=True)
    if any(np.any(v<0) for v in predictions.values()):
        raise RuntimeError("OOF response coverage failed")
    stats={k:float(balanced_accuracy_score(y,v))
           for k,v in predictions.items()}
    a,b,c=(predictions[k] for k in ARMS[:3])
    contrasts={"B_minus_A":paired(y,a,b,prompt),
               "B_minus_C":paired(y,c,b,prompt),
               "B_minus_D":paired(y,predictions["strong_sparse"],b,prompt)}
    wins_a=sum(z["BA18"]["real_context"]>z["BA18"]["bos_unigram"]
               for z in per_fold)
    wins_c=sum(z["BA18"]["real_context"]>z["BA18"]["shuffled_context"]
               for z in per_fold)
    recalls={k:recall_score(y,v,labels=np.arange(K),
           average=None,zero_division=0) for k,v in predictions.items()}
    maxdrop=float((recalls["bos_unigram"]-recalls["real_context"]).max())
    gate=bool(
      contrasts["B_minus_A"]["delta_BA18"]>=.02 and
      contrasts["B_minus_C"]["delta_BA18"]>=.02 and
      contrasts["B_minus_A"]["CI95_prompt_bootstrap"][0]>0 and
      contrasts["B_minus_C"]["CI95_prompt_bootstrap"][0]>0 and
      wins_a>=3 and wins_c>=3 and maxdrop<=.15
    )
    result={
     "experiment":"FCS-v1 actual frozen pretrained DistilGPT2 causal token surprisals",
     "scope":"historical official train 2699 rows, 150 prompts, 18 classes",
     "PRETRAINED_NOT_TARGET_GENERATOR":True,
     "not_fresh_external_holdout":True,
     "not_a_kaggle_public_score":True,
     "no_0715_champion_dependency":True,
     "no_original_competition_test":True,
     "model":MODEL,"revision":REV,"max_bpe_tokens":LIMIT,
     "arms":{k:{"OOF_BA18":round(float(v),7),
      "recall_per_class":{name:round(float(recalls[k][i]),5)
                           for i,name in enumerate(labels)}}
         for k,v in stats.items()},
     "contrasts":contrasts,"wins_B_vs_A":wins_a,"wins_B_vs_C":wins_c,
     "max_class_drop_B_vs_A":round(maxdrop,6),
     "mechanism_pass":gate,
     "competitive_vs_strong_sparse":stats["real_context"]>=stats["strong_sparse"],
     "folds":per_fold,"cpu_forward_seconds":runtime,
     "num_truncated":truncated,
     "total_wall_seconds":round(time.perf_counter()-t0,1),
     "no_GPU":True,"no_foundation_model_training":True,
     "no_Kaggle_competition_submission":True,
    }
    print("NTL_FCS_V1_FINAL="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":main()
