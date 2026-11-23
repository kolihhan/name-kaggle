"""GCTH-v1: source-conditioned categorical next-token generative heads.

Independent CPU-only historical 18-label research. Fits only tiny output heads
to frozen distilgpt2 prior-token causal hidden states. Not proprietary target
LM likelihood, not trained on external generations, not a competition entry.
Contract: docs/research/ntl-gcth-v1-contract-20261123.md.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score, recall_score
from threadpoolctl import threadpool_limits

from scripts.blind_prompt_contrast import load_snapshot
from scripts.ntl_mpcd_v1_20261123 import grouped_ci
from scripts.run_strong_sparse_anchor import build_fold_features, make_svc

SEED = 20261123
MODEL = "distilbert/distilgpt2"
REV = "f241065e938b44ac52db2c5de82c8bd2fafc76d0"
NCLASS = 18
CTX_DIM = 64
LENGTH = 64
BATCH_DOC = 12
OUTPUT_SIZE = 128
TOP_VOCAB = OUTPUT_SIZE - 1
EPOCHS = 3
BATCH_TOKEN = 1024
LR = .01
REGULARIZER = .02
PRIOR_ALPHA = 64.
GLOBAL_ALPHA = .5
ARMS = ("model_unigram", "genuine_context", "shuffled_context", "strong_sparse")


def validate_static_math():
    """All output categories form a genuine probability simplex."""
    counts = np.array([[2, 0, 5], [0, 2, 1]], dtype=np.int64)
    pooled = counts.sum(axis=0)
    p0 = (pooled + .5) / (pooled.sum() + .5 * 3)
    priors = (counts + 64 * p0[None, :]) / (counts.sum(axis=1)[:, None] + 64)
    assert np.allclose(priors.sum(axis=1), 1)
    assert np.all(priors > 0)
    x = np.array([[.1, 0, .2], [.4, .5, .1]])
    normalized = x - logsumexp(x, axis=1, keepdims=True)
    assert np.allclose(np.exp(normalized).sum(axis=1), 1.0)
    print("NTL_GCTH_V1_NORMALIZATION_AND_PRIOR_SELFTEST=PASS", flush=True)


def frozen_hidden(frame):
    """Get GPT2Model hidden state at t-1 to predict token t, all responses."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    begin = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=False)
    net = AutoModel.from_pretrained(MODEL, revision=REV, trust_remote_code=False,
                                    use_safetensors=True, torch_dtype=torch.float32)
    net.eval()
    bos = tok.eos_token_id
    if bos is None:
        raise ValueError("GPT2 BOS/EOS must be available")
    texts = frame.text.astype(str).tolist()
    tokens = [tok.encode(t, add_special_tokens=False, truncation=True,
                         max_length=LENGTH) for t in texts]
    if any(not v for v in tokens):
        raise ValueError("no empty passages supported")
    truncated = 0
    for text, arr in zip(texts, tokens):
        # Tokenizer length is only for numeric metadata; no content leaves job.
        total = len(tok.encode(text, add_special_tokens=False))
        truncated += total > len(arr)
    rng = np.random.default_rng(SEED)
    raw = rng.standard_normal((int(net.config.n_embd), CTX_DIM))
    project, _ = np.linalg.qr(raw, mode="reduced")
    P = torch.tensor(np.asarray(project, dtype=np.float32))
    embeddings, targets, doc_numbers = [], [], []
    with torch.inference_mode():
        for i in range(0, len(tokens), BATCH_DOC):
            rows = tokens[i:i+BATCH_DOC]
            size = max(map(len, rows)) + 1
            # The only context used to predict position j is [BOS] + t_<j.
            data = np.full((len(rows), size), bos, dtype=np.int64)
            masks = np.zeros_like(data)
            for pos, ts in enumerate(rows):
                data[pos, :len(ts)+1] = [bos] + ts
                masks[pos, :len(ts)+1] = 1
            result = net(input_ids=torch.from_numpy(data),
                         attention_mask=torch.from_numpy(masks))
            hidden = result.last_hidden_state[:, :-1, :] @ P
            for pos, ts in enumerate(rows):
                n = len(ts)
                embeddings.append(hidden[pos, :n, :].numpy().astype(np.float32))
                targets.append(np.asarray(ts, dtype=np.int32))
                doc_numbers.append(np.full(n, i+pos, dtype=np.int16))
            if (i//BATCH_DOC + 1) % 55 == 0:
                print("NTL_GCTH_V1_PRETRAINED_CONTEXT_PROGRESS=" +
                      json.dumps({"completed": min(len(tokens), i+BATCH_DOC),
                                  "total": len(tokens)}), flush=True)
    H = np.ascontiguousarray(np.concatenate(embeddings), dtype=np.float32)
    token_ids = np.concatenate(targets)
    doc_ids = np.concatenate(doc_numbers)
    if len(H) != len(token_ids) or len(H) != len(doc_ids) or H.shape[1] != CTX_DIM:
        raise ValueError("flattening mismatch")
    if not np.isfinite(H).all():
        raise FloatingPointError("nonfinite pretrained hidden vectors")
    print("NTL_GCTH_V1_FROZEN_FEATURES=" +
          json.dumps({"rows":len(tokens), "contexts":len(H),
                      "dim": CTX_DIM, "truncated_responses":int(truncated),
                      "seconds":round(time.perf_counter()-begin,1),
                      "model":MODEL,"revision":REV}), flush=True)
    return H, token_ids, doc_ids, {"truncated":int(truncated),
        "extraction_seconds":round(time.perf_counter()-begin,1),
        "total_contexts":len(H)}


def fold_categories(train_ids: np.ndarray, all_ids: np.ndarray,
                    fold: int):
    # Ties go to lower numeric ID. No frequency information from validation.
    counts = Counter(int(x) for x in train_ids)
    top = [id_ for id_, _ in sorted(counts.items(),
                  key=lambda x:(-x[1],x[0]))[:TOP_VOCAB]]
    vocab_map = {id_:i for i,id_ in enumerate(top)}
    out = np.fromiter((vocab_map.get(int(t), TOP_VOCAB) for t in all_ids),
                      dtype=np.int16, count=len(all_ids))
    assert len(top) == TOP_VOCAB and int(out.min()) >= 0
    # Avoid printing token IDs or words to public CI.
    return out


def compute_priors(categories: np.ndarray, model_label: np.ndarray):
    global_ct = np.bincount(categories, minlength=OUTPUT_SIZE)
    p0 = (global_ct+GLOBAL_ALPHA)/(global_ct.sum()+GLOBAL_ALPHA*OUTPUT_SIZE)
    per = np.stack([np.bincount(categories[model_label==m],
                   minlength=OUTPUT_SIZE) for m in range(NCLASS)], axis=0)
    dist = (per+PRIOR_ALPHA*p0[None,:])/(per.sum(axis=1)[:,None]+PRIOR_ALPHA)
    if not np.allclose(dist.sum(axis=1),1.) or np.any(dist<=0):
        raise AssertionError("bad source conditional category prior")
    return np.log(dist).astype(np.float32)


def shuffle_donors_per_model(H:np.ndarray, y:np.ndarray,
                             token_docs:np.ndarray, fold:int):
    X = H.copy()
    valid = []
    for m in range(NCLASS):
        ix = np.flatnonzero(y==m)
        rng = np.random.default_rng(SEED+37*fold+7919*m)
        donors = ix[rng.permutation(len(ix))]
        cross = np.mean(token_docs[ix]!=token_docs[donors])
        valid.append(float(cross))
        X[ix] = H[donors]
    if min(valid)<.95:
        raise AssertionError("shuffle failed distinct-document donor requirement")
    return X, min(valid)


def fit_heads(train_H, train_cat, train_source, prior, fold,
              *, experiment: str):
    """Train only 64x128 source-specific residual token heads, no backbone."""
    import torch
    import torch.nn.functional as F

    Ws = []
    t0 = time.perf_counter()
    for m in range(NCLASS):
        idx = np.flatnonzero(train_source==m)
        x = torch.from_numpy(np.ascontiguousarray(train_H[idx],dtype=np.float32))
        targets = torch.from_numpy(np.ascontiguousarray(train_cat[idx],dtype=np.int64))
        base = torch.tensor(prior[m],dtype=torch.float32)
        W = torch.nn.Parameter(torch.zeros(CTX_DIM,OUTPUT_SIZE,dtype=torch.float32))
        opt = torch.optim.Adam([W], lr=LR, weight_decay=0.)
        rng = np.random.default_rng(SEED+23*fold+173*m)
        for epoch in range(EPOCHS):
            order = rng.permutation(len(idx))
            for left in range(0,len(order),BATCH_TOKEN):
                ix = torch.from_numpy(np.asarray(order[left:left+BATCH_TOKEN],
                                                 dtype=np.int64))
                xx=x[ix]
                yy=targets[ix]
                logits=xx@W+base
                loss=F.cross_entropy(logits,yy)+REGULARIZER*(W*W).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite token output-head training loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        Ws.append(W.detach().numpy().copy())
    results = np.asarray(Ws,dtype=np.float32)
    if results.shape != (NCLASS,CTX_DIM,OUTPUT_SIZE) or not np.isfinite(results).all():
        raise AssertionError("invalid fitted output heads")
    return results, round(time.perf_counter()-t0,1)


def scores_for_docs(H, cat, local_doc_id, num_docs, prior, Ws=None):
    num_tokens=len(H)
    if len(cat)!=num_tokens or len(local_doc_id)!=num_tokens:
        raise ValueError("validation row alignment broken")
    doc_ct=np.bincount(local_doc_id,minlength=num_docs)
    if np.any(doc_ct==0):
        raise AssertionError("empty validation response")
    out=np.empty((num_docs,NCLASS),dtype=np.float64)
    indices=np.arange(num_tokens)
    for m in range(NCLASS):
        if Ws is None:
            vals=prior[m,cat].astype(np.float64)
        else:
            logits=(H @ Ws[m])+prior[m][None,:]
            vals=(logits[indices,cat] - logsumexp(logits,axis=1)).astype(np.float64)
        out[:,m]=np.bincount(local_doc_id,weights=vals,minlength=num_docs)/doc_ct
    if not np.isfinite(out).all():
        raise FloatingPointError("nonfinite source attribution scores")
    return np.argmax(out,axis=1).astype(np.int16)


def paired(y, a, b, prompt):
    return {"delta_BA18":round(float(balanced_accuracy_score(y,b)-
                               balanced_accuracy_score(y,a)),7),
       "CI95_by_prompt":grouped_ci(y,a,b,prompt,seed=SEED,draws=1000),
       "rescues":int(((a!=y)&(b==y)).sum()),
       "regressions":int(((a==y)&(b!=y)).sum()),
       "changed":int((a!=b).sum())}


def main():
    validate_static_math()
    frame = load_snapshot(Path("data/train.csv"),
                          Path("artifacts/phase1/folds.csv"))
    classes,y=np.unique(frame.model.astype(str),return_inverse=True)
    if len(classes)!=NCLASS or len(frame)!=2699 or frame.prompt_id.nunique()!=150:
        raise ValueError("historic competition cohort invalid")
    folds=frame.fold_id.to_numpy()
    prompt=frame.prompt_id.astype(str).to_numpy()
    t0=time.perf_counter()
    with threadpool_limits(limits=2):
        H,tok,doc,meta=frozen_hidden(frame)
        predictions={a:np.full(len(frame),-1,dtype=np.int16) for a in ARMS}
        foldinfo=[]
        for fold in range(5):
            ts=time.perf_counter()
            train_doc=np.flatnonzero(folds!=fold)
            val_doc=np.flatnonzero(folds==fold)
            if set(prompt[train_doc])&set(prompt[val_doc]):
                raise RuntimeError("train/valid prompt overlap")
            train_t=np.flatnonzero(folds[doc]!=fold)
            val_t=np.flatnonzero(folds[doc]==fold)
            z=fold_categories(tok[train_t],tok,fold)
            ct=z[train_t]
            cv=z[val_t]
            src=y[doc[train_t]]
            prior=compute_priors(ct,src)
            mu=H[train_t].mean(axis=0)
            sd=H[train_t].std(axis=0).clip(min=.15)
            norm_train=np.ascontiguousarray(
                np.clip((H[train_t]-mu)/sd,-4,4),dtype=np.float32)
            norm_val=np.ascontiguousarray(
                np.clip((H[val_t]-mu)/sd,-4,4),dtype=np.float32)
            val_local=np.searchsorted(val_doc,doc[val_t])
            if not np.array_equal(val_doc[val_local],doc[val_t]):
                raise RuntimeError("local validation document index mismatch")
            neg_H,cross=shuffle_donors_per_model(
                norm_train,src,doc[train_t],fold=fold)
            w_real,fit_t=fit_heads(norm_train,ct,src,prior,fold,
                    experiment="real")
            w_fake,fit_c=fit_heads(neg_H,ct,src,prior,fold,
                    experiment="shuffled")
            res={}
            for name,w in (("model_unigram",None),
                           ("genuine_context",w_real),
                           ("shuffled_context",w_fake)):
                res[name]=scores_for_docs(norm_val,cv,val_local,
                    len(val_doc),prior,w)
                predictions[name][val_doc]=res[name]
            # True 18-way direct sparse comparator on identical heldout rows.
            fit_text=frame.text.iloc[train_doc].astype(str).tolist()
            val_text=frame.text.iloc[val_doc].astype(str).tolist()
            x,zsp,details=build_fold_features(fit_text,val_text)
            sparse_clf=make_svc(SEED+fold)
            sparse_clf.fit(x,y[train_doc])
            res["strong_sparse"]=sparse_clf.predict(zsp).astype(np.int16)
            predictions["strong_sparse"][val_doc]=res["strong_sparse"]
            fold_ba={name:round(float(balanced_accuracy_score(
                    y[val_doc],result)),7) for name,result in res.items()}
            record={"fold":fold,"train_docs":len(train_doc),
                "valid_docs":len(val_doc),
                "train_contexts":len(train_t),
                "valid_contexts":len(val_t),
                "shuffled_donors_different_response_rate_min":round(cross,5),
                "genuine_head_fit_seconds":fit_t,
                "shuffled_head_fit_seconds":fit_c,
                "fold_seconds":round(time.perf_counter()-ts,1),
                "BA18":fold_ba,
                "sparse_feature_width":details["total_features"]}
            foldinfo.append(record)
            print("NTL_GCTH_V1_FOLD="+json.dumps(record,sort_keys=True),flush=True)
    if any(np.any(v<0) for v in predictions.values()):
        raise RuntimeError("missing 5fold OOF predictions")
    aggregate={name:float(balanced_accuracy_score(y,pred))
               for name,pred in predictions.items()}
    recalls={name:recall_score(y,pred,labels=np.arange(NCLASS),
        average=None,zero_division=0) for name,pred in predictions.items()}
    A=predictions["model_unigram"]
    B=predictions["genuine_context"]
    C=predictions["shuffled_context"]
    D=predictions["strong_sparse"]
    comparisons={"B_minus_A":paired(y,A,B,prompt),
        "B_minus_C":paired(y,C,B,prompt),
        "B_minus_D":paired(y,D,B,prompt)}
    wins_a=sum(r["BA18"]["genuine_context"]>r["BA18"]["model_unigram"]
               for r in foldinfo)
    wins_c=sum(r["BA18"]["genuine_context"]>r["BA18"]["shuffled_context"]
               for r in foldinfo)
    maxdrop=float((recalls["model_unigram"]-recalls["genuine_context"]).max())
    mechanism=bool(
        comparisons["B_minus_A"]["delta_BA18"]>=.02 and
        comparisons["B_minus_C"]["delta_BA18"]>=.02 and
        comparisons["B_minus_A"]["CI95_by_prompt"][0]>0 and
        comparisons["B_minus_C"]["CI95_by_prompt"][0]>0 and
        wins_a>=3 and wins_c>=3 and maxdrop<=.15
    )
    result={"candidate":"GCTH-v1 generator-conditioned pretrained-context token choice",
        "data_scope":"official HISTORICAL TRAIN 2699 rows, 150 prompt groups, 18 sources",
        "arch":"frozen DistilGPT2 causal previous-context hidden states; 18 source-specific normalized 128-category next-token heads",
        "model":MODEL,"revision":REV,
        "real_proprietary_generator_likelihood":False,
        "uses_original_champion":False,"uses_kaggle_test":False,
        "uses_external_generations":False,"competition_submission":False,
        "frozen_pretrained_backbone":True,
        "results":{name:{"OOF_BA18":round(float(b),7),
            "recall_by_model":{label:round(float(recalls[name][i]),5)
                      for i,label in enumerate(classes)}}
            for name,b in aggregate.items()},
        "comparisons":comparisons,
        "folds":foldinfo,
        "wins_B_vs_A":wins_a,"wins_B_vs_C":wins_c,
        "max_class_recall_drop_B_vs_A":round(maxdrop,7),
        "mechanism_gate_PASS":mechanism,
        "competitive_vs_sparse":aggregate["genuine_context"]>=aggregate["strong_sparse"],
        "frozen_hidden":meta,
        "wall_seconds":round(time.perf_counter()-t0,1),
        "source_disjoint_external_holdout":False,
        "no_GPU":True,
        "no_raw_text_or_predictions_export":True}
    print("NTL_GCTH_V1_FINAL="+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
