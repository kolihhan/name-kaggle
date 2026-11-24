"""SCaL: masked full-vocab source-conditioned likelihood and sibling contrast.

Standalone private Kaggle GPU research. NO competition submit, NO raw data export,
NO model export. The main contrast is *train objective*, not another ensemble.
"""
from __future__ import annotations
import bisect
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

SEED = 20261124
REV = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
MODEL_ID = "FacebookAI/roberta-base"
OFFICIAL_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLDS_SHA = "94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"
FOLD_MAP = {}
RANK = 16
LR_AB = 3e-4
LR_BIAS = 1e-3
BATCH_TOKENS = 64
EPOCHS = 2
MAX_SEQ = 256
NEGATIVE_WEIGHT = 0.5
MARGIN = 0.2
PRIMARY_FOLD = 0


def emit(label, data):
    print(label + "=" + json.dumps(data, sort_keys=True, default=str), flush=True)


def selftest():
    p = np.array([[-1., -2.], [-2., -1.]], np.float64)
    assert np.isfinite(p).all() and len(np.unique(p.argmax(1))) == 2
    classes = ["claude-opus-5", "claude-sonnet-5", "gemini-3.6-flash"]
    assert sibling_indices(classes).tolist() == [1, 0, 0]
    m = "example-id"
    seed = int.from_bytes(hashlib.sha256((str(SEED) + "|" + m).encode()).digest()[:8], "big")
    assert seed == int.from_bytes(hashlib.sha256((str(SEED)+"|"+m).encode()).digest()[:8], "big")
    print("NTL_SCAL_STATIC_SELFTEST=PASS", flush=True)


def family(c):
    for k in ("claude-", "gemini-", "glm-", "gpt-", "kimi-", "qwen"):
        if c.startswith(k):
            return k
    return c


def sibling_indices(classes):
    # First alternate label in same generator family, otherwise the next class.
    out = []
    for i, c in enumerate(classes):
        peers = [j for j, other in enumerate(classes) if i != j and family(c) == family(other)]
        out.append(peers[0] if peers else ((i + 1) % len(classes)))
    return np.asarray(out, dtype=np.int64)


def file_sha(path):
    with open(path, "rb") as f:
        import hashlib as h
        return h.file_digest(f, "sha256").hexdigest()


def official_frame():
    import pandas as pd
    fs = [p for p in Path("/kaggle/input").rglob("train.csv")
          if "name-that-llm" in str(p).lower()]
    if len(fs) != 1:
        raise ValueError("unique competition train.csv not mounted: " + str(len(fs)))
    source = fs[0]
    sha = file_sha(source)
    if sha != OFFICIAL_SHA:
        raise ValueError("official TRAIN provenance drift")
    df = pd.read_csv(source, dtype={"id": str, "prompt_id": str, "model": str})
    assert len(df) == 2699 and df.id.is_unique and df.prompt_id.nunique() == 150
    assert all(c in df.columns for c in ("text", "id", "prompt_id", "model"))
    keys = df.prompt_id.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    assert len(FOLD_MAP) == 150 and set(keys) == set(FOLD_MAP)
    df["fold_id"] = keys.map(FOLD_MAP).astype(int)
    assert df.groupby("prompt_id").fold_id.nunique().max() == 1
    assert sorted(df.fold_id.unique()) == [0, 1, 2, 3, 4]
    emit("NTL_SCAL_DATA_PROVENANCE", {"rows": len(df), "prompts": 150,
        "classes":df.model.nunique(), "train_sha_match":True,
        "fold_map_hash_keys_match":True,"folds_sha":FOLDS_SHA,
        "fold0_rows":int((df.fold_id == PRIMARY_FOLD).sum())})
    return df


def mask_document(tokenizer, text, doc_id):
    # Select complete whitespace words, mask *all* their BPE pieces.
    enc = tokenizer(text, truncation=True, max_length=MAX_SEQ,
                    add_special_tokens=True, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    words = list(re.finditer(r"\S+", text))
    starts = [m.start() for m in words]
    groups = {}
    content_right = max((b for a,b in offsets if b>a), default=0)
    for pos,(a,b) in enumerate(offsets):
        if a >= b: continue
        j = bisect.bisect_right(starts,a)-1
        if j < 0 or b > words[j].end() or words[j].end() > content_right:
            continue
        groups.setdefault(j, []).append(pos)
    if not groups:
        return None
    wkeys = sorted(groups)
    seed = int.from_bytes(hashlib.sha256((str(SEED)+"|"+str(doc_id)).encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    n = len(wkeys)
    count = min(24, math.ceil(.15*n), n-1 if n>1 else 1)
    picks = sorted(int(v) for v in rng.choice(n,size=count,replace=False))
    pos = sorted(p for ix in picks for p in groups[wkeys[ix]])
    raw = [ids[p] for p in pos]
    if not raw or any(ids[p] in tokenizer.all_special_ids for p in pos):
        raise ValueError("invalid selected word target")
    masked = list(ids)
    for p in pos: masked[p] = tokenizer.mask_token_id
    # No selected target token survives corruption.
    assert all(masked[p] == tokenizer.mask_token_id for p in pos)
    return (masked, pos, raw, len(ids),len(words),len(wkeys))


def extract(df):
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    assert torch.cuda.is_available(), "GPU unavailable — do not silently fall back to CPU"
    torch.manual_seed(SEED);np.random.seed(SEED)
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV,
                                        trust_remote_code=False, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, revision=REV, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float16).to("cuda").eval()
    assert model.config.vocab_size == 50265
    entries, docs, target, retained = [],[],[],[]
    truncated,skipped=0,0
    for i,row in enumerate(df.itertuples(index=False)):
        text = str(row.text) if row.text is not None else ""
        q = mask_document(tok,text,row.id)
        if q is None:skipped+=1;continue
        masked,pos,tgt,length,wn,wret=q
        truncated+=wn>wret
        entries.append((masked,pos,tgt))
        retained.append(i)
    if skipped: raise ValueError(f"{skipped} official responses have no maskable content")
    max_len=max(len(e[0]) for e in entries)
    emit("NTL_SCAL_MASK_CONTRACT", {"documents":len(entries),"max_tokens":max_len,
         "truncated_word_groups_documents":truncated,"total_masked_target_pieces":sum(len(e[2]) for e in entries)})
    W = model.lm_head.decoder.weight.detach().to(dtype=torch.float32).clone()
    b = model.lm_head.bias.detach().to(dtype=torch.float32).clone()
    hidden, gold, owner = [], [], []
    began=time.monotonic()
    with torch.inference_mode():
        for left in range(0,len(entries),8):
            batch = entries[left:left+8]
            mx=max(len(z[0]) for z in batch)
            input_ids=torch.full((len(batch),mx),tok.pad_token_id,dtype=torch.long)
            attention=torch.zeros((len(batch),mx),dtype=torch.long)
            for j,(ids,pos,targ) in enumerate(batch):
                input_ids[j,:len(ids)] = torch.tensor(ids,dtype=torch.long)
                attention[j,:len(ids)] = 1
            h=model.roberta(input_ids.to("cuda"),attention_mask=attention.to("cuda")).last_hidden_state
            h=model.lm_head.layer_norm(F.gelu(model.lm_head.dense(h)))
            for j,(_,pos,targ) in enumerate(batch):
                v=h[j,torch.tensor(pos,device="cuda"),:].detach().to("cpu",dtype=torch.float16)
                hidden.append(v)
                gold.extend(targ)
                owner.extend([left+j]*len(targ))
            if (left//8+1)%70==0:
                emit("NTL_SCAL_EXTRACTION_PROGRESS", {"docs":min(len(entries),left+8),"total":len(entries)})
    del model
    torch.cuda.empty_cache()
    X=torch.cat(hidden,dim=0)
    ids=np.asarray(gold,dtype=np.int64)
    doc=np.asarray(owner,dtype=np.int32)
    assert len(X)==len(ids)==len(doc) and X.shape[1]==768 and np.isfinite(X.numpy()).all()
    emit("NTL_SCAL_FROZEN_CONTEXT", {"contexts":len(X),"dim":768,
         "seconds":round(time.monotonic()-began,2),"model":MODEL_ID,
         "revision":REV,"pretrained_backbone_frozen":True})
    return X,ids,doc,W,b


def class_weights(doc_ids, labels, train_doc, token_idx):
    # Each TRAIN document has unit mass and each exact label equal document mass.
    dcnt=np.bincount(doc_ids[token_idx],minlength=len(labels))
    f=np.bincount(labels[train_doc],minlength=len(np.unique(labels)))
    assert np.all(f>0) and np.all(dcnt[train_doc]>0)
    per=1.0/(dcnt[doc_ids[token_idx]]*f[labels[doc_ids[token_idx]]])
    return np.asarray(per/per.mean(),dtype=np.float32)


def train_one(X, targets, doc, labels, train_doc, sib, W,b, contrastive):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device="cuda"
    torch.manual_seed(SEED)
    nclass=len(sib);vocab=W.shape[0];hidden=W.shape[1]
    class Adapters(nn.Module):
        def __init__(self):
            super().__init__()
            self.A=nn.Parameter(torch.empty(RANK,hidden,device=device,dtype=torch.float32).normal_(0,.02))
            self.B=nn.Parameter(torch.zeros(nclass,RANK,hidden,device=device))
            self.c=nn.Parameter(torch.zeros(nclass,vocab,device=device))
        def scores(self,z,m,W,b):
            # Indexed source-specific residual, no class label at inference.
            residual=torch.einsum("nr,nrh->nh",F.linear(z,self.A),self.B[m])
            return F.linear(z+residual,W,b)+self.c[m]
    adapter=Adapters().to(device)
    opt=torch.optim.AdamW([{"params":[adapter.A,adapter.B],"lr":LR_AB,"weight_decay":.01},
                          {"params":[adapter.c],"lr":LR_BIAS,"weight_decay":0}],betas=(.9,.999))
    train_tok=np.flatnonzero(np.isin(doc,train_doc))
    weight=class_weights(doc,labels,train_doc,train_tok)
    wmap=np.zeros(len(doc),dtype=np.float32)
    wmap[train_tok]=weight
    source_doc=labels[doc]
    neg_doc=sib[source_doc]
    fixed_W=W.to(device);fixed_b=b.to(device)
    begin=time.monotonic()
    losses=[]
    for epoch in range(EPOCHS):
        rng=np.random.default_rng(SEED+epoch)
        order=rng.permutation(train_tok)
        sums=[]
        for start in range(0,len(order),BATCH_TOKENS):
            ix=order[start:start+BATCH_TOKENS]
            z=X[ix].to(device=device,dtype=torch.float32)
            y=torch.from_numpy(targets[ix]).to(device)
            source=torch.from_numpy(source_doc[ix]).to(device)
            logits=adapter.scores(z,source,fixed_W,fixed_b)
            nll=F.cross_entropy(logits,y,reduction="none")
            loss_token=nll
            if contrastive:
                neg=torch.from_numpy(neg_doc[ix]).to(device)
                neglogits=adapter.scores(z,neg,fixed_W,fixed_b)
                neg_nll=F.cross_entropy(neglogits,y,reduction="none")
                loss_token=nll+NEGATIVE_WEIGHT*F.softplus(MARGIN+nll-neg_nll)
            weights=torch.from_numpy(wmap[ix]).to(device)
            loss=(loss_token*weights).mean()
            if not torch.isfinite(loss):raise FloatingPointError("training nonfinite")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0)
            opt.step()
            sums.append(float(loss.detach()))
        losses.append(round(float(np.mean(sums)),4))
        emit("NTL_SCAL_FIT_EPOCH",{"arm":"counterfactual" if contrastive else "proper_nll",
             "epoch":epoch,"mean_loss":losses[-1],"steps":len(sums)})
    emit("NTL_SCAL_FIT_AGGREGATE",{"arm":"counterfactual" if contrastive else "proper_nll",
        "train_tokens":len(train_tok),"loss_by_epoch":losses,
        "seconds":round(time.monotonic()-begin,2)})
    if losses[-1]>losses[0]*1.02:
        raise RuntimeError("training did not decrease its fixed objective")
    return adapter


def predict(X,targets,doc,valid_doc,classes,adapter,W,b):
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import balanced_accuracy_score
    adapter.eval()
    tokens=np.flatnonzero(np.isin(doc,valid_doc))
    vmap={int(k):i for i,k in enumerate(valid_doc)}
    tally=np.zeros((len(valid_doc),len(classes)),dtype=np.float64)
    counts=np.zeros(len(valid_doc),dtype=np.int32)
    for chunk in range(0,len(tokens),24):
        ix=tokens[chunk:chunk+24]
        z=X[ix].to("cuda",dtype=torch.float32)
        gold=torch.from_numpy(targets[ix]).to("cuda")
        with torch.inference_mode():
            for m in range(len(classes)):
                source=torch.full((len(ix),),m,dtype=torch.long,device="cuda")
                logits=adapter.scores(z,source,W,b)
                lp=(-F.cross_entropy(logits,gold,reduction="none")).cpu().numpy()
                for j,t in enumerate(ix):tally[vmap[int(doc[t])],m]+=lp[j]
        for t in ix:counts[vmap[int(doc[t])]]+=1
    assert np.all(counts>0)
    tally/=counts[:,None]
    assert np.isfinite(tally).all()
    return tally.argmax(axis=1)


def bootstrap(y,a,b,prompt):
    from sklearn.metrics import balanced_accuracy_score
    classes=np.unique(y)
    groups=[np.flatnonzero(prompt==p) for p in np.unique(prompt)]
    rng=np.random.default_rng(SEED)
    deltas=[]
    for _ in range(2000):
        ids=np.concatenate([groups[k] for k in rng.integers(len(groups),size=len(groups))])
        if len(np.unique(y[ids]))!=len(classes):continue
        deltas.append(float(balanced_accuracy_score(y[ids],b[ids])-balanced_accuracy_score(y[ids],a[ids])))
    return [float(z) for z in np.quantile(deltas,[.025,.975])] if len(deltas)>1600 else None


def main():
    import torch
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    selftest()
    began=time.monotonic()
    frame=official_frame()
    labels,source=np.unique(frame.model.astype(str),return_inverse=True)
    assert len(labels)==18
    folds=frame.fold_id.to_numpy(int)
    train_doc=np.flatnonzero(folds!=PRIMARY_FOLD)
    valid_doc=np.flatnonzero(folds==PRIMARY_FOLD)
    assert not set(frame.prompt_id.iloc[train_doc])&set(frame.prompt_id.iloc[valid_doc])
    X,target,doc,W,b=extract(frame)
    sib=sibling_indices(labels)
    doc_embeddings=np.stack([X[doc==i].to(dtype=torch.float32).mean(0).numpy()
                              for i in range(len(frame))])
    doc_clf=__import__("sklearn.pipeline",fromlist=["make_pipeline"]).make_pipeline(
        StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=1600))
    doc_clf.fit(doc_embeddings[train_doc],source[train_doc])
    control=doc_clf.predict(doc_embeddings[valid_doc])
    del doc_embeddings
    base=train_one(X,target,doc,source,train_doc,sib,W,b,contrastive=False)
    pred0=predict(X,target,doc,valid_doc,labels,base,W.to("cuda"),b.to("cuda"))
    del base;torch.cuda.empty_cache()
    new=train_one(X,target,doc,source,train_doc,sib,W,b,contrastive=True)
    pred1=predict(X,target,doc,valid_doc,labels,new,W.to("cuda"),b.to("cuda"))
    y=source[valid_doc]
    pp=frame.prompt_id.iloc[valid_doc].to_numpy(str)
    scores={"proper_source_nll":float(balanced_accuracy_score(y,pred0)),
            "SCaL_sibling_contrastive":float(balanced_accuracy_score(y,pred1)),
            "direct_frozen_doc_CE":float(balanced_accuracy_score(y,control))}
    interval=bootstrap(y,pred0,pred1,pp)
    rescue=int(((pred0!=y)&(pred1==y)).sum())
    regress=int(((pred0==y)&(pred1!=y)).sum())
    gain=scores["SCaL_sibling_contrastive"]-scores["proper_source_nll"]
    gate=bool(gain>=.03 and interval is not None and interval[0]>0 and
             scores["SCaL_sibling_contrastive"]>=scores["direct_frozen_doc_CE"])
    from sklearn.metrics import recall_score
    report={"experiment":"SCaL_18way_full_vocab_source_likelihood_fold0",
      "scores":scores,"delta_vs_source_nll":gain,
      "prompt_bootstrap95":interval,"rescues":rescue,"regressions":regress,
      "changed_labels":int((pred0!=pred1).sum()),
      "num_train_docs":len(train_doc),"num_valid_docs":len(valid_doc),
      "num_prompt_groups":len(np.unique(pp)),
      "class_recall_SCaL":dict(zip(labels,[float(v) for v in recall_score(y,pred1,labels=np.arange(len(labels)),average=None,zero_division=0)])),
      "pilot_gate":gate,"competition_submission":False,
      "independent_transfer":False,"historical_official_fold_only":True,
      "backbone_frozen":True,"full_vocab_source_probabilities":True,
      "model_weights_exported":False,"total_seconds":round(time.monotonic()-began,2)}
    emit("NTL_SCAL_REAL_FOLD0_RESULT",report)
    Path("/kaggle/working/ntl_scal_aggregate.json").write_text(json.dumps(report,sort_keys=True)+"\n")
    if not gate:print("NTL_SCAL_V0_STOP_NO_NEXT_FOLD_NO_SUBMIT=TRUE",flush=True)
    else:print("NTL_SCAL_V0_PILOT_PASS_REQUIRES_NEW_FROZEN_TRANSFER_VALIDATION=TRUE",flush=True)


if __name__=="__main__":
    if os.environ.get("NTL_SCAL_SELFTEST") == "1":
        selftest()
    else:
        main()
