"""Trainable-source encoder falsifier: matched CE vs CE+prompt-contrastive.

Run privately inside a Kaggle T4 notebook. Frozen first ten RoBERTa layers
are cached once; last two transformer layers AND classifier train for both
arms. All 18 classes remain eligible at external inference. No submission.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

SEED = 20261124
MODEL_ID = "FacebookAI/roberta-base"
REV = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLDS_SHA = "94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"
C1_SHA = "8746c4b77c2b09b9abafaeecc381a39321b9919214c005a3657042051c51b177"
C2_SHA = "bf25ef1e953d5c37b177f27b841a04098ae9b35f45fc43fa11036d91e8831e02"
FOLD_MAP = {}
FOLD = 0
MAX_LEN = 160
CACHE_BATCH = 8
EPOCHS = 2
PROMPT_BATCH_LABELS = 6
CONTRAST_WEIGHT = 0.2
TEMPERATURE = 0.15


def emit(marker, payload):
    print(marker+"="+json.dumps(payload,sort_keys=True,default=str),flush=True)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def group_batches(y,group,selected,epoch):
    """Identical two-prompt/two-label-or-more batches for both training arms."""
    banks = {}
    for i in selected:
        p,lab=str(group[i]),str(y[i])
        b=banks.setdefault(p,{})
        if lab in b:
            raise ValueError("duplicate exact source in one prompt bank")
        b[lab]=int(i)
    prompts=np.asarray(sorted(banks),dtype=object)
    if len(prompts)<40:raise ValueError("too few prompt banks")
    rng=np.random.default_rng(SEED+epoch)
    order=rng.permutation(len(prompts))
    batches=[]
    covered=set()
    for rank,which in enumerate(order):
        a=str(prompts[which])
        # Distinct second prompt, deterministic epoch-specific cycle.
        b=str(prompts[order[(rank+1)%len(order)]])
        if a==b:raise ValueError("self prompt pairing")
        common=np.asarray(sorted(set(banks[a])&set(banks[b])),dtype=object)
        rng.shuffle(common)
        for left in range(0,len(common),PROMPT_BATCH_LABELS):
            names=common[left:left+PROMPT_BATCH_LABELS]
            if len(names)<2:continue
            ids=[banks[a][z] for z in names]+[banks[b][z] for z in names]
            labels=[str(y[i]) for i in ids]
            prompts_for_batch=[str(group[i]) for i in ids]
            assert len(ids)<=2*PROMPT_BATCH_LABELS and len(set(labels))==len(names)
            assert len(set(prompts_for_batch))==2
            assert len(ids)==2*len(names)
            covered.update(ids)
            batches.append(np.asarray(ids,dtype=np.int32))
    if len(batches)<150 or len(covered)<0.95*len(selected):
        raise ValueError("insufficient source/prompt train coverage")
    return batches,{"batches":len(batches),"train_rows_covered":len(covered),
                   "training_rows":len(selected),"label_unique":len(set(y[selected]))}


def selftest():
    ys=np.tile(np.array(["a","b","c","d","e","f"],dtype=object),180)
    gs=np.repeat(np.array([f"p{i:02}" for i in range(180)],dtype=object),6)
    bs,meta=group_batches(ys,gs,np.arange(len(ys)),0)
    assert meta["train_rows_covered"]==len(ys)
    assert all(len(z)==12 for z in bs)
    assert len(bs)==180
    assert [list(x) for x in group_batches(ys,gs,np.arange(len(ys)),0)[0]]==[list(x) for x in bs]
    print("NTL_TPC_FOLD0_SELFTEST=PASS",flush=True)


def load_data():
    # Source files are mounted only in private Kaggle. No raw official data
    # nor predictions are downloaded to Github or sent to GitHub job logs.
    root=Path("/kaggle/input")
    official=[x for x in root.rglob("train.csv") if "name-that-llm" in str(x).lower()]
    if len(official)!=1:raise ValueError("expected exactly one official competition train")
    if digest(official[0])!=TRAIN_SHA:raise ValueError("official train source hash mismatch")
    d=pd.read_csv(official[0],dtype={"id":str,"prompt_id":str,"model":str})
    if len(d)!=2699 or d.model.nunique()!=18 or d.prompt_id.nunique()!=150 or not d.id.is_unique:
        raise ValueError("official cohort drift")
    hashes=d.prompt_id.map(lambda v:hashlib.sha256(v.encode()).hexdigest())
    if len(FOLD_MAP)!=150 or set(hashes)!=set(FOLD_MAP):
        raise ValueError("frozen prompt fold map drift")
    d["fold_id"]=hashes.map(FOLD_MAP).astype(int)
    if d.groupby("prompt_id").fold_id.nunique().max()!=1 or set(d.fold_id)!={0,1,2,3,4}:
        raise ValueError("prompt fold leakage")
    labels=np.asarray(sorted(d.model.unique()))
    train_idx=np.flatnonzero(d.fold_id.to_numpy()!=FOLD)
    fold_idx=np.flatnonzero(d.fold_id.to_numpy()==FOLD)
    assert len(train_idx)==2159 and len(fold_idx)==540 and set(d.model.iloc[train_idx])==set(labels)
    external={}
    for name,hash_expected,n,prompts in (("c1",C1_SHA,360,30),("c2",C2_SHA,470,50)):
        paths=[x for x in root.rglob("passages.csv") if "benchmark-"+name in str(x).lower()]
        if len(paths)!=1:raise ValueError("panel "+name+" missing or ambiguous")
        if digest(paths[0])!=hash_expected:raise ValueError("external source hash drift "+name)
        z=pd.read_csv(paths[0],dtype={"prompt_id":str,"label":str})
        if len(z)!=n or z.prompt_id.nunique()!=prompts or z.label.nunique()!=12:
            raise ValueError("external panel counts drift")
        if z.text.isna().any() or not set(z.label)<=set(labels):
            raise ValueError("external label/text validation")
        external[name]=z.rename(columns={"label":"model"})
    if set(external["c1"].model)!=set(external["c2"].model):
        raise ValueError("external panel exact classes differ")
    if set(external["c1"].prompt_id)&set(external["c2"].prompt_id):
        raise ValueError("external panel prompts overlap")
    if any(set(d.prompt_id)&set(z.prompt_id) for z in external.values()):
        raise ValueError("official/external prompt IDs overlap")
    def normtexts(z):
        return {str(s).strip().lower() for s in z.text}
    if any(normtexts(d)&normtexts(z) for z in external.values()):
        raise ValueError("exact official/external response contamination")
    everything=pd.concat([d,external["c1"],external["c2"]],ignore_index=True)
    panel_indices={"official_fold0":fold_idx,
       "c1":np.arange(len(d),len(d)+len(external["c1"])),
       "c2":np.arange(len(d)+len(external["c1"]),len(everything))}
    assert sum(len(v) for v in panel_indices.values())==540+360+470
    emit("NTL_TPC_SOURCE_CHECK",{"official_train_rows":len(train_idx),"fold0_rows":540,
         "c1":360,"c2":470,"classes":18,"frozen_folds":True,"hashes_verified":True,
         "external_same_publisher":True,"source_provider_verified":False,
         "external_18class_coverage":False})
    return everything,labels,train_idx,panel_indices


def frozen_context(texts):
    import torch
    from transformers import AutoTokenizer,AutoModel

    if not torch.cuda.is_available():raise RuntimeError("requires NVIDIA GPU")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=REV,trust_remote_code=False,use_fast=True)
    pretrained=AutoModel.from_pretrained(MODEL_ID,revision=REV,trust_remote_code=False,
                    use_safetensors=True,torch_dtype=torch.float16).cuda().eval()
    if not hasattr(pretrained,"encoder") or len(pretrained.encoder.layer)!=12:
        raise ValueError("RoBERTa architecture contract changed")
    reference_upper=copy.deepcopy(pretrained.encoder.layer[-2:]).cpu().float()
    hs=[];ms=[]
    trunc=0
    start=time.monotonic()
    with torch.inference_mode():
        for left in range(0,len(texts),CACHE_BATCH):
            strings=texts[left:left+CACHE_BATCH]
            encoded=tokenizer(strings,padding="max_length",truncation=True,
                              max_length=MAX_LEN,return_tensors="pt")
            inp=encoded["input_ids"].cuda()
            mask=encoded["attention_mask"].cuda()
            extended=(1-mask[:,None,None,:]).to(torch.float16)*-10000.
            hidden=pretrained.embeddings(input_ids=inp)
            for layer in pretrained.encoder.layer[:10]:
                hidden=layer(hidden,attention_mask=extended)[0]
            hs.append(hidden.cpu().to(dtype=torch.float16))
            ms.append(encoded["attention_mask"].cpu().to(dtype=torch.uint8))
            # Cheap truncation approximation from tokenizer overflow flag is
            # unavailable in batched format; explicitly count lengths separately.
            if (left//CACHE_BATCH+1)%90==0:
                emit("NTL_TPC_FROZEN_CACHE_PROGRESS",{"done":min(len(texts),left+CACHE_BATCH),
                                                        "total":len(texts)})
    H=torch.cat(hs,dim=0);M=torch.cat(ms,dim=0)
    assert len(H)==len(M)==len(texts) and H.shape==(len(texts),MAX_LEN,768)
    assert torch.isfinite(H).all() and M.shape==(len(texts),MAX_LEN)
    del pretrained,hs,ms
    gc.collect()
    torch.cuda.empty_cache()
    emit("NTL_TPC_FROZEN_CACHE_READY",{"documents":len(texts),"shape":list(H.shape),
         "seconds":round(time.monotonic()-start,2),"pretrained_revision":REV,
         "lower10_trainable":False,"upper2_trainable":True,
         "truncation_rate_unmeasured":True})
    return H,M,reference_upper


def train_arm(H,M,top,y,groups,train_idx,nclasses,contrastive):
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.last2=copy.deepcopy(top)
            self.proj=nn.Sequential(nn.Linear(768,256),nn.Tanh(),nn.Dropout(.1))
            self.classifier=nn.Linear(256,nclasses)
        def forward(self,h,mask):
            attn=(1-mask[:,None,None,:]).to(torch.float32)*-10000.0
            x=h
            for block in self.last2:
                x=block(x,attention_mask=attn)[0]
            pool=(x*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            z=self.proj(pool)
            return self.classifier(z),F.normalize(z.float(),dim=1)
    net=Enc().cuda().train()
    opt=torch.optim.AdamW([
        {"params":net.last2.parameters(),"lr":3e-5,"weight_decay":.01},
        {"params":list(net.proj.parameters())+list(net.classifier.parameters()),"lr":1e-4,"weight_decay":.01}
    ])
    scaler=torch.amp.GradScaler("cuda")
    y=torch.as_tensor(y,dtype=torch.long)
    t0=time.monotonic()
    num_steps=0
    epoch_losses=[]
    for epoch in range(EPOCHS):
        batches,meta=group_batches(np.asarray(y),groups,train_idx,epoch)
        accum=[]
        for ids in batches:
            inp=H[ids].cuda().float()
            m=M[ids].cuda().float()
            target=y[ids].cuda()
            ps=np.asarray(groups)[ids]
            with torch.autocast("cuda",dtype=torch.float16):
                logits,z=net(inp,m)
                ce=F.cross_entropy(logits.float(),target)
                loss=ce
                if contrastive:
                    same=target[:,None].eq(target[None,:])
                    different=torch.as_tensor(ps[:,None]!=ps[None,:],device="cuda")
                    positives=same&different
                    if not positives.any() or (positives.sum(1)==0).any():
                        raise ValueError("contrastive batch without cross-prompt exact-label positive")
                    similarities=z.float()@z.float().T/TEMPERATURE
                    eye=torch.eye(len(ids),device="cuda",dtype=torch.bool)
                    similarities=similarities.masked_fill(eye,-1e4)
                    logp=similarities-torch.logsumexp(similarities,dim=1,keepdim=True)
                    sup=-(logp*positives).sum(1).div(positives.sum(1)).mean()
                    loss=ce+CONTRAST_WEIGHT*sup
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite training loss")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
            scaler.step(opt);scaler.update()
            num_steps+=1
            accum.append(float(loss.detach().cpu()))
        epoch_losses.append(round(float(np.mean(accum)),5))
        emit("NTL_TPC_TRAIN_EPOCH",{"arm":"CE_plus_prompt_contrastive" if contrastive else "CE",
            "epoch":epoch,"loss":epoch_losses[-1],"steps":len(batches),
            "input_coverage":meta["train_rows_covered"],"elapsed_s":round(time.monotonic()-t0,1)})
    emit("NTL_TPC_TRAIN_COMPLETE",{"arm":"CE_plus_prompt_contrastive" if contrastive else "CE",
         "total_steps":num_steps,"losses":epoch_losses,"seconds":round(time.monotonic()-t0,2),
         "backbone_top2_trainable":True})
    return net


def predict(net,H,M,idx):
    import torch
    net.eval()
    out=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
        for start in range(0,len(idx),32):
            piece=idx[start:start+32]
            logits,_=net(H[piece].cuda().float(),M[piece].cuda().float())
            out.extend(logits.float().argmax(1).cpu().tolist())
    return np.asarray(out,dtype=np.int16)


def paired(truth,old,new,classes):
    a=old==truth;b=new==truth
    return {"delta_ba":ba(truth,new,classes)-ba(truth,old,classes),
            "rescues":int((~a&b).sum()),"regressions":int((a&~b).sum()),
            "changed_labels":int((new!=old).sum())}


def ba(truth,pred,classes):
    return float(np.mean([(pred[truth==c]==c).mean() for c in classes]))


def external_grouped_ci(y,g,old,new,classes):
    # Group bootstrap over (publisher panel, prompt) groups across C1/C2.
    keys=np.asarray(g)
    chunks=[np.flatnonzero(keys==k) for k in np.unique(keys)]
    denominators=np.asarray([[(y[ix]==c).sum() for c in classes] for ix in chunks])
    effects=np.asarray([[((new[ix]==c)&(y[ix]==c)).sum()-((old[ix]==c)&(y[ix]==c)).sum()
                         for c in classes] for ix in chunks])
    rng=np.random.default_rng(SEED)
    draws=rng.multinomial(len(chunks),np.full(len(chunks),1/len(chunks)),size=2000)
    d=draws@denominators
    n=draws@effects
    valid=np.all(d>0,axis=1)
    if valid.mean()<.95:raise ValueError("under-supported external bootstrap")
    z=(n[valid]/d[valid]).mean(1)
    return [float(v) for v in np.quantile(z,[.025,.975])]


def main():
    import torch
    from scripts.run_strong_sparse_anchor import build_fold_features,make_svc

    selftest()
    all_df,labels,train_idx,panel_idx=load_data()
    sources=all_df.model.to_numpy(str)
    groups=all_df.prompt_id.to_numpy(str)
    lookup={v:i for i,v in enumerate(labels)}
    y=np.array([lookup[v] for v in sources],dtype=np.int16)
    ids={k:np.asarray(v,dtype=np.int32) for k,v in panel_idx.items()}
    begin=time.monotonic()
    texts=all_df.text.astype(str).tolist()
    H,M,upper=frozen_context(texts)

    results={}
    for name,contrast in (("CE",False),("CE_plus_prompt_contrastive",True)):
        net=train_arm(H,M,upper,y,groups,train_idx,len(labels),contrast)
        results[name]={k:predict(net,H,M,v) for k,v in ids.items()}
        del net
        gc.collect();torch.cuda.empty_cache()

    # Fully independent same-example strong published sparse baseline.
    allval=np.concatenate([ids[k] for k in ("official_fold0","c1","c2")])
    x,z,meta=build_fold_features([texts[i] for i in train_idx],[texts[i] for i in allval])
    svc=make_svc(SEED)
    svc.fit(x,sources[train_idx])
    sp=svc.predict(z)
    off=0
    results["strong_sparse"]={}
    for name in ("official_fold0","c1","c2"):
        n=len(ids[name]);results["strong_sparse"][name]=np.array([lookup[v] for v in sp[off:off+n]],dtype=np.int16);off+=n
    del H,M,x,z,upper,svc
    gc.collect();torch.cuda.empty_cache()
    report={}
    for name,ix in ids.items():
        true=y[ix]
        classes=np.unique(true)
        arm={k:v[name] for k,v in results.items()}
        report[name]={
            "n":len(ix),"truth_classes":len(classes),"prediction_classes":len(labels),
            "ba":{k:round(ba(true,p,classes),7) for k,p in arm.items()},
            "B_minus_CE":paired(true,arm["CE"],arm["CE_plus_prompt_contrastive"],classes),
            "B_minus_Strong":paired(true,arm["strong_sparse"],arm["CE_plus_prompt_contrastive"],classes),
            "recall_B":{str(labels[c]):round(float((arm["CE_plus_prompt_contrastive"][true==c]==c).mean()),4)
                        for c in classes},
        }
    c1,c2=ids["c1"],ids["c2"]
    combined=np.concatenate([c1,c2])
    shared=np.unique(y[c1])
    if not np.array_equal(shared,np.unique(y[c2])):raise RuntimeError("external 12 labels differ")
    c1c2_group=np.array(["c1|"+str(groups[i]) if i in set(c1) else "c2|"+str(groups[i]) for i in combined])
    ci=external_grouped_ci(y[combined],c1c2_group,
        np.concatenate([results["CE"]["c1"],results["CE"]["c2"]]),
        np.concatenate([results["CE_plus_prompt_contrastive"]["c1"],results["CE_plus_prompt_contrastive"]["c2"]]),
        shared)
    p=report
    extb=np.mean([p[k]["ba"]["CE_plus_prompt_contrastive"] for k in ("c1","c2")])
    extd=np.mean([p[k]["ba"]["strong_sparse"] for k in ("c1","c2")])
    pass_gate=(p["official_fold0"]["B_minus_CE"]["delta_ba"]>=.02
        and all(p[k]["B_minus_CE"]["delta_ba"]>=.02 for k in ("c1","c2"))
        and ci[0]>0 and p["official_fold0"]["ba"]["CE_plus_prompt_contrastive"]>=p["official_fold0"]["ba"]["strong_sparse"]
        and extb>=extd)
    output={"experiment":"TPC_trainable_roberta_last2_CE_vs_prompt_contrastive",
      "panels":report,"external_pooled_95ci_B_minus_CE":ci,
      "real_backbone_adaptation":True,"fine_tuned_layers":2,
      "metadata_external_provider_verified":False,"full18_external_holdout":False,
      "strong_sparse_feature_width":meta["total_features"],"fold0_historical_spent":True,
      "pass_gate":bool(pass_gate),"promote":False,"official_submit":False,
      "duration_seconds":round(time.monotonic()-begin,2)}
    emit("NTL_TPC_REAL_RESULT",output)
    Path("/kaggle/working/ntl_tpc_aggregate.json").write_text(json.dumps(output,sort_keys=True)+"\n")
    print("NTL_TPC_STOP_NO_SUBMIT="+str(not pass_gate).upper(),flush=True)


if __name__=="__main__":
    if __import__("os").environ.get("NTL_TPC_SELFTEST")=="1":
        selftest()
    else:
        main()
