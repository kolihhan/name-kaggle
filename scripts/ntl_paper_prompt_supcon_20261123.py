"""Prompt-invariant supervised contrastive (SupCon) head on FROZEN strong embeddings.

Paper inspiration: Beigi et al. 2024, arXiv:2407.21264, model attribution
under domain shift via supervised contrastive learning.

This is NOT the same paper's released architecture nor a score claim. Train only
a bounded 18-class neural readout (no QLoRA, no backbone update or API).
Use identical pretrained frozen embeddings for CE and SupCon controls.
For valid competition-style OOF, the embedding backbone MUST NOT have
trained on heldout labels. The 0.715 official-trained Qwen adapter fails that
condition for official-train OOF; its official embeddings may be evaluated
ONLY against truly external heldout labelled data.

Input .npz contract: ids (N unicode), embeddings (N,D) float32.
Input meta .json: encoder_id/revision, label_exposure, source_scope, digest.
No raw text or labels embedded in .npz, and no predictions published.

Design: paired panels of two different prompts, each containing same author
labels. Same-author/cross-prompt = positives; different-author/same-prompt =
hard negatives. Both arms share the same network, seed, optimizer, batch
sampling, and epochs; only lambda_supcon differs.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np

SEED=20261123
EPOCHS=12
WIDTH=128
ZDIM=64
LR=7e-4
TAU=.1
LAMBDA=.3
MIN_SHARED=12
BATCH_PROMPTS=2

def encode_sha(path):
    with open(path,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def validate_feature_contract(features_path,metadata_path,required_ids,expect_disjoint):
    with open(metadata_path,encoding="utf-8") as f:meta=json.load(f)
    necessary=("encoder_id","encoder_revision","label_exposure","source_scope","features_sha256")
    if any(not meta.get(k) for k in necessary):raise ValueError("FEATURE_METADATA_INCOMPLETE")
    if encode_sha(features_path)!=meta["features_sha256"]:
        raise ValueError("FEATURE_BYTES_MISMATCH")
    if not expect_disjoint and meta["label_exposure"]!="never_exposed_to_official_train_labels":
        raise ValueError("INVALID_OOF_ENCODER_LEAKED_OFFICIAL_TRAIN_LABELS")
    if expect_disjoint and meta["label_exposure"] not in (
        "never_exposed_to_official_train_labels",
        "trained_only_on_official_train_labels"):
        raise ValueError("UNKNOWN_ENCODER_TRAIN_LINEAGE")
    with np.load(features_path,allow_pickle=False) as f:
        if set(f.keys())!={"ids","embeddings"}:raise ValueError("BAD_FEATURE_KEYS")
        ids=np.asarray(f["ids"])
        emb=np.asarray(f["embeddings"],dtype=np.float32)
    if ids.dtype.kind not in "US" or len(ids)!=len(set(ids.astype(str))):
        raise ValueError("DUPLICATE_OR_PICKLED_FEATURE_ID")
    if emb.ndim!=2 or emb.shape[0]!=len(ids) or emb.shape[1]<64 or emb.shape[1]>8192:
        raise ValueError("NOT_FROZEN_STRONG_EMBEDDING")
    if not np.isfinite(emb).all() or (np.linalg.norm(emb,axis=1)<1e-7).any():
        raise ValueError("INVALID_FEATURE_VALUES")
    req=list(map(str,required_ids));obs=list(map(str,ids))
    if set(req)!=set(obs):raise ValueError("FEATURE_ID_COVERAGE_OR_MIXED_SOURCE")
    pos={a:i for i,a in enumerate(obs)}
    emb=emb[[pos[k] for k in req]]
    return emb,meta

def panel_index(y,g,min_shared=MIN_SHARED):
    # Build eligibility without examining embedding values or correctness.
    panels={}
    for i,(lab,prompt) in enumerate(zip(y,g)):
        p=panels.setdefault(str(prompt),{})
        if str(lab) in p:raise ValueError("DUPLICATE_LABEL_IN_SINGLE_PROMPT")
        p[str(lab)]=i
    keys=list(sorted(panels))
    pairs=[]
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            common=sorted(set(panels[keys[i]])&set(panels[keys[j]]))
            if len(common)>=min_shared:
                pairs.append((keys[i],keys[j],common))
    if len(pairs)<2:raise ValueError("INSUFFICIENT_PROMPT_PAIRS_FOR_REAL_SUPCON")
    return panels,pairs

def pair_batches(y,g,seed):
    panels,pairs=panel_index(y,g)
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(pairs))
    # Use roughly 3 prompt comparisons per unique prompt per epoch;
    # capped to prevent quadratic O(P^2) overtraining.
    limit=min(len(order),max(30,len(panels)))
    for t in order[:limit]:
        a,b,common=pairs[t]
        # EXACTLY two prompts x same label set. This produces paired
        # cross-prompt same-label positives and same-prompt hard negatives.
        yield np.asarray([panels[a][lab] for lab in common]+
                         [panels[b][lab] for lab in common],dtype=np.int64)

def info_nce(z,y,g,tau=TAU):
    import torch
    z=torch.nn.functional.normalize(z,dim=1)
    sim=z@z.T/tau
    k=len(z)
    diag=torch.eye(k,dtype=torch.bool,device=z.device)
    same_y=y[:,None].eq(y[None,:])
    different_prompt=~g[:,None].eq(g[None,:])
    positive=same_y & different_prompt & ~diag
    if not positive.any():raise ValueError("NO_POSITIVES_IN_PAIR")
    sim=sim.masked_fill(diag,-1e9)
    lse=torch.logsumexp(sim,dim=1,keepdim=True)
    logp=sim-lse
    p_count=positive.sum(dim=1)
    valid=p_count>0
    return -((logp*positive).sum(dim=1)[valid]/p_count[valid]).mean()

def fit_one(x,y,groups,seed=SEED,lambda_supcon=LAMBDA,epochs=EPOCHS):
    import torch
    torch.set_num_threads(min(torch.get_num_threads(),2))
    torch.manual_seed(seed)
    np.random.seed(seed)
    n,d=x.shape
    unique=tuple(sorted(set(map(str,y))))
    if len(unique)!=18:raise ValueError("EXPECTED_18_CLASSES")
    if len(set(groups))<10:raise ValueError("TOO_FEW_PROMPTS")
    label_map={name:i for i,name in enumerate(unique)}
    yi=np.asarray([label_map[str(q)] for q in y],dtype=np.int64)
    _,gidx=np.unique(groups,return_inverse=True)
    inputs=torch.from_numpy(np.asarray(x,dtype=np.float32))
    ys=torch.from_numpy(yi)
    gg=torch.from_numpy(gidx.astype(np.int64))
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj=torch.nn.Sequential(
                torch.nn.LayerNorm(d),torch.nn.Linear(d,WIDTH),
                torch.nn.GELU(),torch.nn.Linear(WIDTH,ZDIM))
            self.out=torch.nn.Linear(ZDIM,len(unique))
        def forward(self,xx):
            zz=torch.nn.functional.normalize(self.proj(xx),dim=1)
            return self.out(zz),zz
    model=Head()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.02)
    # IMPORTANT: both arms use exact same paired-panel minibatches and seed.
    for epoch in range(epochs):
        model.train()
        for batch in pair_batches(y,groups,seed+epoch):
            p,z=model(inputs[batch])
            ce=torch.nn.functional.cross_entropy(p,ys[batch])
            loss=ce+(lambda_supcon*info_nce(z,ys[batch],gg[batch])
                     if lambda_supcon else 0.)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model,unique

def predict(model,x,labels):
    import torch
    with torch.no_grad():
        out=[]
        xx=torch.from_numpy(np.asarray(x,dtype=np.float32))
        for b in range(0,len(xx),128):
            scores,_=model(xx[b:b+128])
            out.extend(scores.argmax(dim=1).tolist())
    return np.asarray([labels[k] for k in out])

def metrics(y,old,new,labels):
    by={lab:float(np.mean(new[np.asarray(y)==lab]==lab)) for lab in labels}
    return {"BA":round(float(np.mean(list(by.values()))),6),
            "correct":int(np.sum(new==y)),
            "rescues":int(np.sum((new==y)&(old!=y))),
            "regressions":int(np.sum((new!=y)&(old==y))),
            "per_class_recall":{k:round(v,4) for k,v in by.items()}}

def run_contract(train_csv,fold_csv,feature_npz,metadata_json,ext_csv=None,
                 ext_npz=None,ext_metadata=None):
    import pandas as pd
    official_sha="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
    if encode_sha(train_csv)!=official_sha:raise ValueError("OFFICIAL_SOURCE_SHA_DRIFT")
    tr=pd.read_csv(train_csv,dtype={"id":str,"prompt_id":str})
    f=pd.read_csv(fold_csv,dtype={"id":str,"prompt_id":str})
    assert len(tr)==2699 and tr.model.nunique()==18 and tr.prompt_id.nunique()==150
    joined=tr.merge(f[["id","fold_id","model","prompt_id"]],on="id",
                    suffixes=("","_f"),validate="one_to_one")
    assert ((joined.model==joined.model_f)&
            (joined.prompt_id==joined.prompt_id_f)).all()
    assert set(joined.fold_id)==set(range(5))
    assert joined.groupby("prompt_id").fold_id.nunique().max()==1
    x,meta=validate_feature_contract(feature_npz,metadata_json,joined.id,False)
    y=joined.model.astype(str).to_numpy()
    groups=joined.prompt_id.astype(str).to_numpy()
    predictions={}
    for k in (0.,LAMBDA):
        o=np.empty(len(joined),dtype=object)
        for fold in range(5):
            tridx=np.flatnonzero(joined.fold_id.to_numpy()!=fold)
            vaidx=np.flatnonzero(joined.fold_id.to_numpy()==fold)
            model,labels=fit_one(x[tridx],y[tridx],groups[tridx],
                                 seed=SEED+fold,lambda_supcon=k)
            o[vaidx]=predict(model,x[vaidx],labels)
        predictions["C0_CE" if k==0 else "C1_PromptSCL"]=o
    zero=predictions["C0_CE"]; scl=predictions["C1_PromptSCL"]
    result={"scope":"OFFICIAL_2699_FIXED_5FOLD_NO_LEAKING_ENCODER",
      "embedding_provenance":meta["encoder_id"]+"/"+meta["encoder_revision"],
      "CE_BA":round(float(np.mean([np.mean(zero[y==q]==q) for q in sorted(set(y))])),6),
      "SCL_vs_CE":metrics(y,zero,scl,tuple(sorted(set(y)))),
      "note":"training folds same, prefit encoder must not have seen official labels",
      "paid_api":False,"QLoRA":False,"new_foundation_model_training":False,
      "competition_submit":False}
    print("NTL_PAPER_SUPCON_REAL="+json.dumps(result,sort_keys=True),flush=True)
    if (ext_csv is None) != (ext_npz is None) or (ext_npz is None)!=(ext_metadata is None):
        raise ValueError("INCOMPLETE_EXTERNAL_FEATURE_SET")
    if ext_csv is not None:
        ext=pd.read_csv(ext_csv,dtype={"id":str})
        assert set(("id","model"))<=set(ext.columns)
        if ext.id.duplicated().any() or set(ext.model)!=set(y):
            raise ValueError("EXTERNAL_COVERAGE_DRIFT")
        ex,ext_meta=validate_feature_contract(ext_npz,ext_metadata,ext.id,True)
        if (ext_meta["encoder_id"]!=meta["encoder_id"] or
            ext_meta["encoder_revision"]!=meta["encoder_revision"]):
            raise ValueError("ENCODER_MISMATCH_BETWEEN_SOURCES")
        out={}
        for k in (0.,LAMBDA):
            model,labels=fit_one(x,y,groups,lambda_supcon=k)
            pred=predict(model,ex,labels)
            out["C0_CE" if k==0 else "C1_PromptSCL"]={
                "BA":round(float(np.mean([
                    np.mean(pred[ext.model.to_numpy()==lab]==lab) for lab in labels
                ])),6)}
        print("NTL_PAPER_SUPCON_EXTERNAL="+json.dumps(out,sort_keys=True),flush=True)

def smoke():
    import torch
    # 18 classes, 12 prompt groups with one sample per class/prompt;
    # synthetic E2E tests mechanics ONLY, not a performance claim.
    g=np.repeat(np.arange(12),18)
    y=np.tile(np.asarray(["class"+str(k) for k in range(18)]),12)
    x=np.random.default_rng(SEED).normal(size=(len(y),96)).astype(np.float32)
    panels=list(pair_batches(y,g,SEED))
    assert len(panels)>=30 and all(len(p)==36 for p in panels)
    for p in panels:
        assert len(set(g[p]))==2 and len(set(y[p]))==18
    m,labels=fit_one(x,y,g,epochs=1)
    p=predict(m,x[:36],labels)
    assert len(p)==36 and set(p)<=set(labels)
    xx=torch.randn(36,ZDIM,requires_grad=True)
    loss=info_nce(xx,torch.tensor(np.tile(np.arange(18),2)),
                  torch.tensor(np.repeat(np.arange(2),18)))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(xx.grad).all()
    print("NTL_PAPER_SUPCON_SMOKE="+json.dumps({
        "pass":True,"groups":12,"labels":18,
        "pair_hard_negatives":True,"cross_prompt_positives":True,
        "gradient_finite":True,"real_BA":None,"no_model_trained":False,
        "synthetic_training_only":True,"not_a_Kaggle_score":True},sort_keys=True),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--smoke",action="store_true")
    a.add_argument("--train",type=Path)
    a.add_argument("--folds",type=Path)
    a.add_argument("--features",type=Path)
    a.add_argument("--metadata",type=Path)
    a.add_argument("--external",type=Path)
    a.add_argument("--external-features",type=Path)
    a.add_argument("--external-metadata",type=Path)
    v=a.parse_args()
    if v.smoke:smoke()
    else:
        if any(z is None for z in (v.train,v.folds,v.features,v.metadata)):
            raise ValueError("REQUIRE_REAL_FROZEN_EMBEDDINGS_SOURCE")
        run_contract(v.train,v.folds,v.features,v.metadata,
                     v.external,v.external_features,v.external_metadata)
