"""Colab T4 decoder hidden-state extractor; not a Kaggle submission.

SHA-pin official 2699-row TRAIN and 5-fold map, model revision and Apache 2.0
license. Only private /content .npz output; DO NOT commit source data,
rowwise features, IDs, or predictions to public GitHub.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLD_SHA="94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="658b6ee46470b73f3e0e04e57fc9fb7031fd7daa"
VERSION="ntl-decoder-fourlayer-prompt-pca8-v1-20261119"
MAX_LENGTH=512

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_train(train,folds):
    if sha(train)!=TRAIN_SHA or sha(folds)!=FOLD_SHA:
        raise ValueError("OFFICIAL_TRAIN_OR_FOLD_SNAPSHOT_DRIFT")
    a=pd.read_csv(train,dtype={"id":str,"prompt_id":str})
    b=pd.read_csv(folds,dtype={"id":str,"prompt_id":str})
    f=a.merge(b,on=["id","prompt_id","model"],validate="one_to_one",how="outer",indicator=True)
    if (len(f)!=2699 or (f._merge!="both").any() or f.isna().any().any()
        or f.id.duplicated().any() or f.model.nunique()!=18 or f.prompt_id.nunique()!=150
        or f.groupby("prompt_id").fold_id.nunique().ne(1).any()):
        raise ValueError("OFFICIAL_TRAIN_VALIDATION_STRUCTURE_INVALID")
    return f.sort_values("id").reset_index(drop=True)

def layer_ids(n):
    if n<8:raise ValueError("NOT_A_MULTILAYER_DECODER")
    indices=[n//4,n//2,(3*n)//4,n]
    if len(set(indices))!=4:raise ValueError("LAYER_INDICES_NOT_UNIQUE")
    return indices

def masked_pooled(hidden_states,attention_mask,indices):
    import torch
    mask=attention_mask.unsqueeze(-1).to(torch.float32)
    den=mask.sum(1).clamp(min=1)
    return torch.cat([(hidden_states[i].float()*mask).sum(1)/den for i in indices],dim=1)

def extract(frame,out,batch):
    import torch
    from transformers import AutoTokenizer,AutoModel
    from huggingface_hub import HfApi
    if out.exists():raise FileExistsError("REFUSE_OVERWRITE_PRIVATE_FEATURES")
    if batch<1 or batch>12:raise ValueError("T4_BATCH_OUT_OF_RANGE")
    if not torch.cuda.is_available():raise RuntimeError("COLAB_GPU_REQUIRED_NO_CPU_FALLBACK")
    if torch.cuda.get_device_properties(0).total_memory<12*1024**3:
        raise RuntimeError("INSUFFICIENT_GPU_RAM")
    info=HfApi().model_info(MODEL,revision=REVISION)
    license=getattr(info.card_data,"license",None)
    if info.sha!=REVISION or license!="apache-2.0":
        raise PermissionError("MODEL_REVISION_OR_APACHE_LICENSE_NOT_VERIFIED")
    print("NTL_F1_PINNED_DECODER_PROVENANCE="+json.dumps({
        "model":MODEL,"revision":REVISION,"license":license,
        "device":str(torch.cuda.get_device_name(0))},sort_keys=True),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,
        trust_remote_code=False,use_fast=True)
    tokenizer.padding_side="right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:raise ValueError("MISSING_PAD_AND_EOS")
        tokenizer.pad_token=tokenizer.eos_token
    torch.manual_seed(20261119)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION,
        torch_dtype=torch.float16,attn_implementation="sdpa",
        trust_remote_code=False,low_cpu_mem_usage=True).to("cuda").eval()
    model.config.use_cache=False
    n=int(model.config.num_hidden_layers)
    layers=layer_ids(n)
    dim=int(model.config.hidden_size)*4
    X=np.empty((len(frame),dim),dtype=np.float16)
    order=sorted(range(len(frame)),key=lambda i:len(str(frame.text.iloc[i])))
    start=time.monotonic()
    with torch.inference_mode():
        for batchnum,lo in enumerate(range(0,len(order),batch)):
            indices=order[lo:lo+batch]
            texts=frame.text.iloc[indices].astype(str).tolist()
            tokens=tokenizer(texts,padding=True,truncation=True,max_length=MAX_LENGTH,
                return_tensors="pt",add_special_tokens=True)
            if tokens["attention_mask"].sum(dim=1).eq(0).any():
                raise ValueError("EMPTY_TOKENIZATION")
            tokens={k:v.cuda(non_blocking=True) for k,v in tokens.items()}
            output=model(**tokens,output_hidden_states=True,use_cache=False,return_dict=True)
            h=output.hidden_states
            if h is None or len(h)!=n+1:raise RuntimeError("MISSING_HIDDEN_STATES")
            v=masked_pooled(h,tokens["attention_mask"],layers)
            if v.shape!=(len(indices),dim) or not torch.isfinite(v).all():
                raise FloatingPointError("EMBEDDING_NONFINITE_OR_BAD_SHAPE")
            X[indices]=v.to(torch.float16).cpu().numpy()
            if batchnum==0 or batchnum%50==49 or lo+len(indices)==len(order):
                print("NTL_F1_GPU_EXTRACT_PROGRESS="+json.dumps({
                    "done":lo+len(indices),"total":len(frame),
                    "seconds":round(time.monotonic()-start,1),
                    "peak_gpu_gb":round(torch.cuda.max_memory_allocated()/1024**3,3)
                },sort_keys=True),flush=True)
            del output,h,v,tokens
    if not np.isfinite(X).all():raise FloatingPointError("EXTRACTED_NONFINITE")
    meta={"schema":VERSION,"model":MODEL,"revision":REVISION,
          "train_sha":TRAIN_SHA,"fold_sha":FOLD_SHA,"license":license,
          "max_length":MAX_LENGTH,"pooling":"attention_mask_mean",
          "row_count":len(frame),"layer_indices":layers,
          "feature_dim":dim,"elapsed_seconds":round(time.monotonic()-start,1),
          "no_training":True,"no_submission":True}
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,ids=frame.id.to_numpy(dtype=str),features=X,
                        metadata=np.asarray(json.dumps(meta,sort_keys=True)))
    print("NTL_F1_PRIVATE_FEATURE_FILE_COMPLETE="+json.dumps({
        "rows":len(frame),"dim":dim,"file_sha256":sha(out),
        "private_local_file":str(out),"never_upload_to_public_repo":True
    },sort_keys=True),flush=True)

def self_test():
    import torch
    h=torch.randn(2,7,4)
    mask=torch.tensor([[1,1,1,0,0,0,0],[1,1,1,1,1,1,1]])
    v=masked_pooled([h,h*2,h*3,h*4,h*5],mask,[1,2,3,4])
    assert layer_ids(28)==[7,14,21,28]
    assert v.shape==(2,16)
    assert torch.allclose(v[0,:4],h[0,:3,:].mean(0)*2)
    print("NTL_F1_GPU_EXTRACTOR_MASKED_POOL_CPU_TEST=PASS")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,default=Path("data/train.csv"))
    p.add_argument("--folds",type=Path,default=Path("artifacts/phase1/folds.csv"))
    p.add_argument("--output",type=Path)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:self_test();return
    if a.output is None:p.error("--output required")
    extract(read_train(a.train,a.folds),a.output,a.batch_size)

if __name__=="__main__":main()
