"""NTL source-invariance experiment on a private, licensed external-only Colab T4.

Two matched training arms differ ONLY in source-head gradient path:
  C0 source head receives a detached text representation;
  C1 source head receives gradient-reversed text representation.
The objective is source-disjoint 18-way balanced accuracy, not Kaggle score.
No official competition data, no online test inference, no paid model API.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

WORK=Path("/content/ntl-source-adversarial-nov17")
DATA=Path("/content/ntl-source-robust-data")
BASE="answerdotai/ModernBERT-base"
MAX_TRAIN_STEPS=250
BATCH=4
ACCUM=4
MAXLEN=256
SOURCE_LOSS_WEIGHT=0.15
SEED=20261117


def install():
    pkgs=[
        "transformers==4.57.1",
        "tokenizers==0.22.1",
        "safetensors==0.6.2",
        "pandas==2.3.3",
        "pyarrow==22.0.0",
        "scikit-learn==1.7.2",
    ]
    subprocess.run([sys.executable,"-m","pip","install","-q",*pkgs],check=True,timeout=360)


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def main():
    install()
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset,DataLoader
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding
    from sklearn.metrics import balanced_accuracy_score

    if not torch.cuda.is_available():
        raise RuntimeError("NO_COLAB_T4_CUDA_ABORT")
    if torch.cuda.get_device_properties(0).total_memory < 13_000_000_000:
        raise RuntimeError("GPU_VRAM_BELOW_13GB")
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    WORK.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((DATA/"manifest.json").read_text())
    train=pd.read_parquet(DATA/"train.parquet")
    valid=pd.read_parquet(DATA/"valid.parquet")
    if sha(DATA/"train.parquet")!=manifest["train_sha256"] or sha(DATA/"valid.parquet")!=manifest["valid_sha256"]:
        raise RuntimeError("FROZEN_DATA_CHANGED")
    labels=list(manifest["labels"])
    if len(labels)!=18 or set(train.model)!=set(labels) or set(valid.model)!=set(labels):
        raise RuntimeError("TRAIN_OR_VALID_NOT_ALL18")
    if set(train.source)&set(valid.source) or set(train.prompt_id)&set(valid.prompt_id):
        raise RuntimeError("SOURCE_OR_PROMPT_LEAKAGE")
    source_names=sorted(train.source.unique())
    source2id={s:i for i,s in enumerate(source_names)}
    label2id={x:i for i,x in enumerate(labels)}
    valid_sources=sorted(valid.source.unique())
    token=AutoTokenizer.from_pretrained(BASE,trust_remote_code=False)

    class TextDataset(Dataset):
        def __init__(self,frame,istrain):
            self.texts=frame.text.astype(str).tolist()
            self.labels=np.array([label2id[x] for x in frame.model.astype(str)],dtype=np.int64)
            self.source_names=frame.source.astype(str).to_numpy()
            self.src=np.array([source2id[x] for x in frame.source] if istrain else [-1]*len(frame),dtype=np.int64)
        def __len__(self):return len(self.texts)
        def __getitem__(self,i):
            enc=token(self.texts[i],truncation=True,max_length=MAXLEN)
            return dict(**enc,labels=int(self.labels[i]),source_idx=int(self.src[i]))
    train_data=TextDataset(train,True)
    valid_data=TextDataset(valid,False)
    pad=DataCollatorWithPadding(tokenizer=token)
    def collate(batch):
        sources=torch.tensor([row.pop("source_idx") for row in batch],dtype=torch.long)
        out=pad(batch)
        out["source_idx"]=sources
        return out
    class Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):
            return x.view_as(x)
        @staticmethod
        def backward(ctx,grad):
            return -grad

    def evaluate(model):
        model.eval()
        gen=DataLoader(valid_data,batch_size=12,shuffle=False,collate_fn=collate,num_workers=0)
        preds=[]
        with torch.no_grad():
            for step,b in enumerate(gen):
                b.pop("source_idx")
                b={k:v.cuda(non_blocking=True) for k,v in b.items()}
                b.pop("labels")
                with torch.autocast("cuda",dtype=torch.float16):
                    logits=model(**b).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
        model.train()
        return np.asarray(preds,dtype=int)

    def train_arm(adversarial):
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        model=AutoModelForSequenceClassification.from_pretrained(
            BASE,num_labels=18,ignore_mismatched_sizes=True,trust_remote_code=False
        ).cuda()
        try:
            model.gradient_checkpointing_enable()
            model.config.use_cache=False
        except Exception as e:
            print("NTL_GRADIENT_CHECKPOINTING_STATE="+type(e).__name__,flush=True)
        hidden=int(model.config.hidden_size)
        source_head=nn.Linear(hidden,len(source_names)).cuda()
        optimizer=torch.optim.AdamW(
            [{"params":model.parameters(),"lr":3e-5},
             {"params":source_head.parameters(),"lr":1e-3}],
            weight_decay=0.01,
        )
        scaler=torch.amp.GradScaler("cuda")
        generator=torch.Generator().manual_seed(SEED)
        dl=DataLoader(train_data,batch_size=BATCH,shuffle=True,
                      generator=generator,collate_fn=collate,num_workers=0)
        steps=0;micro=0;loss_sum=0.;started=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        model.train();source_head.train()
        for raw in dl:
            if steps>=MAX_TRAIN_STEPS:break
            src=raw.pop("source_idx").cuda(non_blocking=True)
            y=raw.pop("labels").cuda(non_blocking=True)
            inputs={k:v.cuda(non_blocking=True) for k,v in raw.items()}
            with torch.autocast("cuda",dtype=torch.float16):
                out=model(**inputs,output_hidden_states=True)
                z=out.hidden_states[-1][:,0,:].float()
                source_in=Reverse.apply(z) if adversarial else z.detach()
                score=source_head(source_in)
                loss_id=F.cross_entropy(out.logits.float(),y)
                loss_source=F.cross_entropy(score,src)
                loss=(loss_id+SOURCE_LOSS_WEIGHT*loss_source)/ACCUM
            if not torch.isfinite(loss):raise RuntimeError("NONFINITE_TRAIN_LOSS")
            scaler.scale(loss).backward()
            loss_sum+=float(loss.detach().cpu())*ACCUM
            micro+=1
            if micro%ACCUM==0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(source_head.parameters()),1.0)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                steps+=1
            if steps and steps%50==0 and micro%ACCUM==0:
                print("NTL_REAL_COLAB_TRAIN_PROGRESS="+json.dumps({
                    "arm":"C1_adversarial" if adversarial else "C0_control",
                    "optimizer_steps":steps,
                    "elapsed_seconds":round(time.monotonic()-started,2)
                },sort_keys=True),flush=True)
        if steps!=MAX_TRAIN_STEPS:
            raise RuntimeError("SHORT_TRAINING_DOES_NOT_MATCH_FROZEN_OPT_STEPS")
        pred=evaluate(model)
        y=np.asarray([label2id[t] for t in valid.model],dtype=int)
        ba=float(np.mean([(pred[y==i]==i).mean() for i in range(18)]))
        by_source={}
        for source in valid_sources:
            ix=np.flatnonzero(valid.source.to_numpy()==source)
            classes=sorted(set(y[ix]))
            by_source[source]={"n":len(ix),"n_labels":len(classes),
               "subset_macro_recall":round(float(np.mean([(pred[ix][y[ix]==i]==i).mean() for i in classes])),6)}
        record={"arm":"C1_adversarial" if adversarial else "C0_control","model":BASE,
                "source_adversarial":adversarial,"optimizer_steps":steps,
                "train_n":len(train),"valid_n":len(valid),
                "valid_balanced_accuracy":round(ba,6),
                "elapsed_seconds":round(time.monotonic()-started,2),
                "by_source":by_source,
                "per_label_recall":{labels[i]:round(float((pred[y==i]==i).mean()),6) for i in range(18)},
                "training_loss_average":round(loss_sum/micro,6)}
        return model,pred,record

    t0=time.monotonic()
    control,pc,rc=train_arm(False)
    del control
    import gc
    gc.collect();torch.cuda.empty_cache()
    candidate,pt,rt=train_arm(True)
    correct=np.asarray([label2id[t] for t in valid.model])
    rescue=int(((pc!=correct)&(pt==correct)).sum())
    regression=int(((pc==correct)&(pt!=correct)).sum())
    source_delta={}
    for s in valid_sources:
        ix=np.flatnonzero(valid.source.to_numpy()==s)
        classes=sorted(set(correct[ix]))
        a=float(np.mean([(pt[ix][correct[ix]==i]==i).mean() for i in classes]))
        b=float(np.mean([(pc[ix][correct[ix]==i]==i).mean() for i in classes]))
        source_delta[s]=round(a-b,6)
    delta=rt["valid_balanced_accuracy"]-rc["valid_balanced_accuracy"]
    decision="ADVANCE_CANDIDATE_ONLY_NOT_OFFICIAL" if (
        delta>=.03 and sum(v>0 for v in source_delta.values())>=2 and rescue>regression
    ) else "REJECT_ADVERSARIAL_MECHANISM"
    report={
        "scope":"colab_t4_new_external_only_source_disjoint_modernbert",
        "data_sha256":manifest["data_sha256"],
        "checkpoint_model":BASE,
        "total_wall_seconds":round(time.monotonic()-t0,2),
        "arms":[rc,rt],
        "candidate_delta_ba":round(delta,6),
        "rescues":rescue,"regressions":regression,
        "by_holdout_source_delta":source_delta,
        "decision":decision,
        "official_competition_submission":False,
        "official_public_score":None,
        "new_training_authorized_ntl_only":True,
        "real_model_training":True,
        "model_identity_independently_proven":False,
        "source_prompt_disjoint":True,
        "upstream_prefit_overlap_with_champion_unverified":True,
        "model_can_be_competition_candidate":False,
        "local_gpu":"Colab T4",
    }
    (WORK/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2))
    print("NTL_REAL_SOURCE_DISJOINT_PAIRED_TRAIN_RESULT="+json.dumps({
       "C0_BA":rc["valid_balanced_accuracy"],
       "C1_BA":rt["valid_balanced_accuracy"],
       "delta":round(delta,6),
       "rescues":rescue,"regressions":regression,
       "source_deltas":source_delta,"decision":decision,
       "not_kaggle_score":True
    },sort_keys=True),flush=True)
    if decision.startswith("ADVANCE"):
        target=WORK/"checkpoint"
        target.mkdir()
        candidate.save_pretrained(target,safe_serialization=True)
        token.save_pretrained(target)
        with zipfile.ZipFile(WORK/"checkpoint.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in sorted(target.rglob("*")):
                if p.is_file():z.write(p,arcname="checkpoint/"+str(p.relative_to(target)))
        print("NTL_COLAB_TRAINED_CHECKPOINT_READY=TRUE",flush=True)
    else:
        print("NTL_COLAB_TRAINED_CHECKPOINT_READY=FALSE_REJECTED_BY_GATE",flush=True)
    del candidate
    gc.collect();torch.cuda.empty_cache()


if __name__=="__main__":
    main()
