"""NTL Prompt-Controlled Contrastive Learning (PC-CL) v0: paired real T4 QLoRA pilot.

Run ONLY in authorized owner's private Colab T4 after data prep; no official
Kaggle train/test inputs, no competition submission, no checkpoint promotion.
C0: CE. C1: identical batches/initialization/optimizer budget + 0.2 SupCon
on frozen 2-prompt x 2-generator rectangles. Contrastive positives share a
generator but NOT prompt; negatives include different generators on SAME prompt.
Neither arm uses source IDs or prompts at inference. Source-disjoint validation.
"""
from __future__ import annotations
import gc, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path("/content")
TRAIN=ROOT/"train.parquet"
VALID=ROOT/"valid.parquet"
BATCHES=ROOT/"batches.json"
MANIFEST=ROOT/"manifest.json"
DEST=ROOT/"ntl-pccl-nov20-result.json"
BASE="Qwen/Qwen3-1.7B-Base"
REVISION="658b6ee46470b73f3e0e04e57fc9fb7031fd7daa"
SEED=20261120
LR=2e-4
BATCH=4
MAXLEN=192
WEIGHT=0.2
TEMPERATURE=0.10
BOOTSTRAPS=1500


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def load():
    if DEST.exists():
        raise FileExistsError("REFUSE_PREVIOUS_COLAB_RESULT_REUSE")
    for p in (TRAIN,VALID,BATCHES,MANIFEST):
        if not p.is_file():raise FileNotFoundError(p.name)
    m=json.loads(MANIFEST.read_text())
    for p,k in ((TRAIN,"train_sha256"),(VALID,"valid_sha256"),(BATCHES,"batches_sha256")):
        if sha(p)!=m[k]:raise RuntimeError("PRIVATE_FROZEN_DATA_OR_SCHEDULE_CHANGED:"+p.name)
    train=pd.read_parquet(TRAIN)
    valid=pd.read_parquet(VALID)
    plan=json.loads(BATCHES.read_text())
    if len(train)!=m["train_rows"] or len(valid)!=m["valid_rows"]:
        raise RuntimeError("TRAIN_VALID_ROW_COUNT_DRIFT")
    if len(plan)!=160 or len(set(m["labels"]))!=18 or len(m["labels"])!=18:
        raise RuntimeError("PCCL_SHAPE_DRIFT")
    if set(train.model)!=set(m["labels"]) or set(valid.model)!=set(m["labels"]):
        raise RuntimeError("PCCL_ALL18_DRIFT")
    if set(train.source)&set(valid.source) or set(train.prompt_key)&set(valid.prompt_key):
        raise RuntimeError("PCCL_SOURCE_OR_PROMPT_DISJOINTNESS_BROKEN")
    if m["model"]!=BASE or m["max_length_tokens"]!=MAXLEN \
        or m["contrastive_weight"]!=WEIGHT or m["contrastive_temperature"]!=TEMPERATURE:
        raise RuntimeError("PCCL_PREDECLARED_HYPERPARAM_DRIFT")
    if any(len(b["indices"])!=BATCH or min(b["indices"])<0 or max(b["indices"])>=len(train)
           for b in plan):
        raise RuntimeError("PCCL_BATCHES_OUT_OF_RANGE")
    for batch in plan:
        if batch["kind"]=="rectangle":
            sub=train.iloc[batch["indices"]]
            a,b,c,d=sub.itertuples(index=False,name=None)
            if a[1]!=c[1] or b[1]!=d[1] or a[1]==b[1] \
                or a[4]!=b[4] or c[4]!=d[4] or a[4]==c[4]:
                raise RuntimeError("ACTUAL_PCCL_RECTANGLE_INVALID")
    return m,train,valid,plan


def deterministic():
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    # SDPA kernels may be nondeterministic, but both arms share exact setup.
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    torch.set_num_threads(4)


def infonce(z):
    """Row order [a@prompt1,b@prompt1,a@prompt2,b@prompt2]."""
    import torch
    import torch.nn.functional as F
    if z.ndim!=2 or len(z)!=4:raise ValueError("PCCL_NEEDS_EXACT_2X2")
    emb=F.normalize(z.float(),dim=-1,eps=1e-8)
    cosine=(emb@emb.T)/TEMPERATURE
    cosine=cosine.masked_fill(torch.eye(4,device=z.device,dtype=torch.bool),-1e4)
    pos=torch.tensor([2,3,0,1],device=z.device)
    return F.cross_entropy(cosine,pos)


def infonce_selftest():
    import torch
    good=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],requires_grad=True)
    bad=torch.tensor([[1.,0.],[0.,1.],[0.,1.],[1.,0.]],requires_grad=True)
    l_good=infonce(good);l_bad=infonce(bad)
    if not torch.isfinite(l_good+l_bad) or not l_good<l_bad:
        raise RuntimeError("CONTRASTIVE_RECTANGLE_LOSS_DIRECTION_WRONG")
    l_good.backward()
    if good.grad is None or not torch.isfinite(good.grad).all():
        raise RuntimeError("CONTRASTIVE_BACKWARD_BROKEN")
    print("NTL_PCCL_REAL_SUPCON_CPU_UNIT=PASS",flush=True)


def make_model(num_labels):
    import torch
    from transformers import AutoModelForSequenceClassification,BitsAndBytesConfig
    from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig,TaskType
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
           bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    net=AutoModelForSequenceClassification.from_pretrained(
       BASE,revision=REVISION,num_labels=num_labels,
       quantization_config=quant,device_map={"":"cuda:0"},
       torch_dtype=torch.float16,attn_implementation="sdpa",
       trust_remote_code=False)
    net.config.use_cache=False
    net=prepare_model_for_kbit_training(net,use_gradient_checkpointing=True)
    lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,
          bias="none",target_modules=["q_proj","k_proj","v_proj","o_proj",
                                      "gate_proj","up_proj","down_proj"],
          modules_to_save=["score"],task_type=TaskType.SEQ_CLS)
    net=get_peft_model(net,lora)
    net.config.pad_token_id=net.get_base_model().config.pad_token_id
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    n=sum(p.numel() for p in net.parameters() if p.requires_grad)
    if n<100_000 or n>60_000_000:
        raise RuntimeError("TRAINABLE_PARAMETER_COUNT_UNEXPECTED:"+str(n))
    return net,n


def train_one(arm,tokenizer,labels,train,valid,plan):
    import torch
    from torch.nn import functional as F
    deterministic()
    net,nparam=make_model(len(labels))
    net.config.pad_token_id=tokenizer.pad_token_id
    net.get_base_model().config.pad_token_id=tokenizer.pad_token_id
    optim=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
            lr=LR,weight_decay=0.01)
    net.train()
    losses=[];first=None
    started=time.monotonic()
    for step,ex in enumerate(plan):
        ix=ex["indices"]
        inputs=tokenizer(train.text.iloc[ix].astype(str).tolist(),
            padding=True,truncation=True,max_length=MAXLEN,
            return_tensors="pt",add_special_tokens=True)
        inputs={k:v.to("cuda") for k,v in inputs.items()}
        yi=torch.tensor([labels.index(x) for x in train.model.iloc[ix]],device="cuda")
        optim.zero_grad(set_to_none=True)
        out=net(**inputs,output_hidden_states=True,return_dict=True)
        if out.logits.shape!=(4,len(labels)):
            raise RuntimeError("MODEL_18CLASS_SCORE_HEAD_SHAPE_WRONG")
        ce=F.cross_entropy(out.logits.float(),yi)
        contrast=torch.zeros((),device="cuda",dtype=torch.float32)
        if arm=="C1" and ex["kind"]=="rectangle":
            final=out.hidden_states[-1]
            last=inputs["attention_mask"].sum(dim=-1)-1
            pooled=final[torch.arange(BATCH,device="cuda"),last,:]
            contrast=infonce(pooled)
        loss=ce+WEIGHT*contrast
        if not torch.isfinite(loss):
            raise RuntimeError("NONFINITE_PCCL_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.)
        optim.step()
        ll=float(loss.detach().item())
        if first is None:first=ll
        losses.append(ll)
        if step==0 or (step+1)%40==0 or step+1==len(plan):
            print("NTL_PCCL_ACTUAL_T4_TRAIN="+json.dumps({
                "arm":arm,"completed_steps":step+1,"total_steps":len(plan),
                "mean_loss_last40":round(float(np.mean(losses[-40:])),5),
                "elapsed_seconds":round(time.monotonic()-started,2),
                "cuda_peak_GB":round(torch.cuda.max_memory_allocated()/1024**3,3)
            },sort_keys=True),flush=True)
        del out,inputs,ce,contrast,loss
    if len(losses)!=160:raise RuntimeError("PCCL_UNEXPECTED_TRAIN_STEPS")
    net.eval()
    preds=[]
    with torch.inference_mode():
        for pos in range(0,len(valid),8):
            inputs=tokenizer(valid.text.iloc[pos:pos+8].astype(str).tolist(),
               padding=True,truncation=True,max_length=MAXLEN,
               return_tensors="pt",add_special_tokens=True)
            inputs={k:v.to("cuda") for k,v in inputs.items()}
            out=net(**inputs,output_hidden_states=False,return_dict=True)
            pred=out.logits.float().argmax(dim=-1).cpu().numpy()
            preds.extend(pred.tolist())
            del out,inputs
    if len(preds)!=len(valid):
        raise RuntimeError("PCCL_INCOMPLETE_VALIDATION")
    y=np.array([labels.index(x) for x in valid.model],dtype=np.int64)
    pred=np.asarray(preds,dtype=np.int64)
    recall={lab:round(float(np.mean(pred[y==i]==i)),6) for i,lab in enumerate(labels)}
    source_recall={}
    for source in sorted(valid.source.unique()):
        ix=valid.source.to_numpy()==source
        sublabs=np.unique(y[ix])
        source_recall[str(source)]={
            "rows":int(ix.sum()),"nclasses":len(sublabs),
            "subgroup_macro_recall":round(float(np.mean([
                np.mean(pred[ix][y[ix]==k]==k) for k in sublabs])),6)
        }
    report={"arm":arm,"optimizer_steps":len(plan),"trainable_parameters":nparam,
       "first_train_loss":round(first,6),
       "final_40step_mean_loss":round(float(np.mean(losses[-40:])),6),
       "BA18":round(float(np.mean(list(recall.values()))),6),
       "per_class_recall":recall,"source_macro_recall":source_recall,
       "elapsed_seconds_including_eval":round(time.monotonic()-started,2)}
    print("NTL_PCCL_PAIRED_ARM_EVAL="+json.dumps(report,sort_keys=True),flush=True)
    del net,optim
    gc.collect()
    torch.cuda.empty_cache()
    return report,pred


def paired_credible(valid,y,a,b,labels):
    correct0=(a==y)
    correct1=(b==y)
    rescued=int((~correct0&correct1).sum())
    regressed=int((correct0&~correct1).sum())
    delta_by_label={
        lab:round(float((correct1[y==k].astype(np.int8)-correct0[y==k].astype(np.int8)).mean()),6)
        for k,lab in enumerate(labels)}
    # Prompt-cluster bootstrap stratified by the three held-out publisher sources.
    rng=np.random.default_rng(SEED)
    group_keys=valid.source.astype(str)+"|"+valid.prompt_id.astype(str)
    group_names=group_keys.to_numpy()
    sources=valid.source.astype(str).to_numpy()
    groups=sorted(set(group_names))
    gid={v:i for i,v in enumerate(groups)}
    counts=np.zeros((len(groups),18),dtype=np.int32)
    resc=np.zeros((len(groups),18),dtype=np.int32)
    for i,key in enumerate(group_names):
        g=gid[key];lab=y[i]
        counts[g,lab]+=1
        resc[g,lab]+=int(correct1[i])-int(correct0[i])
    deltas=[]
    for _ in range(BOOTSTRAPS):
        sampled=np.zeros(len(groups),dtype=np.int32)
        for src in sorted(set(sources)):
            ix=np.array([gid[k] for k in groups if k.startswith(src+"|")],dtype=np.int32)
            if not len(ix):continue
            sampled[ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)))
        tot=sampled@counts
        if (tot==0).any():continue
        change=sampled@resc
        deltas.append(float(np.mean(change/tot)))
    if len(deltas)<int(BOOTSTRAPS*.9):
        raise RuntimeError("BOOTSTRAP_LABEL_SUPPORT_UNRELIABLE")
    ba0=float(np.mean([correct0[y==i].mean() for i in range(18)]))
    ba1=float(np.mean([correct1[y==i].mean() for i in range(18)]))
    lo,hi=np.quantile(deltas,[.025,.975])
    return {"delta_BA":round(ba1-ba0,6),"delta_prompt_cluster_CI95":[round(float(lo),6),round(float(hi),6)],
      "prompt_source_cluster_bootstrap_valid_draws":len(deltas),
      "rescues":rescued,"regressions":regressed,"per_class_recall_delta":delta_by_label,
      "source_count":len(set(sources)),
      "exploratory_disjoint_publisher_source_not_independent_provider_validation":True}


def main():
    import torch
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    if not torch.cuda.is_available():
        raise RuntimeError("REAL_T4_CUDA_MISSING_ABORT_NO_CPU_MODEL")
    if torch.cuda.get_device_properties(0).total_memory<12_000_000_000:
        raise RuntimeError("GPU_NOT_16GB_CLASS")
    if not all(p.is_file() for p in (TRAIN,VALID,BATCHES,MANIFEST)):
        raise RuntimeError("MISSING_OWNER_PRIVATE_REMOTE_DATA")
    infonce_selftest()
    info=HfApi().model_info(BASE,revision=REVISION)
    if info.sha!=REVISION or getattr(info.card_data,"license",None)!="apache-2.0":
        raise RuntimeError("PINNED_APACHE_BACKBONE_NOT_VERIFIED")
    man,train,valid,plan=load()
    labels=man["labels"]
    tokenizer=AutoTokenizer.from_pretrained(BASE,revision=REVISION,
        use_fast=True,trust_remote_code=False)
    tokenizer.padding_side="right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("TOKENIZER_PAD_NOT_CONFIGURABLE")
    reports=[];predictions=[]
    for arm in ("C0","C1"):
        r,p=train_one(arm,tokenizer,labels,train,valid,plan)
        reports.append(r);predictions.append(p)
    y=np.array([labels.index(x) for x in valid.model],dtype=np.int64)
    paired=paired_credible(valid,y,predictions[0],predictions[1],labels)
    if paired["delta_BA"]>=0.04 and paired["delta_prompt_cluster_CI95"][0]>0:
        decision="EXPLORE_SECOND_DISJOINT_SOURCE_AND_SHUFFLED_CONTROL_NOT_SUBMIT"
    elif paired["delta_BA"]<=0:
        decision="REJECT_PC_CL_V0_ON_THIS_COHORT_REASSESS_HYPOTHESIS"
    else:
        decision="INCONCLUSIVE_PAIRED_PILOT_DO_NOT_PROMOTE"
    result={
       "scope":"ntl_real_owner_t4_pccl_external_source_disjoint_v0",
       "backbone":BASE,"revision":REVISION,"official_public_unchanged":0.702,
       "training_same_backbone_data_budget":True,
       "train_sha":man["train_sha256"],"valid_sha":man["valid_sha256"],
       "plan_sha":man["batches_sha256"],
       "train_rows":man["train_rows"],"valid_rows":man["valid_rows"],
       "all_labels":labels,"pilot_not_strong_0702_complementarity":True,
       "arms":reports,"paired":paired,"decision":decision,
       "no_kaggle_gpu":True,"no_competition_submit":True,
       "no_raw_text_predictions_or_model_weights_shared":True,
    }
    DEST.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("NTL_PCCL_REAL_T4_FINAL_AGGREGATE="+json.dumps({
        "C0_BA":reports[0]["BA18"],"C1_BA":reports[1]["BA18"],
        "delta_BA":paired["delta_BA"],"delta_CI95":paired["delta_prompt_cluster_CI95"],
        "rescues":paired["rescues"],"regressions":paired["regressions"],
        "decision":decision,"not_official":True
    },sort_keys=True),flush=True)
    del predictions,train,valid
    gc.collect();torch.cuda.empty_cache()


if __name__=="__main__":
    main()
