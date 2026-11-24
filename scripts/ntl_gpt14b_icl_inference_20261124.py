"""Bounded Qwen3-14B frozen zero-shot vs matched-triplet ICL GPT sibling test.

One private Kaggle T4x2 inference kernel. NO finetune, no gradients,
no learned small classifier, no official Kaggle submission.
No official response/prompt text or logits are logged or published.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
C1_SHA="8746c4b77c2b09b9abafaeecc381a39321b9919214c005a3657042051c51b177"
LABELS=("gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra")
ARMS=("ZERO","PAIR","PLACEBO")
OFFICIAL_HELDOUT_GROUPS=24
DEMO_GROUPS=3
EXEMPLAR_CHARS=480
MAX_INPUT_TOKENS=3072
RESULT_FILE="ntl_qwen14b_gpt_icl_aggregate.json"
KAGGLE_MODEL_REF="qwen-lm/qwen-3/Transformers/14b/1"
SEED=20261124

def _digest(f):
    with Path(f).open("rb") as z:return hashlib.file_digest(z,"sha256").hexdigest()

def _order_key(s):
    return hashlib.sha256(str(s).encode()).hexdigest()

def _normalize(s):
    return re.sub(r"\s+"," ",str(s)).casefold().strip()

def prepare(train_path,c1_path):
    if _digest(train_path)!=TRAIN_SHA or _digest(c1_path)!=C1_SHA:
        raise ValueError("original official or external cohort SHA drift")
    t=pd.read_csv(train_path,dtype={"id":str,"prompt_id":str,"model":str})
    c=pd.read_csv(c1_path,dtype={"prompt_id":str,"label":str})
    if len(t)!=2699 or t.model.nunique()!=18 or t.prompt_id.nunique()!=150 or not t.id.is_unique:
        raise ValueError("original official training schema drift")
    if len(c)!=360 or c.label.nunique()!=12 or c.prompt_id.nunique()!=30:
        raise ValueError("external C1 publisher schema drift")
    if not set(LABELS)<=set(t.model) or not set(LABELS)<=set(c.label):
        raise ValueError("missing GPT 5.6 siblings")
    ts=t[t.model.isin(LABELS)].copy()
    if ts.isna().any().any() or ts.groupby("prompt_id").size().ne(3).any():
        raise ValueError("official GPT triplet must be complete per prompt")
    cg=c[c.label.isin(LABELS)].copy().rename(columns={"label":"model"})
    if cg.groupby("prompt_id").size().ne(3).any() or cg.model.isna().any() or cg.text.isna().any():
        raise ValueError("external C1 GPT triplets must be complete")
    prompts=sorted(ts.prompt_id.unique(),key=_order_key)
    selected=prompts[:OFFICIAL_HELDOUT_GROUPS]
    demos=prompts[OFFICIAL_HELDOUT_GROUPS:OFFICIAL_HELDOUT_GROUPS+DEMO_GROUPS]
    if len(selected)!=24 or len(demos)!=3 or set(selected)&set(demos):
        raise ValueError("official train prompt split drift")
    example_sets={}
    for pid in demos:
        g=ts[ts.prompt_id==pid].set_index("model")
        if len(g)!=3 or set(g.index)!=set(LABELS):raise ValueError("demo prompt missing sibling")
        example_sets[str(pid)]={lab:str(g.loc[lab,"text"])[:EXEMPLAR_CHARS] for lab in LABELS}
    a=ts[ts.prompt_id.isin(selected)].copy()
    b=cg.copy()
    if len(a)!=72 or len(b)!=90:raise ValueError("wrong official/external evaluation size")
    rows=[]
    for frame,tag in ((a,"official_heldout"),(b,"Beroli_C1")):
        if frame[["id","model","text"]].isna().any().any() if "id" in frame.columns else frame[["model","text"]].isna().any().any():
            raise ValueError("missing sample fields")
        for _,z in frame.sort_values(["prompt_id","model"]).iterrows():
            rid=(str(z["id"]) if "id" in frame.columns else tag+"|"+str(z["prompt_id"])+"|"+str(z["model"]))
            rows.append({"id":rid,"group":str(z["prompt_id"]),"domain":tag,"text":str(z["text"]),"truth":str(z["model"])})
    seen=set()
    for x in rows:
        key=_normalize(x["text"])
        if key in seen:raise ValueError("duplicate response inside evaluation panels")
        seen.add(key)
    train_norm={_normalize(z) for g in example_sets.values() for z in g.values()}
    if any(_normalize(r["text"]) in train_norm for r in rows):
        raise ValueError("demo exact-response contamination")
    if len(rows)!=162 or len(set((r["domain"],r["group"],r["truth"]) for r in rows))!=162:
        raise ValueError("incomplete source groups")
    return rows,example_sets,{"off_prompts":24,"demo_prompts":3,"external_prompts":30,
      "official_rows":72,"external_rows":90,"dataset_sha_verified":True,
      "no_test_prompt_grouping_used_at_inference":True}

def choices_for(idstr):
    k=int(_order_key(idstr)[:8],16)%3
    return LABELS[k:]+LABELS[:k]

def make_prompt(row,demos,arm):
    if arm not in ARMS:raise ValueError("unsupported frozen arm")
    ordered=choices_for(row["id"])
    mapping={lab:chr(65+i) for i,lab in enumerate(ordered)}
    header=("You are an LLM text attribution analyst. Identify which exact GPT-5.6 version "
            "most likely wrote a single response. Do not infer the model from the subject "
            "of the response. Compare sentence rhythm, structure, diction and habits. "
            "Answer with exactly one letter, no explanation.\n"
            +"Candidates:\n"+"\n".join(f"{mapping[lab]} = {lab}" for lab in ordered))
    blocks=[]
    if arm!="ZERO":
        # All demonstrations are official TRAIN groups, fixed in advance and
        # heldout/test rows are never chosen as exemplars.
        for idx,pid in enumerate(sorted(demos,key=_order_key)):
            data=demos[pid]
            lines=[]
            for k,source_label in enumerate(LABELS):
                shown=source_label if arm=="PAIR" else LABELS[(k+idx+1)%3]
                lines.append(shown+": "+data[source_label])
            blocks.append("Comparison group "+str(idx+1)
                          +" (the same unseen writing request, independent model outputs):\n"
                          +"\n".join(lines))
    examples="\n\nReference examples:\n"+"\n\n".join(blocks) if blocks else ""
    query="\n\nUnknown RESPONSE:\n"+row["text"]+"\n\nWhich candidate generated it?\nAnswer:"
    return header+examples+query,mapping

def model_dir():
    root=Path("/kaggle/input")
    found=[]
    # Prefer exact pinned mount path, and verify config signature at runtime.
    for path in root.rglob("config.json"):
        txt=str(path.parent).lower().replace("\\","/")
        if "qwen" not in txt or "14b" not in txt:continue
        try:
            c=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if c.get("model_type")=="qwen3" and c.get("hidden_size",0)>=5000:
            found.append(path.parent)
    if len(found)!=1:
        raise FileNotFoundError("exact Kaggle Qwen3 14B model mount not uniquely found")
    return found[0]

def load_qwen14b():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("T4 GPU unavailable for large-model inference")
    path=model_dir()
    tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,
        trust_remote_code=False)
    token_ids={}
    for letter in "ABC":
        ids=tokenizer.encode(letter,add_special_tokens=False)
        if len(ids)!=1:raise RuntimeError("A/B/C choice token must each encode as one exact token")
        token_ids[letter]=ids[0]
    if len(set(token_ids.values()))!=3:raise RuntimeError("choice token IDs collide")
    quant=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(
        str(path),local_files_only=True,trust_remote_code=False,
        quantization_config=quant,torch_dtype=torch.float16,
        device_map="auto",low_cpu_mem_usage=True).eval()
    print("NTL_QWEN14B_REAL_MODEL_LOADED="+json.dumps({
        "model":KAGGLE_MODEL_REF,
        "inference_only":True,"nf4":True,
        "cuda_devices":torch.cuda.device_count(),
        "model_type":model.config.model_type,
        "decoder_layers":model.config.num_hidden_layers,
        "choice_tokens_distinct":True
    },sort_keys=True),flush=True)
    return model,tokenizer,token_ids

def evidence(model,tokenizer,tokens,prompt):
    import torch
    messages=[{"role":"system","content":"You are a forensic classifier. Output a single letter A, B, or C."},
              {"role":"user","content":prompt}]
    try:
        pre=tokenizer.apply_chat_template(messages,tokenize=False,
                 add_generation_prompt=True,enable_thinking=False)
    except (TypeError,ValueError) as e:
        raise RuntimeError("exact Qwen3 no-thinking chat template unsupported") from e
    text=pre+"Answer: "
    ids=tokenizer(text,return_tensors="pt",add_special_tokens=False)
    size=int(ids["input_ids"].shape[1])
    if size>MAX_INPUT_TOKENS:raise RuntimeError("fixed ICL context exceeded 3072, no silent truncation")
    device=next(model.parameters()).device
    ids={k:v.to(device) for k,v in ids.items()}
    with torch.inference_mode():
        out=model(**ids,use_cache=False,return_dict=True).logits[0,-1]
        logits=torch.stack([out[tokens[x]] for x in "ABC"]).float()
        prob=torch.softmax(logits,dim=0).detach().cpu().numpy().astype(float)
    return prob,size

def balanced_acc(y,p):
    if any(np.sum(y==lab)==0 for lab in LABELS):raise ValueError("one class missing")
    return float(np.mean([np.mean(p[y==lab]==lab) for lab in LABELS]))

def evaluate(rows,record):
    rng=np.random.default_rng(SEED)
    out={}
    for domain in ("official_heldout","Beroli_C1"):
        grp=[r for r in rows if r["domain"]==domain]
        groups=sorted({r["group"] for r in grp})
        if len(groups)!=(24 if domain=="official_heldout" else 30):
            raise ValueError("group count drift")
        y=np.asarray([r["truth"] for r in grp])
        predictions={arm:np.asarray([record[r["id"]][arm] for r in grp]) for arm in ARMS}
        arm_scores={arm:balanced_acc(y,predictions[arm]) for arm in ARMS}
        comparisons={}
        for c in ("ZERO","PLACEBO"):
            base=predictions[c];post=predictions["PAIR"]
            gains=(base!=y)&(post==y)
            harm=(base==y)&(post!=y)
            delta=arm_scores["PAIR"]-arm_scores[c]
            boot=np.zeros(2000)
            rowgrp=np.asarray([r["group"] for r in grp])
            for k in range(2000):
                chosen=rng.choice(groups,size=len(groups),replace=True)
                picks=np.concatenate([np.flatnonzero(rowgrp==group) for group in chosen])
                boot[k]=balanced_acc(y[picks],post[picks])-balanced_acc(y[picks],base[picks])
            comparisons["PAIR_minus_"+c]={
                "BA_delta":round(float(delta),6),"rescues":int(gains.sum()),
                "regressions":int(harm.sum()),
                "changed":int((base!=post).sum()),
                "prompt_bootstrap_CI95":np.quantile(boot,[.025,.975]).round(6).tolist()
            }
        out[domain]={
            "rows":len(grp),"prompt_groups":len(groups),
            "BA3":{k:round(float(v),6) for k,v in arm_scores.items()},
            "comparisons":comparisons,
            "per_label":{lab:{arm:round(float(np.mean(predictions[arm][y==lab]==lab)),6)
                for arm in ARMS} for lab in LABELS}
        }
    return out

def run(train,c1):
    start=time.monotonic()
    rows,demos,source_check=prepare(train,c1)
    print("NTL_QWEN14B_PREPARED="+json.dumps(source_check,sort_keys=True),flush=True)
    model,tokenizer,toks=load_qwen14b()
    results={}
    times=[]
    tokens_seen=[]
    for i,row in enumerate(rows):
        probs={}
        predictions={}
        for arm in ARMS:
            prompt,mapletters=make_prompt(row,demos,arm)
            now=time.monotonic()
            prob,tcount=evidence(model,tokenizer,toks,prompt)
            times.append(time.monotonic()-now)
            tokens_seen.append(tcount)
            pletter="ABC"[int(np.argmax(prob))]
            inv={k:v for v,k in mapletters.items()}
            predictions[arm]=inv[pletter]
            probs[arm]=float(np.max(prob))
        results[row["id"]]=predictions
        if (i+1)%12==0 or i+1==len(rows):
            print("NTL_QWEN14B_PROGRESS="+json.dumps({
                "completed":i+1,"total":len(rows),
                "median_forward_seconds":round(float(np.median(times)),3),
                "maximum_input_tokens":int(max(tokens_seen)),
                "accuracy_not_evaluated_until_all_arms_complete":True
            },sort_keys=True),flush=True)
    report={
      "scope":"pretrained_Qwen3_14B_prompt_free_GPT_triad_external_diagnostic",
      "owner_best_public_BA_reference":0.718,
      "official_competition_submission":False,
      "labelled_total":len(rows),"source_guards":source_check,
      "source_is_publisher_claim_only":True,"no_new_model_training":True,
      "no_small_models":True,"pretrained_large_model_ref":KAGGLE_MODEL_REF,
      "same_row_comparisons":evaluate(rows,results),
      "total_forward_calls":len(times),
      "median_forward_seconds":round(float(np.median(times)),3),
      "p95_forward_seconds":round(float(np.quantile(times,.95)),3),
      "largest_input_tokens":max(tokens_seen),
      "naive_full_4000_rows_single_arm_hours":
        round(float(np.median(times))*4000/3600,2),
      "note":"3-class oracle-support experiment, not full18 accuracy or source-ground-truth authentication",
      "elapsed_seconds":round(float(time.monotonic()-start),1)
    }
    Path("/kaggle/working/"+RESULT_FILE).write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("NTL_QWEN14B_REAL_AGGREGATE="+json.dumps(report,sort_keys=True),flush=True)

def selftest():
    demos={"p1":{lab:"One two three." for lab in LABELS},
           "p2":{lab:"Four five six." for lab in LABELS},
           "p3":{lab:"Seven eight nine." for lab in LABELS}}
    row={"id":"abc","text":"Simple one paragraph","group":"q","truth":LABELS[0]}
    results={arm:make_prompt(row,demos,arm) for arm in ARMS}
    assert all(set(mapping)==set(LABELS) and set(mapping.values())==set("ABC")
       for _,mapping in results.values())
    assert "Reference examples" not in results["ZERO"][0]
    assert "Reference examples" in results["PAIR"][0]
    assert results["PAIR"][0]!=results["PLACEBO"][0]
    assert all("Simple one paragraph" in prompt for prompt,_ in results.values())
    assert balanced_acc(np.array(LABELS),np.array(LABELS))==1.0
    print("NTL_QWEN14B_ICL_SYNTHETIC_CPU_SELFTEST=PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--official",type=Path)
    p.add_argument("--c1",type=Path)
    p.add_argument("--selftest",action="store_true")
    args=p.parse_args()
    selftest()
    if not args.selftest:
        if not args.official or not args.c1:p.error("two pinned source inputs required")
        run(args.official,args.c1)
