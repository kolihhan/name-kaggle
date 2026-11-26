"""CGF-v0 real CPU pilot. See frozen contract."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from cgf import sha_ok, contrast, predict, permute_within_prompt

TRAIN="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLDS="94136d7dcb4689b0a8a21b742c20e886b865db67f5643c3402e6c1727b27f66d"
MODEL="FacebookAI/roberta-large"
REV="b91e680b6f95ab812c97ef1d3af8e59d128d84b7"
GROUPS=[("but","yet","however"),("although","though"),("because","since"),
        ("also","additionally","furthermore"),("therefore","thus","hence"),
        ("often","frequently"),("important","significant","crucial"),
        ("can","could","may"),("while","whereas")]
SEED=20261127

def select(d):
    counts=d.groupby("prompt_id").model.agg(["size","nunique"])
    complete=sorted(counts[(counts["size"]==18)&(counts["nunique"]==18)].index)
    chosen=np.random.default_rng(SEED).choice(complete,12,replace=False)
    return d[d.prompt_id.isin(chosen)].sort_values(["prompt_id","model"]).reset_index(drop=True)

def main(train_path,fold_path):
    import torch
    from transformers import AutoTokenizer,AutoModelForMaskedLM
    start=time.monotonic()
    assert sha_ok(Path(train_path).read_bytes(),TRAIN)
    assert sha_ok(Path(fold_path).read_bytes(),FOLDS)
    d=pd.read_csv(train_path,dtype=str)
    f=pd.read_csv(fold_path,dtype=str)
    d=d.merge(f[["id","prompt_id","model","fold_id"]],on=["id","prompt_id","model"],validate="one_to_one")
    assert len(d)==2699 and d.model.nunique()==18 and d.prompt_id.nunique()==150
    assert d.groupby("prompt_id").fold_id.nunique().max()==1
    r=select(d[d.fold_id!="0"]);q=select(d[d.fold_id=="0"])
    assert len(r)==len(q)==216 and not set(r.prompt_id)&set(q.prompt_id)
    assert not set(r.text)&set(q.text)
    rows=pd.concat([r,q],ignore_index=True)
    torch.set_num_threads(4)
    assert not torch.cuda.is_available()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
    model=AutoModelForMaskedLM.from_pretrained(MODEL,revision=REV,use_safetensors=True,trust_remote_code=False).eval()
    groups=[]
    for group in GROUPS:
        ids=[tok.encode(" "+w,add_special_tokens=False) for w in group]
        ids=[i[0] for i in ids if len(i)==1]
        if len(ids)>=2: groups.append(ids)
    assert len(groups)>=5
    lookup={t:(g,j) for g,ids in enumerate(groups) for j,t in enumerate(ids)}
    assert len(lookup)==sum(map(len,groups))
    offsets=np.cumsum([0]+[len(x) for x in groups])
    co=np.cumsum([0]+[len(x)+1 for x in groups])
    n=len(rows)
    surface=np.zeros((n,offsets[-1])); ordinary=np.zeros((n,len(groups)))
    response=np.zeros((n,co[-1])); counts=np.zeros((n,len(groups)))
    jobs=[]
    for i,text in enumerate(rows.text):
        ids=tok.encode(text,add_special_tokens=False)
        hit=[(pos,*lookup[t]) for pos,t in enumerate(ids) if t in lookup]
        for _,g,j in hit: surface[i,offsets[g]+j]+=1/max(1,len(ids))
        # deterministic spread across passage; cap 6 probes per text
        if len(hit)>6: hit=[hit[k] for k in np.linspace(0,len(hit)-1,6,dtype=int)]
        for pos,g,j in hit:
            lo=max(0,pos-46); hi=min(len(ids),pos+47)
            seq=[tok.bos_token_id]+ids[lo:hi]+[tok.eos_token_id]
            target=pos-lo+1; seq[target]=tok.mask_token_id
            jobs.append((i,g,j,target,seq))
    print("CGF_COVERAGE="+json.dumps({"rows":n,"probes":len(jobs),"covered_rows":len(set(x[0] for x in jobs)),"groups":len(groups)}),flush=True)
    for k in range(0,len(jobs),8):
        batch=jobs[k:k+8]
        maxlen=max(len(z[4]) for z in batch)
        inputs=torch.full((len(batch),maxlen),tok.pad_token_id,dtype=torch.long)
        mask=torch.zeros_like(inputs)
        for b,(_,_,_,_,seq) in enumerate(batch):
            inputs[b,:len(seq)]=torch.tensor(seq);mask[b,:len(seq)]=1
        with torch.inference_mode():
            hidden=model.roberta(input_ids=inputs,attention_mask=mask).last_hidden_state
            selected=hidden[torch.arange(len(batch)),torch.tensor([z[3] for z in batch])]
            logp=model.lm_head(selected).float().log_softmax(-1).cpu().numpy()
        for b,(i,g,j,_,_) in enumerate(batch):
            values=logp[b,groups[g]]
            ordinary[i,g]+=values[j]; response[i,co[g]:co[g+1]]+=contrast(values,j);counts[i,g]+=1
        if k%256==0: print("CGF_PROGRESS",k,len(jobs),flush=True)
    for g in range(len(groups)):
        ordinary[:,g]/=np.maximum(counts[:,g],1)
        response[:,co[g]:co[g+1]]/=np.maximum(counts[:,g,None],1)
    assert all(np.isfinite(x).all() for x in (surface,ordinary,response))
    # retain matched coverage indicators in A so that missingness is not a new CGF feature
    surface=np.c_[surface,counts]
    m=len(r); y=q.model.to_numpy();labels=r.model.to_numpy()
    def run(blocks):
        return predict([x[:m] for x in blocks],[x[m:] for x in blocks],labels)
    shuffled=np.concatenate([permute_within_prompt(response[:m],r.prompt_id.to_numpy(),SEED),
                  permute_within_prompt(response[m:],q.prompt_id.to_numpy(),SEED+1)])
    p={"A_surface":run([surface]),"B_observed":run([surface,ordinary]),
       "C_counterfactual":run([surface,ordinary,response]),"D_placebo":run([surface,ordinary,shuffled])}
    def metrics(pred):
        recall={c:float((pred[y==c]==c).mean()) for c in sorted(set(y))}
        return {"BA":float(np.mean(list(recall.values()))),"correct":int((pred==y).sum()),"recall":recall}
    result={name:metrics(pred) for name,pred in p.items()}
    b=p["B_observed"];c=p["C_counterfactual"]
    result.update({"rescues":int(((b!=y)&(c==y)).sum()),"regressions":int(((b==y)&(c!=y)).sum()),
        "gate":result["C_counterfactual"]["BA"]-max(result["B_observed"]["BA"],result["D_placebo"]["BA"])>=.03,
        "seconds":round(time.monotonic()-start,2),"probes":len(jobs),"covered_test":int((counts[m:].sum(1)>0).sum()),"n_test":len(q),"gpu":False,"submission":False})
    print("NTL_CGF_RESULT="+json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    main(*sys.argv[1:])
