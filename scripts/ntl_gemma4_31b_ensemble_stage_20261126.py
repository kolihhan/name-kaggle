"""Gemma 4 31B GGUF inference-only ensemble expert: real offline Kaggle rows.
No fitting, no gradients, no paid API; fail closed if 31B cannot execute.
Only modify the strongest Ensemble's low-margin top-two rows.
"""
import hashlib,json,os,re,subprocess,time,urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

LABELS=set(("claude-fable-5","claude-opus-4-6","claude-opus-4-8","claude-opus-5","claude-sonnet-5",
"gemini-3.1-pro-preview","gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash",
"glm-5.3","glm-5.3-flash","gpt-5.5","gpt-5.6-luna","gpt-5.6-sol",
"gpt-5.6-terra","gpt-6-astra","kimi-k3","qwen3.8-max"))
OUT=Path("/kaggle/working/submission.parquet")
PROOF=Path("/kaggle/working/ntl31_proof.json")
PORT=19031
MAX_CHANGED_CANDIDATES=600
STAGE_LIMIT_S=1700

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for x in iter(lambda:f.read(1048576),b""):h.update(x)
    return h.hexdigest()

def locate(substring,filename):
    matches=[p for p in Path("/kaggle/input").rglob(filename)
             if substring in str(p).lower() and p.is_file()]
    matches.sort(key=lambda p:(len(str(p)),str(p)))
    if not matches:raise RuntimeError("missing required offline asset "+substring+"/"+filename)
    if len(matches)>2:raise RuntimeError("ambiguous 31B model asset")
    return matches[0]

def examples(train_csv):
    df=pd.read_csv(train_csv,usecols=["prompt_id","model","text"])
    if len(df)!=2699 or set(df.model)!=LABELS:raise RuntimeError("not the official 18-label TRAIN")
    ct=df.groupby("prompt_id").model.nunique()
    good=sorted(map(str,ct[ct==18].index))
    if not good:raise RuntimeError("no full 18-label common prompt")
    sample=df[df.prompt_id.astype(str)==good[0]].sort_values("model")
    return "\n\n".join("Generator: "+str(r.model)+"\nWriting: "+" ".join(str(r.text).split()[:42])
                      for r in sample.itertuples(index=False))

def load_votes(ids):
    path="/kaggle/temp/final-logits.npz"
    with np.load(path,allow_pickle=False) as z:
        arrays=[np.asarray(z[k]) for k in z.files]
        a=[x for x in arrays if x.ndim==2 and x.shape==(len(ids),18)]
        b=[x for x in arrays if x.ndim==1 and len(x)==18 and set(map(str,x))==LABELS]
        c=[x for x in arrays if x.ndim==1 and len(x)==len(ids) and list(map(str,x))==ids]
        if len(a)!=1 or len(b)!=1 or not c:
            raise RuntimeError("true final logit alignment unavailable; keys="+repr(z.files))
        return a[0].astype(np.float64),list(map(str,b[0]))

def api_request(body,timeout=125):
    req=urllib.request.Request("http://127.0.0.1:"+str(PORT)+"/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"},
        method="POST")
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.loads(response.read())

def parse_label(v):
    lines=[x.strip().strip(" '\"*").lower() for x in str(v).splitlines() if x.strip()]
    for line in lines[:2]:
        if line.startswith("label:"):line=line[6:].strip()
        if line in LABELS:return line
    return None

def main(test_csv,train_csv,nb_start,preview):
    t0=time.monotonic()
    test=pd.read_csv(test_csv,dtype={"id":str})
    pred=pd.read_parquet(OUT)
    assert list(pred.columns)==["id","model"]
    ids=test.id.astype(str).tolist()
    assert ids==pred.id.astype(str).tolist() and pred.id.is_unique
    assert set(pred.model.astype(str)).issubset(LABELS)
    if "text" not in test:raise RuntimeError("official test text field missing")
    baseline_sha=sha(OUT)
    gguf=locate("gemma-4-31b-it-q4-k-m","gemma-4-31B-it-Q4_K_M.gguf")
    assert gguf.stat().st_size>17_000_000_000
    server_bin=locate("llama-cpp-turing-sm75","llama-server")
    server_bin.chmod(server_bin.stat().st_mode|0o111)
    intro=("Forensic authorship attribution. These are genuine, labelled writing samples "
           "generated for the same prompt by eighteen different LLMs. Compare lexical "
           "and stylistic signals, not provider reputation. Classify a new passage "
           "ONLY between the two named candidates. Return exactly one candidate "
           "label and nothing else.\n\n"+examples(train_csv))
    if preview:
        picks=list(range(min(3,len(ids))))
        other={i:sorted(LABELS-{str(pred.model.iloc[i])})[0] for i in picks}
    else:
        if time.monotonic()-nb_start>8.2*3600:
            raise RuntimeError("insufficient budget for real 31B")
        votes,names=load_votes(ids)
        votes=np.where(np.isfinite(votes),votes,-1e9)
        order=np.argsort(-votes,axis=1)
        names=np.asarray(names)
        options=[]
        other={}
        for i in range(len(ids)):
            pair=names[order[i,:2]].tolist()
            current=str(pred.model.iloc[i])
            if current not in pair:continue
            other[i]=pair[1] if current==pair[0] else pair[0]
            margin=float(votes[i,order[i,0]]-votes[i,order[i,1]])
            if np.isfinite(margin):options.append((margin,ids[i],i))
        picks=[i for _,_,i in sorted(options)[:min(MAX_CHANGED_CANDIDATES,len(options))]]
        if len(picks)<25:raise RuntimeError("not enough real high-uncertainty eligible rows")
    command=[str(server_bin),"-m",str(gguf),"--host","127.0.0.1","--port",str(PORT),
             "-c","4096","-ngl","99","--split-mode","layer",
             "--tensor-split","1,1","-b","256","-ub","128",
             "-t","4","--parallel","1"]
    print("NTL31_START="+json.dumps({"preview":preview,"selected":len(picks),
         "gguf_bytes":gguf.stat().st_size,"baseline_sha":baseline_sha}),flush=True)
    log=open("/kaggle/working/ntl31_llama_server.log","w")
    server=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    calls,valid,changed=0,0,0
    output=pred.model.astype(str).tolist()
    try:
        ready=False
        for _ in range(165):
            if server.poll() is not None:raise RuntimeError("31B server exited during model load")
            try:
                with urllib.request.urlopen("http://127.0.0.1:"+str(PORT)+"/health",timeout=2) as response:
                    if response.status==200:ready=True;break
            except Exception:pass
            time.sleep(2)
        if not ready:raise RuntimeError("Gemma31B never became healthy")
        print("NTL31_REAL_MODEL_LOADED=TRUE",flush=True)
        for i in picks:
            if not preview and time.monotonic()-t0>STAGE_LIMIT_S:break
            if not preview and time.monotonic()-nb_start>8.72*3600:break
            current=str(pred.model.iloc[i]);candidate=str(other[i])
            sample=" ".join(str(test.text.iloc[i]).split()[:200])
            task=("Candidate A: "+current+"\nCandidate B: "+candidate+
                  "\nUnknown passage:\n"+sample+"\nReturn the exact candidate label:")
            body={"model":"local-gemma-31b","messages":[
                 {"role":"system","content":intro},{"role":"user","content":task}],
                 "max_tokens":28,"temperature":0,"stream":False}
            result=api_request(body)
            guess=parse_label(result["choices"][0]["message"].get("content") or "")
            calls+=1
            if guess in (current,candidate):
                valid+=1
                if guess!=current and not preview:
                    output[i]=guess;changed+=1
            if calls%25==0:
                print("NTL31_PROGRESS="+json.dumps({"calls":calls,"changed":changed,
                       "elapsed_s":int(time.monotonic()-t0)}),flush=True)
    finally:
        server.terminate()
        try:server.wait(timeout=20)
        except subprocess.TimeoutExpired:server.kill()
        log.close()
    if calls<1 or valid<1:raise RuntimeError("no real valid 31B predictions")
    if not preview and changed==0:raise RuntimeError("Gemma31B expert made zero hidden changes")
    result=pd.DataFrame({"id":ids,"model":output})
    assert list(result.columns)==["id","model"] and set(result.model).issubset(LABELS)
    temp=Path("/kaggle/working/ntl31_atomic.parquet")
    result.to_parquet(temp,index=False)
    assert len(pd.read_parquet(temp))==len(test)
    os.replace(temp,OUT)
    evidence={"model":"Gemma 4 31B Q4_K_M","no_training":True,
      "loaded_and_inferred":True,"preview":bool(preview),
      "n_test":len(test),"inference_calls":calls,"valid_calls":valid,
      "changed":changed,"baseline_sha256":baseline_sha,
      "final_sha256":sha(OUT),"gguf_bytes":gguf.stat().st_size,
      "time_s":int(time.monotonic()-t0),"early_stop":calls<len(picks)}
    PROOF.write_text(json.dumps(evidence,sort_keys=True))
    print("NTL31_PROOF="+json.dumps(evidence,sort_keys=True),flush=True)

if __name__=="__main__":
    import sys
    main(sys.argv[1],sys.argv[2],float(sys.argv[3]),sys.argv[4]=="1")
