"""Private GPU CGF pilot: Gemma 4 31B Q4 next-token distribution.
No hidden data, training, or competition submission. Aggregate results only.
"""
import hashlib,json,os,shutil,subprocess,time,urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from cgf import contrast,predict,permute_within_prompt,sha_ok

TRAIN="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
GROUPS=[("but","yet","however"),("although","though"),("because","since"),
        ("also","additionally","furthermore"),("therefore","thus","hence"),
        ("often","frequently"),("important","significant","crucial"),
        ("can","could","may"),("while","whereas")]
PORT=19031; SEED=20261127; LIMIT=1800; N=8
def call(path,body=None):
    req=urllib.request.Request("http://127.0.0.1:"+str(PORT)+path,
      data=None if body is None else json.dumps(body).encode(),
      headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=90 if body else 2) as f:return json.load(f)
def tokens(text,add_special=False):
    result=call("/tokenize",{"content":text,"add_special":add_special})
    return result["tokens"]
def parse_distribution(result):
    items=result["completion_probabilities"]
    if len(items)!=1:raise ValueError("expected single generated position")
    item=items[0]; out={}
    for p in item["top_logprobs"]:
        t=int(p["id"]); val=float(p["logprob"])
        if t in out or not np.isfinite(val) or val>1e-5:raise ValueError("invalid probability")
        out[t]=val
    if not out or sum(np.exp(list(out.values())))>1.001:raise ValueError("invalid normalized mass")
    return out
def locate(sub,name):
    found=sorted(p for p in Path("/kaggle/input").rglob(name) if sub in str(p).lower())
    assert len(found)==1,(sub,name,len(found))
    return found[0]
def select(d,keys):
    ids=d.prompt_id.astype(str).map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    d=d[ids.isin(keys)]
    counts=d.groupby("prompt_id").model.agg(["size","nunique"])
    complete=sorted(counts[(counts["size"]==18)&(counts["nunique"]==18)].index)
    chosen=np.random.default_rng(SEED).choice(complete,N,replace=False)
    return d[d.prompt_id.isin(chosen)].sort_values(["prompt_id","model"]).reset_index(drop=True)
def main():
    start=time.monotonic()
    train=locate("name-that-llm","train.csv")
    assert sha_ok(train.read_bytes(),TRAIN)
    d=pd.read_csv(train,dtype=str)
    assert len(d)==2699 and d.model.nunique()==18 and d.prompt_id.nunique()==150
    lock=json.loads(Path("fold-lock.json").read_text())
    assert len(lock)==150 and set(lock.values())=={"0","1","2","3","4"}
    r=select(d,{k for k,v in lock.items() if v!="0"})
    q=select(d,{k for k,v in lock.items() if v=="0"})
    assert len(r)==len(q)==N*18 and not set(r.prompt_id)&set(q.prompt_id)
    assert not set(r.text)&set(q.text)
    rows=pd.concat([r,q],ignore_index=True)
    gguf=locate("gemma-4-31b-it-q4-k-m","gemma-4-31B-it-Q4_K_M.gguf")
    assert gguf.stat().st_size==18323726560
    assert subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).count("T4")>=2
    binary=locate("llama-cpp-bin-cuda13-sm75","llama-server")
    runtime_libdir=str(binary.parent)
    target=Path("/tmp/cgf_llama_server");shutil.copy2(binary,target);target.chmod(0o755)
    env=os.environ.copy();env["LD_LIBRARY_PATH"]=runtime_libdir+":"+env.get("LD_LIBRARY_PATH","")
    logfile=Path("/tmp/cgf_runtime.log")
    with logfile.open("w") as log:
        server=subprocess.Popen([str(target),"-m",str(gguf),"--host","127.0.0.1","--port",str(PORT),
           "-c","4096","-ngl","99","--split-mode","layer","--tensor-split","1,1","-b","256","-ub","128","-t","4","--parallel","1"],
           stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            for _ in range(165):
                if server.poll() is not None:raise RuntimeError("GPU model server exited")
                try:
                    if call("/health").get("status")=="ok":break
                except Exception:pass
                time.sleep(2)
            else:raise RuntimeError("GPU model did not load")
            print("CGF_REAL_31B_LOADED",flush=True)
            # Only alternatives encoded as single exact tokens with a leading space.
            groups=[]
            for words in GROUPS:
                ids=[tokens(" "+w) for w in words]
                ids=[a[0] for a in ids if len(a)==1]
                if len(ids)>=2 and len(set(ids))==len(ids):groups.append(ids)
            lookup={t:(g,j) for g,ids in enumerate(groups) for j,t in enumerate(ids)}
            assert len(groups)>=5 and len(lookup)==sum(map(len,groups))
            offsets=np.cumsum([0]+[len(a) for a in groups]); co=np.cumsum([0]+[len(a)+1 for a in groups])
            n=len(rows); a=np.zeros((n,offsets[-1])); b=np.zeros((n,len(groups)))
            c=np.zeros((n,co[-1])); count=np.zeros((n,len(groups))); attempted=valid=0
            def nextlog(prefix):
                return parse_distribution(call("/completion",{"prompt":prefix,"n_predict":1,
                    "temperature":-1,"n_probs":2048,"post_sampling_probs":False,
                    "cache_prompt":True,"stream":False}))
            probe=nextlog(tokens("The results suggest that",True))
            assert len(probe)>=100 and all(np.isfinite(list(probe.values())))
            print("CGF_REAL_NEXT_TOKEN_LOGPROBS="+json.dumps({"returned":len(probe),"mass":float(sum(np.exp(list(probe.values()))))}),flush=True)
            for i,text in enumerate(rows.text):
                if time.monotonic()-start>LIMIT:raise RuntimeError("fixed GPU pilot exceeded wall budget")
                ids=tokens(text,True)
                hit=[(pos,*lookup[t]) for pos,t in enumerate(ids[:256]) if t in lookup and pos>=8]
                for _,g,j in hit:a[i,offsets[g]+j]+=1/max(len(ids),1)
                if len(hit)>2:hit=[hit[k] for k in np.linspace(0,len(hit)-1,2,dtype=int)]
                for pos,g,j in hit:
                    attempted+=1
                    dist=nextlog(ids[:pos])
                    if not all(t in dist for t in groups[g]):continue
                    x=np.array([dist[t] for t in groups[g]])
                    b[i,g]+=x[j];c[i,co[g]:co[g+1]]+=contrast(x,j);count[i,g]+=1;valid+=1
                if i%36==0:print("CGF_PROGRESS="+json.dumps({"rows":i+1,"attempted":attempted,"valid":valid,"seconds":round(time.monotonic()-start,1)}),flush=True)
            for g in range(len(groups)):
                b[:,g]/=np.maximum(count[:,g],1)
                c[:,co[g]:co[g+1]]/=np.maximum(count[:,g,None],1)
            m=len(r); y=q.model.to_numpy();labels=r.model.to_numpy()
            a=np.c_[a,count]
            def classify(blocks):return predict([x[:m] for x in blocks],[x[m:] for x in blocks],labels)
            sh=np.concatenate([permute_within_prompt(c[:m],r.prompt_id.to_numpy(),SEED),
                  permute_within_prompt(c[m:],q.prompt_id.to_numpy(),SEED+1)])
            preds={"A_surface":classify([a]),"B_observed":classify([a,b]),
                   "C_counterfactual":classify([a,b,c]),"D_placebo":classify([a,b,sh])}
            def metrics(p):
                recall={k:float((p[y==k]==k).mean()) for k in sorted(set(y))}
                return {"BA":float(np.mean(list(recall.values()))),"correct":int((p==y).sum()),"recall":recall}
            result={k:metrics(p) for k,p in preds.items()}
            base=preds["B_observed"];new=preds["C_counterfactual"]
            covered=int((count[m:].sum(1)>0).sum())
            result.update(model="Gemma 4 31B IT Q4_K_M",gpu="T4x2",n_ref=m,n_query=len(q),probes=attempted,valid_probes=valid,covered_test=covered,
                rescues=int(((base!=y)&(new==y)).sum()),regressions=int(((base==y)&(new!=y)).sum()),
                gate=bool(covered>=0.6*len(q) and result["C_counterfactual"]["BA"]-max(result["B_observed"]["BA"],result["D_placebo"]["BA"])>=.03),
                seconds=round(time.monotonic()-start,2),training=False,submission=False,scope="spent official fold0 subsample")
            Path("/kaggle/working/cgf_result.json").write_text(json.dumps(result,sort_keys=True))
            print("CGF_GPU_RESULT="+json.dumps(result,sort_keys=True),flush=True)
        finally:
            server.terminate()
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:server.kill();server.wait()
    logfile.unlink(missing_ok=True)
if __name__=="__main__":main()
