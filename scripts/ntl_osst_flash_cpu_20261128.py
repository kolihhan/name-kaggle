"""OSST Flash 3-way diagnostic on prompt-disjoint spent official rows.
Real Qwen neutralization + exact target-token likelihood, no fitting or GPU.
No current v36 scores available: not a candidate promotion test.
"""
import hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.ntl_rcma_zero_training_20261127 import get,tok,detok,clip,load_train,MODEL_SHA,PORT,SEED

def clean(text):
    return str(text).replace("<|","‹|").replace("|>","|›").replace("<think>","‹think›").replace("</think>","‹/think›")
def chat(system,user):
    return "<|im_start|>system\n"+system+"<|im_end|>\n<|im_start|>user\n"+user+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
def neutralize(s):
    p=chat("Rewrite the text in neutral, plain prose. Preserve facts, meaning and language. Remove distinctive stylistic choices. Return only the rewritten text.","Text:\n"+s)
    r=get("/completion",{"prompt":tok(p,True),"n_predict":128,"temperature":0,"stream":False,"cache_prompt":True,"stop":["<|im_end|>","<|im_start|>"]})
    x=clean(r.get("content","").strip())
    assert x and not r.get("truncated",False),"empty/truncated neutralization"
    assert len(tok(x))<=128
    return x,bool(x==s),bool(r.get("stopped_limit",False))
def summarize(z,y):
    # Shape (panels, queries, 3). Evaluate actual references vs permuted authors.
    assert z.shape==(2,len(y),3) and np.isfinite(z).all()
    a=z.mean(0)
    pred=a.argmax(1)
    def ba(p):return float(np.mean([np.mean(p[y==k]==k) for k in range(3)]))
    rng=np.random.default_rng(SEED+18)
    null=[]
    for _ in range(200):
        t=np.mean([z[i][:,rng.permutation(3)] for i in range(2)],axis=0)
        null.append(ba(t.argmax(1)))
    actual=ba(pred)
    # All 3 labels available; neutralized query is identical for each reference.
    margin=a[np.arange(len(y)),y]-np.max(np.where(np.eye(3)[y].astype(bool),-np.inf,a),axis=1)
    return {"BA":actual,"correct":int((pred==y).sum()),"queries":len(y),
        "per_label_correct":[int(((pred==y)&(y==k)).sum()) for k in range(3)],
        "panel_BA":[ba(x.argmax(1)) for x in z],"null_mean":float(np.mean(null)),
        "null_p95":float(np.quantile(null,.95)),"mean_true_margin":float(margin.mean()),
        "gate":bool(actual>=.5 and actual>np.quantile(null,.95) and all(ba(x.argmax(1))>1/3 for x in z))}
def selftest():
    y=np.tile(np.arange(3),4);z=np.stack([np.eye(3)[y],np.eye(3)[y]])
    assert summarize(z,y)["BA"]==1 and summarize(z,y)["gate"]
    assert not summarize(np.zeros_like(z),y)["gate"]
    print("OSST_SCORE_FIXTURES=PASS",flush=True)
def load_more(root):
    refs,qs=load_train(root)
    # Reuse exact verified data and fold axis; extend to four complete query prompts by same hash ordering.
    import io,zipfile
    path=root/"train.csv"
    if path.exists():data=path.read_bytes()
    else:
        with zipfile.ZipFile(next(root.glob("*.zip"))) as z:data=z.read("train.csv")
    d=pd.read_csv(io.BytesIO(data),dtype=str)
    folds=pd.read_csv("artifacts/phase1/folds.csv",dtype=str)
    d=d.merge(folds[["id","model","prompt_id","fold_id"]],on=["id","model","prompt_id"],validate="one_to_one")
    assert len(d)==2699
    complete=d.groupby("prompt_id").model.agg(["size","nunique"])
    valid=set(complete[(complete["size"]==18)&(complete["nunique"]==18)].index)
    order=sorted(set(d[d.fold_id=="0"].prompt_id)&valid,key=lambda p:hashlib.sha256((str(SEED)+str(p)).encode()).hexdigest())[:4]
    assert len(order)==4
    labs=["gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash"]
    cut=lambda f:f.set_index("model").loc[labs].reset_index()
    r=[cut(f) for f in refs]
    q=[cut(d[d.prompt_id==i]) for i in order]
    assert not set(pd.concat(r).text)&set(pd.concat(q).text)
    return r,q,labs
def main():
    selftest();start=time.monotonic()
    model=Path(os.environ["RCMA_MODEL"])
    with model.open("rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==MODEL_SHA
    with tempfile.TemporaryDirectory(prefix="rcma14b-") as tmp:
        root=Path(tmp);r,q,labs=load_more(root)
        with (root/"server.log").open("w") as out:
            p=subprocess.Popen([os.environ["RCMA_SERVER"],"-m",str(model),"--host","127.0.0.1","--port",str(PORT),
                "-c","2048","-ngl","0","-t","4","-b","256","-ub","128","--parallel","1"],stdout=out,stderr=subprocess.STDOUT)
            try:
                for _ in range(150):
                    if p.poll() is not None:raise RuntimeError("CPU_SERVER_EXITED")
                    try:
                        if get("/health").get("status")=="ok":break
                    except Exception:pass
                    time.sleep(2)
                else:raise RuntimeError("health timeout")
                refs=[[clean(clip(t)) for t in f.text] for f in r]
                query=[clean(clip(t)) for f in q for t in f.text]
                originals=[t for panel in refs for t in panel]+query
                neutral=[];copies=limits=0
                for i,s in enumerate(originals):
                    n,c,l=neutralize(s);neutral.append(n);copies+=c;limits+=l
                    if (i+1)%3==0:print("OSST_NEUTRAL_PROGRESS="+str(i+1),flush=True)
                assert len(neutral)==18 and limits==0,"neutralization incomplete"
                jobs=[];expected=[]
                # First exact-logit crosscheck against the already verified server's full-vocab probabilities.
                check=tok("The capital of France is",True);targ=tok(" Paris")
                assert len(targ)==1
                pr=get("/completion",{"prompt":check,"n_predict":1,"temperature":-1,"n_probs":2048,"post_sampling_probs":False})
                dist={int(x["id"]):float(x["logprob"]) for x in pr["completion_probabilities"][0]["top_logprobs"]}
                assert targ[0] in dist
                jobs.append((check,targ));expected.append(dist[targ[0]])
                # All 72 comparisons with actual token likelihood; same query IDs across references.
                for k in range(2):
                    for i,t in enumerate(query):
                        target=tok(t)
                        assert target
                        for j in range(3):
                            ix=k*3+j
                            user="Neutral text:\n"+neutral[ix]+"\n\nStyled text:\n"+refs[k][j]+"\n\nNeutral text:\n"+neutral[6+i]+"\n\nStyled text:"
                            prefix=tok(chat("Rewrite the final neutral text in the style of the example. Output only the styled text.",user),True)
                            assert len(prefix)+len(target)<=1024
                            jobs.append((prefix,target))
                # Repeat exact identical job to detect state contamination.
                jobs.append(jobs[0])
                f=root/"jobs.txt"
                with f.open("w") as stream:
                    for a,b in jobs:stream.write(" ".join(map(str,[len(a),len(b),*a,*b]))+"\n")
            finally:
                p.terminate()
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        print("OSST_NEUTRALIZATION_DONE="+json.dumps({"passages":18,"exact_copies":copies,"token_limit_stops":limits}),flush=True)
        results=[]
        with (root/"native.log").open("w") as err:
            p=subprocess.Popen([os.environ["RCMA_OSST_SCORER"],str(model),str(f)],stdout=subprocess.PIPE,stderr=err,text=True)
            try:
                for line in p.stdout:
                    o,m,first,n=line.split()
                    assert int(o)==len(results) and int(n)==len(jobs[int(o)][1])
                    results.append(float(m))
                    if len(results)%12==0:print("OSST_TEACHER_FORCED_PROGRESS="+str(len(results)),flush=True)
                assert p.wait()==0
            finally:
                if p.poll() is None:p.kill();p.wait()
        assert len(results)==74
        assert abs(results[0]-expected[0])<.05,"full-vocab scorer mismatch against server"
        assert abs(results[0]-results[-1])<1e-5,"state leakage"
        z=np.array(results[1:-1]).reshape(2,12,3)
        y=np.tile(np.arange(3),4)
        res=summarize(z,y)
        res.update(labels=labs,score_checks_pass=True,exact_copies=copies,seconds=round(time.monotonic()-start,1),
            scope="spent official fold0, four query prompts, 3-way conditional; not independent or full18",gpu=False,training=False,submission=False)
        print("OSST_FLASH_REAL_RESULT="+json.dumps(res,sort_keys=True),flush=True)
if __name__=="__main__":main()
