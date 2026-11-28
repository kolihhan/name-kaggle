"""One fixed spent-publisher matched prompt block; no fit, no GPU.
Same teacher-forced 18xT likelihood matrix for every readout.
"""
import os,json,time,hashlib,tempfile,zipfile,subprocess
from pathlib import Path
import numpy as np,pandas as pd,torch
from scripts.ntl_msf_atlas_cpu_smoke_20261128 import BASE,REV,ADAPTER_SHA,BUNDLE_SHA,sha,log,unique,load_base,install,score,AutoTokenizer,snapshot_download
from scripts.ntl_source_fingerprint import fingerprint
from scripts.ntl_msf_streamed_head import score_streamed
PANEL_SHA="bce34670f4d1170375e7baf85eec5577af4282d5f0ca6cf1573ce46e81c5a45c"
TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
def run(*a):
    p=subprocess.run(["kaggle",*a],capture_output=True,timeout=180)
    assert p.returncode==0,"Kaggle data acquisition failed"
def norm(s):return hashlib.sha256(" ".join(str(s).split()).casefold().encode()).hexdigest()
def main():
    t=time.monotonic()
    assert not torch.cuda.is_available()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    root=Path(os.environ["ATLAS_ASSET_DIR"]);adapter=unique(root,"adapters.safetensors");bundle=unique(root,"atlas_bundle.json")
    assert sha(adapter)==ADAPTER_SHA and sha(bundle)==BUNDLE_SHA
    labels=json.loads(bundle.read_text())["labels"]
    assert len(labels)==len(set(labels))==18
    meta=json.loads(unique(root,"base-fingerprint.json").read_text())
    assert meta["model"]==BASE and meta["revision"]==REV
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp);p=d/"panel";p.mkdir()
        run("datasets","download","-d","nischaydnk/ntl-topic-synthetic-18llms","-f","generations.csv","-p",str(p),"--unzip")
        source=unique(p,"generations.csv");assert sha(source)==PANEL_SHA
        run("competitions","download","-c","name-that-llm","-f","train.csv","-p",tmp,"-q")
        train=d/"train.csv"
        if not train.exists():
            with zipfile.ZipFile(next(d.glob("*.zip"))) as z:train.write_bytes(z.read("train.csv"))
        assert sha(train)==TRAIN_SHA
        seen=set(pd.read_csv(train,usecols=["text"]).text.dropna().map(norm))
        raw=pd.read_csv(source)
        panel=raw.loc[(raw.version=="short")&(raw.topic_id==98),["text","model"]].copy()
        assert len(panel)==18 and set(panel.model)==set(labels) and panel.model.is_unique
        assert panel.text.notna().all() and panel.text.map(norm).is_unique and not set(panel.text.map(norm))&seen
        # Shuffle before scoring so no label order is passed to the model.
        panel=panel.sample(frac=1,random_state=20261128).reset_index(drop=True)
        texts=panel.text.tolist()
        truth=panel.model.tolist()
        del raw,panel,seen
        path=Path(snapshot_download(BASE,revision=REV,allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.tiktoken"],max_workers=2))
        tok=AutoTokenizer.from_pretrained(path,trust_remote_code=False)
        tokens=[tok(s,return_tensors="pt",add_special_tokens=False).input_ids[:,:65] for s in texts]
        lengths=[i.shape[1]-1 for i in tokens]
        assert min(lengths)>=16
        model=load_base(path,meta);route,control,handles=install(model,adapter)
        outputs=[];elapsed=[]
        # The existing runner permits 55 minutes including asset setup. Keep
        # the original cohort and stop cleanly rather than censoring slow rows.
        budget=2400.0
        for i,ids in enumerate(tokens):
            if i and sum(elapsed)+max(elapsed)*(len(tokens)-i)>budget:
                log('MSF_CPU_BUDGET_BLOCKED',{'completed':i,'total':len(tokens),'seconds':sum(elapsed),'projected_inference_seconds':sum(elapsed)+max(elapsed)*(len(tokens)-i),'BA':None})
                for h in handles:h.remove()
                return
            a=time.monotonic();x,perf=score_streamed(model,control,route,ids,list(range(18)),check_reference=(i==0))
            log('MSF_PROFILE',{'row_number':i+1,**perf})
            assert x.shape==(18,ids.shape[1]-1) and np.isfinite(x).all()
            rep=fingerprint(x,labels)
            # ATLAS v36 raw (uncalibrated) product scores:
            # gap_i=sum_j max(lp_j - lp_i,0), averaged across token positions and temperatures.
            gap=np.maximum(x[None,:,:]-x[:,None,:],0).sum(axis=1)
            prod=np.mean([np.exp(-gap/z).mean(axis=1) for z in (1.,2.,4.,8.)],axis=0)
            outputs.append([labels[np.argmax(rep.global_lp)],labels[np.argmax(prod)],rep.prediction])
            elapsed.append(time.monotonic()-a)
            log("MSF_REAL_PROGRESS",{"completed":i+1,"total":18,"tokens":lengths[i],"seconds":round(elapsed[-1],2)})
        for h in handles:h.remove()
        a=np.asarray(outputs);y=np.asarray(truth)
        correct=a==y[:,None]
        def family(s):return s.split("-")[0]
        report={"rows":18,"prompt_clusters":1,"labels":18,"target_token_range":[min(lengths),max(lengths)],"methods":["mean_logp","atlas_raw_product","MSF"],
                "correct":correct.sum(0).tolist(),"BA":correct.mean(0).tolist(),
                "msf_vs_mean":{"rescues":int((~correct[:,0]&correct[:,2]).sum()),"regressions":int((correct[:,0]&~correct[:,2]).sum())},
                "msf_vs_product":{"rescues":int((~correct[:,1]&correct[:,2]).sum()),"regressions":int((correct[:,1]&~correct[:,2]).sum())},
                "family_correct":[int(sum(family(p)==family(q) for p,q in zip(a[:,k],y))) for k in range(3)],
                "mean_inference_seconds":float(np.mean(elapsed)),"total_inference_seconds":float(sum(elapsed)),"seconds":round(time.monotonic()-t,2),
                "scope":"spent one matched publisher prompt block, truncated first 65 tokens, not independent, not v36/calibrated ATLAS","fit":False,"gpu":False,"submission":False,"execution":"streamed_full_vocabulary"}
        log("MSF_REAL_DIAGNOSTIC_RESULT",report)
if __name__=="__main__":main()
