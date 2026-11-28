"""Private T4x2 pilot builder; original fixed 18-source panel and readouts.
Never submit to competition, never expose raw rows or likelihood matrices.
"""
import json,re,sys
from pathlib import Path
DEST="kolihhan/ntl-msf-atlas-t4-paired-20261128"
def build(out):
    root=Path("scripts")
    fp=(root/"ntl_source_fingerprint.py").read_text()
    base=(root/"ntl_msf_atlas_cpu_smoke_20261128.py").read_text()
    base=base[:base.index("def main():")].replace("from scripts.ntl_source_fingerprint import fingerprint","")
    # Replace CPU loading and adapter installation with two-GPU placements.
    start=base.index("def load_base(");end=base.index("@torch.inference_mode()",start)
    gpu=r'''
def load_base(path,fingerprint_meta):
    from transformers import AutoModelForCausalLM
    config=AutoConfig.from_pretrained(path,trust_remote_code=False)
    assert config.model_type=="qwen3" and config.hidden_size==4096
    for name,info in fingerprint_meta["files"].items():
        p=path/name
        assert p.stat().st_size==info["bytes"] and sha(p)==info["sha256"],name
    n=config.num_hidden_layers
    device_map={"model.embed_tokens":0,"model.rotary_emb":0,"model.norm":1,"lm_head":1}
    device_map.update({f"model.layers.{i}":0 if i<n//2 else 1 for i in range(n)})
    if config.tie_word_embeddings:raise ValueError("unexpected tied embeddings")
    model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,
        device_map=device_map,trust_remote_code=False,local_files_only=True,
        attn_implementation="eager",low_cpu_mem_usage=True).eval()
    assert all(p.device.type=="cuda" for p in model.parameters())
    log("GPU_BASE_READY",{"parameters":sum(p.numel() for p in model.parameters()),"dtype":str(model.model.embed_tokens.weight.dtype),"devices":[torch.cuda.get_device_name(i) for i in range(2)]})
    return model
def install(model,adapter):
    route={"labels":None};handles=[]
    with safe_open(adapter,framework="pt",device="cpu") as f:
        keys=set(f.keys());control=f.get_tensor("control")
        assert control.shape==(18,1,4096) and torch.isfinite(control).all()
        akeys=sorted(k for k in keys if k.endswith(".a"))
        assert len(akeys)==144 and keys=={"control"}|set(akeys)|{k[:-2]+".b" for k in akeys}
        for key in akeys:
            assert key.startswith("base.")
            mod=model.get_submodule(key[len("base."):-2])
            assert isinstance(mod,torch.nn.Linear)
            a=f.get_tensor(key);b=f.get_tensor(key[:-2]+".b")
            assert a.shape==(16,mod.in_features) and b.shape==(18,mod.out_features,16)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            a=a.to(mod.weight.device);b=b.to(mod.weight.device)
            def hook(mod,args,out,a=a,b=b):
                labels=route["labels"].to(a.device);x=args[0].to(a.device)
                assert len(labels)==x.shape[0]
                delta=torch.bmm(F.linear(x.float(),a),b.index_select(0,labels).transpose(1,2))
                return out+(delta*2.).to(device=out.device,dtype=out.dtype)
            handles.append(mod.register_forward_hook(hook))
    log("GPU_ADAPTER_READY",{"hooks":len(handles),"control_shape":list(control.shape)})
    return route,control.to(model.get_input_embeddings().weight.device),handles
@torch.inference_mode()
def score(model,control,route,ids,labels):
    outs=[]
    for lo in range(0,len(labels),3):
        chunk=torch.as_tensor(labels[lo:lo+3],dtype=torch.long,device=control.device)
        route["labels"]=chunk
        x=ids.to(control.device).repeat(len(chunk),1)
        tok=model.get_input_embeddings()(x)
        embeds=torch.cat([control.index_select(0,chunk).to(tok.dtype),tok],1)
        mask=torch.ones(embeds.shape[:2],dtype=torch.long,device=embeds.device)
        hid=model.model(inputs_embeds=embeds,attention_mask=mask,use_cache=False).last_hidden_state[:,1:-1]
        target=x[:,1:];assert hid.shape[:2]==target.shape
        lp=[]
        for t in range(0,target.shape[1],16):
            logits=model.lm_head(hid[:,t:t+16].to(model.lm_head.weight.device)).float()
            y=target[:,t:t+16].to(logits.device)
            lp.append(-F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction="none").reshape(y.shape).cpu())
        outs.append(torch.cat(lp,1))
    return torch.cat(outs,0).numpy()
'''
    base=base[:start]+gpu
    pilot=r'''
import pandas as pd
PANEL_SHA="bce34670f4d1170375e7baf85eec5577af4282d5f0ca6cf1573ce46e81c5a45c"
TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
def norm(s):return hashlib.sha256(" ".join(str(s).split()).casefold().encode()).hexdigest()
def main():
    t=time.monotonic()
    assert torch.cuda.device_count()==2,"expected two GPU devices"
    assert all("T4" in torch.cuda.get_device_name(i) for i in range(2))
    torch.set_num_threads(4)
    inp=Path("/kaggle/input")
    adapter=unique(inp,"adapters.safetensors");bundle=unique(inp,"atlas_bundle.json")
    assert sha(adapter)==ADAPTER_SHA and sha(bundle)==BUNDLE_SHA
    labels=json.loads(bundle.read_text())["labels"]
    assert len(labels)==len(set(labels))==18
    meta=json.loads(unique(inp,"base-fingerprint.json").read_text())
    assert meta["model"]==BASE and meta["revision"]==REV
    source=unique(inp,"generations.csv");assert sha(source)==PANEL_SHA
    train=unique(inp,"train.csv");assert sha(train)==TRAIN_SHA
    seen=set(pd.read_csv(train,usecols=["text"]).text.dropna().map(norm))
    raw=pd.read_csv(source)
    panel=raw.loc[(raw.version=="short")&(raw.topic_id==98),["text","model"]].copy()
    assert len(panel)==18 and set(panel.model)==set(labels) and panel.model.is_unique
    assert panel.text.notna().all() and panel.text.map(norm).is_unique and not set(panel.text.map(norm))&seen
    panel=panel.sample(frac=1,random_state=20261128).reset_index(drop=True)
    texts=panel.text.tolist();truth=panel.model.tolist()
    del raw,panel,seen
    path=unique(inp,"model-00001-of-00005.safetensors").parent
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=False,local_files_only=True)
    tokens=[tok(s,return_tensors="pt",add_special_tokens=False).input_ids[:,:65] for s in texts]
    lengths=[i.shape[1]-1 for i in tokens];assert min(lengths)>=16
    model=load_base(path,meta);route,control,handles=install(model,adapter)
    outputs=[];elapsed=[]
    for i,ids in enumerate(tokens):
        for dev in range(2):torch.cuda.synchronize(dev)
        a=time.monotonic()
        x=score(model,control,route,ids,list(range(18)))
        for dev in range(2):torch.cuda.synchronize(dev)
        assert x.shape==(18,ids.shape[1]-1) and np.isfinite(x).all() and np.all(x<=1e-6)
        if i==0:
            single=score(model,control,route,ids,[0])
            err=float(np.max(np.abs(single[0]-x[0])))
            log("MSF_GPU_CROSSCHECK",{"max_abs":err,"pass":err<=.25})
            assert err<=.25
        rep=fingerprint(x,labels)
        gap=np.maximum(x[None,:,:]-x[:,None,:],0).sum(axis=1)
        prod=np.mean([np.exp(-gap/z).mean(axis=1) for z in (1.,2.,4.,8.)],axis=0)
        outputs.append([labels[np.argmax(rep.global_lp)],labels[np.argmax(prod)],rep.prediction])
        elapsed.append(time.monotonic()-a)
        log("MSF_GPU_PROGRESS",{"completed":i+1,"total":18,"tokens":lengths[i],"seconds":round(elapsed[-1],2)})
    for h in handles:h.remove()
    a=np.asarray(outputs);y=np.asarray(truth);c=a==y[:,None]
    def family(s):return s.split("-")[0]
    report={"rows":18,"prompt_clusters":1,"labels":18,"methods":["mean_logp","atlas_raw_product","MSF"],
       "target_token_range":[min(lengths),max(lengths)],"correct":c.sum(0).tolist(),"BA":c.mean(0).tolist(),
       "msf_vs_mean":{"rescues":int((~c[:,0]&c[:,2]).sum()),"regressions":int((c[:,0]&~c[:,2]).sum())},
       "msf_vs_product":{"rescues":int((~c[:,1]&c[:,2]).sum()),"regressions":int((c[:,1]&~c[:,2]).sum())},
       "family_correct":[int(sum(family(p)==family(q) for p,q in zip(a[:,k],y))) for k in range(3)],
       "mean_inference_seconds":float(np.mean(elapsed)),"total_inference_seconds":float(sum(elapsed)),
       "seconds":round(time.monotonic()-t,2),"gpu_peak_allocated_gib":[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)],
       "scope":"spent one matched publisher prompt block, FP16 T4x2, first 65 tokens; not independent, not v36/calibrated ATLAS",
       "fit":False,"gpu":True,"submission":False}
    log("MSF_GPU_REAL_RESULT",report)
    Path("/kaggle/working/msf_aggregate.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__":main()
'''
    code=fp+"\n"+base+"\n"+pilot
    compile(code,"msf_gpu.py","exec")
    out.mkdir(parents=True,exist_ok=False)
    (out/"msf_gpu.py").write_text(code)
    md={"id":DEST,"title":DEST.split("/")[1],"code_file":"msf_gpu.py","language":"python","kernel_type":"script",
      "is_private":True,"enable_gpu":True,"enable_internet":False,"dataset_sources":["josephadamski91/ntl-atlas-qwen3-8b-v1","nischaydnk/ntl-topic-synthetic-18llms"],
      "competition_sources":["name-that-llm"],"model_sources":["qwen-lm/qwen-3/Transformers/8b-base/1"],"kernel_sources":[]}
    (out/"kernel-metadata.json").write_text(json.dumps(md,indent=2))
    print("MSF_GPU_BUILD_PASS",flush=True)
if __name__=="__main__":build(Path(sys.argv[1]))
