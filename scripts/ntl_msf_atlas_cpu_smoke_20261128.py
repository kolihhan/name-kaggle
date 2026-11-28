"""Real ATLAS CPU inference smoke: exact pinned base and adapter, 18 source rows.
Synthetic passage only. No claim about BA, transfer or production runtime.
The readout is independently implemented from the verified ATLAS source contract.
"""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer,AutoConfig,Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from scripts.ntl_source_fingerprint import fingerprint
BASE="Qwen/Qwen3-8B-Base"
REV="49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
ADAPTER_SHA="29e9b32e09679ae4e3b2cdf0729c7011ecee730fdb0f20982593af0b40806106"
BUNDLE_SHA="48f8d7e82bf0b12f9414a314f695216e31b5f4854b5091248d1015b98d9743de"
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def log(k,v):print(k+"="+json.dumps(v,sort_keys=True),flush=True)
def unique(root,n):
    x=list(root.rglob(n));assert len(x)==1,n;return x[0]
def load_base(path,fingerprint_meta):
    config=AutoConfig.from_pretrained(path,trust_remote_code=False)
    assert config.model_type=="qwen3" and config.hidden_size==4096
    config._attn_implementation="eager"
    # No initialized full-size CPU copy. Safetensors CPU tensors are mmap-backed;
    # assign=True keeps their storage instead of copying into a second model.
    with torch.device("meta"):
        model=Qwen3ForCausalLM(config)
    state={}
    for name,info in fingerprint_meta["files"].items():
        p=path/name
        assert p.stat().st_size==info["bytes"] and sha(p)==info["sha256"],name
        with safe_open(p,framework="pt",device="cpu") as f:
            for k in f.keys():
                assert k not in state
                state[k]=f.get_tensor(k)
        log("BASE_SHARD_VERIFIED",{"name":name,"tensor_count":len(state)})
    assert set(state)==set(model.state_dict()),"base state keys differ"
    assert all(state[k].shape==v.shape for k,v in model.state_dict().items())
    model.load_state_dict(state,strict=True,assign=True)
    model.model.rotary_emb=Qwen3RotaryEmbedding(config,device="cpu")
    model.eval()
    assert all(p.device.type=="cpu" for p in model.parameters())
    assert model.model.embed_tokens.weight.dtype==torch.bfloat16 or model.model.embed_tokens.weight.dtype==torch.float16
    log("BASE_MMAP_READY",{"dtype":str(model.model.embed_tokens.weight.dtype),"parameter_count":sum(p.numel() for p in model.parameters()),"device":"cpu"})
    return model
def install(model,adapter):
    route={"labels":None}
    handles=[]
    with safe_open(adapter,framework="pt",device="cpu") as f:
        keys=set(f.keys())
        control=f.get_tensor("control")
        assert control.shape==(18,1,4096) and torch.isfinite(control).all()
        akeys=sorted(k for k in keys if k.endswith(".a"))
        assert len(akeys)==144 and keys=={"control"}|set(akeys)|{k[:-2]+".b" for k in akeys}
        for key in akeys:
            modname=key[:-2]
            assert modname.startswith("base.")
            mod=model.get_submodule(modname[len("base."):])
            assert isinstance(mod,torch.nn.Linear)
            a=f.get_tensor(key);b=f.get_tensor(key[:-2]+".b")
            assert a.shape==(16,mod.in_features) and b.shape==(18,mod.out_features,16)
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            def hook(mod,args,out,a=a,b=b):
                labels=route["labels"];x=args[0]
                assert labels is not None and len(labels)==x.shape[0]
                delta=torch.bmm(F.linear(x.float(),a),b.index_select(0,labels).transpose(1,2))
                return out+(delta*2.).to(out.dtype)
            handles.append(mod.register_forward_hook(hook))
    log("ADAPTER_READY",{"hooks":len(handles),"control_shape":list(control.shape),"alpha":32,"rank":16})
    return route,control,handles
@torch.inference_mode()
def score(model,control,route,ids,labels):
    labels=torch.as_tensor(labels,dtype=torch.long);route["labels"]=labels
    ids=ids.repeat(len(labels),1)
    tok=model.get_input_embeddings()(ids)
    embeds=torch.cat([control.index_select(0,labels).to(tok.dtype),tok],dim=1)
    mask=torch.ones(embeds.shape[:2],dtype=torch.long)
    hid=model.model(inputs_embeds=embeds,attention_mask=mask,use_cache=False).last_hidden_state[:,1:-1]
    target=ids[:,1:]
    assert hid.shape[:2]==target.shape
    # Full vocabulary normalization, chunked by sequence rows and target length.
    out=[]
    for r in range(len(labels)):
        lp=[]
        for t in range(0,target.shape[1],16):
            y=target[r:r+1,t:t+16]
            logits=model.lm_head(hid[r:r+1,t:t+16]).float()
            lp.append(-F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction="none").reshape(y.shape))
        out.append(torch.cat(lp,dim=1))
    return torch.cat(out,dim=0).numpy()
def main():
    t=time.monotonic()
    assert not torch.cuda.is_available()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    root=Path(os.environ["ATLAS_ASSET_DIR"])
    adapter=unique(root,"adapters.safetensors");bundle=unique(root,"atlas_bundle.json")
    assert sha(adapter)==ADAPTER_SHA and sha(bundle)==BUNDLE_SHA
    labels=json.loads(bundle.read_text())["labels"]
    meta=json.loads(unique(root,"base-fingerprint.json").read_text())
    assert meta["model"]==BASE and meta["revision"]==REV
    assert len(labels)==len(set(labels))==18
    log("CPU_ENV",{"torch":torch.__version__,"threads":torch.get_num_threads(),"cuda":False})
    path=Path(snapshot_download(BASE,revision=REV,allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.tiktoken"],max_workers=2))
    model=load_base(path,meta)
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=False)
    route,control,handles=install(model,adapter)
    # This original synthetic text has no generator truth label. It measures
    # numerical execution only, not performance on any competition row.
    ids=tok("A community library opens its doors in the morning. Visitors borrow books, study quietly, and share ideas. A volunteer helps a student find a reference for an assignment.",return_tensors="pt",add_special_tokens=False).input_ids
    assert 17<=ids.shape[1]<=64
    a=time.monotonic()
    out=score(model,control,route,ids,list(range(18)))
    elapsed=time.monotonic()-a
    assert out.shape==(18,ids.shape[1]-1) and np.isfinite(out).all() and np.all(out<=1e-6)
    rep=fingerprint(out,labels)
    log("REAL_MSF_MATRIX",{"shape":list(out.shape),"finite":True,"source_score_span":float(np.ptp(rep.global_lp)),"sources_nonidentical":bool(np.ptp(rep.global_lp)>1e-6),"global_and_local_agree":labels[int(np.argmax(rep.global_lp))]==rep.prediction,"batch_seconds":round(elapsed,2),"dtype":str(model.model.embed_tokens.weight.dtype),"synthetic_text_only":True})
    assert np.ptp(rep.global_lp)>1e-6,"all source scores identical"
    single=score(model,control,route,ids,[0])
    delta=float(np.max(np.abs(out[0]-single[0])))
    log("REAL_MSF_CROSSCHECK",{"source0_batch_vs_single_max_abs":delta,"pass":delta<=.25,"seconds":round(time.monotonic()-t,2),"scope":"synthetic numerical smoke; BF16 or FP16 pinned checkpoint dtype, not validated against original CUDA","gpu":False,"training":False,"submission":False})
    assert delta<=.25
    for h in handles:h.remove()
if __name__=="__main__":main()
