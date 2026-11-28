"""Memory-bounded exact full-vocabulary scoring for the existing MSF backbone.
No quantization, source pruning, or readout change. The LM head's mmap-backed
weight is visited by vocabulary tile, rather than reread for every source and
16-token chunk. Log-sum-exp is reduced in FP32 over every vocabulary item.
"""
import resource,time
import torch
import torch.nn.functional as F

def snapshot():
    u=resource.getrusage(resource.RUSAGE_SELF)
    d={"wall":time.monotonic(),"cpu":time.process_time(),"major_faults":u.ru_majflt,
       "minor_faults":u.ru_minflt,"rss_peak_kib":u.ru_maxrss}
    try:
        with open("/proc/self/io") as f:
            d["disk_read_bytes"]=int(next(l.split(":")[1] for l in f if l.startswith("read_bytes:")))
    except (OSError,StopIteration):pass
    return d

def delta(a,b):
    return {k:round(b[k]-a[k],3) for k in a if k in b and k!="rss_peak_kib"}|{"rss_peak_kib":b["rss_peak_kib"]}

@torch.inference_mode()
def streamed_logp(hid,target,head,tile=4096):
    """Return [source,token] observed-token logp using the full vocabulary."""
    assert hid.ndim==3 and target.shape==hid.shape[:2]
    assert hid.dtype==head.weight.dtype and head.bias is None
    x=hid.reshape(-1,hid.shape[-1]).contiguous()
    y=target.reshape(-1)
    vocab=head.weight.shape[0]
    assert int(y.min())>=0 and int(y.max())<vocab and tile>0
    normalizer=torch.full((len(y),),-torch.inf,dtype=torch.float32,device=x.device)
    observed=torch.full_like(normalizer,torch.nan)
    count=torch.zeros_like(y)
    for lo in range(0,vocab,tile):
        hi=min(lo+tile,vocab)
        z=F.linear(x,head.weight[lo:hi]).float()
        normalizer=torch.logaddexp(normalizer,torch.logsumexp(z,dim=-1))
        inside=(y>=lo)&(y<hi)
        rows=inside.nonzero(as_tuple=True)[0]
        observed[rows]=z[rows,y[rows]-lo]
        count+=inside
    assert bool((count==1).all()) and bool(torch.isfinite(normalizer).all()) and bool(torch.isfinite(observed).all())
    return (observed-normalizer).reshape_as(target)

@torch.inference_mode()
def score_streamed(model,control,route,ids,labels,check_reference=False):
    labels=torch.as_tensor(labels,dtype=torch.long)
    route["labels"]=labels
    ids=ids.repeat(len(labels),1)
    a=snapshot()
    tok=model.get_input_embeddings()(ids)
    embeds=torch.cat([control.index_select(0,labels).to(tok.dtype),tok],dim=1)
    mask=torch.ones(embeds.shape[:2],dtype=torch.long)
    hid=model.model(inputs_embeds=embeds,attention_mask=mask,use_cache=False).last_hidden_state[:,1:-1]
    target=ids[:,1:]
    assert hid.shape[:2]==target.shape
    b=snapshot()
    lp=streamed_logp(hid,target,model.lm_head)
    c=snapshot()
    report={"backbone":delta(a,b),"streamed_head":delta(b,c),"vocab":model.lm_head.weight.shape[0],"tokens":int(target.shape[1]),"sources":int(target.shape[0])}
    if check_reference:
        # Compare exactly the same hidden state and observed targets with the original
        # dense-head/CrossEntropy implementation. No second backbone pass.
        n=min(16,target.shape[1]);ref=model.lm_head(hid[:1,:n]).float()
        expected=-F.cross_entropy(ref.reshape(-1,ref.shape[-1]),target[:1,:n].reshape(-1),reduction="none")
        err=float((expected-lp[0,:n]).abs().max())
        report["reference_max_abs"]=err
        report["reference_tokens"]=n
        assert err<=0.05, "streamed LM head diverges from original dense calculation"
    return lp.cpu().numpy(),report
