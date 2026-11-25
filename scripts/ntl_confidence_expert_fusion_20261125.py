"""Confidence-conditioned expert fusion, frozen no-fit CPU feasibility.

The incumbent's fixed 3% blend stays unchanged unless both models support
the same two candidates, disagree on their order, and the expert has a
larger pairwise log-odds margin than the incumbent. Only then apply a
pairwise logarithmic opinion pool; preserve the pair's original mass.
No labels used by the decision function; no class-specific exceptions.
"""
from __future__ import annotations
import numpy as np

def validate(p,q):
    p,q=np.asarray(p,dtype=np.float64),np.asarray(q,dtype=np.float64)
    if p.ndim!=2 or p.shape!=q.shape or p.shape[1]!=18 or not len(p):
        raise ValueError("require aligned nonempty N x 18 probabilities")
    for x in (p,q):
        if not np.isfinite(x).all() or (x<0).any() or not np.allclose(x.sum(1),1,atol=1e-6,rtol=0):
            raise ValueError("invalid normalized probabilities")
    return p,q

def fuse(p,q):
    p,q=validate(p,q)
    baseline=.97*p+.03*q
    result=baseline.copy()
    a=np.argsort(-p,axis=1,kind="stable")[:,:2]
    b=np.argsort(-q,axis=1,kind="stable")[:,:2]
    i=np.arange(len(p))
    # Competing labels must be mutually proposed; no top-k fishing.
    agree=(a[:,0]==b[:,1])&(a[:,1]==b[:,0])
    eps=1e-12
    m=np.log(np.maximum(p[i,a[:,0]],eps))-np.log(np.maximum(p[i,a[:,1]],eps))
    e=np.log(np.maximum(q[i,a[:,1]],eps))-np.log(np.maximum(q[i,a[:,0]],eps))
    mask=agree&(e>m+1e-10)&(m>0)&(e>0)
    j=i[mask]
    x,y=a[j,0],a[j,1]
    # Equal log-odds pooling within the two-candidate set, no global mass shift.
    # No fitted threshold or new model. Clip solely for numerical stability.
    u=1/(1+np.exp(-np.clip((m[j]-e[j])/2,-60,60)))
    mass=baseline[j,x]+baseline[j,y]
    result[j,x]=mass*u
    result[j,y]=mass*(1-u)
    # Do not introduce a third-label winner from redistribution.
    if len(j):
        winner=result[j].argmax(1)
        invalid=(winner!=x)&(winner!=y)
        result[j[invalid]]=baseline[j[invalid]]
        mask[j[invalid]]=False
    assert np.allclose(result.sum(1),1,atol=1e-6,rtol=0)
    assert np.array_equal(result[~mask],baseline[~mask])
    return baseline,result,mask

def paired_report(p,q,y):
    base,cand,gate=fuse(p,q)
    y=np.asarray(y,dtype=int)
    if y.shape!=(len(base),) or set(y)!=set(range(18)):
        raise ValueError("full 18-class truth required for BA")
    a,b=base.argmax(1),cand.argmax(1)
    ba=lambda x:float(np.mean([(x[y==k]==k).mean() for k in range(18)]))
    return {"rows":len(y),"baseline_ba":ba(a),"candidate_ba":ba(b),
            "gate_rows":int(gate.sum()),"changed":int((a!=b).sum()),
            "rescues":int(((a!=y)&(b==y)).sum()),"regressions":int(((a==y)&(b!=y)).sum()),
            "base_correct":int((a==y).sum()),"candidate_correct":int((b==y).sum())}

def selftest():
    p=np.full((5,18),.01);q=p.copy()
    p[:,0]=.44;p[:,1]=.40
    q[0,0]=.20;q[0,1]=.64
    q[1]=p[1]
    q[2,0]=.415;q[2,1]=.425
    q[3,0]=.10;q[3,1]=.20;q[3,2]=.55;q[3,3:]=.01
    q[4,0]=.40;q[4,1]=.44
    # Last row tie in odds, and non-mutual top2, should stay on baseline.
    a,b,g=fuse(p,q)
    assert g.tolist()==[True,False,False,False,False]
    assert a[0].argmax()==0 and b[0].argmax()==1
    assert np.array_equal(a[1:],b[1:])
    assert np.allclose(a[:,2:],b[:,2:])
    try:fuse(p,q[:-1])
    except ValueError:pass
    else:raise AssertionError("alignment rejection missing")
    try:fuse(p,q*2)
    except ValueError:pass
    else:raise AssertionError("normalization rejection missing")
    print("NTL_CONFIDENCE_FUSION_SYNTHETIC_TESTS=PASS",flush=True)

if __name__=="__main__":
    selftest()
