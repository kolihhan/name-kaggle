"""CPU-only effective LoRA delta merge, not a competition scorer."""
import numpy as np

def factor_mean(aa,bb,weights):
    return sum(w*b for w,b in zip(weights,bb,strict=True)) @ sum(w*a for w,a in zip(weights,aa,strict=True))

def merge_delta(aa,bb,weights,rank):
    """Weights include effective adapter scaling; verify shared base/layer first."""
    aa=[np.asarray(a,dtype=np.float64) for a in aa]
    bb=[np.asarray(b,dtype=np.float64) for b in bb]
    w=np.asarray(weights,dtype=np.float64)
    if not aa or len(aa)!=len(bb) or w.shape!=(len(aa),) or not np.isfinite(w).all():
        raise ValueError('invalid inputs')
    if any(a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[1] or a.shape[1]!=aa[0].shape[1] or b.shape[0]!=bb[0].shape[0] or not np.isfinite(a).all() or not np.isfinite(b).all() for a,b in zip(aa,bb)):
        raise ValueError('incompatible factors')
    if not isinstance(rank,int) or not 1<=rank<=min(bb[0].shape[0],aa[0].shape[1]):
        raise ValueError('invalid rank')
    left=np.concatenate([b*(np.sign(v)*np.sqrt(abs(v))) for b,v in zip(bb,w)],axis=1)
    right=np.concatenate([a*np.sqrt(abs(v)) for a,v in zip(aa,w)],axis=0)
    ql,rl=np.linalg.qr(left,mode='reduced'); qr,rr=np.linalg.qr(right.T,mode='reduced')
    u,s,vt=np.linalg.svd(rl@rr.T,full_matrices=False)
    k=min(rank,len(s)); scale=np.sqrt(s[:k])
    b=(ql@u[:,:k])*scale; a=scale[:,None]*(vt[:k]@qr.T)
    den=np.linalg.norm(s)
    err=float(np.linalg.norm(s[k:])/den) if den else 0.
    if k<rank:
        b=np.pad(b,((0,0),(0,rank-k))); a=np.pad(a,((0,rank-k),(0,0)))
    return b,a,err
