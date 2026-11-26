import hashlib
import numpy as np

def sha_ok(raw, expected):
    return len(expected)==64 and hashlib.sha256(raw).hexdigest()==expected

def contrast(x, observed):
    x=np.asarray(x,dtype=float)
    if len(x)<2 or not 0<=observed<len(x): raise ValueError('invalid options')
    return np.r_[x-x.mean(), x[observed]-np.delete(x,observed).mean()]

def predict(refs, queries, labels):
    classes=np.unique(labels); distances=0
    for r,q in zip(refs,queries,strict=True):
        scale=r.std(axis=0); active=scale>1e-8
        if not active.any(): continue
        r=r[:,active]/scale[active]; q=q[:,active]/scale[active]
        cent=np.stack([r[labels==c].mean(axis=0) for c in classes])
        distances=distances+((q[:,None,:]-cent[None,:,:])**2).mean(axis=-1)
    if np.ndim(distances)==0: return np.repeat(classes[0],len(queries[0]))
    return classes[distances.argmin(axis=1)]

def permute_within_prompt(x,p,seed):
    z=x.copy(); rng=np.random.default_rng(seed)
    for k in np.unique(p):
        ids=np.flatnonzero(p==k);z[ids]=x[rng.permutation(ids)]
    return z
