"""Source-conditioned multiresolution likelihood readout, no fit.

Input is one real teacher-forced token logp matrix [18, T], ordered source
labels and valid target-token mask. This is a standalone scoring mechanism,
not a patch to an ensemble. Never manufacture absent source scores.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Fingerprint:
    global_lp: np.ndarray
    global_relative: np.ndarray
    local_relative: np.ndarray
    local_wins: np.ndarray
    stable_support: np.ndarray
    prediction: str
    blocks: int

def fingerprint(logp, labels, mask=None):
    x=np.asarray(logp,dtype=np.float64)
    labels=tuple(labels)
    if x.ndim!=2 or x.shape[0]!=18 or len(labels)!=18 or len(set(labels))!=18:
        raise ValueError("complete unique 18-source axis required")
    if mask is not None:
        mask=np.asarray(mask)
        if mask.dtype!=np.bool_ or mask.shape!=(x.shape[1],):
            raise ValueError("invalid target mask")
        x=x[:,mask]
    if x.shape[1]<16 or not np.isfinite(x).all() or np.any(x>1e-6):
        raise ValueError("insufficient or invalid observed-token log probabilities")
    # A token's shared difficulty cancels in relative likelihood. No
    # source-specific biases estimated on the test cohort.
    rel=x-x.mean(axis=0,keepdims=True)
    g=x.mean(axis=1)
    gr=rel.mean(axis=1)
    # Fixed disjoint contiguous blocks; retain order, avoid single-token peaks.
    # Aggregate all scales with equal scale weight rather than letting long
    # texts or finer partitions get extra influence.
    local=[];wins=[];support=[]
    scales=[k for k in (2,4,8) if x.shape[1]//k>=8]
    for k in scales:
        b=np.stack([v.mean(axis=1) for v in np.array_split(rel,k,axis=1)],axis=1)
        local.append(np.median(b,axis=1))
        top=b==b.max(axis=0,keepdims=True)
        wins.append((top/top.sum(axis=0,keepdims=True)).mean(axis=1))
        support.append((b>0).mean(axis=1))
    lr=np.mean(local,axis=0)
    lw=np.mean(wins,axis=0)
    st=np.mean(support,axis=0)
    # Standalone frozen candidate: robust across-block evidence, not a
    # tuned ensemble alpha; deterministic label order resolves exact ties.
    prediction=labels[int(np.argmax(lr))]
    return Fingerprint(g,gr,lr,lw,st,prediction,sum(scales))
