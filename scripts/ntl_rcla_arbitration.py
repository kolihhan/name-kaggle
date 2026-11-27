"""RCLA Flash sibling arbitration; fixed inference-only decision contract.
No learned weights, no true labels, no post-hoc thresholds.
Scores are exact mean target-token log probabilities from two fixed reference panels.
An unavailable or incomplete row retains the incumbent decision.
"""
from dataclasses import dataclass
import numpy as np

FLASH=frozenset(("gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash"))

@dataclass(frozen=True)
class Candidate:
    row_id:str
    incumbent:str
    alternative:str

def select_candidates(ids,labels,logits,incumbent):
    ids=np.asarray(ids,dtype=str); labels=np.asarray(labels,dtype=str)
    incumbent=np.asarray(incumbent,dtype=str); z=np.asarray(logits,dtype=float)
    if ids.ndim!=1 or len(set(ids))!=len(ids) or any(not x for x in ids):
        raise ValueError("unique explicit row IDs required")
    if labels.shape!=(18,) or len(set(labels))!=18 or not FLASH<=set(labels):
        raise ValueError("explicit 18-class axis required")
    if z.shape!=(len(ids),18) or incumbent.shape!=(len(ids),) or not np.isfinite(z).all():
        raise ValueError("complete finite aligned native logits required")
    if not set(incumbent)<=set(labels):raise ValueError("invalid incumbent class")
    out=[]
    for i,(rid,base) in enumerate(zip(ids,incumbent)):
        if base not in FLASH:continue
        rank=np.argsort(-z[i],kind="stable")
        if z[i,rank[0]]==z[i,rank[1]] or z[i,rank[1]]==z[i,rank[2]]:continue
        pair=tuple(labels[rank[:2]])
        if base not in pair or not set(pair)<=FLASH:continue
        alt=pair[1] if pair[0]==base else pair[0]
        out.append(Candidate(str(rid),str(base),str(alt)))
    return out

def arbitrate(ids,labels,logits,incumbent,scores):
    """scores: mapping ID -> {class -> (panel1 mean logp, panel2 mean logp)}.
    Both panels must strictly prefer the same alternative to change one row.
    Whole incomplete rows fall back to baseline. Unexpected IDs are rejected.
    """
    candidates=select_candidates(ids,labels,logits,incumbent)
    allowed={c.row_id:c for c in candidates}
    if not set(scores)<=set(allowed):raise ValueError("scores outside frozen eligible ID set")
    result=np.asarray(incumbent,dtype=str).copy()
    idx={str(r):i for i,r in enumerate(ids)}
    summary={"eligible":len(candidates),"complete":0,"changed":0,"fallback_missing":0,"fallback_disagree_or_incumbent":0}
    for c in candidates:
        v=scores.get(c.row_id,{})
        if set(v)-{c.incumbent,c.alternative}:raise ValueError("unexpected candidate class")
        if set(v)!={c.incumbent,c.alternative}:
            summary["fallback_missing"]+=1;continue
        b=np.asarray(v[c.incumbent],dtype=float);a=np.asarray(v[c.alternative],dtype=float)
        if b.shape!=(2,) or a.shape!=(2,) or not np.isfinite(b).all() or not np.isfinite(a).all() or np.any(b>1e-7) or np.any(a>1e-7):
            summary["fallback_missing"]+=1;continue
        summary["complete"]+=1
        if np.all(a>b):
            result[idx[c.row_id]]=c.alternative
            summary["changed"]+=1
        else:summary["fallback_disagree_or_incumbent"]+=1
    assert summary["complete"]+summary["fallback_missing"]==summary["eligible"]
    assert summary["changed"]+summary["fallback_disagree_or_incumbent"]==summary["complete"]
    return result,summary
