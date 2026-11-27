"""Conservative two-channel arbitration prototype.

No fit, no truth use at inference, no mutation of the champion.
Distinct declared lineage is a necessary engineering contract, not proof of statistical independence.
Every score array must carry explicit row/class IDs and coverage. No positional guesses.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Evidence:
    kind: str
    lineage: frozenset
    ids: tuple
    labels: tuple
    scores: np.ndarray
    covered: np.ndarray

def align(e,ids,labels):
    if e.kind not in ("generation","attribution","anchor"):
        raise ValueError("unsupported channel")
    if not e.lineage or any(not isinstance(x,str) or not x for x in e.lineage):
        raise ValueError("missing lineage")
    if len(set(e.ids))!=len(e.ids) or set(e.ids)!=set(ids):
        raise ValueError("row identity mismatch")
    if len(set(e.labels))!=18 or set(e.labels)!=set(labels):
        raise ValueError("class identity mismatch")
    a=np.asarray(e.scores,dtype=float)
    c=np.asarray(e.covered)
    if a.shape!=(len(ids),18) or c.shape!=(len(ids),) or c.dtype.kind!="b":
        raise ValueError("shape/coverage mismatch")
    r={v:i for i,v in enumerate(e.ids)}; col={v:i for i,v in enumerate(e.labels)}
    a=a[[r[v] for v in ids]][:,[col[v] for v in labels]]
    c=c[[r[v] for v in ids]]
    if not np.isfinite(a[c]).all():raise ValueError("invalid covered scores")
    return a,c

def arbitrate(anchor,final,generation,attribution):
    ids=tuple(anchor.ids);labs=tuple(anchor.labels)
    if anchor.kind!="anchor" or len(set(ids))!=len(ids) or len(set(labs))!=18 or len(labs)!=18:
        raise ValueError("invalid anchor contract")
    if set(final)!=set(ids) or not set(final.values())<=set(labs):
        raise ValueError("final baseline IDs or labels mismatch")
    original=np.array([labs.index(final[i]) for i in ids])
    out=original.copy()
    reason=np.full(len(ids),"missing_channel",dtype="<U40")
    a,ac=align(anchor,ids,labs)
    if generation is None or attribution is None:return out,reason
    if generation.kind!="generation" or attribution.kind!="attribution":
        raise ValueError("distinct evidence kinds required")
    if (generation.lineage & attribution.lineage or generation.lineage & anchor.lineage or attribution.lineage & anchor.lineage):
        reason[:]="lineage_overlap"
        return out,reason
    g,gc=align(generation,ids,labs);t,tc=align(attribution,ids,labs)
    for i in range(len(ids)):
        if not(ac[i] and gc[i] and tc[i]):
            reason[i]="uncovered"
            continue
        # A final guard may already have overruled the anchor. Do not undo it.
        order=np.argsort(-a[i],kind="stable")
        if order[0]!=original[i] or a[i,order[0]]<=a[i,order[1]]:
            reason[i]="protected_final"
            continue
        gx=np.argsort(-g[i],kind="stable");tx=np.argsort(-t[i],kind="stable")
        k=gx[0]
        if k!=tx[0] or k==original[i] or k not in order[1:3]:
            reason[i]="no_shared_top3_challenger"
            continue
        # Strict unique winners, no arbitrary cross-channel confidence arithmetic.
        if g[i,k]<=g[i,gx[1]] or t[i,k]<=t[i,tx[1]]:
            reason[i]="evidence_tie"
            continue
        out[i]=k;reason[i]="changed"
    return out,reason

def selftest():
    ids=("a","b","c","d");labs=tuple("m"+str(i) for i in range(18))
    a=np.tile(np.r_[3.,2.,1.,np.zeros(15)],(4,1))
    g=np.tile(np.r_[1.,3.,0.,np.zeros(15)],(4,1))
    t=g.copy();t[2]=np.r_[3.,1.,0.,np.zeros(15)]
    cov=np.ones(4,bool)
    base=Evidence("anchor",frozenset({"base"}),ids,labs,a,cov)
    gen=Evidence("generation",frozenset({"g"}),ids,labs,g,cov)
    att=Evidence("attribution",frozenset({"t"}),ids[::-1],labs[::-1],t[::-1,::-1],np.array([False,True,True,True]))
    final=dict(zip(ids,["m0","m2","m0","m0"]))
    p,r=arbitrate(base,final,gen,att)
    assert p.tolist()==[1,2,0,0],(p,r)
    assert r.tolist()==["changed","protected_final","no_shared_top3_challenger","uncovered"]
    assert arbitrate(base,final,None,att)[0].tolist()==[0,2,0,0]
    overlap=Evidence("attribution",frozenset({"g"}),ids,labs,t,cov)
    assert (arbitrate(base,final,gen,overlap)[1]=="lineage_overlap").all()
    try:
        bad=Evidence("attribution",frozenset({"t"}),("x","b","c","d"),labs,t,cov)
        arbitrate(base,final,gen,bad)
    except ValueError:pass
    else:raise AssertionError("misaligned IDs accepted")
    print("ARBITRATION_CONTRACT_TESTS=PASS",flush=True)
if __name__=="__main__":selftest()
