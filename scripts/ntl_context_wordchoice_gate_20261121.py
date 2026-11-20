"""CPU-only NTL word-choice evaluation gate. Predictions must be genuinely fold-held-out.
C0=vocabulary-only, C1=context, C2=capacity-matched shuffled-context.
This file DOES NOT train a model, establish source independence, or score champion.
"""
import collections, math, random, hashlib

def mask_words(text, seed=20261121):
    words=text.split()
    if not words: raise ValueError("empty passage")
    n=len(words)
    k=min(math.ceil(.15*n),32,n-1) if n>1 else 1
    rng=random.Random(int.from_bytes(hashlib.sha256((str(seed)+"\\0"+text).encode()).digest(),"big"))
    return sorted(rng.sample(range(n),k))

def validate(rows, frozen_prompt_folds):
    if not rows: raise ValueError("no rows")
    seen=set()
    for r in rows:
        if r["id"] in seen: raise ValueError("duplicate id")
        seen.add(r["id"])
        p=r["prompt_id"]
        if p not in frozen_prompt_folds or int(r["fold"])!=int(frozen_prompt_folds[p]): raise ValueError("frozen prompt fold mismatch")
        if not r.get("source"): raise ValueError("missing source")
        if any(int(r[k]) not in range(18) for k in ("label","C0","C1","C2")): raise ValueError("invalid class")

def ba18(rows, pred):
    by=collections.defaultdict(list)
    for r in rows: by[int(r["label"])].append(r)
    if len(by)!=18: raise ValueError("BA18 requires all 18 truth classes")
    return sum(sum(int(int(r[pred])==y) for r in by[y])/len(by[y]) for y in range(18))/18

def sibling_macro(rows,pred,family):
    labels=[y for y in range(18) if sum(family[z]==family[y] for z in range(18))>1]
    by=collections.defaultdict(list)
    for r in rows: by[int(r["label"])].append(r)
    if not labels or any(not by[y] for y in labels): raise ValueError("missing sibling label")
    return sum(sum(int(int(r[pred])==y) for r in by[y])/len(by[y]) for y in labels)/len(labels)

def cluster_ci(rows,a,b,seed=20261121,draws=1000):
    groups=collections.defaultdict(list)
    for r in rows: groups[r["prompt_id"]].append(r)
    keys=sorted(groups); rng=random.Random(seed); diffs=[]
    for _ in range(draws):
        sample=[r for _ in keys for r in groups[rng.choice(keys)]]
        if len({int(r["label"]) for r in sample})!=18: continue
        diffs.append(ba18(sample,a)-ba18(sample,b))
    if len(diffs)<draws//4: raise ValueError("too few complete-class bootstrap draws")
    diffs.sort()
    return diffs[int(.025*len(diffs))],diffs[min(len(diffs)-1,int(.975*len(diffs)))]

def gate(rows,folds,family):
    validate(rows,folds)
    metric={k:{"BA18":ba18(rows,k),"sibling_macro":sibling_macro(rows,k,family)} for k in ("C0","C1","C2")}
    cis={k:cluster_ci(rows,"C1",k) for k in ("C0","C2")}
    passed=all(metric["C1"]["sibling_macro"]-metric[k]["sibling_macro"]>=.02 and cis[k][0]>0 for k in ("C0","C2"))
    return {"metric":metric,"delta_BA18_cluster_ci":cis,"mechanism_pass":passed,"champion_gain":"UNKNOWN: independent aligned 0.702 predictions absent"}

if __name__=="__main__":
    folds={f"p{i}":i%5 for i in range(90)}
    family={i:i//2 for i in range(18)}
    rows=[dict(id=f"{i}-{y}",prompt_id=f"p{i}",fold=i%5,source="synthetic",label=y,C0=y,C1=y,C2=y) for i in range(90) for y in range(18)]
    assert mask_words("a b c")==mask_words("a b c")
    assert mask_words("a")==[0]
    assert len(mask_words(" ".join(map(str,range(300))))==32
    assert gate(rows,folds,family)["mechanism_pass"] is False
    try: validate(rows+[rows[0]],folds)
    except ValueError: pass
    else: raise AssertionError("duplicate accepted")
    try: ba18([r for r in rows if r["label"]!=17],"C1")
    except ValueError: pass
    else: raise AssertionError("missing class accepted")
    for r in rows:
        if r["label"]%2==0: r["C0"]=r["label"]+1; r["C2"]=r["label"]+1
    assert gate(rows,folds,family)["mechanism_pass"] is True
    print("PASS synthetic CPU self-tests; no real training, no official BA change")
