"""Historical OOF cross-prompt calibration diagnostic; not a deployable V90 scorer.

No neural training. Empirical calibration tables are estimated from other prompt
groups; underlying OOF predictions were not generated in this outer protocol,
so this is exploratory cross-fitting, not a fully nested independent result.
"""
import hashlib,json,tempfile,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import ntl_decompose_historical_oof_20261127 as src

NCLASS=18
ALPHA=.30
SEED=20261127

def probabilities(a):
    a=np.asarray(a,dtype=np.float64)
    assert a.shape[-1]==NCLASS and np.isfinite(a).all()
    if np.min(a)>=0 and np.allclose(a.sum(-1),1,atol=1e-5):
        return a
    # Only admit unambiguously normalized log probabilities, not arbitrary activations.
    q=np.exp(a)
    if np.isfinite(q).all() and np.allclose(q.sum(-1),1,atol=1e-5):
        return q
    raise ValueError("unrecognized score space")

def family_groups(names):
    out={}
    for i,n in enumerate(names):
        g=next((x for x in ("qwen14","qwen8","q35","deberta","roberta","style","svm") if n.startswith(x)),None)
        if g is None:raise ValueError("unknown component family "+n)
        out.setdefault(g,[]).append(i)
    return out

def ba(y,pred):
    assert set(y)==set(range(NCLASS))
    return float(np.mean([(pred[y==k]==k).mean() for k in range(NCLASS)]))

def conf_bins(q):
    top=np.sort(q,axis=1)[:,-2:]
    # fixed margin bins, independent of labels and fold composition.
    return np.digitize(top[:,1]-top[:,0],[.05,.15,.30])

def empirical_post(qtr,ytr,qte,context):
    # class-balanced likelihood over the expert's top label and optional margin bin.
    # Uniform target prior is appropriate for BA, not a claim about test sampling.
    a=qtr.argmax(1);b=qte.argmax(1)
    c=conf_bins(qtr) if context else np.zeros(len(a),int)
    d=conf_bins(qte) if context else np.zeros(len(b),int)
    nb=4 if context else 1
    # Back off to class-conditioned unbinned ballot distributions for sparse bins.
    base=np.ones((NCLASS,NCLASS),float)
    np.add.at(base,(ytr,a),1.)
    base/=base.sum(1,keepdims=True)
    counts=np.zeros((NCLASS,NCLASS,nb),float)
    np.add.at(counts,(ytr,a,c),1.)
    likelihood=counts+18.*base[:,:,None]/nb
    likelihood/=likelihood.sum((1,2),keepdims=True)
    p=likelihood[:,b,d].T
    return p/p.sum(1,keepdims=True)

def fusion(q0,qs,y,groups):
    folds=np.unique(groups)
    rng=np.random.default_rng(SEED)
    folds=rng.permutation(folds)
    buckets=np.array_split(folds,5)
    out={k:np.empty_like(q0) for k in ("static_probability","reference_calibration","class_evidence","context_evidence")}
    assigned=np.zeros(len(y),int)
    for g in buckets:
        te=np.isin(groups,g);tr=~te
        assert not set(groups[tr])&set(groups[te])
        assert set(y[tr])==set(y[te])==set(range(NCLASS))
        assigned[te]+=1
        ref_cal=empirical_post(q0[tr],y[tr],q0[te],True)
        out["reference_calibration"][te]=(1-ALPHA)*q0[te]+ALPHA*ref_cal
        out["static_probability"][te]=(1-ALPHA)*q0[te]+ALPHA*qs[:,te].mean(0)
        for name,context in (("class_evidence",False),("context_evidence",True)):
            posts=np.array([empirical_post(q[tr],y[tr],q[te],context) for q in qs])
            # Family-level opinion mixture rather than independence/product assumption.
            evidence=posts.mean(0)
            out[name][te]=(1-ALPHA)*q0[te]+ALPHA*evidence
    assert (assigned==1).all()
    for a in out.values():assert np.allclose(a.sum(1),1)
    return out

def main():
    pins={
        "ntl-oof-validation-cache/diverse_oof.npz":"3db2ada2d57e074ad7507361896a2e976e8a0586f7a89f549e404b3ca4aa455f",
        "ntl-oof-validation-cache/diverse_report.json":"153db6ef08fffe12a2a560e147efb265cc676963cd3837b1b94af3178a945e57",
        "ntl-oof-q35-heads/bundle_manifest.json":"f3106b4274113ee3828987051e0ec5d33746feb1b179f6fa177dc6a95b375fe9",
    }
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);files={}
        for rel,digest in pins.items():
            path=src.get(rel,root);assert path and hashlib.sha256(path.read_bytes()).hexdigest()==digest
            files[rel]=path
        manifest=json.loads(files["ntl-oof-q35-heads/bundle_manifest.json"].read_text())
        scenarios=list(map(str,manifest["scenarios"]));assert scenarios==["1","3","6"]
        train_dir=root/"train";train_dir.mkdir()
        p=src.cmd("competitions","download","name-that-llm","-f","train.csv","-p",str(train_dir),"-q")
        assert p.returncode==0
        paths=list(train_dir.glob("train.csv"))
        if not paths:
            for f in train_dir.glob("*.zip"):
                with zipfile.ZipFile(f) as z:
                    if "train.csv" in z.namelist():
                        out=train_dir/"train.csv";out.write_bytes(z.read("train.csv"));paths.append(out)
        assert len(paths)==1 and hashlib.sha256(paths[0].read_bytes()).hexdigest()==src.SHA
        train=pd.read_csv(paths[0],dtype=str,usecols=["id","model","prompt_id"])
        report=json.loads(files["ntl-oof-validation-cache/diverse_report.json"].read_text())
        arrays,_=src.load_cache(files["ntl-oof-validation-cache/diverse_oof.npz"])
        ids=arrays["ids"].astype(str);labs=arrays["labels"].astype(str)
        assert len(ids)==len(train)==2699 and len(set(ids))==2699 and set(ids)==set(train.id)
        assert len(labs)==18 and set(labs)==set(train.model)
        idx=pd.Index(ids).get_indexer(train.id);y=pd.Index(labs).get_indexer(train.model)
        assert np.array_equal(arrays["y"][idx],y)
        names=arrays["component_names"].astype(str).tolist()
        assert names==report["component_names"]
        groups=family_groups(names)
        deltas=[]
        for si,sc in enumerate(scenarios):
            q0=probabilities(arrays["reference"][si][idx])
            qs=np.array([probabilities(arrays["components"][members,si][:,idx,:]).mean(0) for members in groups.values()])
            pred=q0.argmax(1);baseline=ba(y,pred)
            assert abs(baseline-report["scenario_scores"]["fixed_reference"][sc])<1e-8
            nested=arrays["nested"][si][idx].argmax(1)
            cal=arrays["nested_calibrated"][si][idx].argmax(1)
            stronger=max(ba(y,nested),ba(y,cal))
            result={}
            for method,q in fusion(q0,qs,y,train.prompt_id.to_numpy()).items():
                p=q.argmax(1);gain=ba(y,p)-baseline
                result[method]={"ba":ba(y,p),"delta_reference":gain,"delta_stronger_nested":ba(y,p)-stronger,"rescues":int(((pred!=y)&(p==y)).sum()),"regressions":int(((pred==y)&(p!=y)).sum()),"changed":int((pred!=p).sum())}
            deltas.append(result["context_evidence"]["delta_stronger_nested"])
            src.log("CONTEXT_FUSION",{"scenario":sc,"reference":baseline,"nested":ba(y,nested),"nested_calibrated":ba(y,cal),"methods":result,"families":list(groups),"prompts":train.prompt_id.nunique(),"rows":len(y)})
        src.log("DIAGNOSTIC_GATE",{"context_mean_gain_vs_stronger_nested":float(np.mean(deltas)),"min_gain":float(min(deltas)),"passes_exploratory_gate":bool(np.mean(deltas)>=.005 and min(deltas)>=-.005),"eligible_for_submission":False,"neural_training":False,"empirical_calibration":True,"gpu":False,"scope":"historical OOF cross-prompt diagnostic; upstream exposure not independently ruled out"})
if __name__=="__main__":main()
