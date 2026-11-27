"""Current-lineage retained likelihood artifact admission and paired diagnostic.
No network calls other than authenticated Kaggle artifact retrieval.
No thresholds tuned on external publisher-labelled panel, no training, no submission.
"""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from ntl_fable252_full18_error_profile_20261125 import load_truth_and_predictions

KERNEL="kolihhan/ntl-fable-preservation-252-v2-20261125"
def run(args):
    p=subprocess.run(["kaggle",*args],capture_output=True,text=True,timeout=120)
    if p.returncode:
        reason=next((c for c in ("403","404","429","401","500","503") if c in p.stderr+p.stdout),"other")
        raise RuntimeError("Kaggle artifact retrieval failed: "+reason)
def log(key,x):print(key+"="+json.dumps(x,sort_keys=True),flush=True)
def inspect(path):
    a={}
    with np.load(path,allow_pickle=False) as z:
        for k in z.files:
            try:a[k]=z[k]
            except ValueError:continue
    log("REAL_CACHE_SCHEMA",{"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size,"fields":{k:{"shape":v.shape,"dtype":str(v.dtype)} for k,v in a.items()}})
    return a
def score(a,ids):
    # No positional matching. Refuse ambiguous columns and undocumented directions.
    matches=[]
    for k,v in a.items():
        if v.ndim==1 and v.dtype.kind in "US" and len(v)==len(ids) and len(set(v.astype(str)))==len(ids) and set(v.astype(str))==set(ids):
            matches.append(v.astype(str))
    if not matches or any(not np.array_equal(v,matches[0]) for v in matches):
        return None,"missing or ambiguous explicit row IDs"
    perm=pd.Index(matches[0]).get_indexer(ids)
    # Log-likelihood fields only; higher is better. Do not interpret arbitrary "score".
    candidates=[k for k in ("lp_mean","mean_logprob","mean_log_prob","mean_log_likelihood","avg_logprob") if k in a and a[k].shape==(len(ids),) and np.issubdtype(a[k].dtype,np.number)]
    if len(candidates)!=1:return None,"no unique explicit mean log-likelihood field"
    x=np.asarray(a[candidates[0]],float)[perm]
    if not np.isfinite(x).all() or (x>1e-6).any():return None,"invalid likelihood range"
    return x,None
def main():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);p=root/"panel";p.mkdir();o=root/"outputs";o.mkdir()
        run(["datasets","download","-d","nischaydnk/ntl-topic-synthetic-18llms","-p",str(p),"--unzip","-q"])
        run(["kernels","output",KERNEL,"-p",str(o),"--file-pattern","(^|/)(submission[.]parquet|ntl_fable_preservation_252_aggregate[.]json|pre_apex_submission[.]parquet|final-logits[.]npz|prefix-votes[.]npz|flash_scores[.]npz|glm53_scores[.]npz|report[.]json)$","-o"])
        files={k:next(iter(o.rglob(k)),None) for k in ("submission.parquet","ntl_fable_preservation_252_aggregate.json","pre_apex_submission.parquet","final-logits.npz","prefix-votes.npz","flash_scores.npz","glm53_scores.npz","report.json")}
        log("RETAINED_STAGES",{k:{"bytes":v.stat().st_size} if v else None for k,v in files.items()})
        assert files["submission.parquet"] and files["ntl_fable_preservation_252_aggregate.json"]
        df=load_truth_and_predictions(next(p.rglob("generations.csv")),files["submission.parquet"],files["ntl_fable_preservation_252_aggregate.json"])
        ids=df.id.to_numpy(str)
        y=df.model_true.to_numpy(str);pred=df.model_pred.to_numpy(str)
        correct=y==pred
        assert len(y)==252 and correct.sum()==151
        values={}
        for k in ("final-logits.npz","prefix-votes.npz","flash_scores.npz","glm53_scores.npz"):
            if not files[k]:continue
            a=inspect(files[k])
            if k.endswith("_scores.npz"):
                v,reason=score(a,ids)
                if v is not None:values[k]=v
                else:log("CACHE_NOT_ADMITTED",{"file":k,"reason":reason})
        if len(values)!=2:
            log("LIKELIHOOD_DIAGNOSTIC_BLOCKED","no pair of explicit row-aligned raw mean likelihoods")
        else:
            glm=("glm-5.3","glm-5.3-flash")
            a=values["glm53_scores.npz"];b=values["flash_scores.npz"]
            glmp=np.where(a>=b,glm[0],glm[1])
            # Fixed sibling-only correction on rows that the incumbent already calls GLM.
            # This is a descriptive counterfactual, not promotion on a spent panel.
            candidate=np.where(np.isin(pred,glm),glmp,pred)
            c=candidate==y
            # Within prompt contrasts to control for text topic. No use of test labels to construct predictions.
            affinity=np.maximum(a,b)
            ranks=[]
            for _,block in df.groupby(["version","topic_id"]):
                idx=block.index.to_numpy()
                g=idx[np.isin(y[idx],glm)];other=idx[~np.isin(y[idx],glm)]
                if len(g)==2:
                    ranks.extend(((affinity[g,None]>affinity[other]).astype(float)+.5*(affinity[g,None]==affinity[other])).ravel().tolist())
            log("CURRENT_LIKELIHOOD_RESULT",{"rows":len(y),"base_correct":int(correct.sum()),"sibling_only_correct":int(c.sum()),"rescues":int((~correct&c).sum()),"regressions":int((correct&~c).sum()),"within_prompt_glm_vs_other_rank":float(np.mean(ranks)) if ranks else None,"rank_pairs":len(ranks),"scope":"spent publisher panel, no independent truth certification"})
        log("AUDIT_DONE",{"gpu":False,"training":False,"submit":False})
if __name__=="__main__":main()
