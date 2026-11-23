"""Real prediction-disagreement audit for existing NTL Kaggle kernel outputs.

NO model inference, training, official submission, hidden labels, raw rows printed
or source-code evaluation. Outputs are ephemeral /tmp and aggregated only.
A kernel commit output is NOT necessarily the hidden competition scored rerun.
"""
from __future__ import annotations
import csv
import io
import json
import re
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

REFS={
    "owner_0715_source":"kolihhan/ntl-estopadilla-glm53-0716-repro-20261122",
    "sanpier_v3_source":"sanpier/0-709-320-token-glm-likelihood-qwen-ensemble/3",
    "dennis_source":"dennis6886/glm-user-prefix-guarded-fable-return",
    "areeb_0719_title":"areebbb/0-719-preserve-glm-to-fable-corrections",
    "hcluo_0719_title":"hcluo8888/ntl-0719-glm-prefix-fable-gate",
}
LABELS=[
 "claude-fable-5","claude-opus-4-6","claude-opus-4-8","claude-opus-5","claude-sonnet-5",
 "gemini-3.1-pro-preview","gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash",
 "glm-5.3","glm-5.3-flash","gpt-5.5","gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra",
 "gpt-6-astra","kimi-k3","qwen3.8-max"
]
def family(label):
    if label.startswith("claude-"):return "anthropic"
    if label.startswith("gemini-"):return "google"
    if label.startswith("glm-"):return "zai"
    if label.startswith("gpt-"):return "openai"
    if label.startswith("kimi-"):return "moonshot"
    if label.startswith("qwen-") or label.startswith("qwen3"):return "alibaba"
    raise ValueError("UNKNOWN_FAMILY:"+label)

def emit(tag,obj):
    print("NTL_REALPRED_"+tag+"="+json.dumps(obj,sort_keys=True,ensure_ascii=False),flush=True)

def fetch(ref,path):
    path.mkdir()
    try:
        p=subprocess.run(["kaggle","kernels","output",ref,"-p",str(path),"-o"],
                         capture_output=True,text=True,timeout=125)
    except Exception as e:
        return {"ok":False,"error_type":type(e).__name__}
    names=sorted(x.name for x in path.iterdir() if x.is_file())
    sub=path/"submission.parquet"
    if p.returncode or not sub.is_file():
        return {"ok":False,"returncode":p.returncode,
                "file_names":names[:18],
                "error_category":"output_not_accessible_or_submission_absent"}
    try:
        d=pd.read_parquet(sub)
    except Exception as e:
        return {"ok":False,"error_category":"parquet_unreadable","error_type":type(e).__name__}
    if set(d.columns)!={"id","model"} or d.id.duplicated().any():
        return {"ok":False,"error_category":"bad_schema"}
    d.id=d.id.astype(str)
    d.model=d.model.astype(str)
    if d.id.isna().any() or d.model.isna().any() or not set(d.model).issubset(LABELS):
        return {"ok":False,"error_category":"bad_labels"}
    return {"ok":True,"data":d,"n":len(d),"labels_n":int(d.model.nunique()),
            "output_files":names[:18]}

def compare(base,other,name):
    a=base["data"];b=other["data"]
    A=set(a.index);B=set(b.index)
    if A!=B:
        emit("DIFF",{"target":name,"comparable":False,
                     "reference_n":len(a),"target_n":len(b),
                     "shared_ids":len(A&B),"same_exact_id_set":False})
        return
    b=b.reindex(a.index)
    changed=a.model.to_numpy()!=b.model.to_numpy()
    fam_a=np.array([family(x) for x in a.model])
    fam_b=np.array([family(x) for x in b.model])
    across=changed&(fam_a!=fam_b)
    within=changed&(fam_a==fam_b)
    changed_n=int(changed.sum())
    results={
        "target":name,"comparable":True,"id_set_exact":True,
        "rows":len(a),"changed_predictions":changed_n,
        "changed_fraction":round(changed_n/len(a),6),
        "changed_across_family":int(across.sum()),
        "changed_within_family":int(within.sum()),
        "changed_from_glm_family":int((changed&(fam_a=="zai")).sum()),
        "changed_to_glm_family":int((changed&(fam_b=="zai")).sum()),
        "changed_glm_to_fable":int((changed&(fam_a=="zai")&
                    (b.model.to_numpy()=="claude-fable-5")).sum()),
        "changed_fable_to_glm":int((changed&(a.model.to_numpy()=="claude-fable-5")&
                    (fam_b=="zai")).sum()),
        "reference_predictions_by_family":{f:int((fam_a==f).sum())
                         for f in sorted(set(fam_a))},
        "target_predictions_by_family":{f:int((fam_b==f).sum())
                         for f in sorted(set(fam_b))},
        "no_ground_truth_available":True,"no_rescues_or_regressions_calculable":True,
        "Kaggle_kernel_output_may_be_commit_preview_NOT_hidden_scored_rerun":True,
        "no_accuracy_promotion":True
    }
    emit("DIFF",results)

def main():
    with tempfile.TemporaryDirectory(prefix="ntl-existing-public-output-") as td:
        t=Path(td);out={}
        for key,ref in REFS.items():
            obj=fetch(ref,t/key)
            if obj["ok"]:
                frame=obj.pop("data").set_index("id")
                obj["data"]=frame
                emit("FETCH",{"key":key,"requested_ref":ref,"ok":True,
                              "n_rows":obj["n"],"labels_n":obj["labels_n"],
                              "filenames":obj["output_files"],
                              "these_are_kernel_outputs_not_scored_competition_receipts":True})
            else:
                emit("FETCH",{"key":key,"requested_ref":ref,**obj})
            out[key]=obj
        base=out["owner_0715_source"]
        if base["ok"]:
            for key in REFS:
                if key=="owner_0715_source" or not out[key]["ok"]:continue
                compare(base,out[key],key)
        else:
            emit("NO_BASELINE",{"reason":"owner_kernel_output_missing_or_unreadable",
                  "no_claims_from_public_code_similarity":True})
    emit("END",{"run_complete":True,"no_training":True,"no_inference":True,
                "no_competition_submission":True,"raw_predictions_not_uploaded":True})
if __name__=="__main__":main()
