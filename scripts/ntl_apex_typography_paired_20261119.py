"""Source-faithful Apex typography ON vs OFF, paired and fail-closed.

No new model inference, no official submission, no new data source.
--smoke uses synthetic 18-class logits to prove only that the exact public
Apex code can reverse a label. It cannot measure BA, rescues, or regressions.

--paired requires *real* existing pre-GLM/pre-Apex base logits AND the actual
post-GLM incumbent hard labels for the SAME original sampled rows, plus truth.
This prevents pretending a hard prediction or a pseudo-logit is a model score.

The 0.702 kernel does not currently expose the required real arrays.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

SCORED_CODE_SHA = "2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc"
APEX_STRENGTH = 0.10
ORIGINAL_PENALTY = 0.15
NO_TYPOGRAPHY_PENALTY = 1.0


def extract_exact_source(notebook_path: Path):
    nb = json.loads(notebook_path.read_text(encoding="utf8"))
    cells = []
    for item in nb.get("cells", []):
        if item.get("cell_type") == "code":
            src = item.get("source", [])
            cells.append("".join(src) if isinstance(src, list) else str(src))
    if hashlib.sha256("\n".join(cells).encode()).hexdigest() != SCORED_CODE_SHA:
        raise RuntimeError("STOP_NOT_THE_AUTHENTIC_0702_SCORED_EXECUTABLE_SOURCE")
    modules = {}
    for name in ("pngy_residual.py", "apex_postprocess.py"):
        marker = "%%writefile " + name + "\n"
        matches = [cell for cell in cells if cell.startswith(marker)]
        if len(matches) != 1:
            raise RuntimeError("STOP_EXACTLY_ONE_SCORED_SOURCE_MODULE_REQUIRED:"+name)
        modules[name] = matches[0].split("\n", 1)[1]
    apex=modules["apex_postprocess.py"]
    if not all(t in apex for t in (
        "refined=refine(base,evidence,families,strength=strength)",
        "probs = prompt_pair_adjustment(softmax(refined, axis=1), texts)",
        "keep=np.isin(out,['glm-5.3','glm-5.3-flash'])",
        "probs[i,forbidden]*=penalty",
        "out[i]=labels[probs[i].argmax()]",
    )):
        raise RuntimeError("STOP_APEX_STAGE_SOURCE_CONTRACT_DRIFT")
    return modules


def invoke_exact(modules, run):
    # Import only the pinned, verified source code into an isolated temp dir.
    with tempfile.TemporaryDirectory(prefix="ntl-apex-original-src-") as root:
        for name, content in modules.items():
            (Path(root)/name).write_text(content, encoding="utf8")
        sys.path.insert(0, root)
        try:
            res=importlib.import_module("pngy_residual")
            apex=importlib.import_module("apex_postprocess")
            return run(res, apex)
        finally:
            sys.path.remove(root)
            for name in ("pngy_residual","apex_postprocess"):
                sys.modules.pop(name, None)


def family(label):
    for f in ("claude-","gpt-","gemini-","glm-","kimi-","qwen"):
        if label.startswith(f):
            return f.rstrip("-")
    raise RuntimeError("UNKNOWN_COMPETITION_LABEL:"+label)


def one_row_case(apex, res, labels, *, text, incumbent, challenger, win_prob, runner_prob):
    if incumbent==challenger:raise ValueError("MUST_HAVE_TWO_DISTINCT_CLASSES")
    if win_prob <= runner_prob:raise ValueError("INCUMBENT_MUST_START_TOP")
    z=np.full((1,len(labels)), -18., dtype=np.float64)
    z[0,labels.index(incumbent)]=np.log(win_prob)
    z[0,labels.index(challenger)]=np.log(runner_prob)
    ev=np.ones_like(z)
    families=res.family_columns(labels)
    original,_=apex.apply_arrays(
        z.copy(),ev.copy(),labels,families,[text],np.array([incumbent],dtype=object),
        strength=APEX_STRENGTH,corrected=False,penalty=ORIGINAL_PENALTY)
    disabled,_=apex.apply_arrays(
        z.copy(),ev.copy(),labels,families,[text],np.array([incumbent],dtype=object),
        strength=APEX_STRENGTH,corrected=False,penalty=NO_TYPOGRAPHY_PENALTY)
    return {"original":str(original[0]),"no_typography":str(disabled[0]),
            "predictions_differ":bool(original[0]!=disabled[0])}


def smoke(res,apex):
    labels=sorted([
        "claude-fable-5","claude-opus-4-6","claude-opus-4-8","claude-opus-5","claude-sonnet-5",
        "gemini-3.1-pro-preview","gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash",
        "glm-5.3","glm-5.3-flash",
        "gpt-5.5","gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra","gpt-6-astra",
        "kimi-k3","qwen3.8-max"
    ])
    # These are SYNTHETIC scores, not model outputs on generated passages.
    flip=one_row_case(apex,res,labels,text="A sentence — another idea.",incumbent="kimi-k3",
                      challenger="glm-5.3",win_prob=.60,runner_prob=.35)
    robust=one_row_case(apex,res,labels,text="A sentence — another idea.",incumbent="kimi-k3",
                        challenger="glm-5.3",win_prob=.97,runner_prob=.02)
    allowed=one_row_case(apex,res,labels,text="A sentence — another idea.",incumbent="claude-opus-5",
                         challenger="glm-5.3",win_prob=.60,runner_prob=.35)
    assert flip=={"original":"glm-5.3","no_typography":"kimi-k3","predictions_differ":True},flip
    assert robust["original"]==robust["no_typography"]=="kimi-k3",robust
    assert allowed["original"]==allowed["no_typography"]=="claude-opus-5",allowed
    threshold=np.log(1.0/ORIGINAL_PENALTY)
    return {"source_faithful_exact_module":True,
        "synthetic_flip_can_occur":flip,
        "synthetic_high_margin_no_flip":robust,
        "synthetic_allowed_class_no_flip":allowed,
        "logit_gap_flip_threshold_for_2_class_unprotected_vs_allowed":round(float(threshold),6),
        "no_inference_no_GPU_no_real_model_accuracy_measured":True,
        "official_Public_best_unchanged":0.702}


def paired(res,apex, train_csv:Path, query_csv:Path, base_npz:Path,
           current_parquet:Path):
    from sklearn.metrics import balanced_accuracy_score

    tr=pd.read_csv(train_csv,dtype={"id":str,"prompt_id":str})
    query=pd.read_csv(query_csv,dtype={"id":str})
    incumbent=pd.read_parquet(current_parquet)
    labels=sorted(tr.model.unique().tolist())
    if len(labels)!=18 or len(tr)!=2699 or tr.prompt_id.nunique()!=150:
        raise ValueError("TRAIN_SHA_AND_SCOPE_MUST_BE_VERIFIED_EXTERNALLY")
    if not {"id","text","model"}.issubset(query.columns) or query.id.duplicated().any():
        raise ValueError("REQUIRE_UNIQUE_LABELED_QUERY_WITH_TEXT")
    ids=query.id.astype(str).tolist()
    if incumbent.columns.tolist()!=["id","model"] or incumbent.id.astype(str).tolist()!=ids:
        raise ValueError("STOP_REAL_POSTV1_HARD_LABEL_ROW_ALIGNMENT_MISMATCH")
    with np.load(base_npz,allow_pickle=False) as z:
        if sorted(z.files)!=["ids","labels","logits"]:
            raise ValueError("STOP_REQUIRE_REAL_ID_LABEL_LOGIT_TRIPLE")
        if z["ids"].astype(str).tolist()!=ids:
            raise ValueError("STOP_REAL_PREAPEX_LOGIT_ROW_ALIGNMENT_MISMATCH")
        scorelabels=z["labels"].astype(str).tolist()
        if set(scorelabels)!=set(labels) or len(scorelabels)!=18:
            raise ValueError("STOP_REQUIRED_18_CLASS_SCORE_LABELS")
        base=np.asarray(z["logits"],dtype=np.float64)[:,[scorelabels.index(t) for t in labels]]
    if base.shape!=(len(query),18) or not np.isfinite(base).all():
        raise ValueError("STOP_BROKEN_REAL_PREAPEX_SCORE_MATRIX")
    if not set(query.model).issubset(set(labels)) or len(query)<216:
        raise ValueError("STOP_INSUFFICIENT_LABELED_EVIDENCE")
    model=res.Residual(labels).fit(tr.text.fillna("").astype(str),tr.model)
    x=res.features(query.text.fillna("").astype(str))
    evidence=np.ones_like(base)
    for cols,estimator in model.models:
        probs=estimator.predict_proba(x)
        for j,k in enumerate(estimator.classes_):
            evidence[:,labels.index(k)]=probs[:,j]
    texts=query.text.fillna("").astype(str).tolist()
    orig,_=apex.apply_arrays(base.copy(),evidence.copy(),labels,model.families,texts,
                             incumbent.model.to_numpy(),APEX_STRENGTH,False,ORIGINAL_PENALTY)
    no_typo,_=apex.apply_arrays(base.copy(),evidence.copy(),labels,model.families,texts,
                                incumbent.model.to_numpy(),APEX_STRENGTH,False,NO_TYPOGRAPHY_PENALTY)
    y=query.model.astype(str).to_numpy()
    a=orig==y;b=no_typo==y
    changed=orig!=no_typo
    def errors(pred):
        return {"correct":int((pred==y).sum()),
          "balanced_accuracy":float(balanced_accuracy_score(y,pred))}
    return {"study":"SOURCE_FAITHFUL_APEX_TYPOGRAPHY_PAIRED",
      "source":"Only valid when exact scored same-row existing logits are supplied",
      "rows":len(y),"label_count":len(labels),
      "original":errors(orig),"no_typography":errors(no_typo),
      "changed_predictions":int(changed.sum()),
      "rescues_no_typography":int((b&~a).sum()),
      "regressions_no_typography":int((~b&a).sum()),
      "changed_cross_family":int(sum(family(str(v))!=family(str(w))
                                   for v,w,flag in zip(orig,no_typo,changed) if flag)),
      "not_independent_transfer_by_itself":True,
      "no_GPU_no_submission":True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--notebook",type=Path,required=True)
    p.add_argument("--mode",choices=["smoke","paired"],default="smoke")
    p.add_argument("--train",type=Path)
    p.add_argument("--query",type=Path)
    p.add_argument("--base-logits",type=Path)
    p.add_argument("--current",type=Path)
    a=p.parse_args()
    mods=extract_exact_source(a.notebook)
    if a.mode=="paired" and not all((a.train,a.query,a.base_logits,a.current)):
        p.error("paired mode requires --train --query --base-logits --current")
    out=invoke_exact(mods,
        lambda res,apex:smoke(res,apex) if a.mode=="smoke" else
        paired(res,apex,a.train,a.query,a.base_logits,a.current))
    print("NTL_APEX_TYPOGRAPHY_EXACT_STAGE="+json.dumps(out,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
