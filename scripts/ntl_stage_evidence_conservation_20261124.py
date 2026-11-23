"""Frozen 0.715 decision-stage *candidate*, not a validated model improvement.

HYPOTHESIS: A GLM sibling-focused postprocessor can harm strong provider-family
recognition when it changes family. Preserve the pre-stage family on such
transitions; leave same-family sibling changes untouched. This is a one-shot,
parameter-free *structural falsifier*, NOT a claim that every family edit is bad.

No model fitting, no labels in decision logic, no Kaggle data in repository.
Never write to the original champion submission; use a distinct output path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

FAMILY = {
    "claude-": "Anthropic", "gemini-": "Google", "glm-": "Z.ai",
    "gpt-": "OpenAI", "kimi-": "Moonshot", "qwen": "Alibaba",
}

def family(label: str) -> str:
    for prefix, name in FAMILY.items():
        if label.startswith(prefix):
            return name
    raise ValueError("unknown label " + str(label))

def checked(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if list(df.columns) != ["id", "model"] or df.empty or df.isna().any().any():
        raise ValueError("invalid submission schema")
    df = df.copy()
    df["id"] = df.id.astype(str)
    df["model"] = df.model.astype(str)
    if df.id.duplicated().any():
        raise ValueError("duplicate IDs")
    for label in df.model:
        family(label)
    return df

def candidate(pre: pd.DataFrame, post: pd.DataFrame):
    if not np.array_equal(pre.id.to_numpy(), post.id.to_numpy()):
        raise ValueError("pre/post row IDs differ or are out of order")
    pre_family = np.array([family(x) for x in pre.model])
    post_family = np.array([family(x) for x in post.model])
    cross = pre_family != post_family
    sibling = (pre.model.to_numpy() != post.model.to_numpy()) & ~cross
    result = post.copy()
    result.loc[cross, "model"] = pre.loc[cross, "model"].to_numpy()
    counts = {
        "n_rows": int(len(pre)),
        "cross_family_vetoed": int(cross.sum()),
        "within_family_changes_retained": int(sibling.sum()),
        "candidate_vs_original_post_changed": int((result.model != post.model).sum()),
        "one_shot_fixed_rule": True,
        "model_training": False,
        "no_truth_accessed": True,
        "not_a_BA_claim": True,
    }
    return result, counts

def evaluate_optional(pre, post, treatment, truth: Path):
    """Only for genuinely independent labelled matched rows, NEVER Kaggle hidden."""
    gt = pd.read_parquet(truth)
    if list(gt.columns) != ["id", "model"] or gt.id.duplicated().any():
        raise ValueError("invalid ground truth")
    if not np.array_equal(gt.id.astype(str).to_numpy(), pre.id.to_numpy()):
        raise ValueError("truth ID mismatch")
    if not all(x in set(gt.model) for x in treatment.model):
        # All 18 labels must be represented in this evaluation contract.
        raise ValueError("ground truth does not cover all predicted labels")
    if len(set(gt.model)) != 18:
        raise ValueError("independent evaluation requires all 18 true labels")
    truth_labels = gt.model.to_numpy()
    old_correct = post.model.to_numpy() == truth_labels
    new_correct = treatment.model.to_numpy() == truth_labels
    labels = sorted(set(truth_labels))
    def ba(pred):
        return float(np.mean([np.mean(pred[truth_labels == label] == label) for label in labels]))
    return {
        "control_balanced_accuracy": ba(post.model.to_numpy()),
        "treatment_balanced_accuracy": ba(treatment.model.to_numpy()),
        "rescues": int(np.sum(~old_correct & new_correct)),
        "regressions": int(np.sum(old_correct & ~new_correct)),
        "true_classes": 18,
        "independent_source_provenance_not_automatically_verified": True,
    }

def self_test():
    a = pd.DataFrame({"id":["a","b","c"],"model":["glm-5.3","claude-opus-4-6","glm-5.3"]})
    b = pd.DataFrame({"id":["a","b","c"],"model":["glm-5.3-flash","glm-5.3","glm-5.3"]})
    c, d = candidate(a, b)
    assert c.model.tolist() == ["glm-5.3-flash","claude-opus-4-6","glm-5.3"]
    assert d["cross_family_vetoed"] == 1 and d["within_family_changes_retained"] == 1
    try:
        candidate(a,b.iloc[::-1].reset_index(drop=True))
    except ValueError: pass
    else: raise AssertionError("row-order safety failed")
    print("NTL_EVIDENCE_CONSERVATION_SYNTHETIC=PASS", flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pre",type=Path)
    p.add_argument("--post",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--truth",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    if not all((a.pre,a.post,a.out)):
        p.error("--pre --post --out required")
    if a.out.exists() or a.out.resolve() in (a.pre.resolve(),a.post.resolve()):
        raise FileExistsError("never overwrite frozen scorer or prior candidate")
    pre,post=checked(a.pre),checked(a.post)
    out, counts=candidate(pre,post)
    if a.truth:
        counts["labelled_evaluation"]=evaluate_optional(pre,post,out,a.truth)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    out.to_parquet(a.out,index=False)
    print("NTL_EVIDENCE_CONSERVATION_RESULT="+json.dumps(counts,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
