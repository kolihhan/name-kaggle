"""CPU-only forensic diagnostic of the **actual fitted Apex residual module**.

The comparator is the frozen 0.691 Oleg216 outputs, NOT scored 0.702; R/P0/P1
are explicitly **label-only proxies**, NOT the official Apex final inference.
Public Kaggle source and official train both pass SHA guards. No text/rows
or logits leave ephemeral runner memory. See PREREG.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from scripts.ntl_oleg18_frozen_selector_20261116 import (
    NTL_OLEG18_SHA, NTL_OLEG18_MODELS, ntl_oleg18_select,
)
from scripts.ntl_pro_hierarchical_family_vs_flat_20261116 import family

TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
CODE_SHA = "2343b692d9fbadccd585cd1f7149986f8130be4e8aacffb6cc9902d356b05ffc"
LABELS = tuple(sorted(NTL_OLEG18_MODELS))
GLM = {"glm-5.3", "glm-5.3-flash"}
SEED = 20261119
BOOTSTRAPS = 10_000


def load_apex_module(notebook_file: Path):
    nb = json.loads(notebook_file.read_text(encoding="utf-8"))
    cells = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            v = c.get("source", [])
            cells.append("".join(v) if isinstance(v, list) else str(v))
    joined = "\n".join(cells)
    if hashlib.sha256(joined.encode()).hexdigest() != CODE_SHA:
        raise ValueError("UNVERIFIED_0702_SCORED_CODE_SHA_DRIFT")
    matched = [c for c in cells if c.startswith("%%writefile pngy_residual.py\n")]
    if len(matched) != 1:
        raise ValueError("EXPECTED_ONE_APEX_RESIDUAL_SOURCE_BLOCK")
    source = matched[0].split("\n", 1)[1]
    # Only this pinned source block is imported. Do not run notebook cells.
    with tempfile.TemporaryDirectory(prefix="ntl-frozen-residual-") as td:
        module_path = Path(td) / "pngy_residual.py"
        module_path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("_frozen_apex_residual", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("CANNOT_IMPORT_PINNED_RESIDUAL")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    if not all(hasattr(module, name) for name in ("Residual", "features", "refine")):
        raise RuntimeError("INCOMPLETE_APEX_MODULE")
    # Source is already pinned by the complete champion notebook hash.
    # Avoid inspect.getsource after the temporary module directory is removed.
    print("NTL_APEX_EXACT_REFINER_CODE=" + json.dumps({
        "pinned_pngy_residual_source":source[:11500],
    },ensure_ascii=False,sort_keys=True),flush=True)
    apex_cells = [c for c in cells if c.startswith("%%writefile apex_postprocess.py\n")]
    if len(apex_cells) != 1:
        raise ValueError("EXPECTED_ONE_PINNED_APEX_POSTPROCESS_SOURCE")
    return module, hashlib.sha256(source.encode()).hexdigest(), apex_cells[0].split("\n", 1)[1]


def load_frozen_data(train_path, external_dir, baseline_path):
    if hashlib.sha256(train_path.read_bytes()).hexdigest() != TRAIN_SHA:
        raise ValueError("OFFICIAL_TRAIN_FINGERPRINT_DRIFT")
    for name, expected in NTL_OLEG18_SHA.items():
        path = external_dir / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("FROZEN_OLEG_SOURCE_DRIFT:" + name)
    tr = pd.read_csv(train_path, dtype={"id": str, "prompt_id": str})
    if (len(tr) != 2699 or tr.prompt_id.nunique() != 150
            or set(tr.model) != set(LABELS)):
        raise ValueError("TRAIN_ROWS_LABELS_PROMPTS_DRIFT")
    q = ntl_oleg18_select(
        pd.read_parquet(external_dir / "external_short_wide.parquet"),
        pd.read_parquet(external_dir / "external_cell_metadata.parquet")
    )
    if (len(q) != 216 or set(q.model) != set(LABELS) or
            any(q.model.value_counts().ne(12)) or q.prompt_group_id.nunique() != 216):
        raise ValueError("EXPECTED_FROZEN_216_18CLASS_PER_CLASS_12")
    p = pd.read_parquet(baseline_path)
    if len(p) != 216 or p.id.duplicated().any() or set(p.id) != set(q.id):
        raise ValueError("FROZEN_REFERENCE_ID_MISMATCH")
    if list(p.columns) != ["id", "model"]:
        raise ValueError("FROZEN_REFERENCE_SCHEMA_INVALID")
    p = q[["id"]].merge(p, on="id", validate="one_to_one")
    y = q.model.to_numpy(dtype=str)
    b = p.model.to_numpy(dtype=str)
    if int((b == y).sum()) != 89:
        raise ValueError("HISTORIC_STRONG_MODEL_REFERENCE_NOT_89_OVER_216")
    return tr, q, y, b


def residual_evidence(module, train, query):
    trained = module.Residual(list(LABELS)).fit(
        train.text.fillna("").astype(str),
        train.model
    )
    x = module.features(query.text.fillna("").astype(str))
    evidence = np.ones((len(query), len(LABELS)), dtype=np.float64)
    models = []
    for columns, estimator in trained.models:
        predicted = estimator.predict_proba(x)
        for i, label in enumerate(estimator.classes_):
            if str(label) not in LABELS:
                raise ValueError("UNRECOGNIZED_APEX_RESIDUAL_CLASS")
            evidence[:, LABELS.index(str(label))] = predicted[:, i]
        models.append({
            "estimator":type(estimator).__name__,
            "n_labels":len(estimator.classes_),
            "n_columns":len(columns),
        })
    if not models or not np.isfinite(evidence).all():
        raise ValueError("NO_VALID_APEX_RESIDUAL_EVIDENCE")
    residual = np.asarray(LABELS)[evidence.argmax(axis=1)]
    return residual, models, {
        "evidence_min":float(evidence.min()),
        "evidence_max":float(evidence.max()),
        "rows_with_unchanged_unit_evidence":int(np.any(evidence == 1.0, axis=1).sum()),
    }


def counts(y, p, base):
    truth = p == y
    original = base == y
    cross_err = np.array([family(p[i]) != family(y[i]) for i in range(len(y))])
    cross_change = np.array([family(p[i]) != family(base[i]) for i in range(len(y))])
    changed = p != base
    return {
        "correct":int(truth.sum()),
        "balanced_accuracy":round(float(balanced_accuracy_score(y, p)), 6),
        "wrong_cross_family":int((~truth & cross_err).sum()),
        "wrong_within_family":int((~truth & ~cross_err).sum()),
        "changed_rows":int(changed.sum()),
        "changed_cross_family":int((changed & cross_change).sum()),
        "changed_within_family":int((changed & ~cross_change).sum()),
        "rescues_vs_frozen_0691":int((truth & ~original).sum()),
        "regressions_vs_frozen_0691":int((~truth & original).sum()),
        "unchanged_correct":int((truth & original).sum()),
        "per_class_correct_of_12":{
            label:int((truth & (y == label)).sum()) for label in LABELS
        },
    }


def bootstrap(y, a, b):
    rng = np.random.default_rng(SEED)
    by_class = [np.flatnonzero(y == label) for label in LABELS]
    diffs = (a == y).astype(int) - (b == y).astype(int)
    estimated = np.zeros(BOOTSTRAPS, dtype=np.float64)
    for row in by_class:
        if len(row) != 12:
            raise ValueError("FROZEN_CLASS_BALANCE_DRIFT")
        selected = rng.choice(row, size=(BOOTSTRAPS, len(row)), replace=True)
        estimated += diffs[selected].mean(axis=1) / len(LABELS)
    return np.quantile(estimated, [.025, .975]).round(6).tolist()


def rule_structural_contradictions(query, truth, old_pred, apex_source):
    """Potential correct-label penalization by exact final Apex fixed rules.

    A trigger is *not* an observed prediction flip: the true 0.702 pre-Apex
    probability vector is missing, and the real stage uses post-refine labels.
    """
    required = (
        "allowed={'claude-opus-4-6','claude-opus-5','glm-5.3','glm-5.3-flash'}",
        "(' — ',np.array([x not in allowed for x in labels]))",
        "('’',np.array([x.startswith(('claude-','glm-')) for x in labels]))",
        "('—',np.array([x.startswith('gemini-') for x in labels]))",
        "probs[i,forbidden]*=penalty",
        "out[i]=labels[probs[i].argmax()]",
    )
    if any(part not in apex_source for part in required):
        raise ValueError("SCORING_SOURCE_TYPOGRAPHY_RULE_DRIFT")
    allowed = {"claude-opus-4-6","claude-opus-5","glm-5.3","glm-5.3-flash"}
    cases = [
        ("spaced_emdash", " — ", lambda lab: lab not in allowed),
        ("right_single_quote", "’", lambda lab: lab.startswith(("claude-","glm-"))),
        ("any_emdash", "—", lambda lab: lab.startswith("gemini-")),
    ]
    texts = query.text.fillna("").astype(str).to_numpy()
    y = np.asarray(truth, dtype=str)
    b = np.asarray(old_pred, dtype=str)
    n = len(texts)
    union_true = np.zeros(n,dtype=bool)
    union_base = np.zeros(n,dtype=bool)
    details = {}
    for name,marker,forbidden in cases:
        present = np.array([marker in v for v in texts],dtype=bool)
        truth_at_risk = present & np.array([forbidden(v) for v in y],dtype=bool)
        base_at_risk = present & np.array([forbidden(v) for v in b],dtype=bool)
        union_true |= truth_at_risk
        union_base |= base_at_risk
        details[name] = {
            "marker_rows":int(present.sum()),
            "true_label_forbidden_if_proposed":int(truth_at_risk.sum()),
            "frozen_correct_and_true_forbidden":int((truth_at_risk & (b==y)).sum()),
            "frozen_original_label_forbidden_if_proposed":int(base_at_risk.sum()),
            "false_negative_by_true_family":{
                fam:int((truth_at_risk & (np.array([family(v) for v in y])==fam)).sum())
                for fam in sorted({family(z) for z in LABELS})
            }
        }
    return {
        "source_verified_fixed_typography_rules":True,
        "penalty_if_triggered":0.15,
        "rules":details,
        "union_true_label_forbidden":int(union_true.sum()),
        "union_frozen_correct_and_true_label_forbidden":int((union_true&(b==y)).sum()),
        "union_frozen_prediction_rule_trigger":int(union_base.sum()),
        "union_true_by_label":{
            lab:int((union_true & (y==lab)).sum()) for lab in LABELS
        },
        "row_count":n,
        "no_actual_apex_flips_measured":True,
        "not_from_current_0702_logits":True,
    }


def refine_family_mass_invariant(module):
    """Exactly tests the actual pinned refine mathematical invariant."""
    from scipy.special import logsumexp
    rng = np.random.default_rng(20261119)
    z = rng.normal(0,2,size=(32,len(LABELS)))
    style = rng.uniform(.05,2,size=z.shape)
    f = module.family_columns(list(LABELS))
    after = module.refine(z.copy(),style.copy(),f,strength=.10)
    if after.shape != z.shape or not np.isfinite(after).all():
        raise ValueError("APEX_REFINE_INVALID_SHAPE_OR_NUMBERS")
    errors = [float(np.abs(logsumexp(z[:,cols],axis=1)-
                            logsumexp(after[:,cols],axis=1)).max())
              for cols in f if len(cols)>1]
    grouping = {int(col) for cols in f for col in cols}
    singles = sorted(set(range(len(LABELS)))-grouping)
    singleton_diff = float(np.abs(z[:,singles]-after[:,singles]).max()) if singles else 0.
    if max(errors,default=0)>1e-10 or singleton_diff>1e-12:
        raise AssertionError("EXACT_APEX_REFINE_FAMILY_MASS_PRESERVATION_FAILED")
    return {
        "tested_rows":32,"families":len(errors),
        "max_log_family_mass_change":float(max(errors,default=0)),
        "max_other_class_score_change":singleton_diff,
        "family_mass_preserved":True,
        "cross_family_movement_not_created_by_style_refine_itself":True,
        "synthetic_not_accuracy_evidence":True
    }


def run(args):
    module, residual_source_sha, apex_source = load_apex_module(args.pinned_notebook)
    train, query, y, b = load_frozen_data(args.train, args.external, args.reference)
    residual, fit_info, evidence_info = residual_evidence(module, train, query)
    typography = rule_structural_contradictions(query, y, b, apex_source)
    mass_invariant = refine_family_mass_invariant(module)
    # All policies below are label-only proxies because real 0.702 pre-Apex
    # per-class logits were not saved. Never misname as actual Apex output.
    base_family = np.array([family(x) for x in b])
    res_family = np.array([family(x) for x in residual])
    keep_glm = np.isin(b, list(GLM))
    p0 = np.where(keep_glm, b, residual)
    p1 = np.where(keep_glm | (base_family != res_family), b, residual)
    preds = {"B_frozen_0691":b, "R_residual_only":residual,
             "P0_nonGLM_replace_proxy":p0, "P1_family_veto_proxy":p1}
    metrics = {name:counts(y, pred, b) for name,pred in preds.items()}
    delta = {
        "P1_vs_P0_delta_BA":round(
            metrics["P1_family_veto_proxy"]["balanced_accuracy"] -
            metrics["P0_nonGLM_replace_proxy"]["balanced_accuracy"],6),
        "P1_vs_P0_rescues":int(((p1==y)&(p0!=y)).sum()),
        "P1_vs_P0_regressions":int(((p1!=y)&(p0==y)).sum()),
        "exploratory_95pct_stratified_row_bootstrap":bootstrap(y,p1,p0),
        "P0_vs_B_delta_BA":round(metrics["P0_nonGLM_replace_proxy"]["balanced_accuracy"]-
                                metrics["B_frozen_0691"]["balanced_accuracy"],6),
        "R_vs_B_delta_BA":round(metrics["R_residual_only"]["balanced_accuracy"]-
                               metrics["B_frozen_0691"]["balanced_accuracy"],6)
    }
    return {
        "study":"NTL_20261119_APEX_RESIDUAL_FORENSIC_SPENT_216",
        "truth_scope":"historically_spent_oleg216_source_narrow",
        "comparator":"older_frozen_0691_not_current_0702",
        "actual_0702_Apex_scores":"UNAVAILABLE_NO_BASE_LOGITS",
        "official_best_0702":0.702,
        "source_sha256":residual_source_sha,
        "official_train_rows":len(train), "diagnostic_rows":len(query),
        "labels":len(LABELS), "one_prompt_group_per_row":True,
        "residual_estimators":fit_info,
        "evidence_metadata":evidence_info,
        "actual_apex_refine_invariant":mass_invariant,
        "typography_source_contradictions":typography,
        "predictions":metrics, "comparison":delta,
        "no_Kaggle_GPU_or_submission":True,
        "no_new_external_data_search":True,
        "no_true_Apex_prediction_claim":True,
        "candidate_not_promotable_from_this_test":True,
        "next_gate":"Require real 0.702 pre-Apex logits plus true same-row data before any score claim",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinned-notebook",type=Path,required=True)
    parser.add_argument("--train",type=Path,default=Path("data/train.csv"))
    parser.add_argument("--external",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("NTL_APEX_FORENSIC_TERMINAL="+json.dumps({
        "scope":result["truth_scope"],
        "source_SHA":result["source_sha256"],
        "metrics":result["predictions"],
        "residual_estimators":result["residual_estimators"],
        "evidence_metadata":result["evidence_metadata"],
        "actual_apex_refine_invariant":result["actual_apex_refine_invariant"],
        "typography_source_contradictions":result["typography_source_contradictions"],
        "deltas":result["comparison"],
        "actual_0_702_Apex_scores":"UNAVAILABLE",
    },sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
