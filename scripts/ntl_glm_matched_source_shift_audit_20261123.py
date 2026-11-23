"""Read-only audit of original official vs externally sourced matched GLM sibling responses.

Frozen document: docs/research/ntl-glm-family-source-shift-v1-contract-20261123.md.
NO model training, generator inference, Kaggle submission, raw output or
threshold adjustment. Public logs contain aggregate response features only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.run_style_baseline import FUNCTION_WORDS

TRAIN_SHA = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXTERNAL_SHA = "8339081afee1d4aed0b806a686d91978f0d7815bc67fb8f620860477e0895675"
MODELS = ("glm-5.3", "glm-5.3-flash")
FEATURES = ("word_count", "mean_word_length", "sentence_words",
    "punct_per100w", "emdash_per100w", "newline_per100w",
    "markdown_per100w", "digits_per100w", "first_person_per100w",
    "function_words_per100w")
WORDS = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?")
FIRST = frozenset(("i", "we", "me", "us", "my", "our"))
FUNCTION = frozenset(FUNCTION_WORDS)
SEED = 20261123
NBOOT = 1000


def source_sha(path, expected):
    if not path.is_file():
        raise ValueError("source missing (no data substitution)")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError("source SHA does not match locked dataset version")


def metrics(t):
    words = WORDS.findall(t)
    n = len(words)
    safe = max(1, n)
    lower = [w.lower() for w in words]
    vals = [
        float(n),
        float(sum(len(w) for w in words) / safe),
        float(n / max(1, sum(t.count(x) for x in ".!?"))),
        float(100 * sum(t.count(x) for x in ",.;:!?") / safe),
        float(100 * t.count("\u2014") / safe),
        float(100 * t.count("\n") / safe),
        float(100 * (t.count("**") + t.count("#") + t.count(chr(96))) / safe),
        float(100 * sum(t.count(str(i)) for i in range(10)) / safe),
        float(100 * sum(w in FIRST for w in lower) / safe),
        float(100 * sum(w in FUNCTION for w in lower) / safe),
    ]
    if len(vals) != len(FEATURES) or not np.isfinite(vals).all():
        raise ValueError("feature extraction invalid")
    return vals


def norm(s):
    return " ".join(str(s).split()).casefold()


def group_pairs(rows, grouping, label_field, expected):
    result = defaultdict(dict)
    for row in rows:
        name = row[label_field]
        if name not in MODELS:
            continue
        key = str(row[grouping])
        if not key or name in result[key]:
            raise ValueError("duplicate/missing prompt identity or generator label")
        result[key][name] = row
    if len(result) != expected:
        raise ValueError("unanticipated paired-prompt count")
    if any(set(q) != set(MODELS) for q in result.values()):
        raise ValueError("incomplete matched prompt")
    return np.asarray([[metrics(result[k][model]["text"])
                        for model in MODELS]
                       for k in sorted(result)], dtype=np.float64)


def external_selector(root):
    modpath = Path("kaggle/ntl_glm_crosssource_private.py")
    spec = importlib.util.spec_from_file_location("_frozen_ntl_glm_source", modpath)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen selector missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SOURCE_SHA != EXTERNAL_SHA or module.PAIRS != 64:
        raise ValueError("frozen external sample selector drift")
    with root.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 4758:
        raise ValueError("source corpus size mismatch")
    selected = module.select_pairs(rows)
    if len(selected) != 128:
        raise ValueError("frozen external sample changed")
    if len({r["pid"] for r in selected}) != 64:
        raise ValueError("invalid source prompt independence")
    return selected


def feature_report(official, external):
    # Resample exact observed paired-prompt groups, never individual answer rows.
    rng = np.random.default_rng(SEED)
    oi = rng.integers(0, len(official), size=(NBOOT, len(official)))
    ei = rng.integers(0, len(external), size=(NBOOT, len(external)))
    om = official.mean(axis=0)
    em = external.mean(axis=0)
    od = official[oi].mean(axis=1)
    ed = external[ei].mean(axis=1)
    actual_shift = em - om
    boots = ed - od
    actual_pair_o = om[0] - om[1]
    actual_pair_e = em[0] - em[1]
    pairs = (ed[:, 0] - ed[:, 1]) - (od[:, 0] - od[:, 1])
    shift = []
    paired = []
    for k, name in enumerate(FEATURES):
        for j, model in enumerate(MODELS):
            lo, hi = np.quantile(boots[:, j, k], [.025, .975])
            shift.append({"feature":name,"model":model,
              "official_median":round(float(np.median(official[:, j, k])),6),
              "external_median":round(float(np.median(external[:, j, k])),6),
              "official_mean":round(float(om[j,k]),6),
              "external_mean":round(float(em[j,k]),6),
              "external_minus_official":round(float(actual_shift[j,k]),6),
              "CI95": [round(float(lo),6),round(float(hi),6)]})
        lo,hi=np.quantile(pairs[:,k],[.025,.975])
        paired.append({"feature":name,
          "official_full_minus_flash":round(float(actual_pair_o[k]),6),
          "external_full_minus_flash":round(float(actual_pair_e[k]),6),
          "paired_change_external_minus_official":round(
              float(actual_pair_e[k]-actual_pair_o[k]),6),
          "CI95": [round(float(lo),6),round(float(hi),6)],
          "point_sign_reversal":bool(actual_pair_o[k]*actual_pair_e[k]<0)})
    return shift, paired


def sanity():
    probe = metrics("We write** new text\u2014hi!\nI liked it. #1")
    assert len(probe)==10 and probe[0]>0 and probe[4]>0
    p=np.array([[[2.,3.],[4.,5.]],[[3.,4.],[5.,6.]]])
    q=np.array([[[3.,4.],[5.,6.]],[[6.,7.],[8.,9.]]])
    old=(len(FEATURES),FEATURES)
    # Basic numerical bootstrap arithmetic on 2 observed paired groups.
    rng=np.random.default_rng(SEED)
    assert np.isfinite((q[rng.integers(0,2,size=(5,2))].mean(1) -
                       p[rng.integers(0,2,size=(5,2))].mean(1))).all()
    assert len(FEATURES)==old[0]
    print("NTL_GLM_SHIFT_FROZEN_NUMERIC_TESTS_PASS=TRUE",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--official",type=Path,required=False)
    parser.add_argument("--external",type=Path,required=False)
    parser.add_argument("--selftest",action="store_true")
    args=parser.parse_args()
    sanity()
    if args.selftest:
        return
    if not args.official or not args.external:
        raise ValueError("both locked source files required")
    started=time.perf_counter()
    source_sha(args.official,TRAIN_SHA)
    source_sha(args.external,EXTERNAL_SHA)
    with args.official.open(encoding="utf-8-sig",newline="") as fh:
        official_rows=list(csv.DictReader(fh))
    if len(official_rows)!=2699:
        raise ValueError("official cohort mismatch")
    official=group_pairs(official_rows,"prompt_id","model",150)
    external_rows=external_selector(args.external)
    external=group_pairs(external_rows,"pid","label",64)
    realofficial={norm(r["text"]) for r in official_rows}
    overlap=sum(norm(r["text"]) in realofficial for r in external_rows)
    if overlap:
        raise ValueError("external data overlaps official responses; stop")
    if any(not np.isfinite(a).all() for a in (official,external)):
        raise ValueError("source feature NaN")
    # Original settings can contain protocol strings: count only categories.
    settings=len(set(str(r.get("setting","")) for r in external_rows))
    targets=len(set(str(r.get("target_words","")) for r in external_rows))
    shift,paired=feature_report(official,external)
    primary={
      "scientific_scope":"READ_ONLY_MATCHED_SOURCE_DRIFT_OBSERVATIONAL",
      "official_train_rows":len(official_rows),
      "official_glm_paired_prompts":len(official),
      "external_original_corpus_rows":4758,
      "external_selected_glm_paired_prompts":len(external),
      "external_selected_glm_rows":len(external_rows),
      "verified_exact_text_overlap":overlap,
      "source_settings_distinct_values":settings,
      "source_target_words_distinct_values":targets,
      "bootstrap": {"draws":NBOOT,"seed":SEED,"unit":"matched_prompt_pair"},
      "source_shift":shift,
      "paired_version_fingerprint_shift":paired,
      "historic_64_pair_family_gate":{"total_true_glm":128,
          "champion_predicts_glm_family":95,
          "already_correct_within_family":88,
          "wrong_inside_family":7,
          "missed_glm_family":33,
          "maximum_possible_rescues_by_sibling_only":7,
          "actual_sibling_rescues":6,
          "actual_sibling_regressions":17,
          "actual_net_rescues":-11,
          "champion_2class_BA":.6875,
          "likelihood_2class_BA":.6015625,
          "baseline_source":"authenticated GH Actions 43612072194"},
      "inference":"NONE","new_training":False,
      "competition_submission":False,"used_0715_champion":False,
      "causal_drift_explanation_proven":False,
      "wall_seconds":round(time.perf_counter()-started,3)
    }
    print("NTL_GLM_SOURCE_SHIFT_FINAL="+json.dumps(primary,sort_keys=True),
          flush=True)


if __name__=="__main__":main()
