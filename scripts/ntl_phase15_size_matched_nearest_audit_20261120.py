"""Phase15 frozen-source nearest-neighbor falsification audit (CPU, no training).

This is a one-time post-hoc retrospective diagnostic, not a causal identification
of why one Kaggle Public score decreased. Match sample pool size before interpreting
similarity, do not log raw competition texts, and never call a near neighbor "same prompt"
without the original generation requests. TF-IDF is fit on pooled UNLABELED text only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.ntl_phase15_oof_prompt_leak_audit_20261118 import normalized_hash
from scripts.phase15_terms_clean import build_terms_clean_subset
from scripts.run_phase14_public_sparse import build_conservative_subset, sha256_file, EXPECTED_RAW_SHA256
from scripts.run_strong_sparse_anchor import load_train

RANDOM_SEED = 20261120
N_MATCHED_DRAWS = 30
THRESHOLDS = (0.75, 0.85, 0.95)
EXPECTED_EXT = 26025

def quantiles(values):
    a = np.asarray(values, dtype=np.float32)
    if not len(a): return None
    return {str(q): round(float(np.quantile(a, q)), 6) for q in (0.1, 0.5, 0.9, 0.99)}

def summarize(rows):
    if not rows:
        return {}
    ext = np.array([z["external_all"] for z in rows])
    official = np.array([z["official_other_prompt"] for z in rows])
    matched = np.array([z["external_matched_mean"] for z in rows])
    return {
        "rows": len(rows),
        "external_all_cosine_quantiles": quantiles(ext),
        "official_other_prompt_cosine_quantiles": quantiles(official),
        "external_size_matched_30draw_mean_cosine_quantiles": quantiles(matched),
        "external_all_beats_official_fraction": round(float((ext>official).mean()), 6),
        "external_size_matched_beats_official_fraction": round(float((matched>official).mean()), 6),
        "external_all_threshold_rates": {str(t):round(float((ext>=t).mean()),6) for t in THRESHOLDS},
        "official_threshold_rates": {str(t):round(float((official>=t).mean()),6) for t in THRESHOLDS},
        "external_size_matched_threshold_rates": {str(t):round(float((matched>=t).mean()),6) for t in THRESHOLDS},
    }

def audit(official, external, text_out):
    if len(official)!=2699 or official.prompt_id.nunique()!=150 or official.model.nunique()!=18:
        raise RuntimeError("OFFICIAL_INVARIANTS")
    if len(external)!=EXPECTED_EXT or external.model.nunique()!=11:
        raise RuntimeError("EXTERNAL_INVARIANTS")
    if not set(official.fold_id.astype(int))==set(range(5)):
        raise RuntimeError("FOLD_INVARIANTS")
    rng = np.random.default_rng(RANDOM_SEED)
    all_text = official.text.astype(str).tolist()+external.text.astype(str).tolist()
    v = TfidfVectorizer(
        analyzer="word",ngram_range=(1,2),stop_words="english",
        token_pattern=r"(?u)\b\w+\b",max_features=85000,min_df=2,
        max_df=0.98,sublinear_tf=True,dtype=np.float32,
        strip_accents="unicode",lowercase=True,
    )
    mat = v.fit_transform(all_text).tocsr()
    x_off=mat[:len(official)]
    x_ext=mat[len(official):]
    official_models=official.model.astype(str).to_numpy()
    external_models=external.model.astype(str).to_numpy()
    official_ids=official.id.astype(str).to_numpy()
    external_source = external["source"].astype(str).to_numpy() if "source" in external else np.array(["not_exposed"]*len(external))
    ext_label_idx={label:np.flatnonzero(external_models==label) for label in sorted(set(external_models))}
    official_hashes=set(official.text.map(normalized_hash))
    exact_overlap=int(external.text.map(normalized_hash).isin(official_hashes).sum())
    if exact_overlap:
        raise RuntimeError("PINNED_OLD_EXACT_OVERLAP_CHANGED")

    observed=[]
    best=[]
    fold_summaries=[]
    for fold in range(5):
        ids=np.flatnonzero(official.fold_id.to_numpy()==fold)
        fit=np.flatnonzero(official.fold_id.to_numpy()!=fold)
        if set(official.prompt_id.iloc[ids])&set(official.prompt_id.iloc[fit]):
            raise RuntimeError("OFFICIAL_GROUP_LEAK")
        nfit=len(fit)
        # 30 randomly drawn external subsets of EXACTLY the same size as the
        # available official train fold, independent of labels and scores.
        subsets=np.stack([rng.choice(len(external),size=nfit,replace=False) for _ in range(N_MATCHED_DRAWS)])
        fold_rows=[]
        for start in range(0,len(ids),40):
            bix=ids[start:start+40]
            score_ext=(x_off[bix] @ x_ext.T).toarray()
            score_off=(x_off[bix] @ x_off[fit].T).toarray()
            max_ext=score_ext.max(axis=1)
            arg_ext=score_ext.argmax(axis=1)
            max_off=score_off.max(axis=1)
            # 30x size-matched max, averaged per official response
            max_matched=np.mean(
                np.stack([score_ext[:,sub].max(axis=1) for sub in subsets]),
                axis=0
            )
            for j,idx in enumerate(bix):
                model=official_models[idx]
                ext_ix=int(arg_ext[j])
                record={
                    "fold":fold,
                    "true_model":model,
                    "external_all":float(max_ext[j]),
                    "official_other_prompt":float(max_off[j]),
                    "external_matched_mean":float(max_matched[j]),
                    "external_top_same_class":external_models[ext_ix]==model,
                    "external_top_source":str(external_source[ext_ix]),
                    "external_same_model_max":float(score_ext[j,ext_label_idx[model]].max())
                        if model in ext_label_idx else None,
                }
                observed.append(record)
                fold_rows.append(record)
                if max_ext[j]>=0.75:
                    best.append({
                        "id_hash_sha8":hashlib.sha256(official_ids[idx].encode()).hexdigest()[:8],
                        "official_class":model,
                        "neighbor_model":str(external_models[ext_ix]),
                        "neighbor_source":str(external_source[ext_ix])[:60],
                        "word_tfidf_cosine":round(float(max_ext[j]),6),
                        "official_other_prompt_max":round(float(max_off[j]),6),
                    })
        fold_summaries.append({"fold":fold,**summarize(fold_rows)})
        print("NTL_NEAREST_LEAK_FOLD_AGGREGATE="+json.dumps({
             "fold":fold,"rows":len(fold_rows),
             "ext_all_median":quantiles([z["external_all"] for z in fold_rows])["0.5"],
             "ext_equal_n_median":quantiles([z["external_matched_mean"] for z in fold_rows])["0.5"],
             "official_median":quantiles([z["official_other_prompt"] for z in fold_rows])["0.5"],
        }),flush=True)

    summary=summarize(observed)
    per_class={label:summarize([x for x in observed if x["true_model"]==label])
               for label in sorted(set(official_models))}
    by_class_nearest={label:{
       "external_nearest_same_true_label_fraction":round(float(np.mean([r["external_top_same_class"] for r in observed if r["true_model"]==label])),6),
       "external_same_class_max_cosine_quantiles":
           quantiles([r["external_same_model_max"] for r in observed if r["true_model"]==label
                      and r["external_same_model_max"] is not None])
    } for label in sorted(set(official_models))}
    near=sorted(best,key=lambda x:-x["word_tfidf_cosine"])
    result={
        "study":"NTL_PHASE15_WORD_CONTENT_NEAREST_NEIGHBOR_20261120",
        "model":"Unsupervised TFIDF word(1,2) English stopwords, pooled unlabeled fit; source-trained scorer NOT evaluated here",
        "source_version_sha":EXPECTED_RAW_SHA256,
        "n_official":len(official),"n_official_prompts":150,
        "n_external":len(external),"external_supported_classes":11,
        "n_equal_size_draws_per_fold":N_MATCHED_DRAWS,
        "random_seed":RANDOM_SEED,
        "exact_response_overlap":exact_overlap,
        "overall":summary,"folds":fold_summaries,"per_true_label":per_class,
        "per_label_neighbor_diagnostics":by_class_nearest,
        "nearest_source_distribution_top15":dict(Counter(x["external_top_source"] for x in observed).most_common(15)),
        "flagged_above_word_cosine_075_total":len(near),
        "flagged_top15_anonymized":near[:15],
        "matched_size_null_avoids_26k_vs_2k_maximum_bias":True,
        "similarity_is_not_prompt_identity":True,
        "source_alias_semantic_prompt_reuse_unresolved":True,
        "retrospective_used_train_oof_data":True,
        "no_gpu":True,"no_model_fit":True,"no_competition_submission":True,
        "no_raw_text_or_id_exported":True,
    }
    text_out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NTL_NEAREST_LEAK_SUMMARY="+json.dumps({
        "overall":summary,
        "exact_response_overlap":exact_overlap,
        "flagged_above_075":len(near),
        "external_top_source_top5":dict(Counter(x["external_top_source"] for x in observed).most_common(5)),
        "only_same_label_fraction":round(float(np.mean([x["external_top_same_class"] for x in observed])),6)
    },sort_keys=True),flush=True)

def main():
    a=argparse.ArgumentParser()
    for name in ("train","folds","pool","out"):
        a.add_argument("--"+name,type=Path,required=True)
    args=a.parse_args()
    if sha256_file(args.pool)!=EXPECTED_RAW_SHA256:
        raise RuntimeError("PHASE15_PINNED_SHA_DRIFT")
    official=load_train(args.train,args.folds)
    raw=pd.read_parquet(args.pool)
    external=build_terms_clean_subset(build_conservative_subset(raw,validate_full_contract=True),validate_full_contract=True)
    audit(official,external,args.out)
if __name__=="__main__":
    main()
