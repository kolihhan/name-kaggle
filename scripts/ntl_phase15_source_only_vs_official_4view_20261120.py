"""One-shot retrospective Phase15 source-only transfer and official-only control.
Same fixed four-view TFIDF/LinearSVC recipe as Phase14/15; no parameter tuning.
C0: official-only 5-fold GroupKFold (true official-only OOF).
C1: ONLY legal-filtered Phase15 external 26,025 rows, fit once, test on official
    2,699 TRAIN rows (all official rows unseen for C1, but independent prompt
    origins/serving are unverified). C1 has ONLY 11 train labels, so a full
    18-class BA is capped by 11/18, and should not be compared as a symmetric
    alternative to the full 18-class C0/C2.
C2 historical official+external 0.405322 is a reference, NOT rerun.
"""
from __future__ import annotations
import argparse,gc,json
from pathlib import Path
import numpy as np,pandas as pd
from scripts.run_phase14_public_sparse import (
    EXPECTED_RAW_SHA256, build_conservative_subset, sha256_file,
    build_fold_features, make_svc
)
from scripts.phase15_terms_clean import build_terms_clean_subset
from scripts.run_strong_sparse_anchor import load_train

def summarize(true, pred, labs):
    true=np.asarray(true,dtype=str);pred=np.asarray(pred,dtype=str)
    n={l:int((true==l).sum()) for l in labs}
    per={l:float((pred[true==l]==l).mean()) if n[l] else None for l in labs}
    return {
      "18_label_BA":round(float(np.mean(list(per.values()))),6),
      "per_model_recall":{l:round(v,6) for l,v in per.items()},
      "N":len(true),
      "prediction_label_count":int(len(set(pred))),
      "correct_total":int((true==pred).sum()),
    }

def main():
    p=argparse.ArgumentParser()
    for name in ("train","folds","pool","out"):p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if sha256_file(args.pool)!=EXPECTED_RAW_SHA256: raise RuntimeError("PINNED_POOL_SHA_DRIFT")
    train=load_train(args.train,args.folds)
    raw=pd.read_parquet(args.pool)
    ext=build_terms_clean_subset(build_conservative_subset(raw),validate_full_contract=True)
    if len(ext)!=26025 or len(train)!=2699:raise RuntimeError("FROZEN_SAMPLES_CHANGED")
    labs=sorted(set(train.model.astype(str)))
    eligible=sorted(set(ext.model.astype(str)))
    if len(labs)!=18 or len(eligible)!=11:raise RuntimeError("FROZEN_CLASSES_CHANGED")

    # Exactly same Phase14/15 fitted four-view classifier; only fit sample differs.
    x_train,x_off,meta=build_fold_features(ext.text.astype(str).tolist(),train.text.astype(str).tolist())
    m=make_svc()
    m.fit(x_train,ext.model.astype(str).tolist())
    predicted_ext=m.predict(x_off)
    source_only=summarize(train.model,predicted_ext,labs)
    source_only["supported11_BA"]=round(float(np.mean([source_only["per_model_recall"][k] for k in eligible])),6)
    source_only["unsupported7_predicted_correct_by_design"]=0
    source_only["external_training_rows"]=len(ext)
    del x_train,x_off,m,raw,ext
    gc.collect()
    print("NTL_PHASE15_EXTERNAL_ONLY_TRANSFER_RESULT="+json.dumps({
      "BA18":source_only["18_label_BA"],"BA11":source_only["supported11_BA"],
      "n":source_only["N"],"correct":source_only["correct_total"]
    }),flush=True)

    # Honest official-only GroupKFold: same feature recipe and hyperparams.
    oo=np.full(len(train),"",dtype=object)
    fold_s=[]
    for fold in range(5):
        fit_idx=np.flatnonzero(train.fold_id.to_numpy()!=fold)
        val_idx=np.flatnonzero(train.fold_id.to_numpy()==fold)
        fit=train.iloc[fit_idx];val=train.iloc[val_idx]
        if set(fit.prompt_id.astype(str))&set(val.prompt_id.astype(str)):raise RuntimeError("PROMPT_SPLIT_LEAK")
        x_train,x_val,meta=build_fold_features(fit.text.astype(str).tolist(),val.text.astype(str).tolist())
        model=make_svc()
        model.fit(x_train,fit.model.astype(str).tolist())
        oo[val_idx]=model.predict(x_val)
        fold_score=summarize(val.model,oo[val_idx],labs)
        fold_s.append({"fold":fold,"rows":len(val_idx),"BA18":fold_score["18_label_BA"]})
        print("NTL_PHASE15_OFFICIAL_ONLY_FOLD="+json.dumps(fold_s[-1]),flush=True)
        del x_train,x_val,model
        gc.collect()
    official_only=summarize(train.model,oo,labs)
    result={
      "study":"NTL_PHASE15_OFFICIAL_ONLY_VS_EXTERNAL_ONLY_FOUR_VIEW_SVC",
      "official_train":2699,"official_prompts":150,
      "frozen_pool_sha":EXPECTED_RAW_SHA256,
      "external_training_rows":26025,"external_training_classes":eligible,
      "same_phase14_fourview_featurization_and_svc":True,
      "official_only_true_prompt_disjoint_oof":official_only,
      "official_only_per_fold":fold_s,
      "external_only_no_official_training_rows":source_only,
      "historical_official_plus_external_phase15_true_grouped_oof_BA":0.4053218990802883,
      "historical_phase14_18class_all_source_oof_BA":0.4178100919711658,
      "source_only_missing_7_output_classes_cannot_make_full_18class_prediction":True,
      "source_near_duplicate_and_serving_match_unverified":True,
      "retrospective_spent_official_prompts":True,
      "no_parameter_tuning":True,
      "no_public_submission":True,"no_gpu":True,
      "no_raw_predictions_or_text_exported":True,
    }
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("NTL_PHASE15_ABLATION_AGGREGATE="+json.dumps({
       "official_only_BA18":official_only["18_label_BA"],
       "external_only_BA18":source_only["18_label_BA"],
       "external_only_supported11_BA":source_only["supported11_BA"],
       "original_phase15_BA18":0.4053218990802883,
       "no_new_Kaggle_submission":True
    },sort_keys=True),flush=True)

if __name__=="__main__":main()
