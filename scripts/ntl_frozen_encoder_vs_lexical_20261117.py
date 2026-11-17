"""Source-disjoint dense representation vs identical-row lexical baseline.

This is a NEW representation-level test (not source balancing, no hidden labels).
Uses public pretrained sentence-transformers/all-MiniLM-L6-v2, frozen, with
one 18-way linear readout fitted on approved public rows. CPU only.

PREDECLARED:
- C0: exact historical char_wb(3,5) TF-IDF + class-balanced LinearSVC C=.5.
- C1: frozen mean-pooled MiniLM sentence embeddings + identical readout.
- Advance only if C1 improves same-row source-disjoint macro BA >=.030 AND
  improves at least 2 of the 3 heldout sources AND rescues > regressions.
- No Kaggle submission or Public score claim is authorized by this test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from scripts.ntl_source_shift_linear_control_20261117 import (
    SEED, make_weights, macro,
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_POOL_SHA = "5a3dc82288328abedf49b7dd5e866e5c69eb922f9d6008e6fd78cf5153c12332"
MAX_LEN = 192
BATCH = 24
SOURCE_REQUIRED = 3
MIN_GAIN = .03

def score(y, pred, labels, frame):
    by_source = {}
    for source in sorted(frame.source.unique()):
        ix = frame.source.eq(source).to_numpy()
        present = sorted(set(y[ix]))
        by_source[source] = {
            "rows": int(ix.sum()),
            "label_count": len(present),
            "macro_present": round(macro(y[ix], pred[ix], present), 6),
        }
    return {
        "BA": round(macro(y, pred, labels), 6),
        "per_label_recall": {
            l: round(float(np.mean(pred[y == l] == l)), 6) for l in labels
        },
        "by_source": by_source,
    }

def encode(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
    net = AutoModel.from_pretrained(MODEL, trust_remote_code=False).cpu().eval()
    out = []
    start = time.monotonic()
    with torch.inference_mode():
        for i in range(0, len(texts), BATCH):
            tokens = tokenizer(texts[i:i+BATCH], padding=True, truncation=True,
                               max_length=MAX_LEN, return_tensors="pt")
            a = net(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            vec = (a * mask).sum(1) / mask.sum(1).clamp(min=1)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out.append(vec.float().numpy())
            if (i // BATCH) % 80 == 0:
                print("NTL_ENCODER_CPU_PROGRESS=" + json.dumps({
                    "done": min(i+BATCH, len(texts)),
                    "n": len(texts),
                    "seconds": round(time.monotonic()-start,1)
                }), flush=True)
    return np.concatenate(out).astype(np.float32, copy=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    d=args.data_dir
    manifest=json.loads((d/"manifest.json").read_text())
    if manifest["data_sha256"]!=EXPECTED_POOL_SHA:
        raise RuntimeError("SOURCE_DATA_DRIFT")
    for name in ("train","valid"):
        from scripts.ntl_source_robust_18_trainprep_20261117 import sha
        if sha(d/(name+".parquet"))!=manifest[name+"_sha256"]:
            raise RuntimeError("SPLIT_SHA_DRIFT_"+name)
    tr=pd.read_parquet(d/"train.parquet")
    va=pd.read_parquet(d/"valid.parquet")
    labels=list(manifest["labels"])
    if len(labels)!=18 or set(tr.model)!=set(labels) or set(va.model)!=set(labels):
        raise RuntimeError("INVALID_18_LABEL_COVERAGE")
    if set(tr.source)&set(va.source) or set(tr.prompt_id)&set(va.prompt_id):
        raise RuntimeError("SOURCE_OR_PROMPT_LEAKAGE")
    y=va.model.to_numpy()
    sample_weight=make_weights(tr,False)
    lexical=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,
                            max_features=180000,sublinear_tf=True,dtype=np.float32)
    train_texts=tr.text.astype(str).tolist()
    valid_texts=va.text.astype(str).tolist()
    a=lexical.fit_transform(train_texts)
    b=lexical.transform(valid_texts)
    c0=LinearSVC(C=.5,random_state=SEED,dual="auto",max_iter=3000)
    c0.fit(a,tr.model.to_numpy(),sample_weight=sample_weight)
    pred0=c0.predict(b)
    report0=score(y,pred0,labels,va)
    # MUST reproduce exact previously spent C0 baseline for same frozen data.
    if abs(report0["BA"]-0.213704)>1e-5:
        raise RuntimeError("EXACT_FROZEN_CONTROL_DID_NOT_REPRODUCE: "+str(report0["BA"]))
    del a,b,c0,lexical
    combined = train_texts+valid_texts
    dense=encode(combined)
    if dense.shape!=(len(combined),384) or not np.isfinite(dense).all():
        raise RuntimeError("BAD_FROZEN_EMBEDDINGS")
    train_vec=dense[:len(tr)]
    valid_vec=dense[len(tr):]
    c1=LinearSVC(C=.5,random_state=SEED,dual="auto",max_iter=3000)
    c1.fit(train_vec,tr.model.to_numpy(),sample_weight=sample_weight)
    pred1=c1.predict(valid_vec)
    report1=score(y,pred1,labels,va)
    delta=round(report1["BA"]-report0["BA"],6)
    rescues=int(((pred0!=y)&(pred1==y)).sum())
    regressions=int(((pred0==y)&(pred1!=y)).sum())
    by_source={s:round(report1["by_source"][s]["macro_present"]
                       -report0["by_source"][s]["macro_present"],6)
               for s in report0["by_source"]}
    passes=(delta>=MIN_GAIN and sum(v>0 for v in by_source.values())>=2
            and rescues>regressions and len(by_source)>=SOURCE_REQUIRED)
    report = {
        "experiment":"PRETRAINED_FROZEN_DENSE_VS_CHAR_LEXICAL_NOV17",
        "source": "crtproduction/name-that-llm-open-pool",
        "data_sha256":manifest["data_sha256"],
        "train_sha256":manifest["train_sha256"],
        "valid_sha256":manifest["valid_sha256"],
        "encoder":MODEL,
        "encoder_trainable":False,
        "linear_readout_trained":True,
        "train_n":len(tr),"valid_n":len(va),
        "all_18_truth_classes":True,
        "source_disjoint":True,
        "prompt_disjoint":True,
        "model_version_independently_audited":False,
        "champion_same_row_score_available":False,
        "not_kaggle_public_score":True,
        "new_kaggle_submission":False,
        "arms":{"C0_char_svc":report0,"C1_frozen_MiniLM":report1},
        "delta_BA":delta,
        "rescues":rescues,
        "regressions":regressions,
        "source_deltas":by_source,
        "decision":"ADVANCE_ONLY_TO_INDEPENDENT_SAME_ROW_PAIRED_CHECK" if passes else "KILL_DENSE_REPRESENTATION",
        "hypothesis":"+0.03 BA on source-disjoint from frozen dense pretrained representation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("NTL_FROZEN_DENSE_VS_CHAR_REAL_SOURCE_DISJOINT="+json.dumps({
        "C0_BA":report0["BA"],"C1_BA":report1["BA"],"delta_BA":delta,
        "rescues":rescues,"regressions":regressions,"source_deltas":by_source,
        "decision":report["decision"],
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
