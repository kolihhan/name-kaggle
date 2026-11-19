"""Executed inside the owner's Colab T4 via GitHub Actions google-colab-cli ADC.

One frozen model revision, one full 2,699-row feature extraction, one
C0/C1/C2 grouped evaluation, aggregate-only result. No competition submit.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path("/content")
TRAIN=ROOT/"ntl_f1_train.csv"
FOLDS=ROOT/"ntl_f1_folds.csv"
EXTRACTOR=ROOT/"ntl_decoder_probe_extract_colab_20261119.py"
EVALUATOR=ROOT/"ntl_decoder_probe_eval_cpu_20261119.py"
FEATURES=ROOT/"ntl_f1_decoder_private_2699.npz"
REPORT=ROOT/"ntl_f1_decoder_aggregate_result.json"

def call(args,*,label,timeout=7200):
    print("NTL_F1_REMOTE_STAGE="+label,flush=True)
    subprocess.run(args,check=True,timeout=timeout)

def main():
    for f in (TRAIN,FOLDS,EXTRACTOR,EVALUATOR):
        if not f.is_file():raise RuntimeError("COLAB_UPLOADED_INPUT_MISSING:"+f.name)
    for f in (FEATURES,REPORT):
        if f.exists():raise RuntimeError("REFUSE_REUSING_SAVED_RESULT_OR_FEATURES")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["OMP_NUM_THREADS"]="2"
    os.environ["OPENBLAS_NUM_THREADS"]="2"
    os.environ["MKL_NUM_THREADS"]="2"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"]="1"
    call([sys.executable,"-m","pip","install","-q","--disable-pip-version-check",
         "transformers==4.57.1","huggingface-hub==0.36.0","accelerate==1.11.0",
         "safetensors==0.6.2","numpy==2.2.6","pandas==2.2.3",
         "scipy==1.15.3","scikit-learn==1.6.1"],label="PIN_DEPENDENCIES",timeout=300)
    call([sys.executable,str(EXTRACTOR),"--train",str(TRAIN),"--folds",str(FOLDS),
         "--output",str(FEATURES),"--batch-size","2"],
         label="REAL_PRETRAINED_QWEN3_1P7B_FOUR_LAYER_EXTRACTION",timeout=5700)
    call([sys.executable,str(EVALUATOR),"--train",str(TRAIN),"--folds",str(FOLDS),
         "--features",str(FEATURES),"--out",str(REPORT)],
         label="REAL_18CLASS_GROUPED_OOF_C0_C1_C2",timeout=1400)
    result=json.loads(REPORT.read_text())
    arms=result["results"]
    if len(arms)!=3 or result["rows"]!=2699 or result["prompts"]!=150:
        raise RuntimeError("REAL_DECODER_REPORT_FAIL_CLOSED")
    if abs(arms["C0_char_lr"]["ba"]-.291509)>1e-6:
        raise RuntimeError("REAL_COLAB_C0_CONTROL_DRIFT")
    print("NTL_F1_REAL_COLAB_T4_C0_C1_C2="+json.dumps({
        "ba":{k:round(v["ba"],6) for k,v in arms.items()},
        "comparisons":result["comparisons"],
        "decision":result["decision"],
        "official_public_ba_unchanged":0.702,
        "no_new_official_submission":True,
    },sort_keys=True),flush=True)
    FEATURES.unlink(missing_ok=True)
    print("NTL_F1_PRIVATE_FEATURES_REMOVED_FROM_COLAB=TRUE",flush=True)

if __name__=="__main__":
    main()
