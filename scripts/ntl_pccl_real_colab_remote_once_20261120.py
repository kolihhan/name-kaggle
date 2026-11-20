"""Execute one PC-CL matched owner Colab T4 pilot. All input data ephemeral/private."""
import os,subprocess,sys,time
from pathlib import Path

ROOT=Path("/content")
TRAINER=ROOT/"ntl_pccl_17b_real_qlora_colab_20261120.py"
EXPECTED=("train.parquet","valid.parquet","batches.json","manifest.json")
def main():
    if not TRAINER.is_file() or any(not (ROOT/p).is_file() for p in EXPECTED):
        raise RuntimeError("COLAB_INPUTS_MISSING_NO_TRAIN")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["OMP_NUM_THREADS"]="2"
    os.environ["OPENBLAS_NUM_THREADS"]="2"
    os.environ["MKL_NUM_THREADS"]="2"
    print("NTL_PCCL_REMOTE_DEPENDENCIES_PIN_INSTALL_BEGIN",flush=True)
    pkgs=["transformers==4.57.1","huggingface-hub==0.36.0",
          "accelerate==1.11.0","peft==0.17.1","bitsandbytes==0.48.2",
          "safetensors==0.6.2","numpy==2.2.6","pandas==2.3.3",
          "pyarrow==22.0.0"]
    subprocess.run([sys.executable,"-m","pip","install","-q",
        "--disable-pip-version-check",*pkgs],check=True,timeout=450)
    print("NTL_PCCL_REMOTE_DEPENDENCIES_INSTALLED",flush=True)
    p=subprocess.run([sys.executable,str(TRAINER)],timeout=3000,check=False,
                     capture_output=True,text=True)
    # The Colab CLI may suppress a nested subprocess's stdout/stderr. The only
    # reusable diagnostic is the bounded exception tail; never print raw data.
    markers=[line for line in p.stdout.splitlines()
             if line.startswith("NTL_PCCL_")]
    for line in markers[-18:]:
        print(line[:1200],flush=True)
    if p.returncode:
        error_lines=p.stderr.splitlines()
        print("NTL_PCCL_PRIVATE_PYTHON_EXCEPTION_TAIL_BEGIN",flush=True)
        for line in error_lines[-28:]:
            print(line[:350],flush=True)
        print("NTL_PCCL_PRIVATE_PYTHON_EXCEPTION_TAIL_END",flush=True)
        print("NTL_PCCL_T4_TRAIN_FAILURE_NO_RETRY_NO_SCORE="+str(p.returncode),flush=True)
        raise SystemExit(p.returncode)
    result=ROOT/"ntl-pccl-nov20-result.json"
    if not result.is_file():
        raise RuntimeError("PCCL_COMPLETED_WITHOUT_BOTH_ARMS_AGGREGATE")
    import json
    r=json.loads(result.read_text())
    if len(r["arms"])!=2 or [x["arm"] for x in r["arms"]]!=["C0","C1"]:
        raise RuntimeError("PCCL_RESULTS_NOT_BOTH_REAL_ARMS")
    if r.get("no_competition_submit") is not True:
        raise RuntimeError("UNEXPECTED_COMPETITION_OUTPUT")
    print("NTL_PCCL_REAL_COLAB_TRAINED_PAIR_COMPLETE="+json.dumps({
        "C0":r["arms"][0]["BA18"],"C1":r["arms"][1]["BA18"],
        "delta":r["paired"]["delta_BA"],
        "CI95":r["paired"]["delta_prompt_cluster_CI95"],
        "rescues":r["paired"]["rescues"],"regressions":r["paired"]["regressions"],
        "decision":r["decision"]
    },sort_keys=True),flush=True)
if __name__=="__main__":
    main()
