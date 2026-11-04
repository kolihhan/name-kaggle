"""Colab T4: exact v3 worker interruption/resume equivalence on real external text.

Diagnostic only: predeclared frozen source split; no heldout evaluation,
no official competition data, no Kaggle score/submission.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/content/name-kaggle-v3-resume")
DATA = Path("/content/conservative_public_subset.parquet")
MODEL = Path("/content/v3-resume-qwen3-24")
PINNED_COMMIT = "d9bf4f0f88d396d32348e94033c07759c244ec6d"
OUT = Path("/content/v3_resume_result.json")
MAX_PROOF_STEPS = 2


def install_and_checkout():
    if not DATA.is_file():
        raise FileNotFoundError("Phase14 safe external-data artifact absent")
    if not (ROOT / ".git").is_dir():
        subprocess.run(["git", "init", str(ROOT)], check=True)
        subprocess.run(["git", "-C", str(ROOT), "remote", "add", "origin",
                        "https://github.com/kolihhan/name-kaggle.git"], check=True)
        subprocess.run(["git", "-C", str(ROOT), "sparse-checkout", "init", "--cone"], check=True)
        subprocess.run(["git", "-C", str(ROOT), "sparse-checkout", "set", "scripts"], check=True)
        subprocess.run(["git", "-C", str(ROOT), "fetch", "--depth=1", "--filter=blob:none",
                        "origin", PINNED_COMMIT], check=True)
        subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if head != PINNED_COMMIT:
        raise RuntimeError("worker source revision drift")
    for p in ("data/train.csv", "data/test.csv", "data/sample_submission.parquet"):
        if (ROOT / p).exists():
            raise RuntimeError("Official competition data in proof checkout")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==4.57.1", "peft==0.17.1", "tokenizers==0.22.1",
                    "huggingface-hub==0.36.0", "numpy", "pandas", "pyarrow",
                    "scipy", "scikit-learn", "safetensors", "accelerate"], check=True)
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def exact_equal(a, b, path="state") -> None:
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise AssertionError(f"Tensor resume divergence at {path}")
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.dtype != b.dtype or not np.array_equal(a, b):
            raise AssertionError(f"NumPy resume divergence at {path}")
    elif isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"Dict key drift at {path}")
        for k in a:
            exact_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (tuple, list)) and isinstance(b, type(a)):
        if len(a) != len(b):
            raise AssertionError(f"Sequence size drift at {path}")
        for i, (x, y) in enumerate(zip(a, b)):
            exact_equal(x, y, f"{path}[{i}]")
    elif a != b:
        raise AssertionError(f"Resume mismatch at {path}: {a!r} vs {b!r}")


def run():
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    # PEFT may probe a broken optional torchao package in Colab.
    import importlib.util as importlib_util
    original_find_spec = importlib_util.find_spec
    importlib_util.find_spec = (
        lambda name, *args, **kwargs: None
        if name.split(".")[0] == "torchao"
        else original_find_spec(name, *args, **kwargs)
    )
    import peft
    import peft.import_utils as peft_import_utils
    peft_import_utils.is_torchao_available = lambda *args, **kwargs: False

    from scripts.dct_v3_cpu_feasibility import _download_truncated_snapshot
    from scripts.dct_v3_transfer_split import build_transfer_split
    from scripts.materialize_dct_v3_worker import materialize

    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3:
        raise RuntimeError("Expected real T4 capacity")
    if transformers.__version__ != "4.57.1" or peft.__version__ != "0.17.1":
        raise RuntimeError("Frozen runtime drift")
    safe = pd.read_parquet(DATA)
    split = build_transfer_split(safe)
    for key, expected in {
        "train_rows": 14502, "heldout_rows": 14506,
        "eligible_label_count": 17, "prompt_id_overlap": 0,
        "prompt_key_overlap": 0,
        "selection_digest": "03edcd6d665d5d7992e777b1a83d88845260b06727bc8c7760e73355c0b0af29",
    }.items():
        if split[key] != expected:
            raise RuntimeError(f"Frozen split drift {key}")
    print("EXTERNAL_SPLIT_VERIFIED; no heldout predictions will be computed", flush=True)
    root, dl = _download_truncated_snapshot(MODEL)
    source = (ROOT / "scripts/run_two_stage_lora24_gate_a_gpu.py").read_text()
    rendered = materialize(source)
    generated = ROOT / "scripts/run_dct_v3_gate_a_gpu.py"
    generated.write_text(rendered)
    spec = importlib.util.spec_from_file_location("scripts.run_dct_v3_gate_a_gpu", generated)
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    spec.loader.exec_module(worker)
    worker.build_stage1_split = build_transfer_split
    from scripts.run_two_stage_lora24_gate_a import gate_a_training_spec
    if gate_a_training_spec()["keep_layers"] != 24:
        raise RuntimeError("Frozen model layer drift")

    def one_segment(checkpoint_path: Path, stop: int):
        worker._run_condition_worker(
            condition="A1",
            safe_path=DATA, snapshot_path=root,
            worker_output=ROOT / "MUST_NOT_SCORE_HELDOUT.json",
            checkpoint_dir=ROOT / "MUST_NOT_SAVE_FINAL",
            optimizer_boundary_path=checkpoint_path,
            checkpoint_every_steps=1,
            stop_after_optimizer_steps=stop,
        )
        if (ROOT / "MUST_NOT_SCORE_HELDOUT.json").exists() or (ROOT / "MUST_NOT_SAVE_FINAL").exists():
            raise RuntimeError("Proof leaked into heldout-scoring/final-stage code")
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Missing optimizer state checkpoint after step {stop}")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    uninterrupted = Path("/content/v3-proof-contiguous.pt")
    resumed = Path("/content/v3-proof-resumed.pt")
    for p in (uninterrupted, resumed):
        p.unlink(missing_ok=True)
    start = time.time()
    print("PROOF_UNINTERRUPTED_BEGIN", flush=True)
    one_segment(uninterrupted, MAX_PROOF_STEPS)
    baseline = torch.load(uninterrupted, map_location="cpu", weights_only=False)
    print("PROOF_INTERRUPTED_BEGIN", flush=True)
    one_segment(resumed, 1)
    print("PROOF_RECONSTRUCT_AND_RESUME_BEGIN", flush=True)
    one_segment(resumed, MAX_PROOF_STEPS)
    recovered = torch.load(resumed, map_location="cpu", weights_only=False)
    if baseline["completed_steps"] != MAX_PROOF_STEPS or recovered["completed_steps"] != MAX_PROOF_STEPS:
        raise RuntimeError("Training did not reach two optimizer updates")
    for key in ("fingerprint", "trainable_parameters", "optimizer",
                "scheduler", "grad_scaler", "training_stats", "rng"):
        exact_equal(baseline[key], recovered[key], key)
    result = {
        "scope": "actual_frozen_v3_worker_optimizer_boundary_resume_proof",
        "decision": "verified_bitwise_identical",
        "scientific_transfer_evaluated": False,
        "official_rows_scored": 0,
        "heldout_predictions_computed": False,
        "kaggle_submission_authorized": False,
        "source_commit": PINNED_COMMIT,
        "source_train_rows": int(split["train_rows"]),
        "source_heldout_rows": int(split["heldout_rows"]),
        "optimizer_steps_proven": MAX_PROOF_STEPS,
        "microbatches_processed_per_run": int(baseline["training_stats"]["loss_count"]),
        "trainable_parameter_tensors_verified": len(baseline["trainable_parameters"]),
        "optimizer_state_tensors_verified": len(baseline["optimizer"]["state"]),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "wall_seconds": round(time.time()-start,2),
    }
    OUT.write_text(json.dumps(result, indent=2,sort_keys=True))
    print("V3_CUDA_RESUME_BITWISE_PASS",json.dumps(result,sort_keys=True),flush=True)


if __name__ == "__main__":
    install_and_checkout()
    run()
