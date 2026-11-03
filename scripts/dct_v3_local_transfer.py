from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.dct_v3_source_eval import evaluate_source_transfer
from scripts.dct_v3_transfer_split import build_transfer_split
from scripts.materialize_dct_v3_worker import materialize
from scripts.run_phase14_public_sparse import build_conservative_subset
import scripts.run_two_stage_lora24_gate_a_gpu as v1_worker

CANDIDATE = "dct-temporal-fingerprint-v3"
SCOPE = "paired_external_source_transfer_no_official_rows_no_public_lb"
POOL_DATASET_REF = "nishantdahal/ntl-handwriting-blend"
POOL_FILENAME = "pool_v4.parquet"
EXPECTED_POOL_SHA256 = "37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec"
EXPECTED_HOLDOUT_SOURCES = ["alperen", "hailey", "willrice"]
EXPECTED_TRAIN_ROWS = 14502
EXPECTED_HELDOUT_ROWS = 14506
EXPECTED_ELIGIBLE_ROWS = 14506
EXPECTED_ELIGIBLE_LABELS = 17
MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_snapshot(root: Path) -> dict[str, object]:
    root = Path(root)
    config_path = root / "config.json"
    tokenizer_config = root / "tokenizer_config.json"
    weights = sorted(root.glob("*.safetensors"))
    if not config_path.is_file() or not tokenizer_config.is_file() or not weights:
        raise RuntimeError(f"incomplete local model snapshot: {root}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    observed = {
        "model_type": payload.get("model_type"),
        "hidden_size": int(payload.get("hidden_size", -1)),
        "num_hidden_layers": int(payload.get("num_hidden_layers", -1)),
    }
    expected = {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36}
    if observed != expected:
        raise RuntimeError(f"unexpected local model config: {observed} != {expected}")
    observed["weight_files"] = len(weights)
    observed["root"] = str(root)
    return observed


def validate_local_result_payload(payload: dict[str, Any]) -> dict[str, object]:
    exact = {
        "candidate": CANDIDATE,
        "scope": SCOPE,
        "pool_sha256": EXPECTED_POOL_SHA256,
        "holdout_sources": EXPECTED_HOLDOUT_SOURCES,
        "eligible_rows": EXPECTED_ELIGIBLE_ROWS,
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "v3_stage1_checkpoint_preserved": True,
    }
    for key, expected in exact.items():
        observed = payload.get(key)
        if observed != expected:
            if key == "official_rows_scored":
                raise RuntimeError(f"official rows contamination: {observed}")
            raise RuntimeError(f"local result lineage drift for {key}: {observed!r} != {expected!r}")
    decision = str(payload.get("decision", "")).lower()
    if decision not in {"pass", "fail"}:
        raise RuntimeError(f"unexpected scientific decision: {decision!r}")
    return {
        "candidate": CANDIDATE,
        "decision": decision,
        "eligible_rows": EXPECTED_ELIGIBLE_ROWS,
        "submission_authorized": False,
    }


def _run(args: list[str]) -> None:
    subprocess.run(args, check=True)


def ensure_pool(pool_path: Path | None, cache_dir: Path) -> Path:
    if pool_path is not None:
        candidate = Path(pool_path).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        candidate = cache_dir / POOL_FILENAME
        if not candidate.is_file():
            _run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    POOL_DATASET_REF,
                    "-f",
                    POOL_FILENAME,
                    "-p",
                    str(cache_dir),
                    "--unzip",
                ]
            )
    observed = _sha256_file(candidate)
    if observed != EXPECTED_POOL_SHA256:
        raise RuntimeError(f"pool SHA drift: {observed} != {EXPECTED_POOL_SHA256}")
    return candidate


def ensure_model_snapshot(model_path: Path | None, cache_dir: Path) -> Path:
    if model_path is not None:
        root = Path(model_path).resolve()
    else:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(cache_dir),
            )
        )
    verify_model_snapshot(root)
    return root


def _load_generated_v3(path: Path):
    spec = importlib.util.spec_from_file_location("_dct_v3_generated_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import materialized v3 worker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_safe_pool(pool_path: Path, safe_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(pool_path)
    safe = build_conservative_subset(raw, validate_full_contract=True).reset_index(drop=True)
    split = build_transfer_split(safe)
    checks = {
        "holdout_sources": EXPECTED_HOLDOUT_SOURCES,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "heldout_rows": EXPECTED_HELDOUT_ROWS,
        "eligible_heldout_rows": EXPECTED_ELIGIBLE_ROWS,
        "eligible_label_count": EXPECTED_ELIGIBLE_LABELS,
        "prompt_id_overlap": 0,
        "prompt_key_overlap": 0,
    }
    for key, expected in checks.items():
        observed = split.get(key)
        if observed != expected:
            raise RuntimeError(f"frozen transfer split drift for {key}: {observed!r} != {expected!r}")
    if not bool(split.get("valid_for_transfer_gate")):
        raise RuntimeError("frozen transfer split is invalid")
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe.to_parquet(safe_path, index=False)
    return safe, split


def _run_and_capture(
    module: Any,
    *,
    name: str,
    safe_path: Path,
    snapshot_path: Path,
    worker_output: Path,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    captured: dict[str, Any] = {}
    original_split = module.build_stage1_split
    original_score = module._score_condition
    module.build_stage1_split = build_transfer_split

    def capture(frame, heldout_idx, predictions, eligible_labels, holdout_sources):
        captured["heldout_idx"] = np.asarray(heldout_idx, dtype=np.int64).copy()
        captured["predictions"] = list(predictions)
        captured["eligible_labels"] = list(eligible_labels)
        captured["holdout_sources"] = list(holdout_sources)
        return original_score(frame, heldout_idx, predictions, eligible_labels, holdout_sources)

    module._score_condition = capture
    try:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        module._run_condition_worker(
            condition="A1",
            safe_path=safe_path,
            snapshot_path=snapshot_path,
            worker_output=worker_output,
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        module.build_stage1_split = original_split
        module._score_condition = original_score
    if not captured:
        raise RuntimeError(f"{name} worker did not expose row-level predictions")
    result = json.loads(worker_output.read_text(encoding="utf-8"))
    return captured, result


def _cuda_summary() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("self-hosted v3 transfer requires a CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    memory_gb = float(props.total_memory) / (1024**3)
    if memory_gb < 15.0:
        raise RuntimeError(f"CUDA GPU has insufficient VRAM for frozen recipe: {memory_gb:.1f} GiB")
    return {
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "gpu": str(props.name),
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gib": memory_gb,
        "device_count": int(torch.cuda.device_count()),
    }


def run_local_transfer(
    *,
    work_dir: Path,
    checkpoint_dir: Path,
    pool_path: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, object]:
    work_dir = Path(work_dir).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    runtime = _cuda_summary()

    pool = ensure_pool(pool_path, work_dir / "data")
    safe_path = work_dir / "phase14_safe_transfer.parquet"
    safe, split = _prepare_safe_pool(pool, safe_path)
    snapshot = ensure_model_snapshot(model_path, work_dir / "hf-cache")

    generated_path = work_dir / "run_dct_v3_gate_a_gpu.py"
    source_path = Path(v1_worker.__file__).resolve()
    generated_path.write_text(materialize(source_path.read_text(encoding="utf-8")), encoding="utf-8")
    v3_worker = _load_generated_v3(generated_path)

    v1_checkpoint = work_dir / "v1_stage1_checkpoint_disposable"
    v1_capture, v1_result = _run_and_capture(
        v1_worker,
        name="v1",
        safe_path=safe_path,
        snapshot_path=snapshot,
        worker_output=work_dir / "v1_worker.json",
        checkpoint_dir=v1_checkpoint,
    )
    if v1_checkpoint.exists():
        shutil.rmtree(v1_checkpoint)
    gc.collect()

    v3_capture, v3_result = _run_and_capture(
        v3_worker,
        name="v3",
        safe_path=safe_path,
        snapshot_path=snapshot,
        worker_output=work_dir / "v3_worker.json",
        checkpoint_dir=checkpoint_dir,
    )

    if not np.array_equal(v1_capture["heldout_idx"], v3_capture["heldout_idx"]):
        raise RuntimeError("v1/v3 heldout row alignment differs")
    if v1_capture["eligible_labels"] != v3_capture["eligible_labels"]:
        raise RuntimeError("v1/v3 eligible labels differ")
    if v1_capture["holdout_sources"] != EXPECTED_HOLDOUT_SOURCES:
        raise RuntimeError(f"v1 holdout source drift: {v1_capture['holdout_sources']}")
    if v3_capture["holdout_sources"] != EXPECTED_HOLDOUT_SOURCES:
        raise RuntimeError(f"v3 holdout source drift: {v3_capture['holdout_sources']}")

    heldout_idx = v1_capture["heldout_idx"]
    eval_frame = safe.iloc[heldout_idx][["source", "prompt_id", "model"]].copy().reset_index(drop=True)
    eval_frame["v1_prediction"] = v1_capture["predictions"]
    eval_frame["v3_prediction"] = v3_capture["predictions"]
    eligible = set(v1_capture["eligible_labels"])
    eval_frame = eval_frame.loc[eval_frame["model"].astype(str).isin(eligible)].reset_index(drop=True)
    if len(eval_frame) != EXPECTED_ELIGIBLE_ROWS:
        raise RuntimeError(f"eligible evaluation row drift: {len(eval_frame)} != {EXPECTED_ELIGIBLE_ROWS}")

    decision, details = evaluate_source_transfer(eval_frame)
    result: dict[str, Any] = {
        "candidate": CANDIDATE,
        "scope": SCOPE,
        "execution_backend": "github_actions_self_hosted_cuda_or_local_cuda",
        "runtime": runtime,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pool_sha256": _sha256_file(pool),
        "holdout_sources": EXPECTED_HOLDOUT_SOURCES,
        "eligible_rows": int(len(eval_frame)),
        "transfer_split": {k: v for k, v in split.items() if k not in {"train_idx", "heldout_idx"}},
        "v1_worker": v1_result,
        "v3_worker": v3_result,
        "transfer_gate": details,
        "decision": decision.status,
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "v3_stage1_checkpoint_preserved": checkpoint_dir.is_dir(),
    }
    validate_local_result_payload(result)
    (work_dir / "paired_transfer_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    eval_frame.to_csv(work_dir / "paired_transfer_predictions.csv", index=False)
    return {
        "decision": decision.status,
        "result_path": str(work_dir / "paired_transfer_result.json"),
        "predictions_path": str(work_dir / "paired_transfer_predictions.csv"),
        "checkpoint_dir": str(checkpoint_dir),
        "runtime": runtime,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen DCT-v3 paired transfer on a local/self-hosted CUDA GPU")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_local_transfer(
        work_dir=args.work_dir,
        checkpoint_dir=args.checkpoint_dir,
        pool_path=args.pool,
        model_path=args.model_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"SCIENTIFIC_DECISION={summary['decision']}")
    print("KAGGLE_SUBMISSION_AUTHORIZED=false")


if __name__ == "__main__":
    main()
