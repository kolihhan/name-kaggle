from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from scripts.build_two_stage_lora24_kaggle_kernel import KERNEL_REF, write_kernel_bundle
from scripts.materialize_two_stage_lora24_offline_inputs import (
    BASE_DATASET_REF,
    RUNTIME_DATASET_REF,
    create_private_dataset,
    prepare_base_package,
    prepare_runtime_package,
)
from scripts.two_stage_lora24_submission import CHECKPOINT_DATASET_REF


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, capture_output=True, text=True)


def classify_kernel_status(text: str) -> str:
    value = text.lower()
    if any(token in value for token in ("error", "failed", "cancel")):
        return "failed"
    if any(token in value for token in ("complete", "successful")):
        return "complete"
    if any(token in value for token in ("running", "queued", "pending")):
        return "running"
    return "unknown"


def _dataset_status(ref: str) -> dict:
    result = _run(["kaggle", "datasets", "status", ref, "--format", "json"])
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid Kaggle dataset status JSON for {ref}: {result.stdout!r}") from exc
    return payload


def require_ready_v1(ref: str) -> None:
    payload = _dataset_status(ref)
    if payload.get("status") != "ready":
        raise RuntimeError(f"dataset is not ready: {ref}: {payload}")
    if int(payload.get("current_version_number", -1)) != 1:
        raise RuntimeError(f"dataset version is not 1: {ref}: {payload}")


def wait_dataset_ready_v1(ref: str, *, attempts: int = 90, sleep_seconds: int = 10) -> None:
    for _ in range(attempts):
        payload = _dataset_status(ref)
        status = str(payload.get("status", "")).lower()
        version = int(payload.get("current_version_number", -1))
        if status == "ready" and version == 1:
            return
        if status == "error":
            raise RuntimeError(f"dataset entered error state: {ref}: {payload}")
        if version not in (-1, 1):
            raise RuntimeError(f"unexpected dataset version for {ref}: {payload}")
        time.sleep(sleep_seconds)
    raise RuntimeError(f"dataset did not become ready version 1: {ref}")


def wait_kernel_complete(*, attempts: int = 1080, sleep_seconds: int = 30) -> None:
    for _ in range(attempts):
        result = _run(["kaggle", "kernels", "status", KERNEL_REF], check=False)
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        state = classify_kernel_status(text)
        if state == "complete":
            return
        if state == "failed":
            logs = _run(["kaggle", "kernels", "logs", KERNEL_REF], check=False)
            raise RuntimeError(
                "Kaggle preflight kernel failed:\n"
                + (logs.stdout or "")
                + "\n"
                + (logs.stderr or "")
            )
        if state == "unknown":
            raise RuntimeError(f"unknown Kaggle kernel status: {text.strip()}")
        time.sleep(sleep_seconds)
    raise RuntimeError("Kaggle preflight kernel did not complete before polling limit")


def validate_submission_artifacts(output_dir: Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    submission_path = output_dir / "submission.parquet"
    lineage_path = output_dir / "submission_lineage.json"
    if not submission_path.is_file() or not lineage_path.is_file():
        raise RuntimeError("missing submission.parquet or submission_lineage.json")

    submission = pd.read_parquet(submission_path)
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if submission.columns.tolist() != ["id", "model"]:
        raise RuntimeError(f"unexpected submission columns: {submission.columns.tolist()}")
    if submission[["id", "model"]].isna().any().any():
        raise RuntimeError("submission contains null id/model")
    if submission["id"].duplicated().any():
        raise RuntimeError("submission contains duplicate ids")

    exact = {
        "candidate": "two-stage-lora24-v1",
        "primary_head": "plain",
        "centered_head_used_for_submission": False,
        "competition_test_rows_used_for_training": 0,
        "competition_test_rows_used_for_model_selection": 0,
        "public_lb_tuning": False,
    }
    for key, expected in exact.items():
        observed = lineage.get(key)
        if observed != expected:
            if key == "competition_test_rows_used_for_training":
                raise RuntimeError(f"test rows used for training: {observed}")
            raise RuntimeError(f"submission lineage drift for {key}: {observed!r} != {expected!r}")
    if int(lineage.get("test_rows", -1)) != len(submission):
        raise RuntimeError("lineage test_rows does not match submission rows")
    if int(lineage.get("submission_rows", -1)) != len(submission):
        raise RuntimeError("lineage submission_rows does not match submission rows")
    if lineage.get("submission_columns") != ["id", "model"]:
        raise RuntimeError("lineage submission columns drift")

    return {
        "candidate": lineage["candidate"],
        "primary_head": lineage["primary_head"],
        "rows": int(len(submission)),
        "submission_authorized": False,
        "submission_path": str(submission_path),
        "lineage_path": str(lineage_path),
        "base_manifest_sha256": lineage.get("base_manifest_sha256"),
        "runtime_manifest_sha256": lineage.get("runtime_manifest_sha256"),
    }


def run_preflight(work_dir: Path, repo_root: Path) -> dict[str, object]:
    work_dir = Path(work_dir).resolve()
    repo_root = Path(repo_root).resolve()
    if work_dir.exists():
        raise RuntimeError(f"work directory already exists; refusing reuse: {work_dir}")
    work_dir.mkdir(parents=True)

    require_ready_v1(CHECKPOINT_DATASET_REF)

    base_dir = work_dir / "base-dataset"
    runtime_dir = work_dir / "runtime-dataset"
    base_lineage = prepare_base_package(base_dir)
    runtime_lineage = prepare_runtime_package(runtime_dir)
    create_private_dataset(base_dir, BASE_DATASET_REF)
    create_private_dataset(runtime_dir, RUNTIME_DATASET_REF)
    wait_dataset_ready_v1(BASE_DATASET_REF)
    wait_dataset_ready_v1(RUNTIME_DATASET_REF)

    kernel_dir = work_dir / "kernel"
    write_kernel_bundle(kernel_dir, repo_root=repo_root)
    _run([
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(kernel_dir),
        "--accelerator",
        "NvidiaTeslaT4",
        "--timeout",
        "32400",
    ])
    wait_kernel_complete()

    output_dir = work_dir / "kernel-output"
    output_dir.mkdir()
    _run(["kaggle", "kernels", "output", KERNEL_REF, "-p", str(output_dir), "-o"])
    summary = validate_submission_artifacts(output_dir)
    if summary["base_manifest_sha256"] != base_lineage["file_manifest_sha256"]:
        raise RuntimeError("base manifest in final lineage does not match locally frozen manifest")
    if summary["runtime_manifest_sha256"] != runtime_lineage["file_manifest_sha256"]:
        raise RuntimeError("runtime manifest in final lineage does not match locally frozen manifest")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct no-Actions Kaggle preflight for frozen two-stage LoRA24 v1"
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preflight(args.work_dir, args.repo_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("PRELIGHT_COMPLETE_SUBMISSION_NOT_AUTHORIZED")


if __name__ == "__main__":
    main()
