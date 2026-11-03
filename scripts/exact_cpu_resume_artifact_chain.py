from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.exact_cpu_resume_equivalence import (
    ACCUMULATION_SIZE,
    INTERRUPT_AFTER,
    MICROBATCHES,
    SEED,
    _build_training_objects,
    _microbatches,
    _nested_equal,
    _parameter_state,
    _seed_all,
    _train_until,
)
from scripts.exact_cpu_resume_state import (
    capture_resume_state,
    load_resume_state,
    restore_resume_state,
    save_resume_state,
)

ARTIFACT_CANDIDATE = "synthetic-artifact-chain"
ARTIFACT_SCOPE = "cross-job-resume-proof"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def start_chain(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batches = _microbatches()

    _seed_all(SEED)
    model, optimizer, scheduler = _build_training_objects()
    cursor, position = _train_until(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batches=batches,
        cursor=0,
        accumulation_position=0,
        stop_cursor=INTERRUPT_AFTER,
    )
    if cursor != INTERRUPT_AFTER or position != 1:
        raise RuntimeError("artifact producer did not stop at frozen mid-accumulation point")

    state = capture_resume_state(
        modules={"model": model},
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=cursor,
        accumulation_position=position,
        accumulation_size=ACCUMULATION_SIZE,
        metadata={
            "candidate": ARTIFACT_CANDIDATE,
            "scope": ARTIFACT_SCOPE,
            "microbatches": MICROBATCHES,
            "interrupt_after": INTERRUPT_AFTER,
        },
    )
    state_path = output_dir / "resume.pt"
    save_resume_state(state_path, state)
    state_sha256 = _sha256_file(state_path)
    manifest = {
        "candidate": ARTIFACT_CANDIDATE,
        "scope": ARTIFACT_SCOPE,
        "state_file": state_path.name,
        "state_sha256": state_sha256,
        "cursor": cursor,
        "accumulation_position": position,
        "accumulation_size": ACCUMULATION_SIZE,
        "microbatches": MICROBATCHES,
        "interrupt_after": INTERRUPT_AFTER,
        "scientific_decision_authorized": False,
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "firewall_rows_scored": 0,
        "public_lb_tuning": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def finish_chain(
    *,
    state_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    state_path = Path(state_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact = {
        "candidate": ARTIFACT_CANDIDATE,
        "scope": ARTIFACT_SCOPE,
        "accumulation_size": ACCUMULATION_SIZE,
        "microbatches": MICROBATCHES,
        "interrupt_after": INTERRUPT_AFTER,
        "scientific_decision_authorized": False,
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "firewall_rows_scored": 0,
        "public_lb_tuning": False,
    }
    for key, expected in exact.items():
        observed = manifest.get(key)
        if observed != expected:
            raise RuntimeError(
                f"artifact-chain manifest drift for {key}: {observed!r} != {expected!r}"
            )
    if manifest.get("state_file") != state_path.name:
        raise RuntimeError("artifact-chain state filename drift")
    observed_sha256 = _sha256_file(state_path)
    if observed_sha256 != manifest.get("state_sha256"):
        raise RuntimeError("artifact-chain state SHA256 mismatch")

    batches = _microbatches()

    _seed_all(SEED)
    baseline_model, baseline_optimizer, baseline_scheduler = _build_training_objects()
    baseline_cursor, baseline_position = _train_until(
        model=baseline_model,
        optimizer=baseline_optimizer,
        scheduler=baseline_scheduler,
        batches=batches,
        cursor=0,
        accumulation_position=0,
        stop_cursor=len(batches),
    )
    baseline_parameters = _parameter_state(baseline_model)
    baseline_optimizer_state = copy.deepcopy(baseline_optimizer.state_dict())
    baseline_scheduler_state = copy.deepcopy(baseline_scheduler.state_dict())
    import random

    baseline_rng = (random.random(), float(np.random.rand()), torch.rand(5))

    resumed_model, resumed_optimizer, resumed_scheduler = _build_training_objects(seed=999)
    restored = restore_resume_state(
        load_resume_state(state_path),
        modules={"model": resumed_model},
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        expected_metadata={
            "candidate": ARTIFACT_CANDIDATE,
            "scope": ARTIFACT_SCOPE,
            "microbatches": MICROBATCHES,
            "interrupt_after": INTERRUPT_AFTER,
        },
    )
    if int(restored["cursor"]) != int(manifest["cursor"]):
        raise RuntimeError("artifact-chain cursor differs between manifest and state")
    if int(restored["accumulation_position"]) != int(manifest["accumulation_position"]):
        raise RuntimeError("artifact-chain accumulation position differs between manifest and state")

    final_cursor, final_position = _train_until(
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        batches=batches,
        cursor=int(restored["cursor"]),
        accumulation_position=int(restored["accumulation_position"]),
        stop_cursor=len(batches),
    )
    resumed_rng = (random.random(), float(np.random.rand()), torch.rand(5))

    result = {
        "input_state_sha256": observed_sha256,
        "baseline_cursor": baseline_cursor,
        "baseline_accumulation_position": baseline_position,
        "final_cursor": final_cursor,
        "final_accumulation_position": final_position,
        "parameters_equal": _nested_equal(
            baseline_parameters, _parameter_state(resumed_model)
        ),
        "optimizer_equal": _nested_equal(
            baseline_optimizer_state, resumed_optimizer.state_dict()
        ),
        "scheduler_equal": _nested_equal(
            baseline_scheduler_state, resumed_scheduler.state_dict()
        ),
        "next_python_rng_equal": baseline_rng[0] == resumed_rng[0],
        "next_numpy_rng_equal": baseline_rng[1] == resumed_rng[1],
        "next_torch_rng_equal": torch.equal(baseline_rng[2], resumed_rng[2]),
        "scientific_decision_authorized": False,
    }
    result["all_equal"] = bool(
        result["parameters_equal"]
        and result["optimizer_equal"]
        and result["scheduler_equal"]
        and result["next_python_rng_equal"]
        and result["next_numpy_rng_equal"]
        and result["next_torch_rng_equal"]
        and baseline_cursor == final_cursor == MICROBATCHES
        and baseline_position == final_position == 0
    )
    _write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--output-dir", type=Path, required=True)

    finish = subparsers.add_parser("finish")
    finish.add_argument("--state", type=Path, required=True)
    finish.add_argument("--manifest", type=Path, required=True)
    finish.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "start":
        result = start_chain(args.output_dir)
    else:
        result = finish_chain(
            state_path=args.state,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "finish" and not result["all_equal"]:
        raise SystemExit("cross-job artifact resume equivalence failed")


if __name__ == "__main__":
    main()
