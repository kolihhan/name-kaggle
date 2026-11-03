from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

KERNEL_REF = "kolihhan/ntl-v3-dct-transfer"
EXPECTED_POOL_SHA256 = "37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec"
EXPECTED_HOLDOUT_SOURCES = ["alperen", "hailey", "willrice"]
EXPECTED_ELIGIBLE_ROWS = 14506
EXPECTED_PREDICTION_COLUMNS = [
    "source",
    "prompt_id",
    "model",
    "v1_prediction",
    "v3_prediction",
]


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, capture_output=True, text=True)


def classify_kernel_status(text: str) -> str:
    value = text.lower()
    if any(token in value for token in ("error", "failed", "failure", "cancelled", "canceled")):
        return "failed"
    if any(token in value for token in ("complete", "successful", "succeeded")):
        return "complete"
    if any(token in value for token in ("running", "queued", "pending")):
        return "running"
    return "unknown"


def push_kernel_once(kernel_dir: Path) -> None:
    result = _run(["kaggle", "kernels", "push", "-p", str(Path(kernel_dir))], check=False)
    text = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    lower = text.lower()
    semantic_failure = (
        "kernel push error" in lower
        or "maximum weekly gpu quota" in lower
        or ("quota" in lower and "reached" in lower)
    )
    if result.returncode != 0 or semantic_failure:
        raise RuntimeError(f"Kaggle kernel push failed before execution:\n{text}")


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
                "Kaggle DCT v3 kernel failed:\n"
                + (logs.stdout or "")
                + "\n"
                + (logs.stderr or "")
            )
        if state == "unknown":
            raise RuntimeError(f"unknown Kaggle kernel status: {text.strip()}")
        time.sleep(sleep_seconds)
    raise RuntimeError("Kaggle DCT v3 kernel did not complete before polling limit")


def _count_prediction_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError("paired transfer predictions CSV is empty") from exc
        if header != EXPECTED_PREDICTION_COLUMNS:
            raise RuntimeError(f"prediction columns drift: {header}")
        return sum(1 for _ in reader)


def validate_transfer_artifacts(output_dir: Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    result_path = output_dir / "paired_transfer_result.json"
    predictions_path = output_dir / "paired_transfer_predictions.csv"
    v1_worker_path = output_dir / "v1_worker.json"
    v3_worker_path = output_dir / "v3_worker.json"
    checkpoint_dir = output_dir / "v3_stage1_checkpoint"

    required_files = [result_path, predictions_path, v1_worker_path, v3_worker_path]
    missing = [str(path.name) for path in required_files if not path.is_file()]
    if missing or not checkpoint_dir.is_dir():
        raise RuntimeError(
            f"missing frozen v3 output artifacts: files={missing} checkpoint={checkpoint_dir.is_dir()}"
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    exact = {
        "candidate": "dct-temporal-fingerprint-v3",
        "scope": "paired_external_source_transfer_no_official_rows_no_public_lb",
        "pool_sha256": EXPECTED_POOL_SHA256,
        "holdout_sources": EXPECTED_HOLDOUT_SOURCES,
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "v3_stage1_checkpoint_preserved": True,
    }
    for key, expected in exact.items():
        observed = result.get(key)
        if observed != expected:
            raise RuntimeError(f"transfer lineage drift for {key}: {observed!r} != {expected!r}")

    decision = result.get("decision")
    if decision not in {"pass", "fail"}:
        raise RuntimeError(f"unexpected transfer decision: {decision!r}")
    transfer_gate = result.get("transfer_gate")
    if not isinstance(transfer_gate, dict):
        raise RuntimeError("missing transfer_gate details")
    gate_decision = transfer_gate.get("decision")
    if not isinstance(gate_decision, dict) or gate_decision.get("status") != decision:
        raise RuntimeError("top-level transfer decision disagrees with gate decision")
    if int(transfer_gate.get("rows", -1)) != EXPECTED_ELIGIBLE_ROWS:
        raise RuntimeError("transfer gate row count drift")

    prediction_rows = _count_prediction_rows(predictions_path)
    if prediction_rows != EXPECTED_ELIGIBLE_ROWS:
        raise RuntimeError(
            f"prediction row count drift: {prediction_rows} != {EXPECTED_ELIGIBLE_ROWS}"
        )

    return {
        "candidate": result["candidate"],
        "decision": decision,
        "rows": prediction_rows,
        "delta": gate_decision.get("delta"),
        "bootstrap_ci_lower": gate_decision.get("bootstrap_ci_lower"),
        "bootstrap_ci_upper": gate_decision.get("bootstrap_ci_upper"),
        "oof_gate_unlocked": decision == "pass",
        "submission_authorized": False,
        "result_path": str(result_path),
        "predictions_path": str(predictions_path),
        "checkpoint_dir": str(checkpoint_dir),
    }


def run_direct(work_dir: Path, repo_root: Path) -> dict[str, object]:
    work_dir = Path(work_dir).resolve()
    repo_root = Path(repo_root).resolve()
    if work_dir.exists():
        raise RuntimeError(f"work directory already exists; refusing reuse: {work_dir}")
    work_dir.mkdir(parents=True)

    from scripts.build_dct_v3_transfer_kernel import KERNEL_REF as BUILDER_KERNEL_REF
    from scripts.build_dct_v3_transfer_kernel import write_kernel_bundle

    if BUILDER_KERNEL_REF != KERNEL_REF:
        raise RuntimeError(f"kernel identity drift: {BUILDER_KERNEL_REF} != {KERNEL_REF}")

    kernel_dir = work_dir / "kernel"
    write_kernel_bundle(kernel_dir, repo_root=repo_root)
    push_kernel_once(kernel_dir)
    wait_kernel_complete()

    output_dir = work_dir / "kernel-output"
    output_dir.mkdir()
    _run(["kaggle", "kernels", "output", KERNEL_REF, "-p", str(output_dir), "-o"])
    return validate_transfer_artifacts(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct no-Actions runner for frozen DCT temporal fingerprint v3 transfer gate"
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_direct(args.work_dir, args.repo_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"V3_TRANSFER_COMPLETE_DECISION_{str(summary['decision']).upper()}")
    print("NO_COMPETITION_SUBMISSION_PERFORMED")


if __name__ == "__main__":
    main()
