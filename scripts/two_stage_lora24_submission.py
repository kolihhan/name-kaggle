from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

CANDIDATE = "two-stage-lora24-v1"
MODEL_ID = "Qwen/Qwen3-8B-Base"
MODEL_REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
TRANSFORMERS_VERSION = "4.57.1"
PEFT_VERSION = "0.17.1"
EXPECTED_OFFICIAL_ROWS = 2699
EXPECTED_LABELS = 18
STAGE2_EPOCHS = 2
SEED = 20261101
CHECKPOINT_DATASET_REF = "kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint"
CHECKPOINT_DATASET_VERSION = 1
CHECKPOINT_MANIFEST_SHA256 = "c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0"


def _require_columns(name: str, frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} frame missing required columns: {missing}")


def validate_final_fit_frames(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    _require_columns("official train", train, {"id", "prompt_id", "text", "model"})
    _require_columns("competition test", test, {"id", "text"})
    if len(train) != EXPECTED_OFFICIAL_ROWS:
        raise RuntimeError(
            f"official train row mismatch: {len(train)} != {EXPECTED_OFFICIAL_ROWS}"
        )
    if train[["id", "prompt_id", "text", "model"]].isna().any().any():
        raise RuntimeError("official train contains null id/prompt_id/text/model")
    if test[["id", "text"]].isna().any().any():
        raise RuntimeError("competition test contains null id/text")
    if train["id"].duplicated().any():
        raise RuntimeError("official train contains duplicate ids")
    if test["id"].duplicated().any():
        raise RuntimeError("competition test contains duplicate ids")
    if set(train["id"].astype(str)) & set(test["id"].astype(str)):
        raise RuntimeError("official train and competition test ids overlap")

    labels = sorted(train["model"].astype(str).unique().tolist())
    if len(labels) != EXPECTED_LABELS:
        raise RuntimeError(f"official label mismatch: {len(labels)} != {EXPECTED_LABELS}")
    return np.arange(len(train), dtype=np.int64), labels


def plain_test_batches(row_count: int, *, batch_size: int = 3) -> Iterator[np.ndarray]:
    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, row_count, batch_size):
        yield np.arange(start, min(start + batch_size, row_count), dtype=np.int64)


def build_submission_frame(
    test: pd.DataFrame,
    predictions: Sequence[str],
    *,
    allowed_labels: set[str] | Sequence[str],
) -> pd.DataFrame:
    _require_columns("competition test", test, {"id"})
    predictions = np.asarray(predictions, dtype=object)
    if len(predictions) != len(test):
        raise RuntimeError(
            f"submission prediction count mismatch: {len(predictions)} != {len(test)}"
        )
    submission = pd.DataFrame(
        {"id": test["id"].to_numpy(copy=True), "model": predictions}
    )
    if submission[["id", "model"]].isna().any().any():
        raise RuntimeError("submission contains null id/model")
    if submission["id"].duplicated().any():
        raise RuntimeError("submission contains duplicate ids")
    allowed = set(map(str, allowed_labels))
    predicted = set(submission["model"].astype(str))
    outside = sorted(predicted - allowed)
    if outside:
        raise RuntimeError(f"submission labels outside official label set: {outside}")
    return submission


def build_submission_lineage(
    *,
    official_train_sha256: str,
    test_rows: int,
    base_manifest_sha256: str,
    runtime_manifest_sha256: str,
) -> dict[str, object]:
    return {
        "candidate": CANDIDATE,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "peft_version": PEFT_VERSION,
        "checkpoint_dataset_ref": CHECKPOINT_DATASET_REF,
        "checkpoint_dataset_version": CHECKPOINT_DATASET_VERSION,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "official_training_rows": EXPECTED_OFFICIAL_ROWS,
        "stage2_epochs": STAGE2_EPOCHS,
        "seed": SEED,
        "primary_head": "plain",
        "centered_head_used_for_submission": False,
        "competition_test_rows_used_for_training": 0,
        "competition_test_rows_used_for_model_selection": 0,
        "public_lb_tuning": False,
        "test_rows": int(test_rows),
        "official_train_sha256": str(official_train_sha256),
        "base_manifest_sha256": str(base_manifest_sha256),
        "runtime_manifest_sha256": str(runtime_manifest_sha256),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_plain_test(
    *,
    test: pd.DataFrame,
    tokens: list[list[int]],
    model: object,
    tokenizer: object,
    norms: object,
    plain: object,
    layers: Sequence[int],
    labels: Sequence[str],
    device: object,
    torch: object,
    batch_size: int = 3,
) -> np.ndarray:
    from scripts.run_two_stage_lora24_gate_b_gpu import _make_encoding, _representation

    predictions = np.full(len(test), -1, dtype=np.int64)
    model.eval()
    norms.eval()
    plain.eval()
    with torch.inference_mode():
        for rows in plain_test_batches(len(test), batch_size=batch_size):
            encoding = _make_encoding(tokenizer, tokens, rows, device)
            z = _representation(
                model=model,
                norms=norms,
                layers=layers,
                encoding=encoding,
                torch=torch,
            )
            predictions[rows] = plain(z).argmax(1).detach().cpu().numpy()
    if np.any(predictions < 0):
        raise RuntimeError("one or more competition test rows were not predicted")
    return np.asarray([labels[int(index)] for index in predictions], dtype=object)


def run_final_fit(
    *,
    train_path: Path,
    test_path: Path,
    checkpoint_dir: Path,
    snapshot_path: Path,
    output_dir: Path,
    base_manifest_sha256: str,
    runtime_manifest_sha256: str,
    batch_size: int = 3,
) -> dict[str, object]:
    """Fit the frozen v1 Stage-2 recipe on all official rows, then infer test with plain head."""
    import torch
    import torch.nn.functional as F

    from scripts.run_two_stage_lora24_gate_a import (
        build_prompt_triplets,
        compute_class_weights,
    )
    from scripts.run_two_stage_lora24_gate_b import gate_b_training_spec
    from scripts.run_two_stage_lora24_gate_b_gpu import (
        _seed_all,
        _tokenize_official,
        load_exact_stage1_checkpoint,
        train_two_stage2_epochs,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("two-stage LoRA24 final fit requires CUDA")

    train = pd.read_csv(train_path).reset_index(drop=True)
    test = pd.read_csv(test_path).reset_index(drop=True)
    fit_idx, labels = validate_final_fit_frames(train, test)
    spec = gate_b_training_spec()
    if int(spec["epochs"]) != STAGE2_EPOCHS or int(spec["seed"]) != SEED:
        raise RuntimeError("Gate-B training spec drifted from frozen v1 submission identity")

    _seed_all(SEED, torch)
    device = torch.device("cuda:0")
    weights_map = compute_class_weights(
        train["model"].astype(str).to_numpy(), classes=labels
    )
    class_weights = torch.tensor(
        [weights_map[label] for label in labels], dtype=torch.float32, device=device
    )

    checkpoint = load_exact_stage1_checkpoint(
        checkpoint_dir=checkpoint_dir,
        snapshot_path=snapshot_path,
        official_labels=labels,
        device=device,
    )
    model = checkpoint["model"]
    tokenizer = checkpoint["tokenizer"]
    norms = checkpoint["norms"]
    plain = checkpoint["plain"]
    centered = checkpoint["centered"]
    dropout = checkpoint["dropout"]
    layers = checkpoint["layers"]

    train_tokens = _tokenize_official(
        train,
        tokenizer,
        max_length_before_eos=int(spec["max_length_before_eos"]),
    )
    label_to_idx = {label: index for index, label in enumerate(labels)}
    y = train["model"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64)
    triplets = build_prompt_triplets(train, fit_idx, seed=SEED)
    train_meta = train_two_stage2_epochs(
        frame=train,
        fit_idx=fit_idx,
        triplets=triplets,
        tokens=train_tokens,
        y=y,
        class_weights=class_weights,
        model=model,
        tokenizer=tokenizer,
        norms=norms,
        plain=plain,
        centered=centered,
        dropout=dropout,
        layers=layers,
        device=device,
        torch=torch,
        F=F,
    )

    # Test is tokenized only after the complete two-epoch training call returns.
    test_tokens = _tokenize_official(
        test,
        tokenizer,
        max_length_before_eos=int(spec["max_length_before_eos"]),
    )
    predictions = _infer_plain_test(
        test=test,
        tokens=test_tokens,
        model=model,
        tokenizer=tokenizer,
        norms=norms,
        plain=plain,
        layers=layers,
        labels=labels,
        device=device,
        torch=torch,
        batch_size=batch_size,
    )
    submission = build_submission_frame(test, predictions, allowed_labels=set(labels))

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.parquet"
    submission.to_parquet(submission_path, index=False)
    lineage = build_submission_lineage(
        official_train_sha256=_sha256_file(train_path),
        test_rows=len(test),
        base_manifest_sha256=base_manifest_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
    )
    lineage["training"] = train_meta
    lineage["submission_rows"] = int(len(submission))
    lineage["submission_columns"] = submission.columns.tolist()
    lineage["submission_sha256"] = _sha256_file(submission_path)
    (output_dir / "submission_lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True), encoding="utf-8"
    )
    return lineage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen two-stage LoRA24 v1 final fit and inference")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-manifest-sha256", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lineage = run_final_fit(
        train_path=args.train,
        test_path=args.test,
        checkpoint_dir=args.checkpoint_dir,
        snapshot_path=args.snapshot_path,
        output_dir=args.output_dir,
        base_manifest_sha256=args.base_manifest_sha256,
        runtime_manifest_sha256=args.runtime_manifest_sha256,
        batch_size=args.batch_size,
    )
    print(json.dumps(lineage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
