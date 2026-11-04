"""One frozen Stage-1 LoRA24 public-source holdout inference-only diagnostic.

This evaluates the exact A1 checkpoint on the preselected abdullah/semenoff/zakir
source holdout. It is *not* the Stage-2 model that scored 0.520 on Public LB.
No optimizer, backward pass, additional training, official competition input,
hidden-test inference, or Kaggle competition submission is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_lora24_sibling_holdout import (
    EXPECTED_ELIGIBLE_LABELS,
    EXPECTED_ELIGIBLE_ROWS,
    SOURCES,
    summarize_predictions,
)
from scripts.run_phase14_public_sparse import (
    EXPECTED_EXTERNAL_ROWS,
    EXPECTED_RAW_ROWS,
    EXPECTED_RAW_SHA256,
    build_conservative_subset,
)
from scripts.run_two_stage_lora24_gate_a import (
    build_stage1_split,
    gate_a_training_spec,
    select_eligible_labels,
)
from scripts.run_two_stage_lora24_gate_b_gpu import (
    CHECKPOINT_MANIFEST_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    _make_encoding,
    _representation,
    _tokenize_official,
    load_exact_stage1_checkpoint,
)

EXPECTED_HOLDOUT_SOURCE_SET = frozenset(SOURCES)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_heldout(pool_path: Path) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    if sha256_file(pool_path) != EXPECTED_RAW_SHA256:
        raise RuntimeError("locked public pool content hash mismatch")
    raw = pd.read_parquet(pool_path)
    if len(raw) != EXPECTED_RAW_ROWS:
        raise RuntimeError("locked public pool length mismatch")
    safe = build_conservative_subset(raw, validate_full_contract=True).reset_index(drop=True)
    if len(safe) != EXPECTED_EXTERNAL_ROWS:
        raise RuntimeError("Phase14 conservative subset drift")
    split = build_stage1_split(safe)
    if set(split["holdout_sources"]) != EXPECTED_HOLDOUT_SOURCE_SET:
        raise RuntimeError("frozen Gate-A source identities drifted")
    if split["prompt_id_overlap"] != 0 or split["prompt_key_overlap"] != 0:
        raise RuntimeError("heldout prompt leakage detected")

    heldout_idx = np.asarray(split["heldout_idx"], dtype=int)
    train = safe.iloc[np.asarray(split["train_idx"], dtype=int)]
    heldout = safe.iloc[heldout_idx].copy()
    eligible_labels = select_eligible_labels(train, heldout)
    if len(eligible_labels) != EXPECTED_ELIGIBLE_LABELS:
        raise RuntimeError("frozen Gate-A eligible label count changed")
    heldout["row_index"] = heldout_idx
    heldout = heldout.loc[heldout["model"].astype(str).isin(eligible_labels)].reset_index(drop=True)
    if len(heldout) != EXPECTED_ELIGIBLE_ROWS:
        raise RuntimeError("frozen Gate-A eligible row count changed")
    label_order = sorted(safe["model"].astype(str).unique())
    if len(label_order) != 18:
        raise RuntimeError("frozen 18-class checkpoint label universe drift")
    metadata = {
        "pool_sha256": EXPECTED_RAW_SHA256,
        "full_pool_rows": EXPECTED_RAW_ROWS,
        "safe_rows": len(safe),
        "heldout_eligible_rows": len(heldout),
        "train_rows": len(split["train_idx"]),
        "holdout_sources": list(split["holdout_sources"]),
        "train_heldout_prompt_id_overlap": int(split["prompt_id_overlap"]),
        "train_heldout_prompt_key_overlap": int(split["prompt_key_overlap"]),
    }
    return heldout, eligible_labels, label_order, metadata


def predict_heldout(
    *,
    heldout: pd.DataFrame,
    checkpoint: dict,
    batch_size: int = 6,
) -> list[str]:
    """Plain-head inference only; no privileged same-prompt centering."""
    if not (1 <= batch_size <= 6):
        raise ValueError("inference batch size must be between 1 and 6")
    torch = checkpoint["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("locked Qwen3 LoRA24 source inference requires CUDA")
    model, tokenizer = checkpoint["model"], checkpoint["tokenizer"]
    norms, plain = checkpoint["norms"], checkpoint["plain"]
    labels, layers = checkpoint["labels"], checkpoint["layers"]
    for component in (model, norms, plain, checkpoint["centered"], checkpoint["dropout"]):
        component.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    tokens = _tokenize_official(
        heldout, tokenizer,
        max_length_before_eos=int(gate_a_training_spec()["max_length_before_eos"]),
    )
    order = np.argsort(np.asarray([len(x) for x in tokens]), kind="stable")
    pred_indices = np.full(len(tokens), -1, dtype=np.int64)
    device = torch.device("cuda:0")
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            inputs = _make_encoding(tokenizer, tokens, rows, device)
            z = _representation(
                model=model, norms=norms, layers=layers, encoding=inputs, torch=torch
            )
            pred_indices[rows] = plain(z).argmax(dim=1).cpu().numpy()
    if (pred_indices < 0).any() or len(set(labels)) != len(labels):
        raise RuntimeError("incomplete predictions or duplicate model label")
    return [labels[int(i)] for i in pred_indices]


def run(
    *,
    pool_path: Path,
    checkpoint_dir: Path,
    model_root: Path,
    output_dir: Path,
    batch_size: int = 6,
) -> dict:
    heldout, eligible, labels, split_meta = prepare_heldout(pool_path)
    # This helper validates the exact Stage-1 A1 checkpoint manifest, training
    # provenance, 18-label order, and pinned Qwen runtime before loading it.
    import torch
    ck = load_exact_stage1_checkpoint(
        checkpoint_dir=checkpoint_dir,
        snapshot_path=model_root,
        official_labels=labels,  # parameter name only; NO official data is read
        device=torch.device("cuda:0"),
    )
    if ck["manifest_sha256"] != CHECKPOINT_MANIFEST_SHA256:
        raise RuntimeError("checkpoint transport identity mismatch")

    preds = predict_heldout(heldout=heldout, checkpoint=ck, batch_size=batch_size)
    table = heldout[["row_index", "source", "prompt_id", "model"]].copy()
    table["prediction"] = preds
    report = summarize_predictions(table, eligible_labels=eligible)
    report["lineage"] = {
        **split_meta,
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "stage1_only": True,
        "stage2_train_steps": 0,
        "official_rows_used": 0,
        "competition_rows_used": 0,
        "gpu_only_forward": True,
        "training_performed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Raw external-source predictions stay in the PRIVATE Kaggle kernel output;
    # GitHub workflows must retrieve/report only summary JSON.
    pred_path = output_dir / "stage1_external_holdout_predictions.csv"
    table.to_csv(pred_path, index=False)
    report["private_prediction_csv_sha256"] = sha256_file(pred_path)
    (output_dir / "stage1_sibling_holdout_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SIBLING_STAGE1_AGGREGATE " + json.dumps(
        {"pooled": report["pooled"], "by_source": report["by_source"],
         "source_splits_verified": True, "n_eligible_labels": len(eligible)},
        sort_keys=True,
    ), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 A1 external holdout inference only")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()
    run(
        pool_path=args.pool, checkpoint_dir=args.checkpoint,
        model_root=args.model_root, output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
