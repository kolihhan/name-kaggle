"""Independent frozen A1 LoRA24: 347 fresh-model text inference, no training.

Uses original Stage-1 A1 checkpoint and exact existing tokenizer/encoder/readout.
Input is source-labeled public Model Proxy text only; labels never enter model.
This file never loads official competition train/test or runs an optimizer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lora24_sibling_holdout_inference import predict_heldout
from scripts.run_two_stage_lora24_gate_b_gpu import (
    CHECKPOINT_MANIFEST_SHA256,
    load_exact_stage1_checkpoint,
)

FROZEN = (
    ("nimitha", "generated_proxy_candidates.csv", 240, 20,
     "25d21134cecbddd1b556ec4c3b81fb58dd22cb3102ec7bad967d4191364e167e"),
    ("toan", "passages.jsonl", 107, 9,
     "89a9d0cec7d434dc30bfa6ad303233821eaead8800e396fab6039a7dfeafdf17"),
)

ALL_18 = tuple(sorted((
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "claude-fable-5",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash",
    "kimi-k3",
)))
TRUE_12 = frozenset((
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.8-flash", "gpt-5.5", "gpt-5.6-luna",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_external_rows(nimitha_file: Path, toan_file: Path) -> pd.DataFrame:
    allrows = []
    for source, file, required_count, prompts, expected_sha in (
        (FROZEN[0][0], nimitha_file, FROZEN[0][2], FROZEN[0][3], FROZEN[0][4]),
        (FROZEN[1][0], toan_file, FROZEN[1][2], FROZEN[1][3], FROZEN[1][4]),
    ):
        if _sha(file) != expected_sha:
            raise RuntimeError("public fresh dataset SHA drift: " + source)
        if source == "nimitha":
            frame = pd.read_csv(file)
        else:
            frame = pd.read_json(file, lines=True)
        if not {"model", "text", "prompt_id"}.issubset(frame.columns):
            raise RuntimeError("public fresh dataset schema drift: " + source)
        if len(frame) != required_count or frame.prompt_id.nunique() != prompts:
            raise RuntimeError("public fresh dataset length/prompt drift: " + source)
        if set(frame.model) != TRUE_12:
            raise RuntimeError("public fresh dataset truth-label drift: " + source)
        per_label = frame.groupby("model").size().to_dict()
        if source == "nimitha" and set(per_label.values()) != {20}:
            raise RuntimeError("Nimitha label counts drift")
        if source == "toan" and sorted(per_label.values()) != [8] + [9] * 11:
            raise RuntimeError("Toan label counts drift")
        if frame[["model", "text", "prompt_id"]].isna().any().any():
            raise RuntimeError("public fresh dataset contains null values")
        for idx, row in frame.reset_index(drop=True).iterrows():
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise RuntimeError("public fresh dataset contains empty passage")
            allrows.append({
                "id": source[0] + f"{idx:05d}",
                "source": source,
                "prompt_id": str(row["prompt_id"]),
                "model": str(row["model"]),
                "text": row["text"],
            })
    frame = pd.DataFrame(allrows)
    if len(frame) != 347 or not frame.id.is_unique:
        raise RuntimeError("347 source rows are not uniquely aligned")
    # Champion used the same stable source-local row IDs; prediction order
    # is immaterial because paired analysis joins by ID.
    return frame


def _stats(frame: pd.DataFrame) -> dict:
    truth = frame["model"].astype(str)
    pred = frame["prediction"].astype(str)
    recalls = {
        label: float((pred[truth == label] == label).mean())
        for label in sorted(TRUE_12)
    }
    correct = truth == pred
    wrong = frame[~correct]
    return {
        "rows": int(len(frame)),
        "correct": int(correct.sum()),
        "wrong": int((~correct).sum()),
        "conditional_balanced_accuracy_12_truth_18_predictions":
            float(np.mean(list(recalls.values()))),
        "per_label_recall": recalls,
        "wrong_same_family": int(sum(
            str(a).split("-")[0] == str(b).split("-")[0]
            for a, b in zip(wrong.model, wrong.prediction)
        )),
        "wrong_cross_family": int(sum(
            str(a).split("-")[0] != str(b).split("-")[0]
            for a, b in zip(wrong.model, wrong.prediction)
        )),
    }


def run(*, nimitha_file: Path, toan_file: Path, checkpoint_dir: Path,
        model_root: Path, output_dir: Path, batch_size: int = 6) -> dict:
    frame = build_external_rows(nimitha_file, toan_file)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("one-shot pretrained forward needs Kaggle GPU")
    state = load_exact_stage1_checkpoint(
        checkpoint_dir=checkpoint_dir,
        snapshot_path=model_root,
        official_labels=ALL_18,  # name only: no official examples read
        device=torch.device("cuda:0"),
    )
    if state["manifest_sha256"] != CHECKPOINT_MANIFEST_SHA256:
        raise RuntimeError("frozen A1 checkpoint manifest mismatch")
    result = predict_heldout(heldout=frame, checkpoint=state,
                             batch_size=batch_size)
    if len(result) != 347 or not set(result).issubset(ALL_18):
        raise RuntimeError("incomplete frozen independent predictions")
    frame["prediction"] = result
    output_dir.mkdir(parents=True, exist_ok=True)
    # Private predictions used for same-row pairing. No passages exported.
    frame[["id", "source", "prompt_id", "model", "prediction"]].to_parquet(
        output_dir / "ntl_a1_347_predictions.parquet", index=False)
    report = {
        "contract": "a1-independent-stage1-frozen-347-v1",
        "original_lora24_checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "no_training": True,
        "competition_submission": False,
        "not_final_stage2_public0520_model": True,
        "candidate_predicted_labels": 18,
        "true_labels_observed": 12,
        "n": len(frame),
        "sources": {
            name: _stats(frame[frame.source == name])
            for name in ("toan", "nimitha")
        },
        "combined_exploratory": _stats(frame),
    }
    (output_dir / "ntl_a1_347_summary.json").write_text(
        json.dumps(report, sort_keys=True, indent=2))
    print("NTL_A1_INDEPENDENT_347_AGGREGATE", json.dumps(report, sort_keys=True),
          flush=True)
    return report
