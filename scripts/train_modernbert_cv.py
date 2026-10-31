from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import GroupKFold


EXPECTED_TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_FOLD_FINGERPRINT = "bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d"
DEFAULT_MODEL = "answerdotai/ModernBERT-base"
DEFAULT_SEED = 20261031


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_fingerprint(frame: pd.DataFrame) -> str:
    required = {"id", "prompt_id", "model", "fold_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fold frame missing columns: {missing}")

    ordered = frame.sort_values("id")
    payload = "\n".join(
        f"{row.id},{row.prompt_id},{row.model},{int(row.fold_id)}"
        for row in ordered.itertuples(index=False)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_folds(train: pd.DataFrame, *, n_splits: int = 5) -> pd.DataFrame:
    required = {"id", "prompt_id", "text", "model"}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"train frame missing columns: {missing}")

    folded = train.sort_values("id").reset_index(drop=True).copy()
    folded["fold_id"] = -1
    splitter = GroupKFold(n_splits=n_splits)
    for fold_id, (_, valid_idx) in enumerate(
        splitter.split(folded, y=folded["model"], groups=folded["prompt_id"])
    ):
        folded.loc[valid_idx, "fold_id"] = fold_id

    if (folded["fold_id"] < 0).any():
        raise RuntimeError("at least one row was not assigned to a fold")

    prompt_fold_counts = folded.groupby("prompt_id")["fold_id"].nunique()
    if (prompt_fold_counts > 1).any():
        raise RuntimeError("prompt_id appears in multiple folds")

    observed = set(int(x) for x in folded["fold_id"].unique())
    expected = set(range(n_splits))
    if observed != expected:
        raise RuntimeError(f"expected folds {sorted(expected)}, found {sorted(observed)}")
    return folded


def label_mapping(labels: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    ordered = sorted(set(str(x) for x in labels))
    label2id = {label: idx for idx, label in enumerate(ordered)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def summarize_oof(
    train: pd.DataFrame,
    predictions: np.ndarray,
    *,
    fold_scores: list[float],
) -> dict:
    if len(predictions) != len(train):
        raise ValueError("OOF prediction count must equal official train row count")
    if any(x is None for x in predictions.tolist()):
        raise ValueError("OOF predictions contain unfilled rows")

    labels = sorted(train["model"].astype(str).unique())
    score = float(balanced_accuracy_score(train["model"], predictions))
    recalls = recall_score(
        train["model"],
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "oof_rows": int(len(train)),
        "balanced_accuracy": score,
        "folds": [float(x) for x in fold_scores],
        "per_model_recall": {
            label: float(recall) for label, recall in zip(labels, recalls)
        },
    }


def find_competition_train(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit

    preferred = Path("/kaggle/input/name-that-llm/train.csv")
    if preferred.exists():
        return preferred

    candidates = sorted(Path("/kaggle/input").glob("**/train.csv"))
    valid: list[Path] = []
    for path in candidates:
        try:
            cols = set(pd.read_csv(path, nrows=2).columns)
        except Exception:
            continue
        if {"id", "prompt_id", "text", "model"}.issubset(cols):
            valid.append(path)
    if len(valid) != 1:
        raise RuntimeError(f"expected one compatible train.csv, found {valid}")
    return valid[0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _make_dataset_class():
    from torch.utils.data import Dataset

    class TextDataset(Dataset):
        def __init__(self, frame, tokenizer, label2id, max_length):
            self.texts = frame["text"].astype(str).tolist()
            self.labels = [label2id[str(x)] for x in frame["model"]]
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            item = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_length,
            )
            item["labels"] = self.labels[idx]
            return item

    return TextDataset


def train_one_fold(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    model_name: str,
    label2id: dict[str, int],
    id2label: dict[int, str],
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, float]:
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    TextDataset = _make_dataset_class()
    train_ds = TextDataset(train_frame, tokenizer, label2id, max_length)
    valid_ds = TextDataset(valid_frame, tokenizer, label2id, max_length)
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if device.startswith("cuda") else None,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=2,
        pin_memory=device.startswith("cuda"),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=device.startswith("cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for _epoch in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(**batch)
                loss = output.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    model.eval()
    pred_ids: list[int] = []
    with torch.inference_mode():
        for batch in valid_loader:
            labels = batch.pop("labels")
            del labels
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**batch).logits
            pred_ids.extend(logits.argmax(dim=-1).cpu().tolist())

    predictions = np.array([id2label[int(idx)] for idx in pred_ids], dtype=object)
    score = float(balanced_accuracy_score(valid_frame["model"], predictions))

    del model, tokenizer, optimizer, scheduler, train_loader, valid_loader, train_ds, valid_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, score


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def run_cv(args: argparse.Namespace) -> dict:
    train_path = find_competition_train(args.train)
    observed_train_sha = sha256_file(train_path)
    if observed_train_sha != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(
            f"official train SHA mismatch: {observed_train_sha} != {EXPECTED_TRAIN_SHA256}"
        )

    raw = pd.read_csv(train_path)
    train = build_folds(raw, n_splits=5)
    observed_fingerprint = fold_fingerprint(train)
    if observed_fingerprint != EXPECTED_FOLD_FINGERPRINT:
        raise RuntimeError(
            "fold mapping mismatch: "
            f"{observed_fingerprint} != {EXPECTED_FOLD_FINGERPRINT}"
        )

    label2id, id2label = label_mapping(train["model"])
    if len(label2id) != 18:
        raise RuntimeError(f"expected 18 labels, found {len(label2id)}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and not device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the Phase 4 Kaggle run")

    selected_folds = [int(x) for x in args.folds.split(",") if x.strip()]
    if sorted(selected_folds) != [0, 1, 2, 3, 4]:
        raise ValueError("official Phase 4 baseline must run folds 0,1,2,3,4")

    oof = np.empty(len(train), dtype=object)
    oof[:] = None
    fold_scores: list[float] = []

    for fold in selected_folds:
        train_frame = train.loc[train.fold_id != fold].reset_index(drop=True)
        valid_idx = np.flatnonzero(train.fold_id.to_numpy() == fold)
        valid_frame = train.iloc[valid_idx].reset_index(drop=True)
        predictions, score = train_one_fold(
            train_frame=train_frame,
            valid_frame=valid_frame,
            model_name=args.model,
            label2id=label2id,
            id2label=id2label,
            max_length=args.max_length,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed + fold,
            device=device,
        )
        oof[valid_idx] = predictions
        fold_scores.append(score)
        partial = {
            "completed_folds": fold + 1,
            "fold_scores": fold_scores,
            "fold_fingerprint": observed_fingerprint,
        }
        write_json(args.partial_output, partial)
        print(f"fold={fold} balanced_accuracy={score:.6f}", flush=True)

    oof_summary = summarize_oof(train, oof, fold_scores=fold_scores)
    report = {
        "phase": 4,
        "experiment": "trainable_modernbert_official_only",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": observed_fingerprint,
        },
        "data": {
            "train_path": str(train_path),
            "train_sha256": observed_train_sha,
            "official_rows": int(len(train)),
            "prompt_count": int(train["prompt_id"].nunique()),
            "labels": sorted(label2id),
            "external_rows": 0,
        },
        "model": {
            "name": args.model,
            "trainable": True,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "device": device,
        },
        "oof": oof_summary,
    }
    write_json(args.output, report)

    oof_frame = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof_frame["prediction"] = oof
    args.oof_output.parent.mkdir(parents=True, exist_ok=True)
    oof_frame.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official-only trainable ModernBERT 5-fold CV for Name That LLM."
    )
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/trainable_modernbert_official_cv.json"),
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=Path("/kaggle/working/trainable_modernbert_official_oof.csv"),
    )
    parser.add_argument(
        "--partial-output",
        type=Path,
        default=Path("/kaggle/working/trainable_modernbert_partial.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cv(args)


if __name__ == "__main__":
    main()
