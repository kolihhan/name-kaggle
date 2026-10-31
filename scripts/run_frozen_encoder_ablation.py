from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.svm import LinearSVC


DEFAULT_MODEL = "answerdotai/ModernBERT-base"


def validate_fold_contract(train: pd.DataFrame) -> None:
    required = {"prompt_id", "fold_id"}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"missing fold columns: {missing}")

    fold_counts = train.groupby("prompt_id")["fold_id"].nunique()
    if (fold_counts > 1).any():
        raise ValueError("prompt_id appears in multiple folds")

    observed = set(int(x) for x in train["fold_id"].dropna().unique())
    if observed != {0, 1, 2, 3, 4}:
        raise ValueError(f"expected folds 0..4, found {sorted(observed)}")


def prepare_fold_indices(
    *,
    train_rows: int,
    external_rows: int,
    fold_ids: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(fold_ids) != train_rows:
        raise ValueError("fold_ids length must match train_rows")

    valid_idx = np.flatnonzero(fold_ids == fold)
    official_train_idx = np.flatnonzero(fold_ids != fold)
    external_idx = np.arange(train_rows, train_rows + external_rows)
    augmented_train_idx = np.concatenate([official_train_idx, external_idx])
    return official_train_idx, valid_idx, augmented_train_idx


def _score_condition(
    *,
    name: str,
    train: pd.DataFrame,
    all_embeddings: np.ndarray,
    all_labels: np.ndarray,
    external_rows: int,
    augmented: bool,
    c: float,
) -> dict:
    labels = sorted(train["model"].astype(str).unique())
    fold_ids = train["fold_id"].to_numpy()
    predictions = np.empty(len(train), dtype=object)
    fold_scores: list[float] = []

    for fold in range(5):
        official_train_idx, valid_idx, augmented_train_idx = prepare_fold_indices(
            train_rows=len(train),
            external_rows=external_rows,
            fold_ids=fold_ids,
            fold=fold,
        )
        fit_idx = augmented_train_idx if augmented else official_train_idx

        classifier = LinearSVC(class_weight="balanced", C=c)
        classifier.fit(all_embeddings[fit_idx], all_labels[fit_idx])
        fold_pred = classifier.predict(all_embeddings[valid_idx])
        predictions[valid_idx] = fold_pred
        fold_scores.append(
            float(balanced_accuracy_score(train.iloc[valid_idx]["model"], fold_pred))
        )

    balanced_accuracy = float(balanced_accuracy_score(train["model"], predictions))
    recalls = recall_score(
        train["model"],
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "name": name,
        "oof_rows": int(len(train)),
        "balanced_accuracy": balanced_accuracy,
        "folds": fold_scores,
        "per_model_recall": {
            label: float(recall) for label, recall in zip(labels, recalls)
        },
    }


def run_embedding_ablation(
    *,
    train: pd.DataFrame,
    official_embeddings: np.ndarray,
    external: pd.DataFrame,
    external_embeddings: np.ndarray,
    c: float = 1.0,
) -> dict:
    validate_fold_contract(train)
    if len(official_embeddings) != len(train):
        raise ValueError("official embedding rows must match official train rows")
    if len(external_embeddings) != len(external):
        raise ValueError("external embedding rows must match external rows")
    if official_embeddings.ndim != 2 or external_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if official_embeddings.shape[1] != external_embeddings.shape[1]:
        raise ValueError("official and external embeddings must have the same width")

    official_labels = train["model"].astype(str).to_numpy()
    external_labels = external["model"].astype(str).to_numpy()
    unknown_external = sorted(set(external_labels) - set(official_labels))
    if unknown_external:
        raise ValueError(f"external labels outside competition label set: {unknown_external}")

    all_embeddings = np.vstack([official_embeddings, external_embeddings]).astype(
        np.float32, copy=False
    )
    all_labels = np.concatenate([official_labels, external_labels])

    official = _score_condition(
        name="official_only",
        train=train,
        all_embeddings=all_embeddings,
        all_labels=all_labels,
        external_rows=len(external),
        augmented=False,
        c=c,
    )
    augmented = _score_condition(
        name="official_plus_external",
        train=train,
        all_embeddings=all_embeddings,
        all_labels=all_labels,
        external_rows=len(external),
        augmented=True,
        c=c,
    )
    augmented["delta_vs_official_only"] = (
        augmented["balanced_accuracy"] - official["balanced_accuracy"]
    )

    return {
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
        },
        "classifier": {
            "type": "LinearSVC",
            "class_weight": "balanced",
            "C": c,
        },
        "experiments": [official, augmented],
        "decision": {
            "external_improves_frozen_encoder_oof": (
                augmented["balanced_accuracy"] > official["balanced_accuracy"]
            ),
            "delta": augmented["delta_vs_official_only"],
        },
    }


def encode_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str | None = None,
) -> np.ndarray:
    # Heavy dependencies are intentionally imported lazily so split/metric tests stay cheap.
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()

    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = [str(x) for x in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(resolved_device) for k, v in encoded.items()}
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu().numpy().astype(np.float32, copy=False))

    if not batches:
        raise ValueError("cannot encode an empty text collection")
    return np.vstack(batches)


def load_data(
    *,
    train_path: Path,
    folds_path: Path,
    external_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    folds = pd.read_csv(folds_path)[["id", "fold_id"]]
    train = train.merge(folds, on="id", how="left", validate="one_to_one")
    external = pd.read_parquet(external_path)[["text", "model"]].copy()
    validate_fold_contract(train)
    return train, external


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen encoder A/B on the permanent prompt-grouped folds."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--folds", type=Path, default=Path("artifacts/phase1/folds.csv")
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase3/frozen_modernbert_ablation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train, external = load_data(
        train_path=args.train,
        folds_path=args.folds,
        external_path=args.external,
    )

    all_text = pd.concat([train["text"], external["text"]], ignore_index=True).astype(str)
    embeddings = encode_texts(
        all_text.tolist(),
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    official_embeddings = embeddings[: len(train)]
    external_embeddings = embeddings[len(train) :]

    report = run_embedding_ablation(
        train=train,
        official_embeddings=official_embeddings,
        external=external,
        external_embeddings=external_embeddings,
        c=args.c,
    )
    report["encoder"] = {
        "model": args.model,
        "pooling": "attention_masked_mean_last_hidden_state_then_l2_normalize",
        "frozen": True,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
    }
    report["data"] = {
        "official_rows": int(len(train)),
        "external_rows": int(len(external)),
        "external_labels": sorted(external["model"].astype(str).unique()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
