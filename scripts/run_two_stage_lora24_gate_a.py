from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


SOURCE_HOLDOUT_SALT = "ntl-two-stage-lora24-v1-source-holdout"
MIN_ELIGIBLE_LABELS = 8
MIN_ELIGIBLE_HELDOUT_ROWS = 300
MIN_HELDOUT_ROWS_PER_LABEL = 20


def gate_a_training_spec() -> dict[str, object]:
    """Return the frozen Stage-1 training specification for Gate A."""
    return {
        "backbone": "Qwen3-8B-Base",
        "keep_layers": 24,
        "pooled_layers": [2, 4, 6, 8, 12, 16, 20, 24],
        "max_length_before_eos": 447,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "head_dropout": 0.1,
        "epochs": 1,
        "gradient_accumulation_triplets": 6,
        "adapter_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 0.01,
        "warmup_fraction": 0.10,
        "gradient_clip": 1.0,
        "seed": 20261101,
    }


def gate_a_condition_specs() -> dict[str, dict[str, object]]:
    """Return the two frozen Gate-A conditions; only LoRA trainability differs."""
    return {
        "A0": {
            "name": "frozen_backbone_head_only",
            "train_lora": False,
        },
        "A1": {
            "name": "lora_adaptation",
            "train_lora": True,
        },
    }


def select_holdout_sources(frame: pd.DataFrame) -> list[str]:
    """Select the frozen three-source holdout by full SHA256 digest order."""
    sources = sorted(frame["source"].astype(str).unique().tolist())
    ranked = sorted(
        sources,
        key=lambda source: hashlib.sha256(
            f"{SOURCE_HOLDOUT_SALT}|{source}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:3]


def build_stage1_split(frame: pd.DataFrame) -> dict[str, object]:
    """Build the frozen source holdout and remove prompt-bank leakage from train."""
    holdout_sources = select_holdout_sources(frame)
    source_values = frame["source"].astype(str)
    heldout_mask = source_values.isin(holdout_sources)
    heldout_idx = np.flatnonzero(heldout_mask.to_numpy())

    heldout_prompt_ids = set(frame.loc[heldout_mask, "prompt_id"].astype(str))
    train_mask = ~heldout_mask
    train_mask &= ~frame["prompt_id"].astype(str).isin(heldout_prompt_ids)

    heldout_prompt_keys: set[str] = set()
    if "prompt_key" in frame.columns:
        heldout_prompt_keys = set(frame.loc[heldout_mask, "prompt_key"].astype(str))
        train_mask &= ~frame["prompt_key"].astype(str).isin(heldout_prompt_keys)

    train_idx = np.flatnonzero(train_mask.to_numpy())
    train = frame.iloc[train_idx]
    heldout = frame.iloc[heldout_idx]

    prompt_id_overlap = len(
        set(train["prompt_id"].astype(str)) & set(heldout["prompt_id"].astype(str))
    )
    prompt_key_overlap = 0
    if "prompt_key" in frame.columns:
        prompt_key_overlap = len(
            set(train["prompt_key"].astype(str))
            & set(heldout["prompt_key"].astype(str))
        )

    return {
        "holdout_sources": holdout_sources,
        "train_idx": train_idx,
        "heldout_idx": heldout_idx,
        "prompt_id_overlap": prompt_id_overlap,
        "prompt_key_overlap": prompt_key_overlap,
    }


def select_eligible_labels(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    min_heldout_rows: int = MIN_HELDOUT_ROWS_PER_LABEL,
) -> list[str]:
    """Return labels present in train with enough union-heldout examples."""
    train_labels = set(train["model"].astype(str))
    heldout_counts = heldout["model"].astype(str).value_counts()
    return sorted(
        label
        for label, count in heldout_counts.items()
        if label in train_labels and int(count) >= min_heldout_rows
    )


def compute_class_weights(
    labels: Sequence[object] | np.ndarray,
    *,
    classes: Sequence[str],
) -> dict[str, float]:
    """Inverse-frequency class weights normalized to arithmetic mean 1."""
    label_values = np.asarray(labels, dtype=object).astype(str)
    counts = {str(cls): int(np.sum(label_values == str(cls))) for cls in classes}
    missing = [label for label, count in counts.items() if count <= 0]
    if missing:
        raise ValueError(f"class weights require positive training counts: {missing}")
    raw = {label: 1.0 / count for label, count in counts.items()}
    mean_weight = float(np.mean(list(raw.values())))
    return {label: float(weight / mean_weight) for label, weight in raw.items()}


def build_prompt_triplets(
    frame: pd.DataFrame,
    rows: Sequence[int] | np.ndarray,
    *,
    seed: int,
) -> list[np.ndarray]:
    """Build deterministic microbatches of up to three rows from one prompt only."""
    rows_array = np.asarray(rows, dtype=int)
    rng = np.random.RandomState(seed)
    batches: list[np.ndarray] = []
    prompt_values = frame["prompt_id"].astype(str).to_numpy()

    for prompt_id in sorted(set(prompt_values[rows_array])):
        prompt_rows = rows_array[prompt_values[rows_array] == prompt_id].copy()
        prompt_rows = rng.permutation(prompt_rows)
        for start in range(0, len(prompt_rows), 3):
            batches.append(np.asarray(prompt_rows[start : start + 3], dtype=int))

    if batches:
        order = rng.permutation(len(batches))
        batches = [batches[int(i)] for i in order]
    return batches


def pad_triplets_for_accumulation(
    batches: Sequence[np.ndarray],
    *,
    accumulation: int,
    seed: int,
) -> list[np.ndarray]:
    """Legacy helper retained until the GPU runner migrates to exact grouping."""
    if not batches:
        raise ValueError("at least one triplet is required")
    if accumulation <= 0:
        raise ValueError("accumulation must be positive")

    padded = [np.asarray(batch, dtype=int).copy() for batch in batches]
    remainder = len(padded) % accumulation
    if remainder == 0:
        return padded

    rng = np.random.RandomState(seed)
    needed = accumulation - remainder
    choices = rng.randint(0, len(padded), size=needed)
    padded.extend(padded[int(i)].copy() for i in choices)
    return padded


def group_triplets_for_accumulation(
    batches: Sequence[np.ndarray],
    *,
    accumulation: int,
) -> list[list[np.ndarray]]:
    """Group one exact epoch into optimizer steps without duplicating triplets."""
    if not batches:
        raise ValueError("at least one triplet is required")
    if accumulation <= 0:
        raise ValueError("accumulation must be positive")

    copied = [np.asarray(batch, dtype=int).copy() for batch in batches]
    return [
        copied[start : start + accumulation]
        for start in range(0, len(copied), accumulation)
    ]


def masked_mean_pool(hidden, attention_mask, *, mask_first_token: bool):
    """Mean-pool non-padding tokens, optionally excluding token position zero."""
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype).clone()
    if mask_first_token and mask.shape[1] > 0:
        mask[:, 0] = 0
    denominator = mask.sum(dim=1).clamp(min=1)
    return (hidden * mask).sum(dim=1) / denominator


def center_triplet(features):
    """Center one same-prompt microbatch independently in feature space."""
    return features - features.mean(dim=0, keepdim=True)


def warmup_cosine_multiplier(
    *,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    """Frozen warm-up/cosine multiplier on a conceptual 0..total_steps axis."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be between 0 and total_steps")
    if step < 0:
        raise ValueError("step must be nonnegative")
    if step >= total_steps:
        return 0.0
    if warmup_steps > 0 and step <= warmup_steps:
        return float(step / warmup_steps)

    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        return 0.0
    progress = (step - warmup_steps) / decay_steps
    return float(0.5 * (1.0 + math.cos(math.pi * progress)))


def truncate_qwen_layers(model, *, keep_layers: int):
    """Keep only the first N decoder layers on a Qwen-style model object."""
    if keep_layers <= 0:
        raise ValueError("keep_layers must be positive")

    container = getattr(model, "model", model)
    if not hasattr(container, "layers"):
        raise AttributeError("model has no Qwen-style layers container")
    layers = container.layers
    if len(layers) < keep_layers:
        raise ValueError(
            f"model has fewer layers than requested: {len(layers)} < {keep_layers}"
        )
    container.layers = layers[:keep_layers]
    return model


def score_predictions(
    y_true: Sequence[object] | np.ndarray,
    y_pred: Sequence[object] | np.ndarray,
    *,
    labels: Sequence[str],
) -> dict[str, object]:
    """Compute the competition metric as the mean recall over requested labels."""
    true_values = np.asarray(y_true, dtype=object).astype(str)
    pred_values = np.asarray(y_pred, dtype=object).astype(str)
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")

    recalls: dict[str, float] = {}
    counts: dict[str, int] = {}
    for label in [str(value) for value in labels]:
        mask = true_values == label
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"no evaluation rows for requested label: {label}")
        counts[label] = count
        recalls[label] = float(np.mean(pred_values[mask] == label))

    return {
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "per_label_recall": recalls,
        "label_counts": counts,
    }


def gate_a_decision(
    *,
    eligible_label_count: int,
    eligible_heldout_rows: int,
    pooled_delta: float,
    source_deltas: Sequence[float],
) -> dict[str, object]:
    """Apply the frozen Gate-A PASS/FAIL/INCONCLUSIVE rules exactly."""
    if len(source_deltas) != 3:
        raise ValueError("Gate A requires exactly three source deltas")

    if (
        eligible_label_count < MIN_ELIGIBLE_LABELS
        or eligible_heldout_rows < MIN_ELIGIBLE_HELDOUT_ROWS
    ):
        return {
            "status": "inconclusive",
            "official_oof_authorized": False,
            "reason": "insufficient_eligible_labels_or_rows",
        }

    deltas = np.asarray(source_deltas, dtype=float)
    passes = (
        float(pooled_delta) > 0.0
        and float(deltas.mean()) > 0.0
        and int(np.sum(deltas < 0.0)) <= 1
    )
    if passes:
        return {
            "status": "pass",
            "official_oof_authorized": True,
            "reason": "all_frozen_gate_a_rules_passed",
        }
    return {
        "status": "fail",
        "official_oof_authorized": False,
        "reason": "pooled_or_source_consistency_rule_failed",
    }
