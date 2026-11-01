from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.svm import LinearSVC

from scripts.run_phase14_public_sparse import (
    EXPECTED_RAW_SHA256,
    SEED_SOURCES,
    build_conservative_subset,
    sha256_file,
)
from scripts.phase15_terms_clean import build_terms_clean_subset


DEFAULT_MODEL = "answerdotai/ModernBERT-base"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 16
SVC_C = 1.0
MIN_SOURCE_ROWS = 100
MIN_SOURCE_PROMPTS = 20
MIN_SHARED_LABELS = 3


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _validate_embeddings(frame: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings)
    if values.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if len(values) != len(frame):
        raise ValueError("embedding rows must match frame rows")
    return values.astype(np.float32, copy=False)


def build_prompt_residuals(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Center frozen embeddings within prompt groups and return the LDA-fit mask."""
    _require_columns(frame, {"prompt_id", "model"})
    values = _validate_embeddings(frame, embeddings)

    residuals = np.zeros(values.shape, dtype=np.float32)
    fit_mask = np.zeros(len(frame), dtype=bool)

    prompt_ids = frame["prompt_id"].astype(str).to_numpy()
    labels = frame["model"].astype(str).to_numpy()

    for prompt_id in pd.unique(prompt_ids):
        idx = np.flatnonzero(prompt_ids == prompt_id)
        group = values[idx]
        centered = group - group.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        if np.any(nonzero):
            centered = centered.copy()
            centered[nonzero] /= norms[nonzero]
        residuals[idx] = centered

        if len(set(labels[idx])) >= 2:
            fit_mask[idx] = True

    return residuals, fit_mask


def build_source_holdout(
    frame: pd.DataFrame,
    source: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a source holdout while removing prompt-bank leakage from training."""
    _require_columns(frame, {"source", "prompt_id", "prompt_key", "model"})

    source_values = frame["source"].astype(str)
    heldout_mask = source_values.eq(str(source)).to_numpy()
    heldout_idx = np.flatnonzero(heldout_mask)
    if len(heldout_idx) == 0:
        raise ValueError(f"source has no rows: {source}")

    heldout = frame.iloc[heldout_idx]
    heldout_prompt_ids = set(heldout["prompt_id"].astype(str))
    heldout_prompt_keys = set(heldout["prompt_key"].astype(str))

    prompt_ids = frame["prompt_id"].astype(str)
    prompt_keys = frame["prompt_key"].astype(str)
    leakage_mask = prompt_ids.isin(heldout_prompt_ids) | prompt_keys.isin(heldout_prompt_keys)
    train_mask = (~heldout_mask) & (~leakage_mask.to_numpy())
    train_idx = np.flatnonzero(train_mask)

    train_labels = set(frame.iloc[train_idx]["model"].astype(str))
    heldout_labels = set(heldout["model"].astype(str))
    shared_labels = sorted(train_labels & heldout_labels)
    return train_idx, heldout_idx, shared_labels


def eligible_seed_sources(frame: pd.DataFrame) -> list[str]:
    """Return seed sources satisfying the contract-frozen Gate A eligibility rules."""
    _require_columns(frame, {"source", "prompt_id", "prompt_key", "model"})

    eligible: list[str] = []
    sources = frame["source"].astype(str)
    for source in sorted(SEED_SOURCES):
        source_frame = frame.loc[sources.eq(source)]
        if len(source_frame) < MIN_SOURCE_ROWS:
            continue
        if source_frame["prompt_id"].astype(str).nunique() < MIN_SOURCE_PROMPTS:
            continue
        _, _, shared_labels = build_source_holdout(frame, source)
        if len(shared_labels) < MIN_SHARED_LABELS:
            continue
        eligible.append(source)
    return eligible


def fit_style_projection(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
) -> LinearDiscriminantAnalysis:
    """Fit the contract-frozen LDA style projection on prompt residuals only."""
    values = _validate_embeddings(frame, embeddings)
    residuals, fit_mask = build_prompt_residuals(frame, values)
    labels = frame["model"].astype(str).to_numpy()

    if int(fit_mask.sum()) == 0:
        raise ValueError("no multi-label prompt rows available for style projection")
    fit_labels = labels[fit_mask]
    if len(set(fit_labels)) < 2:
        raise ValueError("style projection requires at least two model labels")

    projection = LinearDiscriminantAnalysis(solver="svd")
    projection.fit(residuals[fit_mask], fit_labels)
    return projection


def _score_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    labels: list[str],
) -> dict:
    classifier = LinearSVC(class_weight="balanced", C=SVC_C)
    classifier.fit(train_x, train_y)
    predictions = classifier.predict(test_x)
    recalls = recall_score(
        test_y,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        "scored_rows": int(len(test_y)),
        "per_label_recall": {
            label: float(value) for label, value in zip(labels, recalls)
        },
    }


def evaluate_source_holdout(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    source: str,
) -> dict:
    """Compare raw frozen embeddings and prompt-residual projection on one source."""
    values = _validate_embeddings(frame, embeddings)
    train_idx, heldout_idx, shared_labels = build_source_holdout(frame, source)
    if len(shared_labels) < 2:
        raise ValueError(f"source holdout has fewer than two shared labels: {source}")

    labels = frame["model"].astype(str).to_numpy()
    train_idx = train_idx[np.isin(labels[train_idx], shared_labels)]
    heldout_idx = heldout_idx[np.isin(labels[heldout_idx], shared_labels)]
    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    train_embeddings = values[train_idx]
    heldout_embeddings = values[heldout_idx]
    train_y = labels[train_idx]
    heldout_y = labels[heldout_idx]

    projection = fit_style_projection(train_frame, train_embeddings)
    projected_train = projection.transform(train_embeddings)
    projected_heldout = projection.transform(heldout_embeddings)

    raw = _score_classifier(
        train_embeddings,
        train_y,
        heldout_embeddings,
        heldout_y,
        shared_labels,
    )
    projected = _score_classifier(
        projected_train,
        train_y,
        projected_heldout,
        heldout_y,
        shared_labels,
    )

    train_prompt_ids = set(train_frame["prompt_id"].astype(str))
    train_prompt_keys = set(train_frame["prompt_key"].astype(str))
    heldout_frame = frame.iloc[heldout_idx]
    heldout_prompt_ids = set(heldout_frame["prompt_id"].astype(str))
    heldout_prompt_keys = set(heldout_frame["prompt_key"].astype(str))

    return {
        "source": str(source),
        "shared_labels": shared_labels,
        "training_rows": int(len(train_idx)),
        "heldout_rows": int(len(heldout_idx)),
        "training_prompt_ids": int(train_frame["prompt_id"].astype(str).nunique()),
        "heldout_prompt_ids": int(heldout_frame["prompt_id"].astype(str).nunique()),
        "prompt_id_overlap": int(len(train_prompt_ids & heldout_prompt_ids)),
        "prompt_key_overlap": int(len(train_prompt_keys & heldout_prompt_keys)),
        "raw": raw,
        "projected": projected,
        "delta_projected_minus_raw": float(
            projected["balanced_accuracy"] - raw["balanced_accuracy"]
        ),
    }


def run_gate_a(frame: pd.DataFrame, embeddings: np.ndarray) -> dict:
    """Run the frozen external-only Gate A and return its predeclared decision."""
    _validate_embeddings(frame, embeddings)
    sources = eligible_seed_sources(frame)
    source_results = [evaluate_source_holdout(frame, embeddings, source) for source in sources]

    if source_results:
        raw_mean = float(np.mean([r["raw"]["balanced_accuracy"] for r in source_results]))
        projected_mean = float(
            np.mean([r["projected"]["balanced_accuracy"] for r in source_results])
        )
        mean_delta = float(
            np.mean([r["delta_projected_minus_raw"] for r in source_results])
        )
        worse_sources = int(
            sum(r["delta_projected_minus_raw"] < 0.0 for r in source_results)
        )
    else:
        raw_mean = projected_mean = mean_delta = float("nan")
        worse_sources = 0

    aggregate = {
        "mean_raw_balanced_accuracy": raw_mean,
        "mean_projected_balanced_accuracy": projected_mean,
        "mean_delta_projected_minus_raw": mean_delta,
        "worse_sources": worse_sources,
        "eligible_source_count": int(len(sources)),
    }

    if len(sources) < 2:
        decision = {
            "status": "inconclusive",
            "official_oof_authorized": False,
            "reason": "fewer_than_two_eligible_sources",
        }
    else:
        passes = mean_delta > 0.0 and worse_sources <= len(sources) // 2
        decision = {
            "status": "pass" if passes else "fail",
            "official_oof_authorized": bool(passes),
            "reason": (
                "frozen_gate_passed"
                if passes
                else "mean_delta_or_source_consistency_rule_failed"
            ),
        }

    return {
        "eligible_sources": sources,
        "source_results": source_results,
        "aggregate": aggregate,
        "decision": decision,
    }


def build_execution_report(
    gate_a: dict,
    *,
    raw_sha256: str,
    phase14_rows: int,
    phase15_rows: int,
    phase15_prompt_ids: int,
    phase15_labels: int,
    model_name: str,
    max_length: int,
) -> dict:
    return {
        "scope": "external_transfer_gate_a_only_no_official_oof",
        "data": {
            "raw_pool_sha256": str(raw_sha256),
            "phase14_rows": int(phase14_rows),
            "phase15_rows": int(phase15_rows),
            "phase15_prompt_ids": int(phase15_prompt_ids),
            "phase15_labels": int(phase15_labels),
        },
        "encoder": {
            "model": str(model_name),
            "frozen": True,
            "max_length": int(max_length),
            "pooling": "attention_mask_mean_then_l2",
        },
        "projection": {
            "type": "LinearDiscriminantAnalysis",
            "solver": "svd",
            "fit_representation": "within_prompt_centered_l2_residuals",
        },
        "classifier": {
            "type": "LinearSVC",
            "class_weight": "balanced",
            "C": SVC_C,
        },
        "official_rows_scored": 0,
        "kaggle_submission_authorized": False,
        "gate_a": gate_a,
    }


def _resolve_raw_pool(raw_pool: str | None) -> Path:
    if raw_pool is not None:
        path = Path(raw_pool)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    kaggle_root = Path("/kaggle/input")
    matches = sorted(kaggle_root.glob("**/pool_v4.parquet")) if kaggle_root.exists() else []
    if len(matches) != 1:
        raise ValueError(f"expected exactly one Kaggle pool_v4.parquet, found {len(matches)}")
    return matches[0]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen prompt-residual style Gate A")
    parser.add_argument("--raw-pool")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_path = _resolve_raw_pool(args.raw_pool)
    observed_sha = sha256_file(raw_path)
    if observed_sha != EXPECTED_RAW_SHA256:
        raise ValueError(
            f"raw pool SHA mismatch: observed={observed_sha} expected={EXPECTED_RAW_SHA256}"
        )

    pool = pd.read_parquet(raw_path)
    phase14 = build_conservative_subset(pool, validate_full_contract=True)
    phase15 = build_terms_clean_subset(phase14, validate_full_contract=True)

    # Heavy encoder dependencies stay lazy so unit tests remain CPU/lightweight.
    from scripts.run_frozen_encoder_ablation import encode_texts

    embeddings = encode_texts(
        phase15["text"].astype(str).tolist(),
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    gate_a = run_gate_a(phase15, embeddings)
    report = build_execution_report(
        gate_a,
        raw_sha256=observed_sha,
        phase14_rows=len(phase14),
        phase15_rows=len(phase15),
        phase15_prompt_ids=phase15["prompt_id"].astype(str).nunique(),
        phase15_labels=phase15["model"].astype(str).nunique(),
        model_name=args.model,
        max_length=args.max_length,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"aggregate": gate_a["aggregate"], "decision": gate_a["decision"]}, indent=2))


if __name__ == "__main__":
    main()
