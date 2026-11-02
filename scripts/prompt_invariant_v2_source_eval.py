from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20261102
SAFETY_MIN_ROWS = 20
SAFETY_MAX_RECALL_DROP = 0.15
MAX_NEGATIVE_SOURCES = 1


@dataclass(frozen=True)
class SourceTransferDecision:
    status: str
    v1_balanced_accuracy: float
    v2_balanced_accuracy: float
    delta: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    mean_source_delta: float
    negative_source_count: int
    safety_veto_labels: tuple[str, ...]
    reasons: tuple[str, ...]


def _recalls(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, float]:
    true = y_true.astype(str).to_numpy()
    pred = y_pred.astype(str).to_numpy()
    result: dict[str, float] = {}
    for label in labels:
        mask = true == label
        if bool(mask.any()):
            result[label] = float(np.mean(pred[mask] == label))
    if not result:
        raise ValueError("no labels available for scoring")
    return result


def _balanced_accuracy(frame: pd.DataFrame, pred_col: str, labels: list[str]) -> float:
    return float(np.mean(list(_recalls(frame["model"], frame[pred_col], labels).values())))


def _prompt_bootstrap(
    frame: pd.DataFrame,
    labels: list[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    groups = sorted(frame["prompt_id"].astype(str).unique().tolist())
    if len(groups) < 2:
        raise ValueError("prompt bootstrap requires at least two prompt groups")
    indices = {
        group: frame.index[frame["prompt_id"].astype(str).eq(group)].to_numpy(dtype=np.int64)
        for group in groups
    }
    rng = np.random.RandomState(seed)
    deltas = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([indices[str(group)] for group in sampled])
        sample = frame.loc[rows]
        deltas[i] = _balanced_accuracy(sample, "v2_prediction", labels) - _balanced_accuracy(
            sample, "v1_prediction", labels
        )
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return float(lower), float(upper)


def evaluate_source_transfer(
    frame: pd.DataFrame,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[SourceTransferDecision, dict[str, object]]:
    required = {"source", "prompt_id", "model", "v1_prediction", "v2_prediction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing transfer prediction columns: {missing}")
    if frame[list(required)].isna().any().any() or len(frame) == 0:
        raise ValueError("transfer prediction frame must be non-empty and null-free")

    labels = sorted(frame["model"].astype(str).unique().tolist())
    v1_ba = _balanced_accuracy(frame, "v1_prediction", labels)
    v2_ba = _balanced_accuracy(frame, "v2_prediction", labels)
    delta = float(v2_ba - v1_ba)
    lower, upper = _prompt_bootstrap(frame, labels, resamples=resamples, seed=seed)

    source_rows: list[dict[str, object]] = []
    source_deltas: list[float] = []
    for source in sorted(frame["source"].astype(str).unique().tolist()):
        part = frame.loc[frame["source"].astype(str).eq(source)]
        source_labels = sorted(part["model"].astype(str).unique().tolist())
        v1_source = _balanced_accuracy(part, "v1_prediction", source_labels)
        v2_source = _balanced_accuracy(part, "v2_prediction", source_labels)
        source_delta = float(v2_source - v1_source)
        source_deltas.append(source_delta)
        source_rows.append(
            {
                "source": source,
                "rows": int(len(part)),
                "labels": int(len(source_labels)),
                "v1_balanced_accuracy": v1_source,
                "v2_balanced_accuracy": v2_source,
                "delta": source_delta,
            }
        )

    mean_source_delta = float(np.mean(source_deltas))
    negative_source_count = int(np.sum(np.asarray(source_deltas) < 0.0))
    counts = frame["model"].astype(str).value_counts().to_dict()
    v1_recalls = _recalls(frame["model"], frame["v1_prediction"], labels)
    v2_recalls = _recalls(frame["model"], frame["v2_prediction"], labels)
    veto = tuple(
        label
        for label in labels
        if int(counts.get(label, 0)) >= SAFETY_MIN_ROWS
        and float(v1_recalls[label] - v2_recalls[label]) > SAFETY_MAX_RECALL_DROP
    )

    reasons: list[str] = []
    if not v2_ba > v1_ba:
        reasons.append("v2_pooled_ba_not_above_v1")
    if not lower > 0.0:
        reasons.append("prompt_bootstrap_lower_bound_not_positive")
    if not mean_source_delta > 0.0:
        reasons.append("mean_source_delta_not_positive")
    if negative_source_count > MAX_NEGATIVE_SOURCES:
        reasons.append("too_many_negative_sources")
    if veto:
        reasons.append("per_label_safety_veto")

    decision = SourceTransferDecision(
        status="pass" if not reasons else "fail",
        v1_balanced_accuracy=v1_ba,
        v2_balanced_accuracy=v2_ba,
        delta=delta,
        bootstrap_ci_lower=lower,
        bootstrap_ci_upper=upper,
        mean_source_delta=mean_source_delta,
        negative_source_count=negative_source_count,
        safety_veto_labels=veto,
        reasons=tuple(reasons),
    )
    details: dict[str, object] = {
        "decision": asdict(decision),
        "rows": int(len(frame)),
        "prompt_groups": int(frame["prompt_id"].astype(str).nunique()),
        "sources": source_rows,
        "per_label_rows": {str(k): int(v) for k, v in sorted(counts.items())},
        "v1_per_label_recall": v1_recalls,
        "v2_per_label_recall": v2_recalls,
        "gate": {
            "bootstrap_resamples": int(resamples),
            "bootstrap_seed": int(seed),
            "safety_min_rows": SAFETY_MIN_ROWS,
            "safety_max_recall_drop": SAFETY_MAX_RECALL_DROP,
            "max_negative_sources": MAX_NEGATIVE_SOURCES,
        },
    }
    return decision, details
