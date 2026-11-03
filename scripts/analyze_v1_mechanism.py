from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


FAMILY_PREFIXES = ("claude", "gemini", "glm", "gpt", "kimi", "qwen")


def family_of_model(label: str) -> str:
    value = str(label).lower()
    for prefix in FAMILY_PREFIXES:
        if value.startswith(prefix):
            return prefix
    return value.split("-", 1)[0]


def _require_columns(name: str, frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _validate_inputs(train: pd.DataFrame, gate_b: pd.DataFrame, sparse: pd.DataFrame) -> None:
    _require_columns("train", train, {"id", "text", "model"})
    _require_columns(
        "gate_b", gate_b, {"id", "model", "plain_prediction", "centered_prediction"}
    )
    _require_columns("sparse", sparse, {"id", "model", "prediction"})
    for name, frame in (("train", train), ("gate_b", gate_b), ("sparse", sparse)):
        if frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ValueError(f"{name} ids must be non-null and unique")
    id_sets = [set(frame["id"].astype(str)) for frame in (train, gate_b, sparse)]
    if not (id_sets[0] == id_sets[1] == id_sets[2]):
        raise ValueError("id sets differ across train/gate_b/sparse")

    truth = train[["id", "model"]].copy()
    truth["id"] = truth["id"].astype(str)
    truth["model"] = truth["model"].astype(str)
    for name, frame in (("gate_b", gate_b), ("sparse", sparse)):
        observed = frame[["id", "model"]].copy()
        observed["id"] = observed["id"].astype(str)
        observed["model"] = observed["model"].astype(str)
        merged = truth.merge(observed, on="id", suffixes=("_train", f"_{name}"))
        if not (merged["model_train"] == merged[f"model_{name}"]).all():
            raise ValueError(f"{name} true-model labels differ from train")


def _safe_accuracy(frame: pd.DataFrame, column: str) -> float | None:
    if len(frame) == 0:
        return None
    return float(frame[column].mean())


def _agreement_pattern(row: pd.Series) -> str:
    p = row["plain_prediction"]
    c = row["centered_prediction"]
    s = row["sparse_prediction"]
    if p == c == s:
        return "all_agree"
    if p == c:
        return "plain_centered_agree"
    if p == s:
        return "plain_sparse_agree"
    if c == s:
        return "centered_sparse_agree"
    return "all_distinct"


def _length_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    ranked = frame["words"].rank(method="first")
    quartile_index = pd.qcut(ranked, 4, labels=False) + 1
    quartiles: list[dict[str, Any]] = []
    for q in range(1, 5):
        part = frame.loc[quartile_index == q]
        quartiles.append(
            {
                "quartile": q,
                "rows": int(len(part)),
                "min_words": int(part["words"].min()),
                "max_words": int(part["words"].max()),
                "mean_words": float(part["words"].mean()),
                "plain_accuracy": float(part["plain_correct"].mean()),
                "centered_accuracy": float(part["centered_correct"].mean()),
                "sparse_accuracy": float(part["sparse_correct"].mean()),
            }
        )

    per_model: list[dict[str, Any]] = []
    for model, group in frame.groupby("model", sort=True):
        median_words = float(group["words"].median())
        short = group.loc[group["words"] <= median_words]
        long = group.loc[group["words"] > median_words]
        if short.empty or long.empty:
            continue
        short_acc = float(short["plain_correct"].mean())
        long_acc = float(long["plain_correct"].mean())
        per_model.append(
            {
                "model": str(model),
                "median_words": median_words,
                "short_rows": int(len(short)),
                "long_rows": int(len(long)),
                "short_accuracy": short_acc,
                "long_accuracy": long_acc,
                "long_minus_short_accuracy": long_acc - short_acc,
            }
        )
    deltas = [item["long_minus_short_accuracy"] for item in per_model]
    return {
        "quartiles": quartiles,
        "within_model_median": {
            "models_evaluable": len(per_model),
            "models_with_positive_long_minus_short": int(sum(delta > 0 for delta in deltas)),
            "macro_long_minus_short_accuracy": float(np.mean(deltas)) if deltas else None,
            "per_model": per_model,
        },
    }


def _complementarity(frame: pd.DataFrame, alt: str) -> dict[str, Any]:
    alt_correct = f"{alt}_correct"
    plain_error = ~frame["plain_correct"]
    rescues = plain_error & frame[alt_correct]
    breaks = frame["plain_correct"] & ~frame[alt_correct]
    same_family_plain_error = frame["plain_prediction_family"] == frame["model_family"]
    return {
        "rescues": int(rescues.sum()),
        "breaks": int(breaks.sum()),
        "net_correct": int(rescues.sum() - breaks.sum()),
        "rescued_same_family_plain_errors": int((rescues & same_family_plain_error).sum()),
        "rescued_cross_family_plain_errors": int((rescues & ~same_family_plain_error).sum()),
    }


def analyze_frames(
    train: pd.DataFrame, gate_b: pd.DataFrame, sparse: pd.DataFrame
) -> dict[str, Any]:
    _validate_inputs(train, gate_b, sparse)

    base = train[["id", "text", "model"]].copy()
    base["id"] = base["id"].astype(str)
    gate = gate_b[["id", "plain_prediction", "centered_prediction"]].copy()
    gate["id"] = gate["id"].astype(str)
    sparse_pred = sparse[["id", "prediction"]].rename(
        columns={"prediction": "sparse_prediction"}
    ).copy()
    sparse_pred["id"] = sparse_pred["id"].astype(str)
    frame = base.merge(gate, on="id", validate="one_to_one").merge(
        sparse_pred, on="id", validate="one_to_one"
    )
    frame = frame.sort_values("id").reset_index(drop=True)

    for column in ("model", "plain_prediction", "centered_prediction", "sparse_prediction"):
        frame[column] = frame[column].astype(str)
        frame[f"{column}_family"] = frame[column].map(family_of_model)
    frame["plain_correct"] = frame["plain_prediction"] == frame["model"]
    frame["centered_correct"] = frame["centered_prediction"] == frame["model"]
    frame["sparse_correct"] = frame["sparse_prediction"] == frame["model"]
    frame["words"] = frame["text"].fillna("").astype(str).str.split().str.len().astype(int)

    errors = frame.loc[~frame["plain_correct"]].copy()
    same_family_errors = errors["plain_prediction_family"] == errors["model_family"]

    pattern_order = (
        "all_agree",
        "plain_centered_agree",
        "plain_sparse_agree",
        "centered_sparse_agree",
        "all_distinct",
    )
    frame["agreement_pattern"] = frame.apply(_agreement_pattern, axis=1)
    agreement: dict[str, Any] = {}
    for pattern in pattern_order:
        part = frame.loc[frame["agreement_pattern"] == pattern]
        agreement[pattern] = {
            "rows": int(len(part)),
            "plain_accuracy": _safe_accuracy(part, "plain_correct"),
            "centered_accuracy": _safe_accuracy(part, "centered_correct"),
            "sparse_accuracy": _safe_accuracy(part, "sparse_correct"),
        }

    confusions = (
        errors.groupby(["model", "plain_prediction"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "model", "plain_prediction"], ascending=[False, True, True])
        .head(20)
    )
    top_confusions = [
        {"true": str(row.model), "predicted": str(row.plain_prediction), "count": int(row.count)}
        for row in confusions.itertuples(index=False)
    ]

    return {
        "status": "POST-HOC DIAGNOSTIC ONLY",
        "rows": int(len(frame)),
        "balanced_accuracy": {
            "plain": float(balanced_accuracy_score(frame["model"], frame["plain_prediction"])),
            "centered": float(balanced_accuracy_score(frame["model"], frame["centered_prediction"])),
            "sparse": float(balanced_accuracy_score(frame["model"], frame["sparse_prediction"])),
        },
        "plain_errors": {
            "count": int(len(errors)),
            "same_family_count": int(same_family_errors.sum()),
            "cross_family_count": int((~same_family_errors).sum()),
            "same_family_rate": float(same_family_errors.mean()) if len(errors) else None,
        },
        "length": _length_diagnostics(frame),
        "complementarity": {
            "centered_vs_plain": _complementarity(frame, "centered"),
            "sparse_vs_plain": _complementarity(frame, "sparse"),
        },
        "agreement_patterns": agreement,
        "top_plain_confusions": top_confusions,
    }


def render_markdown(report: dict[str, Any]) -> str:
    ba = report["balanced_accuracy"]
    errors = report["plain_errors"]
    within = report["length"]["within_model_median"]
    lines = [
        "# V1 Mechanism Audit",
        "",
        "**POST-HOC DIAGNOSTIC ONLY — not promotion evidence and not a routing/blend search.**",
        "",
        "## Headline",
        "",
        f"- Plain BA: `{ba['plain']:.10f}`",
        f"- Centered BA: `{ba['centered']:.10f}`",
        f"- Sparse BA: `{ba['sparse']:.10f}`",
        f"- Plain errors: `{errors['count']}`",
        f"- Same-family plain errors: `{errors['same_family_count']}` / `{errors['count']}` (`{errors['same_family_rate']:.3%}`)",
        "",
        "## Within-model length",
        "",
        f"Longer half beats shorter half for `{within['models_with_positive_long_minus_short']}` / `{within['models_evaluable']}` evaluable labels; macro long-minus-short accuracy delta: `{within['macro_long_minus_short_accuracy']:+.4f}`.",
        "",
        "## Complementarity",
        "",
    ]
    for label, key in (("Centered", "centered_vs_plain"), ("Sparse", "sparse_vs_plain")):
        item = report["complementarity"][key]
        lines.append(
            f"- {label}: rescues `{item['rescues']}`, breaks `{item['breaks']}`, net `{item['net_correct']:+d}`; rescued Same-family / cross-family plain errors = `{item['rescued_same_family_plain_errors']}` / `{item['rescued_cross_family_plain_errors']}`."
        )
    lines.extend(["", "## Agreement patterns", ""])
    for name, item in report["agreement_patterns"].items():
        lines.append(
            f"- `{name}`: n=`{item['rows']}`, plain=`{item['plain_accuracy']}`, centered=`{item['centered_accuracy']}`, sparse=`{item['sparse_accuracy']}`"
        )
    lines.extend(["", "## Top plain confusions", ""])
    for item in report["top_plain_confusions"]:
        lines.append(f"- `{item['true']}` → `{item['predicted']}`: `{item['count']}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These are explanations from already-consumed OOF evidence. They cannot alter the frozen v3 gate or authorize a new ensemble/submission.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc v1 mechanism audit")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--gate-b-oof", type=Path, required=True)
    parser.add_argument("--sparse-oof", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_frames(
        pd.read_csv(args.train),
        pd.read_csv(args.gate_b_oof),
        pd.read_csv(args.sparse_oof),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
