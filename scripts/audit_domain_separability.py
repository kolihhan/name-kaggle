from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HOLDOUT_SALT = "ntl-domain-separability-v1"
RANDOM_STATE = 20261101

STYLE_COLUMNS = [
    "char_count",
    "token_count",
    "line_count",
    "newline_count",
    "avg_token_length",
    "whitespace_per_1k",
    "digit_per_1k",
    "uppercase_per_1k",
    "non_ascii_per_1k",
    "punct_per_1k",
    "period_per_1k",
    "comma_per_1k",
    "question_per_1k",
    "exclamation_per_1k",
    "colon_per_1k",
    "semicolon_per_1k",
    "backtick_per_1k",
    "brace_per_1k",
    "paren_per_1k",
    "code_fence_count",
    "markdown_heading_count",
    "markdown_list_count",
    "url_count",
]


def _rate_per_1k(count: int, denominator: int) -> float:
    return 1000.0 * float(count) / float(max(denominator, 1))


def extract_style_features(texts: Iterable[object]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for value in texts:
        text = "" if pd.isna(value) else str(value)
        chars = len(text)
        tokens = re.findall(r"\S+", text)
        token_lengths = [len(token) for token in tokens]
        punctuation_count = sum(ch in string.punctuation for ch in text)
        rows.append(
            {
                "char_count": float(chars),
                "token_count": float(len(tokens)),
                "line_count": float(text.count("\n") + 1),
                "newline_count": float(text.count("\n")),
                "avg_token_length": float(np.mean(token_lengths)) if token_lengths else 0.0,
                "whitespace_per_1k": _rate_per_1k(sum(ch.isspace() for ch in text), chars),
                "digit_per_1k": _rate_per_1k(sum(ch.isdigit() for ch in text), chars),
                "uppercase_per_1k": _rate_per_1k(sum(ch.isupper() for ch in text), chars),
                "non_ascii_per_1k": _rate_per_1k(sum(ord(ch) > 127 for ch in text), chars),
                "punct_per_1k": _rate_per_1k(punctuation_count, chars),
                "period_per_1k": _rate_per_1k(text.count("."), chars),
                "comma_per_1k": _rate_per_1k(text.count(","), chars),
                "question_per_1k": _rate_per_1k(text.count("?"), chars),
                "exclamation_per_1k": _rate_per_1k(text.count("!"), chars),
                "colon_per_1k": _rate_per_1k(text.count(":"), chars),
                "semicolon_per_1k": _rate_per_1k(text.count(";"), chars),
                "backtick_per_1k": _rate_per_1k(text.count("`"), chars),
                "brace_per_1k": _rate_per_1k(text.count("{") + text.count("}"), chars),
                "paren_per_1k": _rate_per_1k(text.count("(") + text.count(")"), chars),
                "code_fence_count": float(text.count("```")),
                "markdown_heading_count": float(len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+", text))),
                "markdown_list_count": float(len(re.findall(r"(?m)^\s*[-*+]\s+", text))),
                "url_count": float(len(re.findall(r"https?://\S+", text))),
            }
        )
    return pd.DataFrame(rows, columns=STYLE_COLUMNS)


def assign_domain_holdout(
    frame: pd.DataFrame,
    *,
    comparison: str,
    model_label: str,
) -> pd.DataFrame:
    required = {"domain", "prompt_id", "model", "text"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"domain frame missing required columns: {missing}")

    work = frame.copy()
    work["domain"] = work["domain"].astype(str)
    work["prompt_id"] = work["prompt_id"].astype(str)
    work["model"] = work["model"].astype(str)
    if set(work["model"].unique()) != {str(model_label)}:
        raise ValueError("assign_domain_holdout expects exactly one requested model label")

    def is_holdout(row: pd.Series) -> bool:
        payload = (
            f"{HOLDOUT_SALT}|{comparison}|{model_label}|"
            f"{row['domain']}|{row['prompt_id']}"
        )
        value = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
        return value % 5 == 0

    work["is_holdout"] = work.apply(is_holdout, axis=1).astype(bool)
    prompt_nunique = work.groupby(["domain", "prompt_id"])["is_holdout"].nunique()
    if not prompt_nunique.eq(1).all():
        raise RuntimeError("prompt group was split across domain holdout")
    return work


def build_label_domain_split(
    official: pd.DataFrame,
    external: pd.DataFrame,
    *,
    comparison: str,
    model_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"prompt_id", "model", "text"}
    for name, frame in [("official", official), ("external", external)]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} frame missing required columns: {missing}")

    label = str(model_label)
    off = official.loc[official["model"].astype(str).eq(label), list(required)].copy()
    ext = external.loc[external["model"].astype(str).eq(label), list(required)].copy()
    if off.empty or ext.empty:
        raise ValueError(f"label {label!r} is not present in both domains")
    off["domain"] = "official"
    ext["domain"] = "external"
    combined = pd.concat([off, ext], ignore_index=True)
    combined = assign_domain_holdout(combined, comparison=comparison, model_label=label)
    train = combined.loc[~combined["is_holdout"]].reset_index(drop=True)
    test = combined.loc[combined["is_holdout"]].reset_index(drop=True)

    for split_name, split in [("train", train), ("test", test)]:
        if set(split["domain"].unique()) != {"official", "external"}:
            raise RuntimeError(
                f"{comparison}/{label}: {split_name} lacks one domain under frozen holdout"
            )

    for domain in ["official", "external"]:
        train_prompts = set(train.loc[train["domain"].eq(domain), "prompt_id"].astype(str))
        test_prompts = set(test.loc[test["domain"].eq(domain), "prompt_id"].astype(str))
        if train_prompts & test_prompts:
            raise RuntimeError(f"{comparison}/{label}: prompt leakage in {domain}")
    return train, test


def summarize_scores(scores: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize empty scores")
    return {
        "mean": round(float(values.mean()), 15),
        "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _domain_target(frame: pd.DataFrame) -> np.ndarray:
    return frame["domain"].astype(str).eq("external").astype(int).to_numpy()


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "official_recall": float(recall_score(y_true, y_pred, pos_label=0)),
        "external_recall": float(recall_score(y_true, y_pred, pos_label=1)),
    }


def _split_counts(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    def counts(frame: pd.DataFrame) -> dict:
        return {
            "rows": int(len(frame)),
            "rows_by_domain": {
                domain: int(count)
                for domain, count in frame["domain"].value_counts().sort_index().items()
            },
            "prompt_groups_by_domain": {
                domain: int(
                    frame.loc[frame["domain"].eq(domain), "prompt_id"].astype(str).nunique()
                )
                for domain in ["external", "official"]
            },
        }

    return {"train": counts(train), "test": counts(test)}


def _style_smd(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    features = extract_style_features(frame["text"])
    domains = frame["domain"].astype(str).reset_index(drop=True)
    records: list[dict[str, float | str]] = []
    for column in STYLE_COLUMNS:
        off = features.loc[domains.eq("official"), column].to_numpy(dtype=np.float64)
        ext = features.loc[domains.eq("external"), column].to_numpy(dtype=np.float64)
        off_mean = float(off.mean())
        ext_mean = float(ext.mean())
        off_var = float(off.var(ddof=1)) if off.size > 1 else 0.0
        ext_var = float(ext.var(ddof=1)) if ext.size > 1 else 0.0
        pooled = math.sqrt(max((off_var + ext_var) / 2.0, 0.0))
        smd = (ext_mean - off_mean) / pooled if pooled > 0 else 0.0
        records.append(
            {
                "feature": column,
                "official_mean": off_mean,
                "external_mean": ext_mean,
                "standardized_mean_difference_external_minus_official": float(smd),
            }
        )
    return sorted(
        records,
        key=lambda record: abs(
            float(record["standardized_mean_difference_external_minus_official"])
        ),
        reverse=True,
    )


def run_label_audit(
    official: pd.DataFrame,
    external: pd.DataFrame,
    *,
    comparison: str,
    model_label: str,
) -> dict:
    train, test = build_label_domain_split(
        official,
        external,
        comparison=comparison,
        model_label=model_label,
    )
    y_train = _domain_target(train)
    y_test = _domain_target(test)

    style_train = extract_style_features(train["text"])
    style_test = extract_style_features(test["text"])
    style_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    )
    style_model.fit(style_train, y_train)
    style_pred = style_model.predict(style_test)

    lexical_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        min_df=2,
        max_features=20_000,
        sublinear_tf=True,
    )
    lexical_train = lexical_vectorizer.fit_transform(train["text"].astype(str))
    lexical_test = lexical_vectorizer.transform(test["text"].astype(str))
    lexical_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    lexical_model.fit(lexical_train, y_train)
    lexical_pred = lexical_model.predict(lexical_test)

    all_rows = pd.concat([train, test], ignore_index=True)
    return {
        "model_label": str(model_label),
        "counts": _split_counts(train, test),
        "style": _score_predictions(y_test, style_pred),
        "lexical": _score_predictions(y_test, lexical_pred),
        "top_style_standardized_mean_differences": _style_smd(all_rows)[:10],
        "lexical_vocabulary_size": int(len(lexical_vectorizer.vocabulary_)),
    }


def run_comparison(
    official: pd.DataFrame,
    external: pd.DataFrame,
    *,
    comparison: str,
) -> dict:
    official_labels = set(official["model"].astype(str).unique())
    external_labels = set(external["model"].astype(str).unique())
    shared_labels = sorted(official_labels & external_labels)
    if not shared_labels:
        raise RuntimeError(f"{comparison}: no shared labels")

    per_label = [
        run_label_audit(
            official,
            external,
            comparison=comparison,
            model_label=label,
        )
        for label in shared_labels
    ]
    style_scores = [entry["style"]["balanced_accuracy"] for entry in per_label]
    lexical_scores = [entry["lexical"]["balanced_accuracy"] for entry in per_label]

    feature_smds: dict[str, list[float]] = {name: [] for name in STYLE_COLUMNS}
    for entry in per_label:
        for record in entry["top_style_standardized_mean_differences"]:
            feature_smds[str(record["feature"])].append(
                float(record["standardized_mean_difference_external_minus_official"])
            )
    aggregate_smd = []
    for feature, values in feature_smds.items():
        if values:
            aggregate_smd.append(
                {
                    "feature": feature,
                    "mean_signed_smd_over_labels_where_top10": float(np.mean(values)),
                    "mean_absolute_smd_over_labels_where_top10": float(np.mean(np.abs(values))),
                    "labels_where_top10": int(len(values)),
                }
            )
    aggregate_smd.sort(
        key=lambda record: float(record["mean_absolute_smd_over_labels_where_top10"]),
        reverse=True,
    )

    return {
        "comparison": comparison,
        "shared_labels": shared_labels,
        "shared_label_count": int(len(shared_labels)),
        "style_balanced_accuracy": summarize_scores(style_scores),
        "lexical_balanced_accuracy": summarize_scores(lexical_scores),
        "per_label": per_label,
        "aggregate_top_style_smd": aggregate_smd[:10],
    }


def run_audit(
    official: pd.DataFrame,
    phase7_external: pd.DataFrame,
    phase15_external: pd.DataFrame,
) -> dict:
    return {
        "scope": "audit_only_domain_separability_no_competition_model_tuning",
        "holdout_salt": HOLDOUT_SALT,
        "random_state": RANDOM_STATE,
        "phase7_control": run_comparison(
            official,
            phase7_external,
            comparison="phase7",
        ),
        "phase15": run_comparison(
            official,
            phase15_external,
            comparison="phase15",
        ),
        "kaggle_submission_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen domain-separability audit.")
    parser.add_argument("--official", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--phase7-external",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    parser.add_argument("--phase15-external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official = pd.read_csv(args.official)
    phase7_external = pd.read_parquet(args.phase7_external)
    phase15_external = pd.read_parquet(args.phase15_external)
    report = run_audit(official, phase7_external, phase15_external)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
