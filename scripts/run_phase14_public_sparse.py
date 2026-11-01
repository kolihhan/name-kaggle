from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC

from scripts.run_strong_sparse_anchor import load_train, summarize_oof


PHASE7_SCORE = 0.3700521998508576
NUMERIC_ATOL = 1e-12
EXPECTED_RAW_SHA256 = "37750e899ff6634c147daa7888bfed048e92d3fb1381fe5d21f93bf72a84e8ec"
EXPECTED_RAW_ROWS = 75_783
EXPECTED_EXTERNAL_ROWS = 43_781
EXPECTED_SAFE_PROMPT_IDS = 3_736
EXPECTED_SEED_ROWS = 38_210
EXPECTED_INHERITED_ROWS = 5_571
EXPECTED_EXCLUDED_ROWS = 32_002

VIEWS = [
    ("char", 3, 5, 250_000, 1.0),
    ("word", 1, 3, 180_000, 0.7),
    ("shape", 2, 6, 60_000, 0.5),
    ("small", 1, 3, 50_000, 0.5),
]

SEED_SOURCES = {
    "abhinav",
    "hailey",
    "kaplinskiy",
    "keshav",
    "lucasboesen",
    "maniceet",
    "prvsiyan",
    "prvsiyan_v1",
    "semenoff",
    "seyit",
    "willrice",
    "zakir",
    "zbigniew",
}

SMALL_WORDS = set(
    """
    a an the and or but if while although because as so than that which who whom whose
    this these those it its they them their theirs he him his she her hers we us our ours
    you your yours i me my mine in on at to from with without for of by over under between
    through into about is are was were be been being have has had do does did can could will
    would shall should may might must not no nor all any some each every both either neither
    other another such only also just very too more most less least here there when where why
    how what whether then now still yet even much many few little
    """.split()
)

EXPECTED_LABEL_COUNTS = {
    "claude-fable-5": 1_193,
    "claude-opus-4-6": 1_352,
    "claude-opus-4-8": 1_364,
    "claude-opus-5": 1_240,
    "claude-sonnet-5": 1_350,
    "gemini-3.1-pro-preview": 1_409,
    "gemini-3.6-flash": 2_880,
    "gemini-3.7-flash": 1_704,
    "gemini-3.8-flash": 2_778,
    "glm-5.3": 3_240,
    "glm-5.3-flash": 4_236,
    "gpt-5.5": 3_525,
    "gpt-5.6-luna": 4_326,
    "gpt-5.6-sol": 3_879,
    "gpt-5.6-terra": 3_875,
    "gpt-6-astra": 3_044,
    "kimi-k3": 1_509,
    "qwen3.8-max": 877,
}

WORD_TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
SMALL_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
LETTER_RUN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_text(text: str) -> str:
    text = str(text)
    return LETTER_RUN_RE.sub(
        lambda match: "X" if match.group(0)[0].isupper() else "x",
        text,
    )


def small_words_text(text: str) -> str:
    tokens = SMALL_TOKEN_RE.findall(str(text))
    rendered: list[str] = []
    for token in tokens:
        lower = token.lower()
        if lower in SMALL_WORDS or not token.isalnum():
            rendered.append(lower)
        else:
            rendered.append("CONTENT")
    return " ".join(rendered)


def _required_pool_columns() -> set[str]:
    return {"prompt_id", "prompt_key", "text", "model", "source", "setting"}


def build_conservative_subset(
    pool: pd.DataFrame,
    *,
    validate_full_contract: bool = True,
) -> pd.DataFrame:
    missing = sorted(_required_pool_columns() - set(pool.columns))
    if missing:
        raise RuntimeError(f"public pool missing required columns: {missing}")

    if validate_full_contract and len(pool) != EXPECTED_RAW_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_RAW_ROWS} raw public rows, found {len(pool)}"
        )

    sources = pool["source"].astype(str)
    prompt_ids = pool["prompt_id"].astype(str)

    if validate_full_contract:
        prvsiyan = pool.loc[sources.eq("prvsiyan")]
        unique_keys = prvsiyan["prompt_key"].astype(str).nunique()
        if unique_keys != 231:
            raise RuntimeError(
                f"expected 231 clean prvsiyan prompt keys, found {unique_keys}"
            )
        if not prvsiyan["prompt_key"].astype(str).str.startswith("prvsiyan:n").all():
            raise RuntimeError("prvsiyan prompt-key provenance drifted")

    seed_mask = sources.isin(SEED_SOURCES)
    safe_prompt_ids = set(prompt_ids.loc[seed_mask])
    inherited_mask = prompt_ids.isin(safe_prompt_ids)
    keep_mask = seed_mask | inherited_mask
    safe = pool.loc[keep_mask].copy().reset_index(drop=True)

    if validate_full_contract:
        seed_rows = int(seed_mask.sum())
        inherited_rows = int((keep_mask & ~seed_mask).sum())
        excluded_rows = int((~keep_mask).sum())
        safe_prompt_count = int(safe["prompt_id"].astype(str).nunique())
        labels = {
            str(label): int(count)
            for label, count in safe["model"].astype(str).value_counts().sort_index().items()
        }

        expected_scalars = {
            "seed rows": (seed_rows, EXPECTED_SEED_ROWS),
            "retained rows": (len(safe), EXPECTED_EXTERNAL_ROWS),
            "safe prompt ids": (safe_prompt_count, EXPECTED_SAFE_PROMPT_IDS),
            "inherited rows": (inherited_rows, EXPECTED_INHERITED_ROWS),
            "excluded rows": (excluded_rows, EXPECTED_EXCLUDED_ROWS),
        }
        for name, (observed, expected) in expected_scalars.items():
            if int(observed) != int(expected):
                raise RuntimeError(
                    f"Phase 14 subset drift for {name}: {observed} != {expected}"
                )
        if labels != EXPECTED_LABEL_COUNTS:
            raise RuntimeError(
                "Phase 14 retained label counts drifted: "
                f"{labels} != {EXPECTED_LABEL_COUNTS}"
            )
        if safe["text"].isna().any() or safe["model"].isna().any():
            raise RuntimeError("Phase 14 subset contains null text/model")
        if (safe["text"].astype(str).str.strip() == "").any():
            raise RuntimeError("Phase 14 subset contains empty text")

    return safe


def _view_texts(name: str, texts: Sequence[str]) -> list[str]:
    values = [str(text) for text in texts]
    if name == "shape":
        return [shape_text(text) for text in values]
    if name == "small":
        return [small_words_text(text) for text in values]
    return values


def build_fold_features(
    train_text: Sequence[str],
    valid_text: Sequence[str],
    *,
    min_df: int = 2,
    feature_cap: int | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict]:
    train_parts: list[sparse.csr_matrix] = []
    valid_parts: list[sparse.csr_matrix] = []
    widths: dict[str, int] = {}

    for name, lo, hi, frozen_max_features, weight in VIEWS:
        max_features = (
            frozen_max_features
            if feature_cap is None
            else min(int(feature_cap), int(frozen_max_features))
        )
        analyzer = "char" if name in {"char", "shape"} else "word"
        kwargs = {
            "analyzer": analyzer,
            "ngram_range": (lo, hi),
            "lowercase": False,
            "sublinear_tf": True,
            "min_df": min_df,
            "max_features": max_features,
            "dtype": np.float32,
        }
        if analyzer == "word":
            kwargs["token_pattern"] = WORD_TOKEN_PATTERN

        vectorizer = TfidfVectorizer(**kwargs)
        transformed_train = _view_texts(name, train_text)
        transformed_valid = _view_texts(name, valid_text)
        x_train = vectorizer.fit_transform(transformed_train).tocsr().astype(np.float32)
        x_valid = vectorizer.transform(transformed_valid).tocsr().astype(np.float32)
        if weight != 1.0:
            x_train = x_train.multiply(np.float32(weight)).tocsr()
            x_valid = x_valid.multiply(np.float32(weight)).tocsr()

        train_parts.append(x_train)
        valid_parts.append(x_valid)
        widths[name] = int(x_train.shape[1])

    combined_train = sparse.hstack(train_parts, format="csr", dtype=np.float32)
    combined_valid = sparse.hstack(valid_parts, format="csr", dtype=np.float32)
    meta = {
        "widths": widths,
        "total_features": int(combined_train.shape[1]),
        "views": [
            {
                "name": name,
                "ngram_range": [lo, hi],
                "max_features": cap,
                "weight": weight,
            }
            for name, lo, hi, cap, weight in VIEWS
        ],
    }
    return combined_train, combined_valid, meta


def make_svc() -> LinearSVC:
    return LinearSVC(
        C=0.5,
        random_state=42,
        max_iter=5000,
        class_weight="balanced",
    )


def promotion_decision(score: float) -> dict[str, float | bool]:
    delta = float(score) - PHASE7_SCORE
    real_gain = float(score) > PHASE7_SCORE + NUMERIC_ATOL
    return {
        "phase7_reference": PHASE7_SCORE,
        "balanced_accuracy": float(score),
        "delta_vs_phase7": delta,
        "beats_phase7_beyond_numeric_equality": real_gain,
        "promote_phase14": real_gain,
    }


def run_cv(
    train: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    labels = sorted(train["model"].astype(str).unique())
    if len(train) != 2699:
        raise RuntimeError(f"expected 2699 official rows, found {len(train)}")
    if len(labels) != 18:
        raise RuntimeError(f"expected 18 official labels, found {len(labels)}")
    if len(external) != EXPECTED_EXTERNAL_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_EXTERNAL_ROWS} external rows, found {len(external)}"
        )
    if set(external["model"].astype(str).unique()) != set(labels):
        raise RuntimeError("external label set does not match official 18-label set")

    fold_ids = train["fold_id"].to_numpy()
    predictions = np.full(len(train), None, dtype=object)
    fold_scores: list[float] = []
    feature_meta: list[dict] = []

    ext_text = external["text"].astype(str).tolist()
    ext_y = external["model"].astype(str).tolist()

    for fold in range(5):
        fit_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        fit = train.iloc[fit_idx]
        valid = train.iloc[valid_idx]

        fit_text = fit["text"].astype(str).tolist() + ext_text
        fit_y = fit["model"].astype(str).tolist() + ext_y
        valid_text = valid["text"].astype(str).tolist()

        x_train, x_valid, meta = build_fold_features(fit_text, valid_text)
        model = make_svc()
        model.fit(x_train, fit_y)
        pred = model.predict(x_valid)
        predictions[valid_idx] = pred
        score = float(balanced_accuracy_score(valid["model"], pred))
        fold_scores.append(score)
        meta.update(
            {
                "fold": fold,
                "official_training_rows": int(len(fit)),
                "external_training_rows": int(len(external)),
                "total_training_rows": int(len(fit_text)),
                "official_validation_rows": int(len(valid)),
                "external_validation_rows": 0,
            }
        )
        feature_meta.append(meta)
        print(
            f"phase14 fold={fold} score={score:.6f} "
            f"train={len(fit_text)} valid={len(valid)} "
            f"features={meta['total_features']}",
            flush=True,
        )

        del x_train, x_valid, model, fit_text, fit_y, valid_text

    result = summarize_oof(train, predictions, fold_scores)
    decision = promotion_decision(result["balanced_accuracy"])

    oof = train[["id", "prompt_id", "model", "fold_id"]].copy()
    oof["prediction"] = predictions
    if len(oof) != 2699 or oof["id"].nunique() != 2699:
        raise RuntimeError("Phase 14 OOF row/ID invariant failed")
    if oof["prediction"].isna().any():
        raise RuntimeError("Phase 14 OOF contains null predictions")

    report = {
        "phase": 14,
        "experiment": "conservative_public_four_view_svc",
        "validation": {
            "folds": 5,
            "group_key": "prompt_id",
            "metric": "balanced_accuracy",
            "scored_rows": "official_validation_only",
            "fold_fingerprint": "bf6f47bfaeebdf817a1f7972b54deabbe5e54fb087c77753b6361970d0c9ee0d",
        },
        "data": {
            "official_rows": int(len(train)),
            "external_rows": int(len(external)),
            "external_prompt_ids": int(external["prompt_id"].astype(str).nunique()),
            "external_labels": {
                str(label): int(count)
                for label, count in external["model"].astype(str).value_counts().sort_index().items()
            },
            "seed_sources": sorted(SEED_SOURCES),
        },
        "configuration": {
            "views": [
                {
                    "name": name,
                    "ngram_range": [lo, hi],
                    "max_features": cap,
                    "weight": weight,
                }
                for name, lo, hi, cap, weight in VIEWS
            ],
            "tfidf": {
                "lowercase": False,
                "sublinear_tf": True,
                "min_df": 2,
                "dtype": "float32",
                "word_token_pattern": WORD_TOKEN_PATTERN,
            },
            "classifier": "LinearSVC(C=0.5,random_state=42,max_iter=5000,class_weight=balanced)",
            "new_tuned_parameters": 0,
        },
        "fold_feature_meta": feature_meta,
        "result": result,
        "decision": decision,
    }
    return report, oof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen Phase 14 conservative public four-view sparse CV."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/phase1/folds.csv"))
    parser.add_argument("--external-raw", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase14/public_sparse.json"),
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=Path("artifacts/phase14/public_sparse_oof.csv"),
    )
    parser.add_argument(
        "--subset-output",
        type=Path,
        default=None,
        help="Optional path to persist the mechanically rebuilt 43,781-row subset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observed_sha = sha256_file(args.external_raw)
    if observed_sha != EXPECTED_RAW_SHA256:
        raise RuntimeError(
            f"public raw pool SHA mismatch: {observed_sha} != {EXPECTED_RAW_SHA256}"
        )

    pool = pd.read_parquet(args.external_raw)
    external = build_conservative_subset(pool, validate_full_contract=True)
    if args.subset_output is not None:
        args.subset_output.parent.mkdir(parents=True, exist_ok=True)
        external.to_parquet(args.subset_output, index=False)

    train = load_train(args.train, args.folds)
    report, oof = run_cv(train, external)
    report["data"]["raw_pool_sha256"] = observed_sha
    report["data"]["raw_pool_rows"] = int(len(pool))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    oof.to_csv(args.oof_output, index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
