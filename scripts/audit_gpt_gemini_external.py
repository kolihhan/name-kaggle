from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def audit_external(
    source_path: Path,
    train_path: Path,
    phase1_audit_path: Path,
    report_path: Path,
    parquet_path: Path,
) -> dict:
    ext = pd.read_csv(source_path)
    train = pd.read_csv(train_path)
    target_labels = set(json.loads(phase1_audit_path.read_text())["model_counts"])

    required = {
        "label",
        "prompt_id",
        "setting",
        "target_words",
        "text",
        "model_seen",
        "finish_reason",
        "cost_usd",
        "batch",
    }
    missing_cols = sorted(required - set(ext.columns))
    if missing_cols:
        raise RuntimeError(f"Missing expected columns: {missing_cols}")

    ext = ext.rename(columns={"label": "model"}).copy()
    ext["text"] = ext["text"].astype(str)
    ext["model"] = ext["model"].astype(str)

    ext_norm = ext["text"].map(_norm)
    train_norm = train["text"].astype(str).map(_norm)
    train_set = set(train_norm)
    label_counts = ext["model"].value_counts().sort_index()
    invalid_labels = sorted(set(label_counts.index) - target_labels)
    missing_competition_labels = sorted(target_labels - set(label_counts.index))
    word_counts = ext_norm.str.split().map(len)
    exact_overlap_mask = ext_norm.isin(train_set)

    all_text = pd.concat([train_norm, ext_norm], ignore_index=True)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=80000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(all_text)
    train_matrix = matrix[: len(train)]
    ext_matrix = matrix[len(train) :]
    max_similarity = np.zeros(len(ext), dtype=np.float32)
    for start in range(0, len(ext), 256):
        sims = cosine_similarity(
            ext_matrix[start : start + 256],
            train_matrix,
            dense_output=True,
        )
        max_similarity[start : start + len(sims)] = sims.max(axis=1)

    prompt_counts = ext["prompt_id"].astype(str).value_counts()
    model_seen = ext["model_seen"].fillna("").astype(str).str.strip()
    finish_reason = ext["finish_reason"].fillna("").astype(str).str.strip()

    audit = {
        "source": {
            "name": "Extra GPT and Gemini generations (free Kaggle Benchmarks quota)",
            "upstream_repo": "RohanExploit/kaggle-name-that-llm-top20",
            "upstream_path": "data/extra_gpt_gemini/ntl_generations.csv",
            "git_blob_sha1": "8dccb6d8f8b8388b134a6aa8f8ea42e99df08c10",
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "license_reported": "CC BY 4.0 (best-effort statement in upstream README)",
            "prompt_bank": "seyitkaangunes/ntl-extra-api-passages — 1,000 new prompts, CC BY 4.0",
        },
        "shape": {
            "rows": int(len(ext)),
            "columns": list(map(str, ext.columns)),
            "duplicate_full_rows": int(ext.duplicated().sum()),
            "duplicate_normalized_texts": int(ext_norm.duplicated().sum()),
            "prompt_ids": int(ext["prompt_id"].nunique()),
            "prompt_min_rows": int(prompt_counts.min()),
            "prompt_max_rows": int(prompt_counts.max()),
        },
        "labels": {
            "n_present": int(label_counts.size),
            "counts": {str(k): int(v) for k, v in label_counts.items()},
            "invalid_labels": invalid_labels,
            "missing_competition_labels": missing_competition_labels,
        },
        "provenance": {
            "model_seen_blank_rows": int(model_seen.eq("").sum()),
            "model_seen_unique": int(
                model_seen.replace("", np.nan).dropna().nunique()
            ),
            "settings": {
                str(k): int(v)
                for k, v in ext["setting"]
                .fillna("<NA>")
                .astype(str)
                .value_counts()
                .items()
            },
            "finish_reasons": {
                str(k): int(v)
                for k, v in finish_reason.replace("", "<blank>").value_counts().items()
            },
            "batches": int(ext["batch"].nunique()),
        },
        "protocol": {
            "word_count_min": int(word_counts.min()),
            "word_count_p05": float(word_counts.quantile(0.05)),
            "word_count_median": float(word_counts.median()),
            "word_count_p95": float(word_counts.quantile(0.95)),
            "word_count_max": int(word_counts.max()),
            "fraction_40_to_250_words": float(word_counts.between(40, 250).mean()),
            "rows_with_newline": int(
                ext["text"].str.contains(r"\n", regex=True).sum()
            ),
        },
        "official_train_overlap": {
            "exact_normalized_text_matches": int(exact_overlap_mask.sum()),
            "tfidf_word_1_2_max_similarity_max": float(max_similarity.max()),
            "tfidf_word_1_2_max_similarity_p99": float(
                np.quantile(max_similarity, 0.99)
            ),
            "rows_similarity_gt_0_25": int((max_similarity > 0.25).sum()),
            "rows_similarity_gt_0_50": int((max_similarity > 0.50).sum()),
            "rows_similarity_gt_0_80": int((max_similarity > 0.80).sum()),
        },
    }

    hard_failures: list[str] = []
    if invalid_labels:
        hard_failures.append(f"invalid competition labels: {invalid_labels}")
    if int(exact_overlap_mask.sum()) > 0:
        hard_failures.append("exact normalized text overlap with official train")
    if int(model_seen.eq("").sum()) > 0:
        hard_failures.append("blank model_seen provenance rows")

    audit["gate"] = {
        "hard_failures": hard_failures,
        "passed_raw_identity_gate": not hard_failures,
        "note": "Missing competition labels are expected for this targeted source and are not a hard failure.",
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    ext.to_parquet(parquet_path, index=False)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--phase1-audit",
        type=Path,
        default=Path("artifacts/phase1/audit.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/phase2/gpt_gemini_raw_audit.json"),
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/external/gpt_gemini_generations.parquet"),
    )
    args = parser.parse_args()

    audit = audit_external(
        source_path=args.source,
        train_path=args.train,
        phase1_audit_path=args.phase1_audit,
        report_path=args.report,
        parquet_path=args.parquet,
    )
    print(
        json.dumps(
            {
                "rows": audit["shape"]["rows"],
                "prompts": audit["shape"]["prompt_ids"],
                "labels": audit["labels"]["n_present"],
                "exact_overlap": audit["official_train_overlap"][
                    "exact_normalized_text_matches"
                ],
                "near_gt_0_50": audit["official_train_overlap"][
                    "rows_similarity_gt_0_50"
                ],
                "gate_pass": audit["gate"]["passed_raw_identity_gate"],
            },
            indent=2,
        )
    )
    if not audit["gate"]["passed_raw_identity_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
