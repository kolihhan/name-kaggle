from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


SHADOW_SALTS = {
    "shadow_a": "ntl-shadow-v1-a",
    "shadow_b": "ntl-shadow-v1-b",
    "shadow_c": "ntl-shadow-v1-c",
}


def assign_prompt_folds(prompt_ids: Iterable[object], salt: str) -> dict[str, int]:
    prompts = sorted({str(prompt_id) for prompt_id in prompt_ids})
    ordered = sorted(
        prompts,
        key=lambda prompt_id: (
            hashlib.sha256(f"{salt}|{prompt_id}".encode("utf-8")).hexdigest(),
            prompt_id,
        ),
    )
    return {prompt_id: rank % 5 for rank, prompt_id in enumerate(ordered)}


def build_shadow_folds(train: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "prompt_id", "model"}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"train data missing required columns: {missing}")
    if train["id"].duplicated().any():
        raise ValueError("train ids must be unique")

    prompt_strings = train["prompt_id"].astype(str)
    out = train[["id", "prompt_id"]].copy()
    for column, salt in SHADOW_SALTS.items():
        mapping = assign_prompt_folds(prompt_strings.unique(), salt)
        out[column] = prompt_strings.map(mapping).astype(int)

        prompt_map = pd.DataFrame(
            {"prompt_id": prompt_strings, "fold": out[column]}
        ).drop_duplicates()
        if prompt_map.groupby("prompt_id")["fold"].nunique().max() != 1:
            raise RuntimeError(f"{column} split prompt groups across folds")
        counts = prompt_map["fold"].value_counts().sort_index().to_dict()
        if len(mapping) == 150 and counts != {0: 30, 1: 30, 2: 30, 3: 30, 4: 30}:
            raise RuntimeError(f"{column} prompt fold counts drifted: {counts}")

    return out


def canonical_csv_bytes(shadow: pd.DataFrame) -> bytes:
    columns = ["id", "prompt_id", *SHADOW_SALTS]
    if shadow.columns.tolist() != columns:
        raise ValueError(f"unexpected shadow columns: {shadow.columns.tolist()}")
    rendered = shadow.sort_values("id", kind="stable").to_csv(
        index=False,
        lineterminator="\n",
    )
    return rendered.encode("utf-8")


def _prompt_mapping(frame: pd.DataFrame, fold_column: str) -> dict[str, int]:
    work = pd.DataFrame(
        {
            "prompt_id": frame["prompt_id"].astype(str),
            "fold": frame[fold_column].astype(int),
        }
    ).drop_duplicates()
    if work.groupby("prompt_id")["fold"].nunique().max() != 1:
        raise RuntimeError(f"{fold_column} splits a prompt across folds")
    return dict(zip(work["prompt_id"], work["fold"]))


def build_manifest(
    train: pd.DataFrame,
    shadow: pd.DataFrame,
    *,
    original_folds: pd.DataFrame | None = None,
) -> dict:
    csv_bytes = canonical_csv_bytes(shadow)
    prompt_count = int(train["prompt_id"].astype(str).nunique())

    prompt_maps = {column: _prompt_mapping(shadow, column) for column in SHADOW_SALTS}
    prompt_fold_counts = {
        column: {
            str(fold): int(count)
            for fold, count in pd.Series(mapping).value_counts().sort_index().items()
        }
        for column, mapping in prompt_maps.items()
    }

    pairwise_different_prompt_counts: dict[str, int] = {}
    columns = list(SHADOW_SALTS)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            pairwise_different_prompt_counts[f"{left}__vs__{right}"] = sum(
                prompt_maps[left][prompt_id] != prompt_maps[right][prompt_id]
                for prompt_id in prompt_maps[left]
            )

    permanent_comparison = None
    if original_folds is not None:
        required = {"id", "prompt_id", "fold_id"}
        missing = sorted(required - set(original_folds.columns))
        if missing:
            raise ValueError(f"original folds missing required columns: {missing}")
        permanent = _prompt_mapping(original_folds, "fold_id")
        if set(permanent) != set(next(iter(prompt_maps.values()))):
            raise RuntimeError("permanent and shadow prompt sets differ")
        permanent_comparison = {
            column: {
                "different_prompt_count": int(
                    sum(
                        mapping[prompt_id] != permanent[prompt_id]
                        for prompt_id in permanent
                    )
                ),
                "same_prompt_count": int(
                    sum(
                        mapping[prompt_id] == permanent[prompt_id]
                        for prompt_id in permanent
                    )
                ),
            }
            for column, mapping in prompt_maps.items()
        }

    return {
        "rows": int(len(train)),
        "prompt_ids": prompt_count,
        "labels": int(train["model"].astype(str).nunique()),
        "salts": SHADOW_SALTS,
        "prompt_fold_counts": prompt_fold_counts,
        "shadow_folds_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "pairwise_different_prompt_counts": pairwise_different_prompt_counts,
        "comparison_vs_permanent": permanent_comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen Phase 19 shadow folds.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--original-folds",
        type=Path,
        default=Path("artifacts/phase1/folds.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase19/shadow_folds.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/phase19/shadow_fold_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    shadow = build_shadow_folds(train)
    original = pd.read_csv(args.original_folds)
    manifest = build_manifest(train, shadow, original_folds=original)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_csv_bytes(shadow))
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
