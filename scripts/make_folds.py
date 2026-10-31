from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold


def build_folds(train_path: Path, output_path: Path) -> pd.DataFrame:
    train = pd.read_csv(train_path).sort_values("id").reset_index(drop=True)
    required = {"id", "prompt_id", "model"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"train.csv missing required columns: {sorted(missing)}")

    train["fold_id"] = -1
    splitter = GroupKFold(n_splits=5)
    for fold_id, (_, valid_idx) in enumerate(
        splitter.split(train, y=train["model"], groups=train["prompt_id"])
    ):
        train.loc[valid_idx, "fold_id"] = fold_id

    if (train["fold_id"] < 0).any():
        raise RuntimeError("at least one row was not assigned to a fold")

    for fold_id in range(5):
        tr_prompts = set(train.loc[train.fold_id != fold_id, "prompt_id"])
        va_prompts = set(train.loc[train.fold_id == fold_id, "prompt_id"])
        overlap = tr_prompts & va_prompts
        if overlap:
            raise RuntimeError(f"prompt leakage in fold {fold_id}: {sorted(overlap)[:5]}")
        if len(va_prompts) != 30:
            raise RuntimeError(
                f"fold {fold_id} has {len(va_prompts)} validation prompts; expected 30"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = train[["id", "prompt_id", "model", "fold_id"]].copy()
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/phase1/folds.csv")
    )
    args = parser.parse_args()
    folds = build_folds(args.train, args.output)
    print(
        f"wrote {len(folds)} rows to {args.output}; "
        f"fold sizes={folds.fold_id.value_counts().sort_index().to_dict()}"
    )


if __name__ == "__main__":
    main()
