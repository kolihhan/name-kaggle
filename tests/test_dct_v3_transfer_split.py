from __future__ import annotations

import pandas as pd

from scripts.dct_v3_transfer_split import (
    EXCLUDED_HOLDOUT_SOURCES,
    TRANSFER_HOLDOUT_SOURCE_COUNT,
    build_transfer_split,
    select_transfer_holdout_sources,
)


def _frame() -> pd.DataFrame:
    rows = []
    labels = [f"model-{i}" for i in range(8)]
    sources = sorted(EXCLUDED_HOLDOUT_SOURCES) + [
        "fresh-a",
        "fresh-b",
        "fresh-c",
        "fresh-d",
        "fresh-e",
    ]
    rid = 0
    for source_idx, source in enumerate(sources):
        for label in labels:
            for j in range(20):
                # Every source has its own prompt bank except one deliberate
                # cross-source prompt used to verify train-side leakage removal.
                prompt = f"{source}-p-{label}-{j}"
                if source == "fresh-e" and label == "model-0" and j == 0:
                    prompt = "shared-probe"
                if source == "fresh-a" and label == "model-0" and j == 0:
                    prompt = "shared-probe"
                rows.append(
                    {
                        "id": f"r{rid}",
                        "source": source,
                        "model": label,
                        "prompt_id": prompt,
                        "prompt_key": prompt,
                        "text": f"{source} {label} {j}",
                    }
                )
                rid += 1
    return pd.DataFrame(rows)


def test_selection_is_deterministic_and_excludes_spent_sources() -> None:
    frame = _frame()
    first = select_transfer_holdout_sources(frame)
    second = select_transfer_holdout_sources(frame.sample(frac=1.0, random_state=7))
    assert first == second
    assert len(first) == TRANSFER_HOLDOUT_SOURCE_COUNT
    assert set(first).isdisjoint(EXCLUDED_HOLDOUT_SOURCES)


def test_split_has_coverage_and_zero_prompt_leakage() -> None:
    split = build_transfer_split(_frame())
    assert split["valid_for_transfer_gate"] is True
    assert split["eligible_label_count"] >= 8
    assert split["eligible_heldout_rows"] >= 300
    assert split["prompt_id_overlap"] == 0
    assert split["prompt_key_overlap"] == 0
    assert all(v > 0 for v in split["source_eligible_rows"].values())
    assert set(split["holdout_sources"]).isdisjoint(EXCLUDED_HOLDOUT_SOURCES)


def test_split_records_selection_without_model_scores() -> None:
    split = build_transfer_split(_frame())
    forbidden = {"prediction", "score", "balanced_accuracy", "public_lb"}
    assert forbidden.isdisjoint(split)
    assert "selection_digest" in split
    assert "candidate_valid_combinations" in split
