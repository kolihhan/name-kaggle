import numpy as np
import pandas as pd
import pytest

from scripts.run_strong_sparse_external_ablation import (
    EXPECTED_EXTERNAL_COUNTS,
    prepare_condition_rows,
    summarize_ablation,
    validate_external,
)


def sample_train() -> pd.DataFrame:
    rows = []
    row_id = 0
    for prompt_id, fold_id in [("p0", 0), ("p1", 1), ("p2", 2), ("p3", 3), ("p4", 4)]:
        for model in ["a", "b"]:
            rows.append({
                "id": row_id,
                "prompt_id": prompt_id,
                "model": model,
                "text": f"{prompt_id}-{model}",
                "fold_id": fold_id,
            })
            row_id += 1
    return pd.DataFrame(rows)


def test_prepare_condition_rows_keeps_external_train_only():
    train = sample_train()
    external = pd.DataFrame({"text": ["e0", "e1", "e2"], "model": ["a", "b", "a"]})

    official_fit, valid, augmented_fit = prepare_condition_rows(train, external, fold=2)

    assert set(valid["prompt_id"]) == {"p2"}
    assert "p2" not in set(official_fit["prompt_id"])
    assert len(official_fit) == 8
    assert len(augmented_fit) == 11
    assert augmented_fit["is_external"].sum() == 3
    assert valid.get("is_external", pd.Series(False, index=valid.index)).sum() == 0


def test_validate_external_requires_exact_audited_counts():
    rows = []
    for label, count in EXPECTED_EXTERNAL_COUNTS.items():
        rows.extend({"text": f"{label}-{i}", "model": label} for i in range(count))
    external = pd.DataFrame(rows)
    official_labels = sorted(set(EXPECTED_EXTERNAL_COUNTS) | {"other-label"})

    validated = validate_external(external, official_labels=official_labels)
    assert len(validated) == 2503
    assert validated["model"].value_counts().to_dict() == EXPECTED_EXTERNAL_COUNTS

    broken = validated.iloc[:-1].copy()
    with pytest.raises(ValueError, match="audited external counts"):
        validate_external(broken, official_labels=official_labels)


def test_summarize_ablation_scores_only_official_rows_and_reports_covered_delta():
    train = pd.DataFrame({"model": ["a", "b", "a", "b"]})
    pred_a = np.array(["a", "b", "b", "b"], dtype=object)
    pred_b = np.array(["a", "b", "a", "b"], dtype=object)

    report = summarize_ablation(
        train=train,
        official_predictions=pred_a,
        augmented_predictions=pred_b,
        official_fold_scores=[0.75],
        augmented_fold_scores=[1.0],
        covered_labels={"a"},
    )

    assert report["official_only"]["oof_rows"] == 4
    assert report["official_plus_external"]["oof_rows"] == 4
    assert report["official_only"]["balanced_accuracy"] == pytest.approx(0.75)
    assert report["official_plus_external"]["balanced_accuracy"] == pytest.approx(1.0)
    assert report["delta"] == pytest.approx(0.25)
    assert report["covered_mean_recall_delta"] == pytest.approx(0.5)
    assert report["uncovered_mean_recall_delta"] == pytest.approx(0.0)
