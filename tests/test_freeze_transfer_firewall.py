import pandas as pd
import pytest

from scripts.freeze_transfer_firewall import build_firewall_lock


def _pool() -> pd.DataFrame:
    rows = []
    for source, prompt, label, count in [
        ("c1", "p1", "a", 25),
        ("c1", "p2", "b", 25),
        ("c2", "p3", "a", 25),
        ("c2", "p4", "b", 25),
        ("r1", "p5", "a", 25),
        ("r1", "p6", "b", 25),
        ("r2", "p7", "a", 25),
        ("r2", "p8", "b", 25),
    ]:
        for index in range(count):
            rows.append(
                {
                    "source": source,
                    "prompt_id": prompt,
                    "prompt_key": prompt,
                    "model": label,
                    "text": f"{source}-{prompt}-{index}",
                }
            )
    rows.append(
        {
            "source": "c1",
            "prompt_id": "p5",
            "prompt_key": "p5",
            "model": "a",
            "text": "leak",
        }
    )
    return pd.DataFrame(rows)


def test_reserves_every_unconsumed_source_and_removes_prompt_overlap() -> None:
    lock = build_firewall_lock(
        _pool(), consumed_sources={"c1", "c2"}, min_label_rows=20
    )
    assert lock["firewall_sources"] == ["r1", "r2"]
    assert lock["consumed_sources"] == ["c1", "c2"]
    assert lock["heldout_rows"] == 100
    assert lock["train_rows_before_leakage_removal"] == 101
    assert lock["train_rows_after_leakage_removal"] == 100
    assert lock["removed_training_rows_for_prompt_overlap"] == 1
    assert lock["prompt_id_overlap_after_removal"] == 0
    assert lock["prompt_key_overlap_after_removal"] == 0
    assert lock["eligible_labels"] == ["a", "b"]
    assert lock["eligible_heldout_rows"] == 100
    assert lock["training_labels_after_leakage_removal"] == ["a", "b"]
    assert lock["source_rows"] == {"c1": 51, "c2": 50, "r1": 50, "r2": 50}
    assert lock["model_scores_used_for_selection"] is False


def test_rejects_consumed_source_not_present() -> None:
    with pytest.raises(RuntimeError, match="consumed sources missing"):
        build_firewall_lock(
            _pool(), consumed_sources={"c1", "missing"}, min_label_rows=20
        )
