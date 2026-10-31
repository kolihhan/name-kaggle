import numpy as np
import pandas as pd
import pytest

from scripts.run_phase12_healthbench_anthropic_specialist import (
    OPUS_LABEL,
    SONNET_LABEL,
    TARGET_LABELS,
    gate_pair_predictions,
    paired_training_rows,
    phase12_promotion_decision,
    validate_healthbench_external,
)


def _valid_external():
    return pd.DataFrame(
        {
            "text": [f"opus response {i}" for i in range(600)]
            + [f"sonnet response {i}" for i in range(610)],
            "model": [OPUS_LABEL] * 600 + [SONNET_LABEL] * 610,
        }
    )


def test_validate_healthbench_external_accepts_exact_frozen_counts_and_unique_rows():
    validate_healthbench_external(_valid_external())


def test_validate_healthbench_external_rejects_count_model_empty_or_duplicate_drift():
    with pytest.raises(RuntimeError, match="1210"):
        validate_healthbench_external(_valid_external().iloc[:-1].copy())

    wrong_model = _valid_external()
    wrong_model.loc[0, "model"] = "claude-opus-4-8"
    with pytest.raises(RuntimeError, match="model"):
        validate_healthbench_external(wrong_model)

    empty = _valid_external()
    empty.loc[0, "text"] = "   "
    with pytest.raises(RuntimeError, match="empty"):
        validate_healthbench_external(empty)

    duplicate = _valid_external()
    duplicate.loc[1, "text"] = duplicate.loc[0, "text"]
    with pytest.raises(RuntimeError, match="unique"):
        validate_healthbench_external(duplicate)


def test_paired_training_rows_keep_only_official_pair_and_append_all_external_rows():
    official_fit = pd.DataFrame(
        {
            "text": ["official opus", "official sonnet", "official fable", "official gpt"],
            "model": [OPUS_LABEL, SONNET_LABEL, "claude-fable-5", "gpt-5.6-sol"],
        }
    )
    external = _valid_external()

    texts, labels = paired_training_rows(official_fit, external)

    assert texts[:2] == ["official opus", "official sonnet"]
    assert labels[:2] == [OPUS_LABEL, SONNET_LABEL]
    assert len(texts) == 1212
    assert len(labels) == 1212
    assert labels.count(OPUS_LABEL) == 601
    assert labels.count(SONNET_LABEL) == 611
    assert set(labels) == TARGET_LABELS


def test_pair_gate_changes_only_rows_phase7_already_put_inside_pair():
    phase7 = np.array(
        [OPUS_LABEL, SONNET_LABEL, "claude-fable-5", "gpt-5.6-sol"],
        dtype=object,
    )
    specialist = np.array(
        [SONNET_LABEL, OPUS_LABEL, SONNET_LABEL, OPUS_LABEL],
        dtype=object,
    )

    final, changed = gate_pair_predictions(phase7, specialist)

    assert final.tolist() == [SONNET_LABEL, OPUS_LABEL, "claude-fable-5", "gpt-5.6-sol"]
    assert changed.tolist() == [True, True, False, False]


def test_pair_gate_rejects_invalid_specialist_labels_or_alignment():
    phase7 = np.array([OPUS_LABEL], dtype=object)

    with pytest.raises(RuntimeError, match="specialist"):
        gate_pair_predictions(phase7, np.array(["claude-fable-5"], dtype=object))

    with pytest.raises(RuntimeError, match="length"):
        gate_pair_predictions(phase7, np.array([], dtype=object))


def test_promotion_rejects_float_noise_and_requires_pair_mean_gain_and_invariants():
    phase7_ba = 0.3700521998508576
    noise = 0.37005219985085763

    decision = phase12_promotion_decision(
        phase7_ba=phase7_ba,
        final_ba=noise,
        phase7_pair_mean=0.49666666666666665,
        final_pair_mean=0.51,
        other_16_recalls_preserved=True,
        only_pair_rewrites=True,
    )

    assert noise > phase7_ba
    assert decision["material_ba_gain"] is False
    assert decision["promote_phase12"] is False

    base = dict(
        phase7_ba=0.3700521998508576,
        final_ba=0.371,
        phase7_pair_mean=0.49666666666666665,
        final_pair_mean=0.51,
        other_16_recalls_preserved=True,
        only_pair_rewrites=True,
    )
    assert phase12_promotion_decision(**base)["promote_phase12"] is True
    assert phase12_promotion_decision(
        **{**base, "final_pair_mean": 0.49666666666666665}
    )["promote_phase12"] is False
    assert phase12_promotion_decision(
        **{**base, "other_16_recalls_preserved": False}
    )["promote_phase12"] is False
    assert phase12_promotion_decision(
        **{**base, "only_pair_rewrites": False}
    )["promote_phase12"] is False
