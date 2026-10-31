import numpy as np
import pandas as pd
import pytest

from scripts.run_phase12_anthropic_pair_specialist import (
    OPUS_LABEL,
    PAIR_LABELS,
    PHASE7_PAIR_MEAN_RECALL,
    PHASE7_SCORE,
    SONNET_LABEL,
    gate_pair_predictions,
    promotion_decision,
    specialist_training_rows,
    validate_healthbench_external,
)


def test_pair_label_set_is_exactly_opus5_and_sonnet5():
    assert PAIR_LABELS == {OPUS_LABEL, SONNET_LABEL}
    assert OPUS_LABEL == "claude-opus-5"
    assert SONNET_LABEL == "claude-sonnet-5"


def test_pair_gate_only_changes_rows_phase7_predicted_inside_pair():
    baseline = np.array([OPUS_LABEL, "gpt-5.5", SONNET_LABEL, "kimi-k3"], dtype=object)
    specialist = np.array([SONNET_LABEL, OPUS_LABEL, OPUS_LABEL, SONNET_LABEL], dtype=object)

    final = gate_pair_predictions(baseline, specialist)

    assert final.tolist() == [SONNET_LABEL, "gpt-5.5", OPUS_LABEL, "kimi-k3"]


def test_pair_gate_rejects_specialist_output_outside_pair():
    baseline = np.array([OPUS_LABEL], dtype=object)
    specialist = np.array(["gpt-5.5"], dtype=object)
    with pytest.raises(RuntimeError, match="outside pair"):
        gate_pair_predictions(baseline, specialist)


def test_specialist_training_rows_use_only_official_pair_plus_all_external():
    official_fit = pd.DataFrame(
        {
            "text": ["official opus", "official sonnet", "official other"],
            "model": [OPUS_LABEL, SONNET_LABEL, "gpt-5.5"],
        }
    )
    external = pd.DataFrame(
        {
            "text": ["external opus", "external sonnet"],
            "model": [OPUS_LABEL, SONNET_LABEL],
        }
    )

    texts, labels = specialist_training_rows(official_fit, external)

    assert texts == ["official opus", "official sonnet", "external opus", "external sonnet"]
    assert labels == [OPUS_LABEL, SONNET_LABEL, OPUS_LABEL, SONNET_LABEL]


def test_healthbench_external_validation_enforces_frozen_counts_and_unique_text():
    rows = []
    for i in range(600):
        rows.append({"text": f"opus {i}", "model": OPUS_LABEL, "prompt_id": f"o{i}"})
    for i in range(610):
        rows.append({"text": f"sonnet {i}", "model": SONNET_LABEL, "prompt_id": f"s{i}"})
    external = pd.DataFrame(rows)

    validate_healthbench_external(external)

    external.loc[0, "text"] = external.loc[1, "text"]
    with pytest.raises(RuntimeError, match="unique"):
        validate_healthbench_external(external)


def test_promotion_rejects_float_noise_and_requires_pair_gain_and_invariant():
    noise = promotion_decision(
        PHASE7_SCORE + 5e-17,
        PHASE7_PAIR_MEAN_RECALL,
        other_16_recalls_preserved=True,
        pair_only_changes=True,
    )
    assert noise["promote_phase12"] is False

    good = promotion_decision(
        PHASE7_SCORE + 0.001,
        PHASE7_PAIR_MEAN_RECALL + 0.009,
        other_16_recalls_preserved=True,
        pair_only_changes=True,
    )
    assert good["promote_phase12"] is True

    bad_invariant = promotion_decision(
        PHASE7_SCORE + 0.001,
        PHASE7_PAIR_MEAN_RECALL + 0.009,
        other_16_recalls_preserved=False,
        pair_only_changes=True,
    )
    assert bad_invariant["promote_phase12"] is False
