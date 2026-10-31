import numpy as np
import pandas as pd
import pytest

from scripts.run_phase11_fable_binary_override import (
    FABLE_LABEL,
    OTHER_LABEL,
    apply_fable_override,
    binary_training_rows,
    phase11_promotion_decision,
    validate_fable_external,
)


def _valid_external():
    return pd.DataFrame(
        {
            "text": [f"fable response {i}" for i in range(344)],
            "model": [FABLE_LABEL] * 344,
        }
    )


def test_validate_fable_external_accepts_exact_frozen_344_unique_rows():
    external = _valid_external()

    validate_fable_external(external)


def test_validate_fable_external_rejects_count_model_empty_or_duplicate_drift():
    with pytest.raises(RuntimeError, match="344"):
        validate_fable_external(_valid_external().iloc[:-1].copy())

    wrong_model = _valid_external()
    wrong_model.loc[0, "model"] = "claude-opus-4-8"
    with pytest.raises(RuntimeError, match="model"):
        validate_fable_external(wrong_model)

    empty = _valid_external()
    empty.loc[0, "text"] = "   "
    with pytest.raises(RuntimeError, match="empty"):
        validate_fable_external(empty)

    duplicate = _valid_external()
    duplicate.loc[1, "text"] = duplicate.loc[0, "text"]
    with pytest.raises(RuntimeError, match="unique"):
        validate_fable_external(duplicate)


def test_binary_training_rows_use_all_official_rows_and_external_only_as_positive():
    official_fit = pd.DataFrame(
        {
            "text": ["official fable", "official opus", "official gemini"],
            "model": [FABLE_LABEL, "claude-opus-5", "gemini-3.8-flash"],
        }
    )
    external = _valid_external()

    texts, labels = binary_training_rows(official_fit, external)

    assert len(texts) == 347
    assert len(labels) == 347
    assert labels[:3] == [FABLE_LABEL, OTHER_LABEL, OTHER_LABEL]
    assert labels[3:] == [FABLE_LABEL] * 344
    assert set(labels) == {FABLE_LABEL, OTHER_LABEL}


def test_override_only_adds_fable_and_never_removes_existing_fable():
    phase7 = np.array(
        [FABLE_LABEL, "claude-opus-5", "gpt-5.6-sol", "kimi-k3"],
        dtype=object,
    )
    detector = np.array(
        [OTHER_LABEL, FABLE_LABEL, OTHER_LABEL, FABLE_LABEL],
        dtype=object,
    )

    final, changed = apply_fable_override(phase7, detector)

    assert final.tolist() == [FABLE_LABEL, FABLE_LABEL, "gpt-5.6-sol", FABLE_LABEL]
    assert changed.tolist() == [False, True, False, True]
    assert final[0] == FABLE_LABEL
    assert all(final[i] == FABLE_LABEL for i in np.flatnonzero(changed))


def test_override_rejects_invalid_detector_labels_or_alignment():
    phase7 = np.array(["claude-opus-5"], dtype=object)

    with pytest.raises(RuntimeError, match="detector"):
        apply_fable_override(phase7, np.array(["claude-opus-4-8"], dtype=object))

    with pytest.raises(RuntimeError, match="length"):
        apply_fable_override(phase7, np.array([], dtype=object))


def test_promotion_rejects_numerical_equality_even_when_python_float_is_greater():
    phase7_ba = 0.3700521998508576
    numerical_noise = 0.37005219985085763

    decision = phase11_promotion_decision(
        phase7_ba=phase7_ba,
        final_ba=numerical_noise,
        phase7_fable_recall=0.14093959731543623,
        final_fable_recall=0.2,
        no_fable_removed=True,
        only_changes_to_fable=True,
    )

    assert numerical_noise > phase7_ba
    assert decision["material_ba_gain"] is False
    assert decision["promote_phase11"] is False


def test_promotion_requires_ba_and_fable_recall_and_both_invariants():
    base_kwargs = dict(
        phase7_ba=0.37,
        final_ba=0.371,
        phase7_fable_recall=0.14,
        final_fable_recall=0.20,
        no_fable_removed=True,
        only_changes_to_fable=True,
    )

    assert phase11_promotion_decision(**base_kwargs)["promote_phase11"] is True

    assert phase11_promotion_decision(
        **{**base_kwargs, "final_fable_recall": 0.14}
    )["promote_phase11"] is False
    assert phase11_promotion_decision(
        **{**base_kwargs, "no_fable_removed": False}
    )["promote_phase11"] is False
    assert phase11_promotion_decision(
        **{**base_kwargs, "only_changes_to_fable": False}
    )["promote_phase11"] is False
