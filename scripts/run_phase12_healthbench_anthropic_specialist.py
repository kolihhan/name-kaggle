from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.phase12_audit_healthbench_anthropic import normalize_text
from scripts import run_phase12_anthropic_pair_specialist as pair


OPUS_LABEL = pair.OPUS_LABEL
SONNET_LABEL = pair.SONNET_LABEL
TARGET_LABELS = set(pair.PAIR_LABELS)
GAIN_EPSILON = pair.GAIN_EPSILON


def validate_healthbench_external(external: pd.DataFrame) -> None:
    required = {"text", "model"}
    missing = sorted(required - set(external.columns))
    if missing:
        raise RuntimeError(f"HealthBench external missing columns: {missing}")
    if len(external) != pair.EXPECTED_HEALTHBENCH_ROWS:
        raise RuntimeError(
            f"expected {pair.EXPECTED_HEALTHBENCH_ROWS} HealthBench rows, found {len(external)}"
        )
    if external[["text", "model"]].isna().any().any():
        raise RuntimeError("HealthBench external contains null text/model")
    if (external["text"].astype(str).str.strip() == "").any():
        raise RuntimeError("HealthBench external contains empty text")

    observed_labels = set(external["model"].astype(str))
    if observed_labels != TARGET_LABELS:
        raise RuntimeError(
            f"HealthBench external model label set drifted: {sorted(observed_labels)}"
        )

    counts = external["model"].astype(str).value_counts().to_dict()
    expected_counts = {
        OPUS_LABEL: pair.EXPECTED_OPUS_EXTERNAL_ROWS,
        SONNET_LABEL: pair.EXPECTED_SONNET_EXTERNAL_ROWS,
    }
    if counts != expected_counts:
        raise RuntimeError(
            f"HealthBench external model counts drifted: expected={expected_counts} observed={counts}"
        )

    normalized = external["text"].astype(str).map(normalize_text)
    if normalized.nunique() != pair.EXPECTED_HEALTHBENCH_ROWS:
        raise RuntimeError("HealthBench external response text must be globally normalized-unique")

    if "prompt_id" in external.columns:
        if external["prompt_id"].isna().any():
            raise RuntimeError("HealthBench external contains null prompt_id")
        for label, expected_count in expected_counts.items():
            subset = external.loc[external["model"].astype(str).eq(label)]
            if subset["prompt_id"].astype(str).nunique() != expected_count:
                raise RuntimeError(
                    f"HealthBench prompt_id values must be unique within {label}"
                )


def paired_training_rows(
    official_fit: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    return pair.specialist_training_rows(official_fit, external)


def gate_pair_predictions(
    phase7_predictions: np.ndarray,
    specialist_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    final = pair.gate_pair_predictions(phase7_predictions, specialist_predictions)
    baseline = np.asarray(phase7_predictions, dtype=object)
    changed = baseline != final
    return final, changed


def phase12_promotion_decision(
    *,
    phase7_ba: float,
    final_ba: float,
    phase7_pair_mean: float,
    final_pair_mean: float,
    other_16_recalls_preserved: bool,
    only_pair_rewrites: bool,
) -> dict[str, bool]:
    material_ba_gain = float(final_ba) - float(phase7_ba) > GAIN_EPSILON
    material_pair_gain = (
        float(final_pair_mean) - float(phase7_pair_mean) > GAIN_EPSILON
    )
    promote = (
        material_ba_gain
        and material_pair_gain
        and bool(other_16_recalls_preserved)
        and bool(only_pair_rewrites)
    )
    return {
        "material_ba_gain": material_ba_gain,
        "material_pair_mean_recall_gain": material_pair_gain,
        "other_16_recalls_preserved": bool(other_16_recalls_preserved),
        "only_pair_rewrites": bool(only_pair_rewrites),
        "promote_phase12": promote,
    }


def main() -> None:
    pair.main()


if __name__ == "__main__":
    main()
