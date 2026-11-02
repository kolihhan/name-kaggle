from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/two-stage-lora24-gate-b-phase7-preflight.yml")


def test_phase7_preflight_regenerates_rows_and_fails_closed_on_frozen_drift():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "regenerate_phase7_oof" in text
    assert "artifacts/phase7/gated_specialist.json" in text
    assert "data/external/gpt_gemini_generations.parquet" in text
    assert "artifacts/phase1/folds.csv" in text
    assert "phase7_regenerated_oof.csv" in text
    assert "phase7_preflight.json" in text
    assert "two-stage-lora24-gate-b-phase7-preflight" in text

    lowered = text.lower()
    assert "kaggle kernels push" not in lowered
    assert "kaggle competitions submit" not in lowered
    assert "[gate-b-run]" not in text
    assert "public lb" not in lowered
