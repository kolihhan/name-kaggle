"""Apply frozen GLM-sibling likelihood to a scored champion's actual test outputs.

This production-only module never reads training labels, gold answers, topic IDs,
prompt IDs, or the development diagnostic's truth.csv.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ntl_glm_conditional_runner import eligible_inputs, routed_predictions


def finalize(data, payload, *, output_root=Path('/kaggle/working')):
    import pandas as pd
    import ntl_glm_runner as scorer

    data = Path(data)
    output = Path(output_root) / "submission.parquet"
    champion = pd.read_parquet(output)
    if list(champion.columns) != ["id", "model"] or len(champion) == 0:
        raise ValueError("unexpected frozen champion submission format")
    if champion.id.isna().any() or champion.model.isna().any():
        raise ValueError("null champion prediction")
    prediction_ids = champion.id.astype(str)
    if not prediction_ids.is_unique:
        raise ValueError("duplicate champion IDs")
    # All non-GLM predictions are retained, including original id dtype/order.
    preds = [{"id": id_, "model": str(model)}
             for id_, model in zip(prediction_ids, champion.model)]
    test = pd.read_csv(data / "test.csv", dtype={"id": str}, usecols=["id", "text"])
    if len(test) != len(champion):
        raise ValueError("test and prediction lengths disagree")
    texts = test[["id", "text"]].to_dict(orient="records")
    selected = eligible_inputs(preds, texts)

    if selected:
        if scorer.WORK.exists():
            raise RuntimeError("GLM scratch directory unexpectedly exists")
        scorer.WORK.mkdir(parents=True, exist_ok=False)
        pd.DataFrame(selected, columns=["id", "text"]).to_csv(
            scorer.WORK / "ratio-input.csv", index=False)
        ratios, stages = scorer.scores(payload)
        if not ratios.index.is_unique or set(map(str, ratios.index)) != {r["id"] for r in selected}:
            raise ValueError("incomplete likelihood rows")
        differences = {str(key): float(value) for key, value in ratios.items()}
    else:
        differences, stages = {}, {"flash": 0.0, "glm53": 0.0}
    treated = routed_predictions(preds, differences)
    if [r["id"] for r in treated] != list(prediction_ids):
        raise ValueError("prediction order drift")
    output_models = [r["model"] for r in treated]
    changes = sum(a != b for a, b in zip(champion.model, output_models))
    champion["model"] = output_models

    # Only overwrite once all model runs, ID checks and routing succeed.
    temporary = output.with_suffix(".glm-tmp.parquet")
    champion.to_parquet(temporary, index=False)
    if list(pd.read_parquet(temporary).columns) != ["id", "model"]:
        raise RuntimeError("invalid candidate parquet")
    os.replace(temporary, output)

    metrics = {
        "frozen_gate": "positive_loglikelihood_full_else_flash",
        "scored_champion_ref": "58295802",
        "total": len(champion), "eligible_glm": len(selected),
        "changed": changes, "stage_seconds": stages,
        "new_training": False, "used_truth_labels": False,
    }
    (Path(output_root) / "glm-candidate-runtime.json").write_text(
        json.dumps(metrics, sort_keys=True))
    print("NTL_CONDITIONAL_CANDIDATE_COMPLETED=" +
          json.dumps(metrics, sort_keys=True), flush=True)
