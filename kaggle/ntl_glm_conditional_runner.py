"""No-fit conditional GLM scoring after the frozen eight-expert champion predicts.

Selection is text+champion-prediction only. No true label, prompt ID, topic,
target length, or validation score is read before likelihood inference.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

FAMILY = frozenset({"glm-5.3", "glm-5.3-flash"})
FULL, FLASH = "glm-5.3", "glm-5.3-flash"
RESULT = "glm-conditional-summary.json"


def eligible_inputs(predictions: list[dict], inputs: list[dict]) -> list[dict]:
    """Only pass (id,text) for IDs predicted as a GLM sibling; fail closed."""
    if (len(predictions) != len(inputs) or not predictions
            or len({r.get("id") for r in predictions}) != len(predictions)
            or len({r.get("id") for r in inputs}) != len(inputs)):
        raise ValueError("mismatched or duplicate input IDs")
    by_id = {}
    for row in predictions:
        if set(row) != {"id", "model"} or not isinstance(row["id"], str) or not isinstance(row["model"], str):
            raise ValueError("invalid champion prediction fields")
        by_id[row["id"]] = row["model"]
    if set(by_id) != {r.get("id") for r in inputs}:
        raise ValueError("champion and input ID mismatch")
    for row in inputs:
        if set(row) != {"id", "text"} or not isinstance(row["id"], str) or not isinstance(row["text"], str):
            raise ValueError("GLM inference may receive only id and text")
    return [dict(row) for row in inputs if by_id[row["id"]] in FAMILY]


def routed_predictions(predictions: list[dict], differences: dict[str, float]) -> list[dict]:
    """Zero threshold is frozen; keep all non-GLM predictions untouched."""
    allowed = {row["id"] for row in predictions if row["model"] in FAMILY}
    if set(differences) != allowed:
        raise ValueError("missing, additional or stale likelihood IDs")
    for ratio in differences.values():
        if isinstance(ratio, bool) or not isinstance(ratio, (float, int)) or not math.isfinite(ratio):
            raise ValueError("invalid likelihood")
    out = []
    for row in predictions:
        model = row["model"]
        if model in FAMILY:
            ratio = differences[row["id"]]
            if ratio > 0:
                model = FULL
            elif ratio < 0:
                model = FLASH
        out.append({"id": row["id"], "model": model})
    return out


def run(data, payload):
    """Run the already-pinned scoring modules only on champion-selected GLM texts."""
    import pandas as pd
    import ntl_glm_runner as base

    champion = pd.read_parquet(base.OUT / "submission.parquet")
    if list(champion.columns) != ["id", "model"]:
        raise ValueError("unexpected champion output schema")
    test = pd.read_csv(Path(data) / "test.csv", dtype={"id": str})
    if set(test.columns) != {"id", "text"} or len(test) != 252:
        raise ValueError("unexpected observational development test")
    if len(champion) != len(test) or champion["id"].isna().any() or champion["model"].isna().any():
        raise ValueError("champion row mismatch")

    predictions = champion[["id", "model"]].to_dict(orient="records")
    inputs = test[["id", "text"]].to_dict(orient="records")
    selected = eligible_inputs(predictions, inputs)
    selected_count = len(selected)

    # The third-party likelihood worker sees only the subset's ID/text.
    # The labels in the diagnostic truth file are not passed into this process.
    if selected_count:
        pd.DataFrame(selected, columns=["id", "text"]).to_csv(base.WORK / "ratio-input.csv", index=False)
        ratios, seconds = base.scores(payload)
        if set(ratios.index) != {x["id"] for x in selected}:
            raise ValueError("incomplete or unexpected likelihood outputs")
        differences = {str(k): float(v) for k, v in ratios.items()}
    else:
        differences, seconds = {}, {"flash": 0.0, "glm53": 0.0}

    treated = routed_predictions(predictions, differences)
    if [x["id"] for x in treated] != [x["id"] for x in predictions]:
        raise ValueError("prediction order changed")

    # Truth labels are used ONLY after the frozen inference rule has finished.
    truth = pd.read_csv(base.WORK / "truth.csv", dtype={"id": str})
    panel = truth.loc[truth.panel == "panel"]
    if len(panel) != 252 or set(panel.id) != {x["id"] for x in inputs}:
        raise ValueError("invalid development truth")
    joined = panel.set_index("id").join(
        pd.DataFrame(predictions).set_index("id").rename(columns={"model": "champion"})
    ).join(
        pd.DataFrame(treated).set_index("id").rename(columns={"model": "routed"})
    )
    by_label = {label: base.outcome(g) for label, g in joined.groupby("model")}
    if len(by_label) != 18 or not all(v["rows"] == 14 for v in by_label.values()):
        raise ValueError("invalid observational class support")
    overall = base.outcome(joined)
    if overall["routed_correct"] - overall["champion_correct"] != overall["rescues"] - overall["regressions"]:
        raise AssertionError("incorrect paired contingency")
    report = {
        "scope": "OBSERVATIONAL_DEVELOPMENT_ONLY", "champion_ref": "58295802",
        "official_submission": False, "new_training": False, "independent_holdout": False,
        "conditional_only": True, "rows": 252, "eligible_glm": selected_count,
        "selected_fraction": selected_count / len(predictions),
        "overall": overall, "by_label": by_label, "runtime_seconds": seconds,
        "champion_panel_ba": sum(v["champion_correct"] / v["rows"] for v in by_label.values()) / 18,
        "routed_panel_ba": sum(v["routed_correct"] / v["rows"] for v in by_label.values()) / 18,
        "source_comparison": "same frozen 252-row panel; prior full-scan GLM result is spent",
    }
    (base.OUT / RESULT).write_text(json.dumps(report, sort_keys=True, indent=2))
    print("NTL_GLM_CONDITIONAL_ONLY_RESULT_READY=" + str(selected_count), flush=True)
