"""Frozen 64-prompt GLM sibling transfer pilot, source distinct from prior panel.

No training, no threshold fitting, no leaderboard submission. Observational,
not certified independent of the champion's upstream prefit training sources.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path

FULL, FLASH = "glm-5.3", "glm-5.3-flash"
LABELS = {FULL, FLASH}
SOURCE_SHA = "8339081afee1d4aed0b806a686d91978f0d7815bc67fb8f620860477e0895675"
SEED = "ntl-external-glm-paired64-20261114-v0"
PAIRS = 64
RESULT = "glm-external-paired64-result.json"


def select_pairs(rows: list[dict]) -> list[dict]:
    """Select disjoint-pid, complete, same-prompt/settings pairs before inference."""
    glm = [r for r in rows if r.get("label") in LABELS]
    if not glm:
        raise ValueError("no GLM labels in source")
    for row in glm:
        if (row.get("returned_model") != row["label"]
            or row.get("provider") != "zhipu"
            or not row.get("user_message") or not row.get("pid")
            or not row.get("text") or not row.get("target_words")):
            raise ValueError("served-model identity/protocol gate failed")
    groups = defaultdict(list)
    for row in glm:
        key = (row["user_message"], row["target_words"], row.get("setting", ""))
        groups[key].append(row)
    good = []
    for key, group in groups.items():
        if (len(group) == 2 and set(r["label"] for r in group) == LABELS
            and len({r["pid"] for r in group}) == 1):
            h = hashlib.sha256((SEED + "|" + json.dumps(key, ensure_ascii=False)).encode()).hexdigest()
            good.append((h, key, group))
    good.sort(key=lambda v: v[0])
    chosen, seen_pids = [], set()
    for _, _, group in good:
        pid = group[0]["pid"]
        if pid in seen_pids:
            continue
        seen_pids.add(pid)
        chosen.extend(sorted(group, key=lambda r: r["label"]))
        if len(chosen) == 2 * PAIRS:
            break
    if len(chosen) != 2 * PAIRS or len(seen_pids) != PAIRS:
        raise ValueError("insufficient unique independent prompt IDs")
    normalized = [" ".join(r["text"].split()).casefold() for r in chosen]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate external responses")
    return chosen


def prepare(data):
    import pandas as pd
    import ntl_glm_runner as base

    sources = [p for p in base.DATA_ROOT.rglob("generations.csv")
               if "ntl-synth-generations-v1" in p.parts and base._digest(p) == SOURCE_SHA]
    if len(sources) != 1:
        raise ValueError("missing or drifted exact external source CSV")
    external = pd.read_csv(sources[0], dtype=str, keep_default_na=False)
    expected = {"label", "text", "user_message", "target_words", "setting",
                "provider", "returned_model", "requested_model", "pid"}
    if not expected <= set(external.columns) or len(external) != 4758:
        raise ValueError("external schema/denominator changed")
    selected = select_pairs(external.to_dict(orient="records"))

    official = pd.read_csv(Path(data) / "train.csv", usecols=["text", "model"])
    if len(set(official.model)) != 18:
        raise ValueError("invalid competition model label set")
    normal_train = {" ".join(str(x).split()).casefold() for x in official.text.dropna()}
    if any(" ".join(r["text"].split()).casefold() in normal_train for r in selected):
        raise ValueError("exact external response duplicates official train")
    records = [{"id": f"glmx-{i:03d}", "text": r["text"]} for i, r in enumerate(selected)]
    truth = [{"id": records[i]["id"], "model": r["label"],
              "pid": r["pid"], "run": r.get("run", "")}
             for i, r in enumerate(selected)]
    base.WORK.mkdir(parents=True, exist_ok=False)
    target = base.WORK / "data"
    target.mkdir()
    (target / "train.csv").symlink_to((Path(data) / "train.csv").resolve())
    pd.DataFrame(records).to_csv(target / "test.csv", index=False)
    pd.DataFrame(truth).to_csv(base.WORK / "truth.csv", index=False)
    print("NTL_GLM_EXTERNAL_INPUT_READY=128_TWO_LABELS_64_UNIQUE_PID", flush=True)
    return str(target)


def compare(data, payload):
    import pandas as pd
    import ntl_glm_runner as base
    from ntl_glm_conditional_runner import eligible_inputs, routed_predictions

    champion = pd.read_parquet(base.OUT / "submission.parquet")
    test = pd.read_csv(Path(data) / "test.csv", dtype={"id": str})
    truth = pd.read_csv(base.WORK / "truth.csv", dtype={"id": str})
    if list(champion.columns) != ["id", "model"] or len(champion) != 2 * PAIRS:
        raise ValueError("invalid frozen champion output")
    predictions = champion[["id", "model"]].to_dict("records")
    inputs = test[["id", "text"]].to_dict("records")
    selected = eligible_inputs(predictions, inputs)
    if not selected:
        raise ValueError("champion did not predict any GLM sibling")
    pd.DataFrame(selected, columns=["id", "text"]).to_csv(
        base.WORK / "ratio-input.csv", index=False)
    ratios, times = base.scores(payload)
    if not ratios.index.is_unique or set(ratios.index) != {r["id"] for r in selected}:
        raise ValueError("incomplete GLM real-model likelihood")
    treated = routed_predictions(predictions, {str(k): float(v) for k, v in ratios.items()})
    joined = (truth.set_index("id")
              .join(pd.DataFrame(predictions).set_index("id").rename(columns={"model": "champion"}))
              .join(pd.DataFrame(treated).set_index("id").rename(columns={"model": "routed"})))
    if joined.isna().any().any():
        raise ValueError("unmatched pair truth")
    by_label = {lab: base.outcome(group) for lab, group in joined.groupby("model")}
    if set(by_label) != LABELS or any(by_label[k]["rows"] != PAIRS for k in LABELS):
        raise ValueError("invalid GLM pair class balance")
    overall = base.outcome(joined)
    report = {
        "scope": "EXTERNAL_SOURCE_GLM_PAIR_DIAGNOSTIC_NOT_FULL_18",
        "dataset": "lxdlxdlxd/ntl-synth-generations-v1",
        "csv_sha256": SOURCE_SHA, "pairs": PAIRS, "rows": PAIRS * 2,
        "distinct_prompt_ids": truth.pid.nunique(), "eligible_glm": len(selected),
        "champion_ref": "58295802", "new_training": False,
        "competition_submission": False, "independence_proven": False,
        "overall": overall, "by_label": by_label,
        "two_label_ba_champion": sum(v["champion_correct"] / PAIRS for v in by_label.values()) / 2,
        "two_label_ba_routed": sum(v["routed_correct"] / PAIRS for v in by_label.values()) / 2,
        "model_stage_seconds": times,
    }
    if overall["routed_correct"] - overall["champion_correct"] != overall["rescues"] - overall["regressions"]:
        raise AssertionError("paired contingency mismatch")
    (base.OUT / RESULT).write_text(json.dumps(report, sort_keys=True, indent=2))
    print("NTL_GLM_EXTERNAL_PAIR_RESULT_READY=TRUE", flush=True)
