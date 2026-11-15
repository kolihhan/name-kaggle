"""No-training, no-prediction-export gate for frozen NTL paired aggregate evidence.

This gate cannot turn a spent 12-class experiment into independent 18-class
balanced accuracy. All inputs are private Kaggle aggregate counts only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = ("rows", "true_labels", "champ_correct", "wemm_correct",
            "both_right", "both_wrong", "rescues", "regressions",
            "prediction_disagreements")


def check_counts(row: dict) -> None:
    if any(type(row.get(k)) is not int or row[k] < 0 for k in REQUIRED):
        raise ValueError("missing_or_invalid_paired_aggregate_counts")
    n = row["rows"]
    if not (1 <= row["true_labels"] <= 18 and
            row["both_right"] + row["both_wrong"] +
            row["rescues"] + row["regressions"] == n and
            row["both_right"] + row["regressions"] == row["champ_correct"] and
            row["both_right"] + row["rescues"] == row["wemm_correct"] and
            row["rescues"] + row["regressions"] <= row["prediction_disagreements"] <= n):
        raise ValueError("paired_contingency_inconsistent")


def evaluate(summary: dict) -> dict:
    combined = summary["combined"]
    check_counts(combined)
    sources = summary["per_source"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("no_source_disaggregation")
    for row in sources.values():
        check_counts(row)
    for k in ("rows", "champ_correct", "wemm_correct", "both_right",
              "both_wrong", "rescues", "regressions", "prediction_disagreements"):
        if sum(x[k] for x in sources.values()) != combined[k]:
            raise ValueError("aggregated_source_counts_mismatch")
    if summary.get("champion_official_submission_ref") != "58295802":
        raise ValueError("not_official_pinned_0683_champion")
    if summary.get("scope") != "SPENT_TWO_SOURCE_12_LABEL_DIAGNOSTIC_ONLY":
        raise ValueError("source_scope_drift_or_unqualified_new_source")
    if summary.get("wemm_folds_evaluated") != [0]:
        raise ValueError("not_same_frozen_single_fold")
    if summary.get("official_submission") is not False or summary.get("no_new_training") is not True:
        raise ValueError("provenance_or_no_training_drift")
    rescued, regressed = combined["rescues"], combined["regressions"]
    net = rescued - regressed
    return {
        "source_type": "EXISTING_SPENT_12_LABEL_CONDITIONAL_DIAGNOSTIC",
        "matched_rows": combined["rows"],
        "true_label_support": combined["true_labels"],
        "required_true_labels": 18,
        "champion_correct": combined["champ_correct"],
        "candidate_correct": combined["wemm_correct"],
        "rescues": rescued,
        "regressions": regressed,
        "net_rescues_minus_regressions": net,
        "per_source_net": {s: d["rescues"]-d["regressions"]
                           for s,d in sorted(sources.items())},
        "independent_full18_gate": "BLOCKED_SPENT_12_OF_18",
        "candidate_decision": "REJECT_WE_MM_FOLD0_CURRENT_RELEASE" if net <= 0
                              else "PRELIMINARY_ONLY_NOT_PROMOTABLE",
        "promote": False,
        "new_training": False,
        "new_inference": False,
        "new_competition_submission": False,
        "limitation": "This compares only Areeb8 vs single WeMM fold0, NOT Lightning0694. Oracle union cannot be deployed. No fresh 18-class validation.",
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    args=parser.parse_args()
    x=json.loads(args.aggregate.read_text(encoding="utf-8"))
    print("NTL_COMPLEMENTARITY_EXISTING_REAL_GATE="+json.dumps(evaluate(x),sort_keys=True),flush=True)


if __name__=="__main__":
    main()
