"""Fail-closed evidence classification for new NTL research decisions.

This does not certify the truth of supplied attestations. Missing evidence
blocks claims; engineering success is never itself a scientific verdict.
"""
from dataclasses import dataclass
from enum import Enum
from math import isfinite

class Outcome(str, Enum):
    BLOCKED = "BLOCKED_BY_EVIDENCE"
    ENGINEERING = "ENGINEERING_ONLY"
    DIAGNOSTIC = "DIAGNOSTIC_ONLY"
    PROMPT_RETIRED = "EXACT_INTERVENTION_RETIRED"
    PROMOTION_ELIGIBLE = "PROMOTION_ELIGIBLE"

@dataclass(frozen=True)
class Evidence:
    run_completed: bool
    outputs_verified: bool
    intervention: str
    metric: str
    truth_classes: int
    same_rows: bool = False
    independent_source: bool = False
    provenance_verified: bool = False
    baseline_sha: str = ""
    full18_delta: float | None = None
    paired_ci_low: float | None = None
    runtime_verified: bool = False
    exact_intervention_failed: bool = False
    diagnostic_only: bool = True

def classify(e: Evidence) -> Outcome:
    if not e.intervention.strip():
        raise ValueError("Name the exact intervention, not the entire model family")
    if not 0 <= e.truth_classes <= 18:
        raise ValueError("Invalid truth class count")
    if not e.run_completed or not e.outputs_verified:
        return Outcome.BLOCKED
    if e.metric == "none":
        return Outcome.ENGINEERING
    if e.exact_intervention_failed:
        return Outcome.PROMPT_RETIRED
    if (e.diagnostic_only or e.metric != "18class_BA" or e.truth_classes != 18
        or not (e.same_rows and e.independent_source and e.provenance_verified
                and e.runtime_verified)
        or len(e.baseline_sha) != 64
        or any(c not in "0123456789abcdef" for c in e.baseline_sha)
        or e.full18_delta is None or e.paired_ci_low is None
        or not all(isfinite(x) for x in (e.full18_delta,e.paired_ci_low))
        or e.full18_delta < .02 or not 0 < e.paired_ci_low <= e.full18_delta):
        return Outcome.DIAGNOSTIC
    return Outcome.PROMOTION_ELIGIBLE

def require_promotion(e: Evidence) -> None:
    result=classify(e)
    if result is not Outcome.PROMOTION_ELIGIBLE:
        raise ValueError(f"{result.value}: need independently admitted full-18 same-row BA, pinned incumbent, paired CI and measured runtime; do not submit as a promoted candidate")
