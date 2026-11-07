"""Synthetic tests: unique prompts != repeated seeds; no source fetch/GPU."""
from scripts.audit_stablepair_ma_bench_data_gate import (
    metadata_audit, MODEL_PAIR, CONDITIONS, META_FIELDS
)


def fixture(n_cases=10, seeds=5, metadata=False):
    for case in range(n_cases):
        for seed in range(seeds):
            for model in MODEL_PAIR:
                for condition in CONDITIONS:
                    x={
                        "brief_id":f"{condition}/{case}/{model}/{seed}",
                        "condition":condition,"case_id":f"C{case}",
                        "scenario":f"case-{case}",
                        "model":model,"model_slug":model,"seed":seed,
                        "text":"test visible model output " * (50 + case)}
                    if metadata:
                        x.update({key:"fixture" for key in META_FIELDS})
                    yield x


def test_repeats_are_not_independent_prompts():
    x=metadata_audit(fixture(),source_sha256="fixture",read_bytes=0)
    assert x["records"]==200
    assert x["unique_scenarios_global"]==10
    assert x["matched_distinct_cases_both_conditions"]==10
    assert x["matched_case_seed_both_conditions"]==50
    assert x["gate"]=="NOT_ELIGIBLE_FOR_PAIRED_PROTOCOL_PILOT"
    assert x["full_stablepair_source_gate"]=="NOT_MET_ONE_SOURCE_ONLY"
    assert "test visible" not in str(x)


def test_hundred_real_cases_plus_metadata_would_only_pass_pilot():
    x=metadata_audit(fixture(n_cases=101,seeds=1,metadata=True),
                     source_sha256="fixture",read_bytes=0)
    assert x["matched_distinct_cases_both_conditions"]==101
    assert x["eligible_for_pilot"] is True
    assert x["gate"]=="PASS_FOR_PILOT_CONTRACT_ONLY"
    assert x["full_stablepair_source_gate"]=="NOT_MET_ONE_SOURCE_ONLY"


def test_duplicates_fail_closed():
    import pytest
    rows=list(fixture(n_cases=1,seeds=1))
    with pytest.raises(RuntimeError,match="duplicate brief_id"):
        metadata_audit(rows+[rows[0]],source_sha256="fixture",read_bytes=0)
