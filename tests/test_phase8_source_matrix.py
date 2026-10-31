import pandas as pd

from scripts.phase8_audit_response_corpus import (
    PRECOMPETITION_SOURCES,
    build_source_matrix,
    select_precompetition_sources,
    select_strict_sources,
)


def test_source_matrix_counts_unique_responses_and_models():
    provenance = pd.DataFrame(
        {
            "response_id": ["r1", "r1", "r2", "r3"],
            "source": ["A", "A", "A", "B"],
            "model": ["m1", "m1", "m2", "m1"],
            "license": ["MIT", "MIT", "MIT", "CC-BY-4.0"],
            "license_unknown": [False, False, False, False],
            "output_terms_review": [False, False, False, True],
            "model_evidence_class": ["request_and_returned_model", "request_and_returned_model", "dataset_card", "row_or_run_metadata"],
            "prompt_exact": [True, True, False, True],
        }
    )
    matrix = build_source_matrix(provenance)
    a = matrix["A"]
    assert a["occurrences"] == 3
    assert a["unique_responses"] == 2
    assert a["models"] == ["m1", "m2"]
    assert a["license_known_fraction"] == 1.0
    assert a["output_terms_clear_fraction"] == 1.0
    assert a["prompt_exact_fraction"] == 2 / 3


def test_strict_source_rule_requires_all_three_fractions_to_be_one():
    matrix = {
        "keep": {"license_known_fraction": 1.0, "output_terms_clear_fraction": 1.0, "prompt_exact_fraction": 1.0},
        "license_unknown": {"license_known_fraction": 0.99, "output_terms_clear_fraction": 1.0, "prompt_exact_fraction": 1.0},
        "terms_review": {"license_known_fraction": 1.0, "output_terms_clear_fraction": 0.5, "prompt_exact_fraction": 1.0},
        "prompt_not_exact": {"license_known_fraction": 1.0, "output_terms_clear_fraction": 1.0, "prompt_exact_fraction": 0.9},
    }
    assert select_strict_sources(matrix) == ["keep"]


def test_precompetition_gate_is_intersection_with_frozen_evidence_set():
    strict = sorted(PRECOMPETITION_SOURCES | {"post_competition_or_unpinned"})
    assert select_precompetition_sources(strict) == sorted(PRECOMPETITION_SOURCES)
