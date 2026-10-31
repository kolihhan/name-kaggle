import pandas as pd

from scripts.phase8_audit_response_corpus import (
    build_clean_mask,
    normalize_text,
    summarize_overlap,
)


def test_clean_mask_rejects_quality_and_cross_label_flags():
    df = pd.DataFrame(
        {
            "no_train": [False, True, False, False],
            "mechanically_eligible": [True, True, False, True],
            "text_only_suitable": [True, True, True, True],
            "possible_truncation": [False, False, False, True],
            "possible_reasoning_in_answer": [False, False, False, False],
            "text_shared_across_labels": [False, False, False, False],
        }
    )
    assert build_clean_mask(df).tolist() == [True, False, False, False]


def test_normalize_text_collapses_case_unicode_and_whitespace():
    assert normalize_text("  Café\nTEST  ") == normalize_text("cafe\u0301 test")


def test_overlap_summary_detects_raw_and_normalized_overlap():
    official = ["Hello world", "Different answer"]
    external = ["hello   world", "brand new"]
    result = summarize_overlap(official, external)
    assert result["exact_overlap_count"] == 0
    assert result["normalized_overlap_count"] == 1
