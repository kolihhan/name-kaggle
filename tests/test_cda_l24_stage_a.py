"""CDA-L24 Stage-A preparation tests. Synthetic only; no official text used."""
from __future__ import annotations

import hashlib
import json
import pandas as pd
import pytest

from scripts.cda_l24_stage_a import (
    CANDIDATE, EXPECTED_POOL_SHA256, RESERVED_SOURCE,
    build_clean_partitions, validate_clearance, _text_hash
)

TRAIN1="origin/training-1"
TRAIN2="origin/training-2"
LABELS=["model-1","model-2","model-3","model-4",
        "model-5","model-6","model-7","model-8"]


def _manifest():
    sources=[TRAIN1,TRAIN2,RESERVED_SOURCE]
    evidence={s:{
        "origin_ref":s,"original_license":"CC BY 4.0",
        "license_evidence_url":"https://example.org/license",
        "model_versions_evidence_url":"https://example.org/models",
        "generation_protocol_evidence_url":"https://example.org/protocol",
        "independent_source_lineage_evidence_url":"https://example.org/lineage",
    } for s in sources}
    return {"candidate":CANDIDATE,"source_pool_sha256":EXPECTED_POOL_SHA256,
        "reserved_transfer_source":RESERVED_SOURCE,"source_lineage_reviewed":True,
        "original_terms_reviewed":True,"model_identity_reviewed":True,
        "overlap_reviewed":True,"provenance_status":"APPROVED",
        "approved_train_source_ids":[TRAIN1,TRAIN2],
        "original_source_evidence":evidence}


def test_manifest_fail_closed_until_real_review():
    m=_manifest()
    assert validate_clearance(m)==(TRAIN1,TRAIN2)
    del m["original_source_evidence"][RESERVED_SOURCE]["original_license"]
    with pytest.raises(PermissionError,match="original_license"):
        validate_clearance(m)
    m=_manifest()
    m["source_lineage_reviewed"]=False
    with pytest.raises(PermissionError,match="source_lineage_reviewed"):
        validate_clearance(m)
    m=_manifest()
    m["approved_train_source_ids"]=[TRAIN1,RESERVED_SOURCE]
    with pytest.raises(PermissionError,match="exclude transfer"):
        validate_clearance(m)


def _synthetic_rows():
    entries=[]
    for source in [TRAIN1,TRAIN2,RESERVED_SOURCE]:
        for lab in LABELS:
            for k in range(3):
                entries.append({
                  "source_dataset":source,"text":f"Response {source} {lab} {k}",
                  "model":lab,"prompt_group":f"{lab}-{k}",
                  "prompt_canon":f"{source} :: {lab} prompt {k}",
                  "source_license":"CC BY 4.0",
                  "flag_near_train_prompt":False,
                  "flag_train_prompt_regen":False,
                  "flag_kb_proxy":False,
                  "flag_gemini37_routing_risk":False,
                })
    return pd.DataFrame(entries)


def test_strict_source_separation_and_distinct_prompt_keys():
    frame=_synthetic_rows()
    # Some exact labels are no longer independent; expected reject flags/dups.
    frame.loc[frame.source_dataset.eq(TRAIN1)&frame.model.eq(LABELS[0])&frame.prompt_group.eq(
        f"{LABELS[0]}-0"),"flag_train_prompt_regen"]=True
    frame.loc[frame.source_dataset.eq(TRAIN2)&frame.model.eq(LABELS[1])&frame.prompt_group.eq(
        f"{LABELS[1]}-0"),"text"]="Known old train"
    # a shared prompt key across two different sources must not survive.
    shared="the same unseen but shared prompt"
    frame.loc[frame.source_dataset.eq(TRAIN1)&frame.model.eq(LABELS[2])&frame.prompt_group.eq(
        f"{LABELS[2]}-0"),"prompt_canon"]=shared
    frame.loc[frame.source_dataset.eq(RESERVED_SOURCE)&frame.model.eq(LABELS[2])&frame.prompt_group.eq(
        f"{LABELS[2]}-0"),"prompt_canon"]=shared
    a,b,stats=build_clean_partitions(frame,labels=LABELS,
         approved_sources=(TRAIN1,TRAIN2),
         old_fit_text_hashes={_text_hash("Known old train")},
         official_text_hashes=set(),old_prompt_keys=set())
    assert stats["flagged_rows"]==1
    assert stats["known_answer_overlap_rows"]==1
    assert stats["shared_train_transfer_prompt_keys_removed"]==1
    assert not set(a.prompt_key)&set(b.prompt_key)
    assert set(b.source)=={RESERVED_SOURCE}
    assert set(a.source)=={TRAIN1,TRAIN2}
    assert stats["train_labels"]==8
    assert stats["transfer_labels"]==8
    assert _text_hash("KNOWN    old Train")==_text_hash("known old train")


def test_missing_risk_column_or_unknown_official_label_fails():
    frame=_synthetic_rows()
    with pytest.raises(ValueError,match="source schema"):
        build_clean_partitions(frame.drop(columns=["flag_kb_proxy"]),
            labels=LABELS,approved_sources=(TRAIN1,TRAIN2),
            old_fit_text_hashes=set(),official_text_hashes=set(),old_prompt_keys=set())
    frame.loc[0,"model"]="random-injected-label"
    with pytest.raises(ValueError,match="unknown official labels"):
        build_clean_partitions(frame,labels=LABELS,approved_sources=(TRAIN1,TRAIN2),
            old_fit_text_hashes=set(),official_text_hashes=set(),old_prompt_keys=set())


def test_unknown_license_rows_are_excluded():
    frame=_synthetic_rows()
    frame.loc[frame.index[0],"source_license"]="unknown"
    a,b,s=build_clean_partitions(frame,labels=LABELS,
         approved_sources=(TRAIN1,TRAIN2),
         old_fit_text_hashes=set(),official_text_hashes=set(),old_prompt_keys=set())
    assert s["license_unknown_rows"]==1
    assert s["train_rows"]==47


def test_frozen_v1_invalidated_even_if_user_supplies_clearance_flag():
    """README-grounded scientific invalidity cannot be overridden by a JSON flag."""
    from scripts.cda_l24_stage_a import reject_invalid_frozen_v1_science_cohort
    with pytest.raises(PermissionError, match="COHORT DISQUALIFIED"):
        reject_invalid_frozen_v1_science_cohort()
