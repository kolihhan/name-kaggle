"""Synthetic NDA-safe verification of new private checkpoint and gates."""
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.cda_l24_checkpoint import (
    directory_manifest_digest,validate_cda_checkpoint
)
from scripts.cda_l24_stage_a import (
    CANDIDATE,EXPECTED_POOL_SHA256,RESERVED_SOURCE
)
from scripts.cda_l24_stage_b import assert_promoted_stage_a
from scripts.cda_l24_transfer_gate import assess

LABELS = [f"gpt-test-{i}" for i in range(4)] + [
    f"claude-test-{i}" for i in range(4)]
ALL_LABELS = [f"m{i:02}" for i in range(18)]
SHA = "a"*64


def _fake_checkpoint(tmp_path:Path)->tuple[Path,str]:
    root=tmp_path/"saved"
    (root/"adapter").mkdir(parents=True)
    (root/"tokenizer").mkdir(parents=True)
    (root/"adapter/adapter_model.safetensors").write_bytes(b"placeholder-weights")
    (root/"adapter/adapter_config.json").write_text("{}")
    (root/"tokenizer/tokenizer.json").write_text("{}")
    (root/"heads.safetensors").write_bytes(b"source-head-not-present")
    (root/"checkpoint.sha256").write_text("mock\n")
    from scripts.run_two_stage_lora24_gate_b_gpu import CHECKPOINT_MANIFEST_SHA256
    (root/"meta.json").write_text(json.dumps({
       "candidate":CANDIDATE,"arm":"C1","labels":ALL_LABELS,
       "source_ids":["source-a","source-b"],
       "source_pool_sha256":EXPECTED_POOL_SHA256,
       "base_stage1_sha256":CHECKPOINT_MANIFEST_SHA256,
       "source_transfer_reserved":RESERVED_SOURCE,
       "source_head_persisted":False,
       "provenance_manifest_sha256":SHA
    }))
    return root,directory_manifest_digest(root)


def test_checkpoint_validates_all_artifacts_and_rejects_modified_bytes(tmp_path):
    root,digest=_fake_checkpoint(tmp_path)
    m=validate_cda_checkpoint(root,labels=ALL_LABELS,
           expected_sha256=digest,arm="C1",expected_clearance_sha256=SHA)
    assert m["source_head_persisted"] is False
    (root/"heads.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError,match="integrity mismatch"):
        validate_cda_checkpoint(root,labels=ALL_LABELS,
           expected_sha256=digest,arm="C1",expected_clearance_sha256=SHA)


def test_stage_a_promotion_gate_fails_closed(tmp_path):
    p=tmp_path/"gate.json"
    p.write_text(json.dumps({"decision":"REJECT_CDA_L24_SOURCE_TRANSFER"}))
    with pytest.raises(PermissionError,match="not passed"):
        assert_promoted_stage_a(p)
    p.write_text(json.dumps({
      "decision":"PROMOTE_TO_GROUPED_OFFICIAL_OOF",
      "C1_minus_C0":0.022,"prompt_bootstrap_CI95":[0.003,0.04],
      "provider_macro_delta":-0.005,"eligible_true_labels":8,
      "official_competition_OOF":False}))
    assert assert_promoted_stage_a(p)["C1_minus_C0"]==.022


def _synth():
    rows=[]
    for label in LABELS:
        for k in range(40):
            rows.append({"row_index":len(rows),
                         "source":RESERVED_SOURCE,
                         "prompt_group":f"unseen-{label}-{k}",
                         "label":label,
                         "prediction":label if k<26 else "glm-bad"})
    a=pd.DataFrame(rows)
    b=a.copy(deep=True)
    # Correct additional errors; never change prompt or true model.
    b.loc[b.prediction=="glm-bad","prediction"]=b.loc[
        b.prediction=="glm-bad","label"]
    return a,b


def test_grouped_transfer_gate_rescues_without_posthoc_fitting():
    a,b=_synth()
    z=assess(a,b,reps=250,seed=20261108)
    assert z["decision"]=="PROMOTE_TO_GROUPED_OFFICIAL_OOF"
    assert z["eligible_true_labels"]==8
    assert z["C1_minus_C0"] > 0.3
    assert z["prompt_bootstrap_CI95"][0]>0
    assert z["C1_only_correct"]==112
    assert z["C0_only_correct"]==0


def test_unpaired_transfer_rows_rejected():
    a,b=_synth()
    b.loc[0,"label"]="different"
    with pytest.raises(ValueError,match="unpaired"):
        assess(a,b,reps=100)
