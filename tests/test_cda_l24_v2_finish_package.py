"""No real model or Kaggle data: validate the private one-shot C0/C1 handoff."""
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from scripts.cda_l24_v2_finish_package import prepare,find_checkpoint,check_metadata


def _write_arm(root,arm,shas,labels,preds):
    ck=root/f"cda_l24_v2_{arm}_checkpoint"
    (ck/"adapter").mkdir(parents=True)
    (ck/"tokenizer").mkdir()
    for p,data in {
       "adapter/adapter_model.safetensors":b"synthetic-adapter",
       "adapter/adapter_config.json":b"{}",
       "heads.safetensors":b"synthetic-heads",
       "tokenizer/tokenizer.json":b"{}",
       "checkpoint.sha256":b"fake",
    }.items():
        (ck/p).write_bytes(data)
    source_sha=hashlib.sha256(json.dumps(shas,sort_keys=True).encode()).hexdigest()
    meta={
       "arm":arm,"candidate":"cda-l24-direct-sources-v2",
       "source_transfer_reserved":"kilupy/ntl-18-model-corpus",
       "base_stage1_sha256":"c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0",
       "source_head_persisted":False,"source_ids":sorted(list(shas)[:2]),
       "source_pool_sha256":"DIRECT_ORIGINAL_ARX_TRN_TRAIN_KIL_80x2_HELDOUT_V2",
       "labels":labels,"provenance_manifest_sha256":source_sha,
       "run_stats":{"arm":arm,"strength":0.1 if arm=="C1" else 0.,
          "train_rows":1000,"transfer_rows":len(preds),
          "crossed_labels":8,"train_sources":2,"optimizer_steps":10,
          "transfer_labels_present":18,"transfer_source":"kilupy/ntl-18-model-corpus",
          "stage_A_source_missing_classes":["gemini-3.7-flash"],
          "stage_A_identity_CE_uses_only_observed_classes":True,
          "official_all_18_class_logits_retained_for_inference":True},
       "train_stats":{"dataset_sha256":shas}
    }
    (ck/"meta.json").write_text(json.dumps(meta))
    preds.to_csv(root/f"cda_l24_v2_{arm}_private_predictions.csv",index=False)


def _dataset(tmp_path):
    labels=[f"fake-{i}" for i in range(17)]+["gemini-3.7-flash"]
    rows=[]
    for i,name in enumerate(labels):
        for j in range(21):
            rows.append({"row_index":len(rows),"source":"kilupy/ntl-18-model-corpus",
                         "prompt_group":f"test-group-{i}-{j}",
                         "label":name,"prediction":name if j<10 else "wrong-provider"})
    a=pd.DataFrame(rows);z=a.copy()
    z["prediction"]=z["label"]
    shas={"arxitekton/ntl-claude-gen-v1":"a"*64,
          "trnhip/ntl-api-gen":"b"*64,
          "kilupy/ntl-18-model-corpus":"c"*64}
    c0=tmp_path/"C0";c1=tmp_path/"C1"
    c0.mkdir();c1.mkdir()
    _write_arm(c0,"C0",shas,labels,a)
    _write_arm(c1,"C1",shas,labels,z)
    return c0,c1


def test_one_shot_pairing_private_dataset_manifest_and_provenance(tmp_path):
    c0,c1=_dataset(tmp_path)
    report=prepare(c0,c1,tmp_path/"package",tmp_path/"report.json")
    assert report["eligible_true_labels"]==18
    assert report["decision"]=="PROMOTE_TO_GROUPED_OFFICIAL_OOF"
    assert report["C1_minus_C0"] > .4
    assert report["checkpoint_archive_bytes"]>0
    m=json.loads((tmp_path/"package/dataset-metadata.json").read_text())
    assert m["id"]=="kolihhan/ntl-cda-l24-v2-c1-weights"
    assert m["isPrivate"] is True
    with zipfile.ZipFile(tmp_path/"package/cda-v2-C1-checkpoint.zip") as z:
        assert "meta.json" in z.namelist()
        assert "heads.safetensors" in z.namelist()
        assert "adapter/adapter_model.safetensors" in z.namelist()
        assert "tokenizer/tokenizer.json" in z.namelist()


def test_refuse_different_source_fingerprints_or_unsafe_changed_checkpoint(tmp_path):
    c0,c1=_dataset(tmp_path)
    p=c1/"cda_l24_v2_C1_checkpoint/meta.json"
    j=json.loads(p.read_text())
    j["train_stats"]["dataset_sha256"]["trnhip/ntl-api-gen"]="f"*64
    p.write_text(json.dumps(j))
    with pytest.raises(RuntimeError,match="source SHA"):
        prepare(c0,c1,tmp_path/"package",tmp_path/"report.json")
