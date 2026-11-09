"""Final Kaggle v2 checkpoint, frozen source SHA, and offline packaging tests."""
import hashlib,json
from pathlib import Path
import pytest

from scripts.cda_l24_v2_stage_a import (
    CANDIDATE,EXPECTED_POOL_SHA256,RESERVED_SOURCE,
)
from scripts.cda_l24_v2_checkpoint import (
    validate_cda_v2_checkpoint,directory_manifest_digest
)
from scripts.run_two_stage_lora24_gate_b_gpu import CHECKPOINT_MANIFEST_SHA256
from scripts.build_cda_l24_v2_kaggle_submission import (
    REQUIRED_SOURCE_PATHS,write_kernel_bundle,KERNEL_REF,CHECKPOINT_RESOLVER_SOURCE
)

LABELS=[f"m{i:02}" for i in range(18)]
SOURCES={
    "arxitekton/ntl-claude-gen-v1":"a"*64,
    "trnhip/ntl-api-gen":"b"*64,
    "kilupy/ntl-18-model-corpus":"c"*64,
}
SOURCE_SHA=hashlib.sha256(json.dumps(SOURCES,sort_keys=True).encode()).hexdigest()

def _mock_model(root:Path):
    for path,data in {
        "adapter/adapter_model.safetensors":b"frozen adapter placeholder",
        "adapter/adapter_config.json":b'{}',
        "heads.safetensors":b"three heads",
        "tokenizer/tokenizer.json":b"tokenizer",
        "checkpoint.sha256":b"local",
    }.items():
        p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    meta={
        "candidate":CANDIDATE,"arm":"C1",
        "source_pool_sha256":EXPECTED_POOL_SHA256,
        "base_stage1_sha256":CHECKPOINT_MANIFEST_SHA256,
        "source_transfer_reserved":RESERVED_SOURCE,
        "source_head_persisted":False,
        "provenance_manifest_sha256":SOURCE_SHA,
        "source_ids":["arxitekton/ntl-claude-gen-v1","trnhip/ntl-api-gen"],
        "labels":LABELS,
        "run_stats":{"crossed_labels":8,"transfer_labels_present":18,
                     "transfer_source":RESERVED_SOURCE},
        "train_stats":{"dataset_sha256":SOURCES},
    }
    (root/"meta.json").write_text(json.dumps(meta))
    return directory_manifest_digest(root)

def test_private_weight_manifest_and_served_model_metadata_guards(tmp_path):
    root=tmp_path/"checkpoint"
    digest=_mock_model(root)
    x=validate_cda_v2_checkpoint(root,labels=LABELS,expected_sha256=digest,
                                 arm="C1",expected_source_manifest_sha256=SOURCE_SHA)
    assert x["candidate"]==CANDIDATE
    assert x["run_stats"]["crossed_labels"]>=8
    (root/"heads.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError,match="integrity mismatch"):
        validate_cda_v2_checkpoint(root,labels=LABELS,expected_sha256=digest,
                                   arm="C1",expected_source_manifest_sha256=SOURCE_SHA)

def test_final_kernel_requires_signed_model_and_no_internet(tmp_path):
    sources={p:"# packaged source file\n" for p in REQUIRED_SOURCE_PATHS}
    metadata=write_kernel_bundle(tmp_path,stage_a_sha256="a"*64,
                                 source_sha256="b"*64,source_texts=sources)
    assert metadata["id"]==KERNEL_REF
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["competition_sources"]==["name-that-llm"]
    assert metadata["dataset_sources"]==[
        "kolihhan/ntl-cda-l24-v2-c1-weights",
        "kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2"]
    code=(tmp_path/"kernel_main.py").read_text()
    compile(code,"kernel_main.py","exec")
    assert 'cda_l24_v2_submission' in code
    assert 'FROZEN_C1_STAGE_A_SHA256' in code
    assert 'cda-v2-C1-checkpoint.zip' in code
    assert 'resolve_checkpoint_input' in code
    assert 'no_index' not in code  # real pip uses --no-index

def test_rejects_unpinned_checkpoint(tmp_path):
    with pytest.raises(ValueError,match="SHA256"):
        write_kernel_bundle(tmp_path,stage_a_sha256="unknown",
                            source_sha256="b"*64,
                            source_texts={p:"# synthetic" for p in REQUIRED_SOURCE_PATHS})

def test_kaggle_auto_expanded_checkpoint_mount_and_zip_fallback(tmp_path):
    namespace={"Path":Path}
    exec(CHECKPOINT_RESOLVER_SOURCE, namespace)
    resolve=namespace["resolve_checkpoint_input"]
    root=tmp_path/"input"
    mount=root/"ntl-cda-l24-v2-c1-weights"
    (mount/"adapter").mkdir(parents=True)
    (mount/"meta.json").write_text("{}")
    (mount/"heads.safetensors").write_bytes(b"synthetic-head")
    (mount/"adapter"/"adapter_model.safetensors").write_bytes(b"synthetic-lora")
    assert resolve(root,"ntl-cda-l24-v2-c1-weights")==("directory",mount)
    archive=root/"another-mount"/"cda-v2-C1-checkpoint.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"synthetic-zip")
    with pytest.raises(RuntimeError,match="expected one frozen checkpoint"):
        resolve(root,"ntl-cda-l24-v2-c1-weights")
    (mount/"meta.json").unlink()
    assert resolve(root,"ntl-cda-l24-v2-c1-weights")==("zip",archive)
    archive.unlink()
    with pytest.raises(RuntimeError,match="expected one frozen checkpoint"):
        resolve(root,"ntl-cda-l24-v2-c1-weights")
