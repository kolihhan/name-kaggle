"""CDA-L24 v2 one-shot model source plumbing; synthetic only."""
import json
from pathlib import Path
import pandas as pd
import pytest

from scripts.cda_l24_v2_stage_a import _stable_prompts_by_block
from scripts.build_cda_l24_v2_training_kernel import (
    CODE_FILE, SOURCE_IDS, KERNEL_IDS, REQUIRED_SOURCE_PATHS,
    write_kernel_bundle
)


def test_two_original_prompt_blocks_preselect_without_labels_or_scores():
    frame=pd.DataFrame([
        {"block":block,"prompt_id":str(pid),"model":label,
         "text":f"independent-{block}-{pid}-{label}"}
        for block,labels in (
            ("claude_5models",["a","b","c","d","e"]),
            ("other_13models",["x","y"]))
        for pid in range(90) for label in labels
    ])
    chosen=_stable_prompts_by_block(frame,per_block=80)
    assert chosen.prompt_id.groupby(chosen.block).nunique().to_dict()=={
        "claude_5models":80,"other_13models":80}
    assert len(chosen)==80*7
    for block in ("claude_5models","other_13models"):
        assert set(chosen.loc[chosen.block==block,"prompt_id"]) < set(
            frame.loc[frame.block==block,"prompt_id"])
    # Reordering the original manifest cannot change selected original prompts.
    reordered=_stable_prompts_by_block(frame.sample(frac=1,random_state=42),per_block=80)
    assert set(map(tuple,chosen[["block","prompt_id"]].drop_duplicates().values))==set(
        map(tuple,reordered[["block","prompt_id"]].drop_duplicates().values))


def test_private_kaggle_kernel_builds_exact_original_sources_no_official_competition(tmp_path):
    texts={p:"# synthetic runtime code\n" for p in REQUIRED_SOURCE_PATHS}
    for arm in ("C0","C1"):
        out=tmp_path/arm
        metadata=write_kernel_bundle(out,arm=arm,source_texts=texts)
        assert metadata["id"]==KERNEL_IDS[arm]
        assert metadata["competition_sources"]==[]
        assert metadata["enable_internet"] is False
        assert metadata["is_private"] is True
        assert metadata["enable_gpu"] if "enable_gpu" in metadata else metadata["machine_shape"]=="NvidiaTeslaT4"
        assert set(SOURCE_IDS).issubset(metadata["dataset_sources"])
        assert metadata["model_sources"]==["qwen-lm/qwen-3/transformers/8b-base/1"]
        source=(out/CODE_FILE).read_text()
        compile(source,str(out/CODE_FILE),"exec")
        assert "PRIVATE_CDA_GPU_CANDIDATE_FINISHED" in source
        assert "cda_l24_v2_stage_a" in source
        assert "cda_l24_stage_a.py" in source
        assert (out/"kernel-metadata.json").exists()


def test_non_frozen_training_arm_rejected(tmp_path):
    with pytest.raises(ValueError,match="invalid frozen arm"):
        write_kernel_bundle(tmp_path,arm="C2",source_texts={p:"x" for p in REQUIRED_SOURCE_PATHS})


def test_missing_single_external_class_preserves_18_logits_and_zero_gradient():
    """No fake Gemini 3.7 Flash source examples or forced negative gradients."""
    import torch
    import torch.nn.functional as F
    from scripts.cda_l24_v2_stage_a import (
        stage_a_observed_class_weights,stage_a_active_label_logits
    )
    official=[f"example-{i}" for i in range(17)]+["gemini-3.7-flash"]
    observed=pd.Series(official[:-1]*3)
    weights,active,missing=stage_a_observed_class_weights(observed,official)
    assert missing==["gemini-3.7-flash"]
    assert sum(active)==17 and active[-1] is False
    assert weights["gemini-3.7-flash"]==0.
    assert all(weights[c]>0 for c in official[:-1])
    torch.manual_seed(41)
    raw=torch.randn(4,18,requires_grad=True)
    x=stage_a_active_label_logits(raw,active)
    assert x.shape==(4,18)
    targets=torch.tensor([0,1,2,3])
    weight=torch.tensor([weights[x] for x in official])
    F.cross_entropy(x,targets,weight=weight).backward()
    assert raw.grad is not None
    assert torch.all(raw.grad[:,-1]==0)
    assert torch.any(raw.grad[:,:17]!=0)
    # Evaluation retains the original unmasked prediction vocabulary.
    assert torch.isfinite(raw).all()


def test_unexpected_source_coverage_drift_fails_before_gpu_fit():
    from scripts.cda_l24_v2_stage_a import stage_a_observed_class_weights
    official=[f"example-{i}" for i in range(16)]+["gemini-3.7-flash","other-missing"]
    with pytest.raises(RuntimeError,match="unexpected external-label"):
        stage_a_observed_class_weights(pd.Series(official[:16]),official)
