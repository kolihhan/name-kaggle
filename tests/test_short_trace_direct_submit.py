"""CPU-only preflight: no official data, no GPU, no submission."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts.run_short_trace_direct_submit import assert_inputs, fit_readout, LABEL_COUNT
from scripts.build_short_trace_direct_submit import build, KERNEL_REF


def test_competition_schema_guards():
    train=pd.DataFrame({
        "id":[f"train_{x}" for x in range(2699)],
        "prompt_id":[f"p{x//18}" for x in range(2699)],
        "text":["synthetic"]*2699,
        "model":[f"label_{x%18}" for x in range(2699)]})
    test=pd.DataFrame({"id":["a","b"],"text":["x","y"]})
    assert len(assert_inputs(train,test))==LABEL_COUNT
    with pytest.raises(RuntimeError):
        assert_inputs(train, pd.DataFrame({"id":["a","a"],"text":["x","y"]}))
    with pytest.raises(RuntimeError):
        assert_inputs(train.iloc[:-1],test)


def test_identical_fixed_cpu_readout():
    x=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.],[2.,0.],[2.,1.]],dtype=np.float32)
    y=np.array(["a","a","b","b","c","c"])
    a=fit_readout(x,y)
    b=fit_readout(x,y)
    assert a.named_steps["logisticregression"].C==1
    assert np.array_equal(a.predict(x),b.predict(x))


def test_competition_kernel_bundle_builds_without_model(tmp_path):
    build(Path("."),tmp_path)
    code=(tmp_path/"kernel_main.py").read_text()
    compile(code,"kernel_main.py","exec")
    metadata=json.loads((tmp_path/"kernel-metadata.json").read_text())
    assert metadata["id"]==KERNEL_REF
    assert metadata["competition_sources"]==["name-that-llm"]
    assert metadata["is_private"] and not metadata["enable_internet"]
    assert metadata["model_sources"]==["qwen-lm/qwen-3/transformers/8b-base/1"]
    assert metadata["dataset_sources"]==[
        "kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2",
        "kolihhan/ntl-short-trace-offline-bnb-0502"]
    assert "submission.parquet" in code
    assert "bitsandbytes-0.50.2" in code
    assert "SHORT_TRACE_PREVIEW_SCHEMA_ONLY_NO_8B_MODEL_INFERENCE" in code
