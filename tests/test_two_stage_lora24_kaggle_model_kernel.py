import json
import py_compile

from scripts.build_two_stage_lora24_kaggle_model_kernel import (
    KERNEL_REF,
    MODEL_SOURCE,
    offline_pip_args,
    write_kernel_bundle,
)


def _sources():
    return {
        "scripts/__init__.py": "",
        "scripts/run_style_baseline.py": "X=1\n",
        "scripts/run_strong_sparse_anchor.py": "X=1\n",
        "scripts/run_two_stage_lora24_gate_a.py": "X=1\n",
        "scripts/run_two_stage_lora24_gate_b.py": "X=1\n",
        "scripts/run_two_stage_lora24_gate_b_gpu.py": "X=1\n",
        "scripts/two_stage_lora24_submission.py": "X=1\n",
    }


def test_metadata_uses_offline_kaggle_model_source(tmp_path):
    write_kernel_bundle(tmp_path, source_texts=_sources())
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == KERNEL_REF
    assert meta["enable_internet"] is False
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["model_sources"] == [MODEL_SOURCE]
    assert meta["dataset_sources"] == [
        "kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint",
        "kolihhan/name-that-llm-two-stage-lora24-v1-runtime-py313-v2",
    ]
    py_compile.compile(str(tmp_path / meta["code_file"]), doraise=True)


def test_kernel_finds_qwen_model_and_uses_plain_final_fit(tmp_path):
    write_kernel_bundle(tmp_path, source_texts=_sources())
    text = (tmp_path / "kernel_main.py").read_text()
    assert "model_type" in text and "qwen3" in text
    assert "hidden_size" in text and "4096" in text
    assert "num_hidden_layers" in text and "36" in text
    assert "run_final_fit" in text
    assert "kaggle competitions submit" not in text
    assert "kaggle-model-source:qwen-lm/qwen-3/transformers/8b-base/1" in text


def test_offline_pip_is_index_free_and_compatible():
    args = offline_pip_args("/runtime")
    assert "--no-index" in args
    assert "--find-links" in args
    assert "--no-deps" in args
    assert "transformers==4.57.1" in args
    assert "peft==0.17.1" in args
    assert "tokenizers==0.22.1" in args
    assert "huggingface-hub==0.36.0" in args
