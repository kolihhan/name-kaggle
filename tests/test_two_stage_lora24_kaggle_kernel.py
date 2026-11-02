import ast
import json
import py_compile

from scripts.build_two_stage_lora24_kaggle_kernel import (
    BASE_DATASET_REF,
    CHECKPOINT_DATASET_REF,
    COMPETITION,
    KERNEL_REF,
    RUNTIME_DATASET_REF,
    offline_pip_args,
    write_kernel_bundle,
)


def test_offline_pip_args_never_use_an_index():
    args = offline_pip_args("/kaggle/input/runtime/wheelhouse")
    assert args[:4] == ["-m", "pip", "install", "-q"]
    assert "--no-index" in args
    assert "--find-links" in args
    assert "transformers==4.57.1" in args
    assert "peft==0.17.1" in args


def test_kernel_bundle_is_private_offline_gpu_and_exact_inputs(tmp_path):
    sources = {
        "scripts/__init__.py": "",
        "scripts/run_style_baseline.py": "X=1\n",
        "scripts/run_strong_sparse_anchor.py": "X=1\n",
        "scripts/run_two_stage_lora24_gate_a.py": "X=1\n",
        "scripts/run_two_stage_lora24_gate_b.py": "X=1\n",
        "scripts/run_two_stage_lora24_gate_b_gpu.py": "X=1\n",
        "scripts/two_stage_lora24_submission.py": "X=1\n",
    }
    out = tmp_path / "kernel"
    write_kernel_bundle(out, source_texts=sources)
    meta = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert meta["id"] == KERNEL_REF
    assert meta["is_private"] is True
    assert meta["enable_internet"] is False
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["competition_sources"] == [COMPETITION]
    assert meta["dataset_sources"] == [
        CHECKPOINT_DATASET_REF,
        BASE_DATASET_REF,
        RUNTIME_DATASET_REF,
    ]
    assert meta["kernel_sources"] == []
    assert meta["model_sources"] == []
    script = out / meta["code_file"]
    py_compile.compile(str(script), doraise=True)


def test_generated_kernel_has_no_submission_side_effect(tmp_path):
    sources = {
        "scripts/__init__.py": "",
        "scripts/run_style_baseline.py": "X=1\n",
        "scripts/run_strong_sparse_anchor.py": "X=1\n",
        "scripts/run_two_stage_lora24_gate_a.py": "X=1\n",
        "scripts/run_two_stage_lora24_gate_b.py": "X=1\n",
        "scripts/run_two_stage_lora24_gate_b_gpu.py": "X=1\n",
        "scripts/two_stage_lora24_submission.py": "X=1\n",
    }
    out = tmp_path / "kernel"
    write_kernel_bundle(out, source_texts=sources)
    source = (out / "kernel_main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not any("kaggle competitions submit" in value for value in string_literals)
    assert "/kaggle/working/submission.parquet" in source
