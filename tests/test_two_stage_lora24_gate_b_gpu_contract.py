from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pandas as pd
import pytest


MODULE = "scripts.run_two_stage_lora24_gate_b_gpu"


def _gpu():
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        pytest.fail(f"Gate B GPU runner is missing: {exc}")


def test_gate_b_gpu_runner_pins_exact_runtime_and_checkpoint_lineage():
    gpu = _gpu()
    assert gpu.MODEL_ID == "Qwen/Qwen3-8B-Base"
    assert gpu.MODEL_REVISION == "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
    assert gpu.TRANSFORMERS_VERSION == "4.57.1"
    assert gpu.PEFT_VERSION == "0.17.1"
    assert gpu.CHECKPOINT_DATASET_REF == "kolihhan/name-that-llm-two-stage-lora24-v1-a1-checkpoint"
    assert gpu.CHECKPOINT_DATASET_VERSION == 1
    assert gpu.CHECKPOINT_MANIFEST_SHA256 == "c9ba8f77cc0ec84800a8d3508c15939396353772189c573c597bdcf223eaafc0"
    assert gpu.SOURCE_ARTIFACT_ID == 13094069155
    assert gpu.SOURCE_ARTIFACT_DIGEST == "sha256:a2c4dca4ca5de2b283e5e6806526485f0d92377584d7e621b35a717770c95e66"
    assert gpu.FOLDS == (0, 1, 2, 3, 4)


def test_gate_b_gpu_runner_scope_is_official_oof_only():
    gpu = _gpu()
    assert gpu.execution_scope() == {
        "scope": "gate_b_official_oof",
        "official_oof_rows_expected": 2699,
        "public_stage1_rows_mixed_into_stage2": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "kaggle_submission_authorized": False,
    }


def test_checkpoint_loader_restores_exact_trainable_adapter_and_saved_readouts():
    gpu = _gpu()
    source = inspect.getsource(gpu.load_exact_stage1_checkpoint)
    assert "validate_gate_a_checkpoint_meta" in source
    assert "checkpoint_manifest_sha256" in source
    assert "CHECKPOINT_MANIFEST_SHA256" in source
    assert "PeftModel.from_pretrained" in source
    assert "is_trainable=True" in source
    assert "load_file" in source
    assert 'checkpoint_dir / "heads.safetensors"' in source
    assert 'checkpoint_dir / "tokenizer"' in source
    assert "labels != official_labels" in source


def test_fold_worker_uses_only_permanent_fit_partition_for_training():
    gpu = _gpu()
    source = inspect.getsource(gpu.run_fold_worker)
    assert "load_train" in source
    assert "split_official_fold" in source
    assert "fit_idx" in source
    assert "valid_idx" in source
    assert "compute_class_weights" in source
    assert "fit_frame" in source
    assert "build_prompt_triplets" in source
    assert "seed=seed" in source
    assert "public_stage1_rows_mixed_into_stage2" in source


def test_stage2_training_replays_one_triplet_order_for_exactly_two_epochs():
    gpu = _gpu()
    signature = inspect.signature(gpu.train_two_stage2_epochs)
    assert "valid_idx" not in signature.parameters
    assert "validation" not in signature.parameters
    source = inspect.getsource(gpu.train_two_stage2_epochs)
    assert "gate_b_training_spec" in source
    assert 'int(spec["epochs"])' in source
    assert "step_groups" in source
    assert "total_steps" in source
    assert "warmup_steps" in source
    assert "scheduler.step()" in source
    assert "early_stopping" not in source.lower()
    assert "best_epoch" not in source.lower()


def test_gate_b_gpu_runner_has_no_public_retraining_or_submission_path():
    gpu = _gpu()
    source = inspect.getsource(gpu)
    forbidden = [
        "pool_v4.parquet",
        "build_conservative_subset",
        "build_stage1_split",
        "kaggle competitions submit",
        "sample_submission",
        "data/test.csv",
    ]
    for text in forbidden:
        assert text not in source


def test_training_accounting_spans_both_epochs_without_padding():
    gpu = _gpu()
    accounting = gpu.stage2_training_accounting(
        triplet_microbatches_per_epoch=13,
        accumulation=6,
        epochs=2,
    )
    assert accounting == {
        "epochs": 2,
        "triplet_microbatches_per_epoch": 13,
        "triplet_microbatches_total": 26,
        "optimizer_steps_per_epoch": 3,
        "optimizer_steps_total": 6,
        "final_partial_accumulation_group_size": 1,
    }


def test_validate_fold_predictions_requires_each_validation_id_exactly_once():
    gpu = _gpu()
    valid = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "prompt_id": ["p0", "p0", "p1"],
            "model": ["a", "b", "a"],
            "fold_id": [2, 2, 2],
        }
    )
    predictions = valid.copy()
    predictions["plain_prediction"] = ["a", "b", "b"]
    predictions["centered_prediction"] = ["a", "a", "a"]
    gpu.validate_fold_predictions(valid, predictions, labels=["a", "b"], fold=2)

    duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="exactly once"):
        gpu.validate_fold_predictions(valid, duplicate, labels=["a", "b"], fold=2)

    missing = predictions.iloc[:-1].copy()
    with pytest.raises(RuntimeError, match="exactly once"):
        gpu.validate_fold_predictions(valid, missing, labels=["a", "b"], fold=2)

    invalid_label = predictions.copy()
    invalid_label.loc[0, "plain_prediction"] = "other"
    with pytest.raises(RuntimeError, match="outside label set"):
        gpu.validate_fold_predictions(valid, invalid_label, labels=["a", "b"], fold=2)


def _tiny_complete_oof() -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    official_rows = []
    fold_predictions: dict[int, pd.DataFrame] = {}
    row_id = 0
    for fold in range(5):
        rows = []
        for offset, label in enumerate(("a", "b")):
            row = {
                "id": f"id{row_id}",
                "prompt_id": f"p{fold}",
                "model": label,
                "fold_id": fold,
            }
            official_rows.append(row)
            rows.append(
                {
                    **row,
                    "plain_prediction": label,
                    "centered_prediction": "a" if offset == 0 else "b",
                }
            )
            row_id += 1
        fold_predictions[fold] = pd.DataFrame(rows)
    return pd.DataFrame(official_rows), fold_predictions


def test_combine_fold_predictions_requires_exactly_five_complete_disjoint_folds():
    gpu = _gpu()
    train, folds = _tiny_complete_oof()
    combined = gpu.combine_fold_predictions(train, folds)
    assert len(combined) == len(train)
    assert combined["id"].is_unique
    assert set(combined["fold_id"].astype(int)) == set(gpu.FOLDS)
    assert combined["plain_prediction"].notna().all()
    assert combined["centered_prediction"].notna().all()

    missing_fold = dict(folds)
    del missing_fold[4]
    with pytest.raises(RuntimeError, match="fold set"):
        gpu.combine_fold_predictions(train, missing_fold)

    duplicate = {key: value.copy() for key, value in folds.items()}
    duplicate[1] = pd.concat([duplicate[1], duplicate[1].iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="exactly once"):
        gpu.combine_fold_predictions(train, duplicate)


def test_orchestrator_launches_every_fold_in_fresh_subprocess_and_never_skips():
    gpu = _gpu()
    source = inspect.getsource(gpu.run_orchestrator)
    assert "FOLDS" in source
    assert "subprocess" in inspect.getsource(gpu)
    assert "CUDA_VISIBLE_DEVICES" in inspect.getsource(gpu)
    assert "fold_results" in source
    assert "set(fold_results) != set(FOLDS)" in source
    assert "combine_fold_predictions" in source
    assert "snapshot_download" in source


def test_fold_worker_command_is_fresh_and_explicit_per_fold():
    gpu = _gpu()
    command = gpu.fold_worker_command(
        fold=3,
        train_path=Path("train.csv"),
        folds_path=Path("folds.csv"),
        checkpoint_dir=Path("checkpoint"),
        snapshot_path=Path("snapshot"),
        output_dir=Path("output"),
    )
    assert "--fold" in command
    assert command[command.index("--fold") + 1] == "3"
    assert "--worker" in command
    assert "--checkpoint-dir" in command
    assert "--snapshot-path" in command


def test_gpu_runner_cli_supports_orchestrator_and_worker_modes():
    gpu = _gpu()
    source = Path(gpu.__file__).read_text(encoding="utf-8")
    assert 'parser.add_argument("--worker", action="store_true")' in source
    assert 'parser.add_argument("--orchestrate", action="store_true")' in source
    assert 'parser.add_argument("--fold", type=int, default=None)' in source
    assert "run_fold_worker(" in source
    assert "run_orchestrator(" in source
