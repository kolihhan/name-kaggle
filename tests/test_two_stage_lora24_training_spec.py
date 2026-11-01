from __future__ import annotations

from scripts import run_two_stage_lora24_gate_a as gate


def test_gate_a_training_spec_matches_frozen_contract():
    spec = gate.gate_a_training_spec()

    assert spec == {
        "backbone": "Qwen3-8B-Base",
        "keep_layers": 24,
        "pooled_layers": [2, 4, 6, 8, 12, 16, 20, 24],
        "max_length_before_eos": 447,
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "head_dropout": 0.1,
        "epochs": 1,
        "gradient_accumulation_triplets": 6,
        "adapter_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 0.01,
        "warmup_fraction": 0.10,
        "gradient_clip": 1.0,
        "seed": 20261101,
    }


def test_gate_a_conditions_differ_only_by_lora_trainability():
    conditions = gate.gate_a_condition_specs()

    assert conditions == {
        "A0": {
            "name": "frozen_backbone_head_only",
            "train_lora": False,
        },
        "A1": {
            "name": "lora_adaptation",
            "train_lora": True,
        },
    }
