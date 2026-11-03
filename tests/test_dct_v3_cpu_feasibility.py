import json

from scripts.dct_v3_cpu_feasibility import (
    KEEP_LAYERS,
    is_required_weight_key,
    select_required_shards,
    validate_probe_payload,
)


def test_required_weight_key_keeps_embedding_norm_and_first_24_layers():
    assert KEEP_LAYERS == 24
    assert is_required_weight_key("model.embed_tokens.weight")
    assert is_required_weight_key("model.norm.weight")
    assert is_required_weight_key("model.layers.0.self_attn.q_proj.weight")
    assert is_required_weight_key("model.layers.23.mlp.down_proj.weight")
    assert not is_required_weight_key("model.layers.24.self_attn.q_proj.weight")
    assert not is_required_weight_key("model.layers.35.mlp.down_proj.weight")
    assert not is_required_weight_key("lm_head.weight")


def test_select_required_shards_is_deterministic():
    weight_map = {
        "model.embed_tokens.weight": "model-00001-of-00005.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00005.safetensors",
        "model.layers.23.mlp.down_proj.weight": "model-00004-of-00005.safetensors",
        "model.layers.24.self_attn.q_proj.weight": "model-00004-of-00005.safetensors",
        "model.norm.weight": "model-00005-of-00005.safetensors",
        "lm_head.weight": "model-00005-of-00005.safetensors",
    }
    assert select_required_shards(weight_map) == [
        "model-00001-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
    ]


def test_validate_probe_payload_accepts_clean_success():
    payload = {
        "candidate": "dct-temporal-fingerprint-v3",
        "scope": "cpu_execution_feasibility_only_no_scoring",
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "holdout_predictions_computed": False,
        "model_loaded": True,
        "backward_step_completed": True,
    }
    summary = validate_probe_payload(payload)
    assert summary["scientific_decision_authorized"] is False
    assert summary["backward_step_completed"] is True


def test_validate_probe_payload_rejects_scoring_contamination():
    payload = {
        "candidate": "dct-temporal-fingerprint-v3",
        "scope": "cpu_execution_feasibility_only_no_scoring",
        "official_rows_scored": 0,
        "competition_test_rows_used": 0,
        "public_lb_tuning": False,
        "holdout_predictions_computed": True,
        "model_loaded": True,
        "backward_step_completed": True,
    }
    try:
        validate_probe_payload(payload)
    except RuntimeError as exc:
        assert "holdout_predictions_computed" in str(exc)
    else:
        raise AssertionError("expected contamination rejection")
